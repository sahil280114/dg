import random
from supervised_dataset import (
    SupervisedDataset,
    DataCollatorForSupervisedDataset,
    make_supervised_data_module
)
from google.cloud import storage
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datetime import datetime
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import functools
import torch.distributed as dist
import wandb
import uuid
import torch
import transformers
import os
import math
import numpy as np
import time


def disable_model_dropout(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def setup_model(model_name, max_length):
    config = transformers.AutoConfig.from_pretrained(
        model_name,
    )

    config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token="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",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=4000,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        token="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",
    )

    # special_tokens_dict = dict()
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = "<|end_of_text|>"
    # special_tokens_dict["eos_token"] = "<|im_end|>"
    # # if tokenizer.unk_token is None:
    # #     special_tokens_dict["unk_token"] = "<unk>"

    # # experimental thought token
    # # special_tokens_dict["eos_token"] = "<|end_of_text|>"
    # # special_tokens_dict["additional_special_tokens"] = ["<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>", "<|end_of_text|>"]

    # tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer






def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]

    result += list(model._parameters.keys())
    return result


def get_optimizer(model, lr, weight_decay):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )


def should_run_eval(total_steps, times_to_run, current_step):
    return current_step % (total_steps // times_to_run) == 0


def log_stats(pbar, wandb, epoch, loss_tensor, grad_norm, scheduler):
    last_lr = scheduler.get_last_lr()[0]

    wandb.log(
        {
            "current_loss": loss_tensor,
            "current_epoch": epoch,
            "learning_rate": last_lr,
            "grad_norm": grad_norm,
        },
    )

    current_loss = f"{loss_tensor:.4f}"
    current_lr = f"{last_lr:.10f}"

    pbar.set_description(f"Epoch {epoch:.2f}, Loss: {current_loss}, LR: {current_lr}")


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_warmup_steps(num_training_steps, warmup_ratio=0.05):
    return math.ceil(num_training_steps * warmup_ratio)


def clip_model_gradients(model, max_grad_norm):
    return model.clip_grad_norm_(max_grad_norm).item()


def get_scheduler(local_rank, scheduler_type, optimizer, max_steps):
    warmup_steps = get_warmup_steps(max_steps)

    if local_rank == 0:
        print(f"[WARMUP STEPS]: {warmup_steps}")
        print(f"[MAX STEPS]: {max_steps}")
        print(f"[SCHEDULER]: {scheduler_type}")

    return transformers.get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )


def save_model(local_rank, model, tokenizer, outpath, current_epoch, current_step):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()

    if local_rank == 0:
        print(f"SAVING MODEL")
        outpath += f"_{current_step}/"
        model.save_pretrained(outpath, state_dict=cpu_state,max_shard_size="20GB")
        tokenizer.save_pretrained(outpath)

def get_dataloader(
    use_multipack_sampler,
    max_length,
    dataset,
    world_size,
    local_rank,
    shuffle,
    seed,
    collator,
    batch_size,
):

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=shuffle,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        batch_size=batch_size,
        collate_fn=collator,
        sampler=sampler,
    )

    return sampler, loader

def download_data_file(source_blob_name):
    """Downloads a file from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket("glaive-data")
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename("train.parquet")

def upload_blob(source_file_name,destination_blob_name):
    """
    Uploads a file to the specified Google Cloud Storage bucket.
    """
    bucket_name = "glaive-model-weights"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    url ="" #blob.generate_signed_url(datetime.timedelta(seconds=864000), method='GET')
    return url

if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcs.json"
    os.environ["WANDB_PROJECT"] = "function_calling_llama_8b"
    os.environ["WANDB_API_KEY"] = "8178101716b4f4b65d6923fd79705f245ce570ea"
    # download_data_file(os.environ["DATA_PATH"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    scheduler_type = "cosine"
    seed = 873645  # set your seed
    transformers.set_seed(seed)

    run_id = "llama_8b_func_large_special_vocab_llama_template_1"
    output_dir = f"llama_8b_func_large_special_vocab_llama_template_1"
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I_%M_%S_%p")
    max_length = 4000  # adjust as needed
    disable_dropout = False
    gradient_checkpointing = True
    clip_gradients = True
    shuffle = True  # multipack sampler already does random sampling
    train_batch_size = 2  # adjust as needed
    validation_batch_size = 4  # adjust as needed
    epochs = 3  # adjust as needed
    acc_steps = 8  # TODO: not implemented here yet
    lr = 1e-5  # adjust as needed
    weight_decay = 0.01  # adjust as needed
    gradient_clipping = 1.0  # adjust as needed
    train_on_inputs = False  # whether to train on instruction tokens
    use_multipack_sampler = (
        False  # whether to use the multipack sampler or torch sampler
    )

    model, tokenizer = setup_model(model_name, max_length)
    num_params = sum([p.numel() for p in model.parameters()])
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )

    fsdp_config = dict(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=None,
        param_init_fn=None,
        cpu_offload=None
    )

    model = FSDP(model, **fsdp_config)
    optimizer = get_optimizer(model, lr, weight_decay)

    data_module = make_supervised_data_module(tokenizer=tokenizer)

    train_dataset = data_module["train_dataset"]
    collator = data_module["data_collator"]


    train_sampler, train_loader = get_dataloader(
        use_multipack_sampler,
        max_length,
        train_dataset,
        world_size,
        local_rank,
        shuffle,
        seed,
        collator,
        train_batch_size,
    )


    total_steps_per_epoch = len(train_loader)

    max_steps = total_steps_per_epoch * epochs
    scheduler = get_scheduler(local_rank, scheduler_type, optimizer, max_steps)

    if local_rank == 0:
        run = wandb.init(
            project="function_calling_llama_8b",
            name=run_id,
            config={
                "model_name": model_name,
                "run_id": run_id,
                "date": date_of_run,
                "dataset_size": len(train_dataset),
                "weight_decay": weight_decay,
                "clip_gradients": clip_gradients,
                "learning_rate": lr,
                "shuffle": shuffle,
                "seed": seed,
                "disable_dropout": disable_dropout,
                "use_multipack_sampler": use_multipack_sampler,
                "train_on_inputs": train_on_inputs,
                "epochs": epochs,
                "acc_steps": acc_steps,
                "batch_size": train_batch_size,
                "total_batch_size": train_batch_size * world_size,
                "scheduler_type": scheduler_type,
            },
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if disable_dropout:
        disable_model_dropout(model)

    model.train()
    dist.barrier()

    for epoch in range(0, epochs):
        train_sampler.set_epoch(epoch)
        current_epoch = epoch + 1

        pbar = tqdm(
            enumerate(train_loader),
            total=total_steps_per_epoch,
            colour="blue",
            desc=f"Epoch {epoch}.00",
            disable=(local_rank != 0),
        )

        for step, batch in pbar:
            current_step = step + 1

            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "labels": batch["labels"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
            }

            # forward
            outputs = model(**inputs)
            loss = outputs.loss

            # backward
            loss.backward()

            # clipping
            if clip_gradients:
                grad_norm = clip_model_gradients(model, gradient_clipping)

            # weight update
            optimizer.step()
            scheduler.step()

            # zero gradients after weight update
            optimizer.zero_grad(set_to_none=True)

            # detach from graph
            loss = loss.detach()

            # avg loss over all processes
            loss = get_all_reduce_mean(loss).item()

            if local_rank == 0:
                log_stats(
                    pbar,
                    wandb,
                    round((current_step / total_steps_per_epoch), 2) + epoch,
                    loss,
                    grad_norm,
                    scheduler,
                )

            model.train()
        save_model(local_rank, model, tokenizer, output_dir, epochs, f"{epoch}_epochs")



    # save final model
    save_model(local_rank, model, tokenizer, output_dir, epochs, "final")
    time.sleep(30)
    # upload_blob("model_run/pytorch_model.bin",f"{run_id}.bin")
