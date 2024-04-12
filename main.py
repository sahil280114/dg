import argparse
from model import load_model,format_prompt_data_gen,inference,process_variables_batch_job,inference_with_context
import concurrent
import pandas as pd
import datetime
from google.cloud import storage
from typing import Any, Dict, List, Optional
import json
import tqdm
from seed_data import SeedDataPoint,web_documents_to_seed_data,keywords_to_seed_data,save_seed_data
from search import init_search_client,search_docs
import uuid
import random

def upload_blob(source_file_name:str)->str:
    """
    Uploads a file to the specified Google Cloud Storage bucket.
    """
    bucket_name = "glaive-data"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file_name)

    blob.upload_from_filename(source_file_name, if_generation_match=0)
    url = blob.generate_signed_url(datetime.timedelta(seconds=864000), method='GET')
    return url

def upload_artifact(source_file_name:str)->str:
    """
    Uploads a file to the specified Google Cloud Storage bucket.
    """
    bucket_name = "data_generation_artifacts"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file_name)

    blob.upload_from_filename(source_file_name, if_generation_match=0)

def generate_sample_for_web_docs(model,use_case:str,input_format:str,output_format:str,seed_data:SeedDataPoint)->Dict[str,str]:
    """
    Generate a data sample based on the use case and the document.
    """
    try:
        
        prompt = format_prompt_data_gen(use_case,input_format,output_format,keyword="",web_doc=seed_data.content)
        output = inference(model,prompt)
        split_example = output.split('-----------')
        prompt = split_example[1].strip()
        response = split_example[3].strip()
        generated_sample = {"prompt":prompt,"response":response}
        return generated_sample
    
    except Exception as e:
        print(f"Error generating sample: {e}")
        return ""
    

def generate_sample_for_keyword(model,use_case:str,input_format:str,output_format:str,seed_data:SeedDataPoint)->Dict[str,str]:
    """
    Generate a data sample based on the use case and the document.
    """
    try:

        generated_examples = []

        prompt = format_prompt_data_gen(use_case,input_format,output_format,keyword=seed_data.content,web_doc="")
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ]
        for i in range(5):
            output = inference_with_context(model,messages)
            messages.append({
                "role": "assistant",
                "content": output
            })
            generated_examples.append(output)
        processed_examples = []
        for sample in generated_examples:
            split_example = sample.split('-----------')
            prompt = split_example[1].strip()
            response = split_example[3].strip()
            generated_sample = {"prompt":prompt,"response":response}
            processed_examples.append(generated_sample)
        return processed_examples
    
    except Exception as e:
        print(f"Error generating sample: {e}")
        return ""

def generate_and_filter_data(model,public_id:str, use_case:str, input_format:str, output_format:str, keywords:List[SeedDataPoint],web_docs:List[SeedDataPoint],debug_mode:bool=False):
    """
    Main function to generate and filter data based on the cosine similarity of the responses.
    """
    try: 
        result_samples = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
            tasks = [executor.submit(generate_sample_for_web_docs,model, use_case,input_format,output_format,seed_data) for seed_data in web_docs]
            for future in tqdm.tqdm(concurrent.futures.as_completed(tasks),total=len(tasks),desc="Generating samples web"):
                result = future.result()
                if result!="":
                    result_samples.append(result)
        with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
            tasks = [executor.submit(generate_sample_for_keyword,model, use_case,input_format,output_format,seed_data) for seed_data in keywords]
            for future in tqdm.tqdm(concurrent.futures.as_completed(tasks),total=len(tasks),desc="Generating samples keywords"):
                result = future.result()
                if result!="":
                    result_samples.extend(result)
        random.shuffle(result_samples)
        # Processing of data samples
        prompts, responses = [], []
        for sample in result_samples:
            if "glaive_variable" in sample["prompt"] or "glaive_variable" in sample["response"] or "glaive_input" in sample["prompt"]:
                continue
            try:
                prompts.append(sample['prompt'])
                responses.append(sample['response'])
            except Exception as e:
                print(f"Error processing sample: {e}")

        df = pd.DataFrame({'prompt': prompts, 'response': responses}).drop_duplicates()

        df = df.drop_duplicates()

        df.to_parquet(f'{public_id}.parquet',index=False)

        if not debug_mode:
            url = upload_blob(f'{public_id}.parquet')
            print("Data uploaded to ", url)
    except Exception as e:
        print(f"Data generation error: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate training data for AI models.')
    parser.add_argument("--id", type=str, required=True, help="ID of the task")
    parser.add_argument('--use_case', type=str, required=True, help='Use case for the AI model')
    parser.add_argument('--input_variables', type=str, required=False,default=None, help='Input variables for the model')
    parser.add_argument('--output_variables', type=str, required=False,default=None, help='Output variables for the model')
    parser.add_argument('--input_template', type=str, required=True,default=None, help='Input format for the model')
    parser.add_argument('--output_template', type=str, required=True,default=None, help='Output format for the model')
    parser.add_argument('--keywords', type=str, default=None, required=True,help='Keywords for doc search')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--output_complexity', type=int, required=False, help='Complexity to add to the data samples')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')
    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.input_variables is None:
        full_input = args.input_template
    else:
        try:
            input_variables = json.loads(args.input_variables)["variables"]
            full_input = process_variables_batch_job(args.input_template, input_variables)
        except:
            full_input = args.input_template
    if args.output_variables is None:
        full_output = args.output_template
    else:
        try:
            output_variables = json.loads(args.output_variables)["variables"]
            full_output = process_variables_batch_job(args.output_template, output_variables)
        except:
            full_output = args.output_template
   
    model = load_model()
    search_client = init_search_client()

    keywords_list = args.keywords.split(",")
    seed_data_keywords = keywords_to_seed_data(model,args.use_case,keywords_list)
    web_documents = []
    for keyword in keywords_list:
        result = search_docs(search_client,keyword)
        web_documents.extend(result)
    seed_data_web_docs = web_documents_to_seed_data(web_documents)

    full_seed_data = seed_data_keywords + seed_data_web_docs
    if args.debug:
        save_seed_data(full_seed_data, args.id)
    
    generate_and_filter_data(model,args.id,args.use_case, full_input, full_output, seed_data_keywords,seed_data_web_docs ,args.debug)

    artifact = {
        "id": args.id,
        "use_case": args.use_case,
        "input_template": full_input,
        "output_template": full_output,
        "seed_data": [data_point.to_dict() for data_point in full_seed_data],
    }

    uuid4 = uuid.uuid4()
    with open(f'{uuid4}_artifact.json', 'w') as f:
        json.dump(artifact, f, indent=4)
    upload_artifact(f'{uuid4}_artifact.json')

if __name__ == "__main__":
    main()



