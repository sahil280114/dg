import argparse
from edit_types import EditType, EditSamplesAction
from google.cloud import storage
import pandas as pd
import json
from model import load_model,inference,format_prompt_conditional_removal,format_prompt_data_gen_conditional,format_prompt_data_gen,inference_with_context,format_prompt_edit_schema,process_variables_batch_job
import concurrent
import tqdm
import datetime
from seed_data import SeedDataPoint,web_documents_to_seed_data,keywords_to_seed_data,save_seed_data
from search import init_search_client,search_docs
import random
from typing import List
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

def download_data_file(source_blob_name):
    """Downloads a file from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket("glaive-data")
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename("parent.parquet")

def read_parquet_to_dict_list(file_path):
    """Load a .parquet file into a list of dictionaries."""
    df = pd.read_parquet(file_path)
    dict_list = df.to_dict('records')
    return dict_list



def check_condition(model,sample,condition):
    try:
        prompt = format_prompt_conditional_removal(sample,condition)
        response = inference(model,prompt)
        if "true" in response or "True" in response:
            return {"sample":sample,"condition":True}
        else:
            return {"sample":sample,"condition":False}
    except:
        return {"sample":sample,"condition":False}
    
def conditional_removal(model,parent_data:List[dict],condition,scope):
    """
    loop through inut and output
    check for condition
    delete if true
    retu new data"""
    edited_dataset = []
    removed_samples = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
        tasks = [executor.submit(check_condition,model,sample,condition) for sample in parent_data]
        for future in tqdm.tqdm(concurrent.futures.as_completed(tasks),total=len(tasks),desc="Filtering samples"):
            result = future.result()
            edited_dataset.append(result)

    if scope == -1:
        return_dataset = []
        for item in edited_dataset:
            if item["condition"] == False:
                return_dataset.append(item["sample"])
        return return_dataset
    else:
        return_dataset = []
        for item in edited_dataset:
            if item["condition"] == False and removed_samples < scope:
                return_dataset.append(item["sample"])
            else:
                removed_samples += 1
        return return_dataset


def generate_sample_for_web_docs(model,use_case:str,input_format:str,output_format:str,seed_data:SeedDataPoint):
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
    

def generate_sample_for_keyword(model,use_case:str,input_format:str,output_format:str,seed_data:SeedDataPoint):
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

def knowledge_addition(model,parent_data,use_case,input_format,output_format,new_keywords):
    """
    Run search + data generation on new keywords similat to main.py
    """
    search_client = init_search_client()
    seed_data_keywords = keywords_to_seed_data(model,use_case,new_keywords)
    web_documents = []
    for keyword in new_keywords:
        result = search_docs(search_client,keyword)
        web_documents.extend(result)
    seed_data_web_docs = web_documents_to_seed_data(web_documents)
    result_samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
        tasks = [executor.submit(generate_sample_for_web_docs,model, use_case,input_format,output_format,seed_data) for seed_data in seed_data_web_docs]
        for future in tqdm.tqdm(concurrent.futures.as_completed(tasks),total=len(tasks),desc="Generating samples web"):
            result = future.result()
            if result!="":
                result_samples.append(result)
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
        tasks = [executor.submit(generate_sample_for_keyword,model, use_case,input_format,output_format,seed_data) for seed_data in seed_data_keywords]
        for future in tqdm.tqdm(concurrent.futures.as_completed(tasks),total=len(tasks),desc="Generating samples keywords"):
            result = future.result()
            if result!="":
                result_samples.extend(result)
    random.shuffle(result_samples)
    parent_data.extend(result_samples)
    return parent_data
    

def convert_and_upload(edited_dataset,public_id):
    prompts, responses = [], []
    for sample in edited_dataset:
        try:
            prompts.append(sample['prompt'])
            responses.append(sample['response'])
        except Exception as e:
            print(f"Error processing sample: {e}")

    df = pd.DataFrame({'prompt': prompts, 'response': responses}).drop_duplicates()

    df = df.drop_duplicates()

    df.to_parquet(f'{public_id}.parquet',index=False)
    url = upload_blob(f'{public_id}.parquet')
    print("Data uploaded to ", url)
def generate_sample_for_web_docs_condition(model,use_case:str,input_format:str,output_format:str,seed_data:SeedDataPoint,condition:str):
    """
    Generate a data sample based on the use case and the document.
    """
    try:
        
        prompt = format_prompt_data_gen_conditional(use_case,input_format,output_format,keyword="",web_doc=seed_data.content,condition=condition)
        output = inference(model,prompt)
        split_example = output.split('-----------')
        prompt = split_example[1].strip()
        response = split_example[3].strip()
        generated_sample = {"prompt":prompt,"response":response}
        return generated_sample
    
    except Exception as e:
        print(f"Error generating sample: {e}")
        return ""
    

def generate_sample_for_keyword_condition(model,use_case:str,input_format:str,output_format:str,seed_data:SeedDataPoint,condition:str):
    """
    Generate a data sample based on the use case and the document.
    """
    try:

        generated_examples = []

        prompt = format_prompt_data_gen_conditional(use_case,input_format,output_format,keyword=seed_data.content,web_doc="",condition=condition)
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ]
        output = inference_with_context(model,messages)
            
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

def conditional_addition(model,parent_data,use_case,input_format,output_format,condition,scope,keywords):
    search_client = init_search_client()
    keywords = keywords.split(",")
    seed_data_keywords = keywords_to_seed_data(model,condition,keywords)
    web_documents = []
    for keyword in keywords:
        result = search_docs(search_client,keyword)
        web_documents.extend(result)
    seed_data_web_docs = web_documents_to_seed_data(web_documents)
    half_scope = int(scope/2)
    seed_keywords = seed_data_keywords[:half_scope]
    seed_web_docs = seed_data_web_docs[:half_scope]
    result_samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
        tasks = [executor.submit(generate_sample_for_web_docs_condition,model, use_case,input_format,output_format,seed_data,condition) for seed_data in seed_web_docs]
        for future in tqdm.tqdm(concurrent.futures.as_completed(tasks),total=len(tasks),desc="Generating samples web"):
            result = future.result()
            if result!="":
                result_samples.append(result)
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
        tasks = [executor.submit(generate_sample_for_keyword_condition,model, use_case,input_format,output_format,seed_data,condition) for seed_data in seed_keywords]
        for future in tqdm.tqdm(concurrent.futures.as_completed(tasks),total=len(tasks),desc="Generating samples keywords"):
            result = future.result()
            if result!="":
                result_samples.extend(result)
    random.shuffle(result_samples)
    parent_data.extend(result_samples)
    return parent_data

def change_schema(model,sample,old_input_schema,old_output_schema,new_input_schema,new_output_schema):

    try:
        prompt = format_prompt_edit_schema(sample["prompt"],old_input_schema,old_output_schema,new_input_schema,new_output_schema)
        edited_input = inference(model,prompt)

        prompt = format_prompt_edit_schema(sample["response"],old_input_schema,old_output_schema,new_input_schema,new_output_schema)
        edited_output = inference(model,prompt)
        return {"prompt":edited_input,"response":edited_output}
    except Exception as e:
        print(f"Error generating sample: {e}")
        return ""

def edit_schema(model,parent_data,old_input_schema,old_output_schema,new_input_schema,new_output_schema):
    result_samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
        tasks = [executor.submit(change_schema,model,sample,old_input_schema,old_output_schema,new_input_schema,new_output_schema) for sample in parent_data]
        for future in tqdm.tqdm(concurrent.futures.as_completed(tasks),total=len(tasks),desc="Editing schema"):
            result = future.result()
            if result!="":
                result_samples.append(result)
    return result_samples
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate training data for AI models.')
    parser.add_argument("--id", type=str, required=True, help="ID of the new dataset")
    parser.add_argument("--parent_id", type=str, required=True, help="parent dataset")
    parser.add_argument('--edit_type', type=int, required=True, help='Edit type for the task')
    parser.add_argument('--edit_parameters', type=str, required=False, help='edit parameters as json string')
    return parser.parse_args()

def main():
    args = parse_arguments()
    edit_type = EditType(args.edit_type)
    download_data_file(f"{args.parent_id}.parquet")
    parent_data = read_parquet_to_dict_list("parent.parquet")
    edit_parameters = json.loads(args.edit_parameters)
    
    model = load_model()
    print(edit_type.name)
    print(edit_parameters)
    

    if edit_type == EditType.AddKnowledge:
        new_keywords = edit_parameters["keywords"]
        use_case = edit_parameters["use_case"]
        input_format = edit_parameters["input_template"]
        output_format = edit_parameters["output_template"]
        input_variables = edit_parameters.get("input_variables",None)
        output_variables = edit_parameters.get("output_variables",None)
        if input_variables is None:
            full_input = input_format
        else:
            try:
                input_variables = json.loads(args.input_variables)["variables"]
                full_input = process_variables_batch_job(input_format, input_variables)
            except:
                full_input = input_format
        if output_variables is None:
            full_output = output_format
        else:
            try:
                output_variables = json.loads(args.output_variables)["variables"]
                full_output = process_variables_batch_job(output_format, output_variables)
            except:
                full_output = output_format
        edited_dataset = knowledge_addition(model,parent_data,use_case,full_input,full_output,new_keywords)
        convert_and_upload(edited_dataset,args.id)

    elif edit_type == EditType.EditSchema:
        old_input_template = edit_parameters["old_input_template"]
        old_input_variables = edit_parameters.get("old_input_variables",None)
        if old_input_variables is None:
            full_old_input = old_input_template
        else:
            try:
                old_input_variables = old_input_variables["variables"]
                full_old_input = process_variables_batch_job(old_input_template, old_input_variables)
            except:
                full_old_input = old_input_template

        old_output_template = edit_parameters["old_output_template"]
        old_output_variables = edit_parameters.get("old_output_variables",None)
        if old_output_variables is None:
            full_old_output = old_output_template
        else:
            try:
                old_input_variables = old_output_variables["variables"]
                full_old_output = process_variables_batch_job(old_output_template, old_output_variables)
            except:
                full_old_output = old_output_template

        new_input_template = edit_parameters["new_input_template"]
        new_input_variables = edit_parameters.get("new_input_variables",None)
        if new_input_variables is None:
            full_new_input = new_input_template
        else:
            try:
                new_input_variables = new_input_variables["variables"]
                full_new_input = process_variables_batch_job(new_input_template, new_input_variables)
            except:
                full_new_input = old_input_template
        

        new_output_variables = edit_parameters.get("new_output_variables",None)
        new_output_template = edit_parameters["new_output_template"]
        if new_output_variables is None:
            full_new_output = new_output_template
        else:
            try:
                new_input_variables = new_output_variables["variables"]
                full_new_output = process_variables_batch_job(new_output_template, new_output_variables)
            except:
                full_new_output = new_output_template

        edited_dataset = edit_schema(model,parent_data,full_old_input,full_old_output,full_new_input,full_new_output)
        convert_and_upload(edited_dataset,args.id)
    
    elif edit_type == EditType.UpdateSamples:
        action = EditSamplesAction(edit_parameters["action"])
        scope = int(edit_parameters["scope"]) # either num samples, or -1 for all samples
        condition = edit_parameters.get("condition") #string
        if action == EditSamplesAction.Remove:
            edited_dataset = conditional_removal(model,parent_data,condition,scope)
            convert_and_upload(edited_dataset,args.id)
        else:
            keywords = edit_parameters["keywords"]
            use_case = edit_parameters["use_case"]
            input_template = edit_parameters["input_template"]
            input_variables = edit_parameters.get("input_variables",None)
            if input_variables is None:
                full_input = input_template
            else:
                try:
                    input_variables = input_variables["variables"]
                    full_input = process_variables_batch_job(input_template, input_variables)
                except:
                    full_input = input_template

            output_template = edit_parameters["output_template"]
            output_variables = edit_parameters.get("output_variables",None)
            if output_variables is None:
                full_output = output_template
            else:
                try:
                    output_variables = output_variables["variables"]
                    full_output = process_variables_batch_job(output_template, output_variables)
                except:
                    full_output = output_template
            edited_dataset = conditional_addition(model,parent_data,use_case,full_input,full_output,condition,scope,keywords)
            convert_and_upload(edited_dataset,args.id)

if __name__ == "__main__":
    main()