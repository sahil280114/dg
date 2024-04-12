from flask import Flask, request, jsonify, Response, stream_with_context
import os
import concurrent
from model import load_model,format_prompt_data_gen,inference,process_variables
import random
from search import init_search_client,search_docs
from seed_data import save_seed_data

app = Flask(__name__)

def generate_sample_for_web_docs(model,use_case:str,input_format:str,output_format:str,web_doc):
    """
    Generate a data sample based on the use case and the document.
    """
    try:
        
        prompt = format_prompt_data_gen(use_case,input_format,output_format,keyword="",web_doc=web_doc)
        output = inference(model,prompt)
        split_example = output.split('-----------')
        prompt = split_example[1].strip()
        response = split_example[3].strip()
        generated_sample = {"prompt":prompt,"response":response}
        return generated_sample
    
    except Exception as e:
        print(f"Error generating sample: {e}")
        return ""
def generate_sample_for_keyword(model,use_case:str,input_format:str,output_format:str,keyword):
    """
    Generate a data sample based on the use case and the document.
    """
    try:
        prompt = format_prompt_data_gen(use_case,input_format,output_format,keyword=keyword,web_doc="")
        output = inference(model,prompt)
        split_example = output.split('-----------')
        prompt = split_example[1].strip()
        response = split_example[3].strip()
        generated_sample = {"prompt":prompt,"response":response}
        return generated_sample
        
    
    except Exception as e:
        print(f"Error generating sample: {e}")
        return ""

def genereate_data(model,use_case, input_format, output_format,num_samples,keywords,web_docs):
    print("Generating {} samples".format(num_samples),flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            # Create a list of tasks
            tasks = [executor.submit(generate_sample_for_web_docs,model,use_case,input_format,output_format,web_doc) for web_doc in web_docs]
            for future in concurrent.futures.as_completed(tasks):
                # Process results here
                result = future.result()
                if result != "":
                    yield result
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            # Create a list of tasks
            tasks = [executor.submit(generate_sample_for_keyword,model,use_case,input_format,output_format,keyword) for keyword in keywords]
            for future in concurrent.futures.as_completed(tasks):
                # Process results here
                result = future.result()
                if result != "":
                    yield result
    except Exception as e:
        yield "Error"
                    


@app.route('/generate_data', methods=['POST'])
def generate_data():
    if request.is_json:

        model = load_model()

        data = request.get_json()
        dataset_parameters = data["parameters"]

        num_samples = data.get("limit", 15)
        use_case = dataset_parameters["description"]
        keywords = dataset_parameters["knowledge"]["keyphrases"]

        input_template = dataset_parameters["input"]["template"]
        output_template = dataset_parameters["output"]["template"]

        try:
            input_variables = dataset_parameters["input"]["variables"]["variables"]
        except:
            input_variables = None
        try:    
            output_variables = dataset_parameters["output"]["variables"]["variables"]
        except:
            output_variables = None
        #Assemble input and output format  
        if input_variables is None:
            input_format = input_template
        else:
            input_format = process_variables(input_template, input_variables)   
        if output_variables is None:
            output_format = output_template
        else:
            output_format = process_variables(output_template, output_variables)

        if num_samples<len(keywords):
            keywords = random.choices(keywords,k=num_samples)   
        elif num_samples>len(keywords):
            num_samples = len(keywords)

        
        mid_index = int(len(keywords) / 2)

        # Split the list into two halves
        seed_keywords = keywords[:mid_index]
        seed_for_web_docs = keywords[mid_index:]

        seed_web_docs = []

        for seed in seed_for_web_docs:
            seed_web_docs.extend(search_docs(init_search_client(),seed,num_results=1))

        data_generator = genereate_data(model,use_case,input_format,output_format,num_samples,seed_keywords,seed_web_docs)

        try:
            def generate():
                for sample in data_generator:
                    try:
                        if "glaive_variable" in sample["prompt"] or "glaive_variable" in sample["response"] or "glaive_input" in sample["prompt"]:
                            continue
                        response_object= f'''<glaive_input>{sample["prompt"]}</glaive_input><glaive_output>{sample["response"]}</glaive_output><|endofsample|>'''
                    except:
                        continue
                    yield response_object
            return Response(stream_with_context(generate()), mimetype='application/json')
        except Exception as e:
            print(e)
            return jsonify({"error": "Error generating data"}), 500
    else:
        return jsonify({"error": "Request must be JSON"}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5090))
    app.run(debug=False, host='0.0.0.0',port=port)