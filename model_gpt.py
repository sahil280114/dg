from typing import Any, Dict
from openai import OpenAI

def load_model():
    """
    Loads the specified LLM model.

    :param model_name_or_path: The name or path of the model to load.
    :return: Loaded LLM model.
    """
    client = OpenAI(
    api_key="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"
    )
    return client
def extract_description(data, key):
    try:
        return data[key]["templated_description"]
    except:
        return ""
def extract_type(data, key):
    try:
        return data[key]["type"]
    except:
        return ""
def extract_size(data, key):
    try:
        return data[key]["size"]
    except:
        return ""

def assemble_variable(description, type, size):
    return f"<glaive_variable> <description>{description}</description><type>{type}</type><size>{size}</size></glaive_variable>"

def assemble_description(template, data):
    if template == "":
        for key in data.keys():
            description = extract_description(data, key)
            typee = extract_type(data, key)
            size = extract_size(data, key)
            assembled_variable = assemble_variable(description, typee, size)
            template += assembled_variable + "\n"
        return template
    for key in data.keys():
        description = extract_description(data, key)
        typee = extract_type(data, key)
        size = extract_size(data, key)
        assembled_variable = assemble_variable(description, typee, size)
        template = template.replace(f"{{{{{key}}}}}", assembled_variable)
    return template
def format_prompt(use_case: str,input_format:str,output_format:str,keyword:str="") -> str:
    """
    Formats a prompt for generating search terms related to a specific use case.

    :param use_case: A description of the use case for which search terms are generated.
    :return: A formatted prompt string.
    """
    if keyword!="":
        base_prompt = f'''You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train which contains information about the use case and requirements,the format of inputs, the format of outputs and from that, you will generate one data sample, with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model. Change the diversity of output each turn based on previous ones.\n\nHere is the use case of the model we want to train:\n```{use_case}```.\n Here's the input format:\n```{input_format}```\n Here's the output format:\n```{output_format}```. \n The xml tags with <glaive_variable> are variables where you will put generated information of the given type and size. Do not include those tags in samples. Use the following subtopic - {keyword}'''
        return base_prompt
    base_prompt = f'''You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train which contains information about the use case and requirements,the format of inputs, the format of outputs and from that, you will generate one data sample, with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model. Change the diversity of output each turn based on previous ones.\n\nHere is the use case of the model we want to train:\n```{use_case}```.\n Here's the input format:\n```{input_format}```\n Here's the output format:\n```{output_format}```. \n The xml tags with <glaive_variable> are variables where you will put generated information of the given type and size. Do not include those tags in samples.'''
    #full_prompt = f'[INST] {base_prompt} [/INST]'
    return base_prompt

def inference(client, prompt: str) -> str:
    """
    Generates an inference from the model based on the given prompt and sampling parameters.

    :param model: The LLM model to use for inference.
    :param prompt: The prompt to provide to the model.
    :param sampling_params: A dictionary of parameters to control the sampling of the model's response.
    :return: The model's generated output as text.
    """
    messages=[
            {
                "role": "system",
                "content": prompt
            }
        ]
    output =  client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
            top_p=1,
        )
    return output.choices[0].message.content
