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

def process_variables(string_template, variables_dict):
    result = string_template

    for key, value in variables_dict.items():
        glaive_variable = f"<glaive_variable><name>{value['name']}</name><description>{value['description']}</description>"
        type_var = list(value['type']["Type"].keys())[0].lower()
        if 'enum' in type_var:
            enum_values = value['type']["Type"]
            glaive_variable += f"<type>{enum_values}</type>"
        elif 'string' in type_var:
            type_details = value['type']["Type"]
            glaive_variable += f"<type>{type_details}</type>"
        elif 'int32' in type_var:
            type_details = value['type']["Type"]
            glaive_variable += f"<type>{type_details}</type>"
        elif 'float' in type_var:
            type_details = value['type']["Type"]
            glaive_variable += f"<type>{type_details}</type>"
        elif 'bool' in type_var:
            glaive_variable += "<type>Bool</type>"
        elif 'array' in type_var:
            array_variable = value['type']["Type"]['Array']['variable']
            array_type = array_variable['type']
            array_type_str = next(iter(array_type.keys()))
            array_size = value['type']['array'].get('size', {})
            min_size = array_size.get('min')
            max_size = array_size.get('max')
            size_str = f"min={min_size} max={max_size}" if min_size is not None and max_size is not None else ""
            glaive_variable += f"<type>Array</type><arrayType>{array_type_str}</arrayType><size>{size_str}</size>"
        elif 'struct' in type_var:
            struct_fields = value['type']["Type"]['struct']['fields']
            force_json = value['type']['struct'].get('force_json', False)
            struct_fields_str = ""
            for field_name, field_variable in struct_fields.items():
                field_type = field_variable['type']
                field_type_str = next(iter(field_type.keys()))
                struct_fields_str += f"<field><name>{field_name}</name><type>{field_type_str}</type></field>"
            glaive_variable += f"<type>Struct</type><fields>{struct_fields_str}</fields><forceJson>{force_json}</forceJson>"

        glaive_variable += "</glaive_variable>"
        result = result.replace(f"{{{{{key}}}}}", glaive_variable)

    return result

def format_prompt_keyword_expansion(use_case: str, keyword: str) -> str:
    prompt = f'''We are building a training set for an AI model for the use case - {use_case} . For this we have come up with the following keyword - {keyword}. Generate a list of subtopics for this keyword which cover all possibilites. Provide as json in this format {{'search_terms': []}}. Do not return anything else. Utilise depth first search internally'''
    return prompt

def format_prompt_data_gen(use_case: str,input_format:str,output_format:str,keyword:str="",web_doc:str="") -> str:
    """
    Formats a prompt for generating search terms related to a specific use case.

    :param use_case: A description of the use case for which search terms are generated.
    :return: A formatted prompt string.
    """
    if keyword!="":
        base_prompt = f'''You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train which contains information about the use case and requirements,the format of inputs, the format of outputs and from that, you will generate one data sample, with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model. Change the diversity of output each turn based on previous ones.\n\nHere is the use case of the model we want to train:\n```{use_case}```.\n Here's the input format:\n```{input_format}```\n Here's the output format:\n```{output_format}```. \n The xml tags with <glaive_variable> are variables where you will put generated information of the given type and size. Do not include those tags in samples. Wherever possbile generate a diversity of variable types available in enums. In this context, generate sample on the following topic within the usecase- {keyword}.\nDo not repeat previous generated samples, make it different'''
        return base_prompt
    elif web_doc!="":
        base_prompt = f'''You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train which contains information about the use case and requirements,the format of inputs, the format of outputs and from that, you will generate one data sample, with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model. Change the diversity of output each turn based on previous ones.\n\nHere is the use case of the model we want to train:\n```{use_case}```.\n Here's the input format:\n```{input_format}```\n Here's the output format:\n```{output_format}```. \n The xml tags with <glaive_variable> are variables where you will put generated information of the given type and size. Do not include those tags in samples. Wherever possbile generate a diversity of variable types available in enums. Use the following text from a website to generate the sample, if web text is irrelevant to the usecase then ignore it. - {web_doc}.'''
        return base_prompt
    base_prompt = f'''You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train which contains information about the use case and requirements,the format of inputs, the format of outputs and from that, you will generate one data sample, with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model. Change the diversity of output each turn based on previous ones.\n\nHere is the use case of the model we want to train:\n```{use_case}```.\n Here's the input format:\n```{input_format}```\n Here's the output format:\n```{output_format}```. \n The xml tags with <glaive_variable> are variables where you will put generated information of the given type and size. Do not include those tags in samples. Wherever possbile generate a diversity of variable types.'''
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

def inference_with_context(client, messages) -> str:
    """
    Generates an inference from the model based on the given prompt and sampling parameters.

    :param model: The LLM model to use for inference.
    :param prompt: The prompt to provide to the model.
    :param sampling_params: A dictionary of parameters to control the sampling of the model's response.
    :return: The model's generated output as text.
    """
    output =  client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
            top_p=1,
        )
    return output.choices[0].message.content