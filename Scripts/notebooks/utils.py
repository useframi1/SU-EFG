import numpy as np
import requests
import inspect


def log_transform(x):
    return np.log(np.abs(x.flatten()) + 1)


API_URL = "http://nexusraven.nexusflow.ai"
headers = {"Content-Type": "application/json"}


def query(payload):
    """
    Sends a payload to a TGI endpoint.
    """
    import requests

    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def query_raven(prompt):
    """
    This function sends a request to the TGI endpoint to get Raven's function call.
    This will not generate Raven's justification and reasoning for the call, to save on latency.
    """

    output = query(
        {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.001,
                "stop": ["<bot_end>"],
                "do_sample": False,
                "max_new_tokens": 2000,
                "return_full_text": False,
            },
        }
    )
    call = output[0]["generated_text"].replace("Call:", "").strip()
    return call


def query_raven_with_reasoning(prompt):
    """
    This function sends a request to the TGI endpoint to get Raven's function call AND justification for the call
    """

    output = query(
        {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.001,
                "do_sample": False,
                "max_new_tokens": 2000,
                "return_full_text": False,
            },
        }
    )
    call = output[0]["generated_text"].replace("Call:", "").strip()
    return call


def build_raven_prompt(function_list, user_query):
    raven_prompt = ""
    for function in function_list:
        signature = inspect.signature(function)
        docstring = function.__doc__
        prompt = f'''
Function:
def {function.__name__}{signature}
    """
    {docstring.strip()}
    """
    
'''
        raven_prompt += prompt

    raven_prompt += f"User Query: {user_query}<human_end>"
    return raven_prompt
