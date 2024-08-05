import numpy as np
import requests
import json
from dotenv import load_dotenv
import os

_ = load_dotenv()


def log_transform(x):
    return np.log(np.abs(x.flatten()) + 1)


def get_client_data(
    age: int = None,
    risk_rate: str = None,
    gender: str = None,
    completed_orders: str = None,
    canceled_orders: str = None,
    average_price: float = None,
    most_frequent_order_type: str = None,
    most_frequent_execution_status: str = None,
    most_frequent_sector_name: str = None,
    avg_order_rate_difference: str = None,
    avg_order_quantity_rate_difference: str = None,
) -> dict:
    """
    Constructs a dict with all the client details specified in the user query.
    """
    provided_params = locals().copy()
    provided_params["gender"] = 1 if provided_params["gender"] == "Male" else 0
    key_mapping = {
        "age": "Age",
        "gender": "IsMale",
        "average_price": "AvgPrice",
        "risk_rate": "RiskRate",
        "avg_order_rate_difference": "AvgOrderRate_Difference",
        "avg_order_quantity_rate_difference": "AvgQuantityOrderedRate_Difference",
        "completed_orders": "CompletedOrdersRatio",
        "canceled_orders": "CanceledOrdersRatio",
        "most_frequent_order_type": "Most_Frequent_OrderType",
        "most_frequent_execution_status": "Most_Frequent_ExecutionStatus",
        "most_frequent_sector_name": "Most_Frequent_SectorName",
    }

    for old_key, new_key in key_mapping.items():
        if old_key in provided_params:
            provided_params[new_key] = provided_params.pop(old_key)

    try:
        response = requests.post(
            url=os.getenv("MODEL_API_BASE_URL") + "/predict", json=provided_params
        )
        if response.status_code == 200:
            data = response.json()
            if "prediction" in data.keys():
                if data["prediction"] == 1:
                    return json.dumps({"prediction": "The client will churn"})
                elif data["prediction"] == 0:
                    return json.dumps({"prediction": "The client will not churn"})
    except Exception as e:
        return json.dumps({"error": e.args[0]})


# API_URL = "http://nexusraven.nexusflow.ai"
# headers = {"Content-Type": "application/json"}


# def query(payload):
#     """
#     Sends a payload to a TGI endpoint.
#     """
#     import requests

#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()


# def query_raven(prompt):
#     """
#     This function sends a request to the TGI endpoint to get Raven's function call.
#     This will not generate Raven's justification and reasoning for the call, to save on latency.
#     """

#     output = query(
#         {
#             "inputs": prompt,
#             "parameters": {
#                 "temperature": 0.001,
#                 "stop": ["<bot_end>"],
#                 "do_sample": False,
#                 "max_new_tokens": 2000,
#                 "return_full_text": False,
#             },
#         }
#     )
#     call = output[0]["generated_text"].replace("Call:", "").strip()
#     return call


# def query_raven_with_reasoning(prompt):
#     """
#     This function sends a request to the TGI endpoint to get Raven's function call AND justification for the call
#     """

#     output = query(
#         {
#             "inputs": prompt,
#             "parameters": {
#                 "temperature": 0.001,
#                 "do_sample": False,
#                 "max_new_tokens": 2000,
#                 "return_full_text": False,
#             },
#         }
#     )
#     call = output[0]["generated_text"].replace("Call:", "").strip()
#     return call


# def build_raven_prompt(function_list, conversation, user_input):
#     raven_prompt = ""
#     for function in function_list:
#         signature = inspect.signature(function)
#         docstring = function.__doc__
#         prompt = f'''
# Function:
# def {function.__name__}{signature}
#     """
#     {docstring.strip()}
#     """

# '''
#         raven_prompt += prompt

#     raven_prompt += f"""
# {conversation}
# User input:{user_input}<human_end>
# """
#     return raven_prompt


# def get_age(age: int) -> int:
#     """
#     Validates and returns the age value.

#     Args:
#         age (int): The age of the client. Must be a non-negative integer. If not provided, defaults to None.

#     Returns:
#         int: The validated age. If no age is provided, None is returned.
#     """
#     if age is not None and age < 0:
#         return None

#     return age


# def get_gender(gender: str) -> str:
#     """
#     Will return the gender of the client.

#     Args:
#         gender (str): the gender of the client. Can only be one of the following: 'Male', 'Female'. If not provided, defaults to None.

#     Returns:
#         str: The gender of the client.
#     """

#     if gender is not None and gender not in ["Male", "Female"]:
#         return None

#     return gender


# def get_most_frequent_sector_name(most_frequent_sector_name: str) -> str:
#     """
#     Will return the most frequent sector that the clients orders are in.

#     Args:
#         most_frequent_sector_name (str): The most frequent sector that the clients orders are in. Can only be one of the following:
#             - Industries
#             - Financials
#             - Real Estate
#             - Materials
#             - Energy
#             - INVESTMENT
#             - Consumer Discretionary
#             - INDUSTRIAL
#             - Information Technology
#             - Health Care
#             - Consumer Staples
#             - REAL ESTATE
#             - Telecommunication Services
#             - Basic Materials
#             - Others
#             - FOOD
#             - Tourism
#             - Telecommunications
#             - SERVICES

#     Returns:
#         str: The sector name if valid. If no sector name is provided, None is returned.
#     """
#     valid_sectors = [
#         "Industries",
#         "Financials",
#         "Real Estate",
#         "Materials",
#         "Energy",
#         "INVESTMENT",
#         "Consumer Discretionary",
#         "INDUSTRIAL",
#         "Information Technology",
#         "Health Care",
#         "Consumer Staples",
#         "REAL ESTATE",
#         "Telecommunication Services",
#         "Basic Materials",
#         "Others",
#         "FOOD",
#         "Tourism",
#         "Telecommunications",
#         "SERVICES",
#     ]
#     if most_frequent_sector_name and most_frequent_sector_name not in valid_sectors:
#         return None

#     return most_frequent_sector_name


# def get_risk_rate(risk_rate: str) -> str:
#     """
#     Will return the risk rate of the client

#     Args:
#         risk_rate (str): The risk rate of the client. Can only be one of the following:
#             - High
#             - Low
#             - Medium
#             - Not Assigned

#     Returns:
#         str: A valid category for the risk rate of the client
#     """
#     if risk_rate is not None and risk_rate not in [
#         "High",
#         "Low",
#         "Medium",
#         "Not Assigned",
#     ]:
#         return None

#     return risk_rate


# def get_completed_orders(completed_orders: str) -> str:
#     """
#     Will return the completed orders of the client

#     Args:
#         completed_orders (str): The completed orders of the client. can only be one of the following:
#             - All
#             - More Than Half
#             - Less Than Half
#             - None

#     Returns:
#         str: A valid category that shows the trend of the clients completed orders

#     Raises:
#         ValueError: If `completed_orders` is not one of the valid options.
#     """
#     if completed_orders and completed_orders not in [
#         "All",
#         "More Than Half",
#         "Less Than Half",
#         "None",
#     ]:
#         return None

#     return completed_orders


# def get_canceled_orders(canceled_orders: str) -> str:
#     """
#     Will return the canceled orders of the client

#     Args:
#         canceled_orders (str): The canceled orders of the client. Can only be one of the following:
#             - All
#             - Most
#             - Moderate
#             - Little
#             - None

#     Returns:
#         str: A valid category that shows the trend of the clients canceled orders

#     Raises:
#         ValueError: If `canceled_orders` is not one of the valid options.
#     """
#     if canceled_orders and canceled_orders not in [
#         "All",
#         "Most",
#         "Moderate",
#         "Little",
#         "None",
#     ]:
#         return None

#     return canceled_orders


# def get_average_price(average_price: float) -> float:
#     """
#     Will return the average price of the clients orders

#     Args:
#         average_price (int): The average price of the clients orders

#     Returns:
#         float: The validated average price of the clients orders

#     Raises:
#         ValueError: If `average_price` is a negative value
#     """
#     if average_price and average_price < 0:
#         return None

#     return average_price


# def get_most_frequent_order_type(most_frequent_order_type: str) -> str:
#     """
#     Will return the most frequent order type of the client

#     Args:
#         most_frequent_order_type (str): The most frequent order type of the client. Can only be one of the following:
#             - Buy
#             - Sell

#     Returns:
#         str: A valid category for the most frequent order type of the client

#     Raises:
#         ValueError: If `most_frequent_order_type` is not one of the valid options.
#     """
#     if most_frequent_order_type and most_frequent_order_type not in ["Buy", "Sell"]:
#         return None

#     return most_frequent_order_type


# def get_most_frequent_execution_status(
#     most_frequent_execution_status: str,
# ) -> str:
#     """
#     Will return the most frequent execution status of the clients orders

#     Args:
#         most_frequent_execution_status (str): The most frequent execution status of the clients orders.
#         Can only be one of the following:
#             - Executed
#             - Not Executed
#             - Partially Executed

#     Returns:
#         str: A valid category for the most frequent execution status of the clients orders

#     Raises:
#         ValueError: If `most_frequent_execution_status` is not one of the valid options.
#     """
#     if most_frequent_execution_status and most_frequent_execution_status not in [
#         "Executed",
#         "Not Executed",
#         "Partially Executed",
#     ]:
#         return None

#     return most_frequent_execution_status


# def get_avg_order_rate_difference(avg_order_rate_difference: str) -> str:
#     """
#     Will return the change in the client's order activity

#     Args:
#         avg_order_rate_difference (str): The change in the client's order activity.
#         Can only be one of the following:
#             - Increased
#             - Decreased
#             - Constant

#     Returns:
#         str: A valid category for the change in the client's order activity

#     Raises:
#         ValueError: If `avg_order_rate__difference` is not one of the valid options.
#     """
#     if avg_order_rate_difference and avg_order_rate_difference not in [
#         "Increased",
#         "Decreased",
#         "Constant",
#     ]:
#         return None

#     return avg_order_rate_difference


# def get_avg_order_quantity_rate_difference(
#     avg_order_quantity_rate_difference: str,
# ) -> str:
#     """
#     Will return the change in the client's order quantity

#     Args:
#         avg_order_quantity_rate_difference (str): The change in the client's order quantity.
#         Can only be one of the following:
#             - Increased
#             - Decreased
#             - Constant

#     Returns:
#         str: A valid category for the change in the client's order quantity

#     Raises:
#         ValueError: If `avg_order_quantity_rate__difference` is not one of the valid options.
#     """
#     if (
#         avg_order_quantity_rate_difference
#         and avg_order_quantity_rate_difference
#         not in [
#             "Increased",
#             "Decreased",
#             "Constant",
#         ]
#     ):
#         return None
#     return avg_order_quantity_rate_difference


# def handle_unspecified_client_data(missing_client_data: list[str]) -> str:
#     """
#     Handles the case where client details are not provided by the user.

#     Args:
#         missing_client_data (list[str]): The list of the missing client details.

#     Returns:
#         str: A message prompting the user to provide the missing arguments.
#     """
#     return (
#         f"Please provide the following missing values: {', '.join(missing_client_data)}"
#     )


# def no_relevant_function(prompt: str) -> dict:
#     """
#     Call this when no other provided function can be called to answer the user query.

#     Args:
#        prompt: The prompt that cannot be answered by any other function calls.
#     """
#     no_function_calling_prompt = f"""
#     <s> [INST] {prompt} [/INST]
#     <s> [INST] I am called Raven. How can i assist you today?[/INST]
#     """
#     return {"message": query_raven(no_function_calling_prompt)}


# def unknown_arguments(unknown_arg: str) -> str | int:
#     """
#     Provides default values for unknown or unspecified client details in a user query.

#     This function returns a default value for specific client details that are not known
#     or provided by the user.

#     Args:
#         unknown_arg (str): The name of the unknown or unspecified client detail. Could be one of the following:
#             - age
#             - risk_rate
#             - gender
#             - completed_orders
#             - canceled_orders
#             - avg_price
#             - most_frequent_order_type
#             - most_frequent_execution_status
#             - most_frequent_sector_name
#             - avg_order_rate_difference
#             - avg_order_quantity_rate_difference

#     Returns:
#         str: The default value corresponding to the provided unknown client detail.
#     """
#     if unknown_arg == "age":
#         return 40
#     if unknown_arg == "gender":
#         return "Male"
#     if unknown_arg == "most_frequent_sector_name":
#         return "Financials"
#     if unknown_arg == "risk_rate":
#         return "Not Assigned"
#     if unknown_arg == "completed_orders":
#         return "Most"
#     if unknown_arg == "canceled_orders":
#         return "Moderate"
#     if unknown_arg == "avg_price":
#         return 9.56
#     if unknown_arg == "most_frequent_order_type":
#         return "Sell"
#     if unknown_arg == "most_frequent_execution_status":
#         return "Executed"
#     if unknown_arg == "avg_order_rate_difference":
#         return "constant"
#     if unknown_arg == "avg_order_quantity_rate_difference":
#         return "constant"


# def construct_client_dict(
#     age: int = None,
#     risk_rate: str = None,
#     gender: str = None,
#     completed_orders: str = None,
#     canceled_orders: str = None,
#     average_price: float = None,
#     most_frequent_order_type: str = None,
#     most_frequent_execution_status: str = None,
#     most_frequent_sector_name: str = None,
#     avg_order_rate_difference: str = None,
#     avg_order_quantity_rate_difference: str = None,
# ) -> dict:
#     """
#     Constructs a dict with all the client details specified in the user query.

#     Args:
#         age (int, optional): The age of the client. If not provided, defaults to None.
#         risk_rate (str, optional): The risk rate of the client. If not provided, defaults to None.
#         gender (str, optional): The gender of the client. If not provided, defaults to None.
#         completed_orders (str, optional): The completed orders of the client. If not provided, defaults to None.
#         canceled_orders_ratio (str, optional): The canceled orders ratio of the client. If not provided, defaults to None.
#         average_price (float, optional): The average price of the clients orders. If not provided, defaults to None.
#         most_frequent_order_type (str, optional): The most frequent order type of the client. If not provided, defaults to None.
#         most_frequent_execution_status (str, optional): The most frequent execution status of the clients orders. If not provided, defaults to None.
#         most_frequent_sector_name (str, optional): The most frequent sector that the clients orders are in. If not provided, defaults to None.
#         avg_order_rate_difference (str, optional): The change in the client's order activity. If not provided, defaults to None.
#         avg_order_quantity_rate_difference (str, optional): The change in the client's order quantity. If not provided, defaults to None.

#     Returns:
#         dict: The client data.
#     """

#     provided_params = locals().copy()

#     message = ""
#     missing_args = [key for key, value in provided_params.items() if value is None]
#     if missing_args:
#         message = handle_unspecified_client_data(missing_args)

#     return {"client_data": provided_params, "message": message}
