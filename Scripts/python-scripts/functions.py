# Functions to be called by llm so that the features can be extracted from natural language prompt


def get_age(age: int = 40) -> int:
    """
    Will return the age of the client

    Parameters:
    age (int): The age of the client
    """
    # if age < 10 or age > 90:
    #     return ValueError("The given age is invalid")

    return age


def get_risk_rate(risk_rate: str = "Not Assigned") -> str:
    """
    Will return the risk rate of the client

    Parameters:
    risk_rate (str): The risk rate of the client. Can only be one of the following: 'High', 'Low', 'Medium', 'Not Assigned'

    """
    # if risk_rate not in ['High', 'Low', 'Medium', 'Not Assigned']:
    #     return ValueError("Risk rate is invalid")

    return risk_rate


def get_gender(gender: str = "Male") -> int:
    """
    Will return if the client is male or not

    Parameters:
    gender: the gender of the client. Can only be one of the following: 'Male', 'Female'
    """

    # if gender not in ['Male', 'Female']:
    #     return ValueError("Gender not given")
    if gender == "Male":
        return 1
    else:
        return 0


def get_completed_orders_ratio(completed_orders_ratio: str = "Most") -> str:
    """
    Will return the completed orders ratio of the client

    Parameters:
    completed_orders_ratio (str): The completed orders ratio of the client. can only be one of the following: 'All', 'Most', 'Little', 'None'
    """
    # if completed_orders_ratio not in ['All', 'More Than Half', 'Less Than Half', 'None']:
    #     return ValueError("Completed orders ratio is invalid")

    return completed_orders_ratio


def get_canceled_orders_ratio(canceled_orders_ratio: str = "Moderate") -> str:
    """
    Will return the canceled orders ratio of the client

    Parameters:
    canceled_orders_ratio (str): The canceled orders ratio of the client. Can only be one of the following: 'All', 'Most', 'Moderate', 'Little', 'None'
    """
    # if canceled_orders_ratio not in ['All', 'Most', 'Moderate', 'Little', 'None']:
    #     return ValueError("Canceled orders ratio is invalid")

    return canceled_orders_ratio


def get_avg_price(avg_price: float = 9.56) -> float:
    """
    Will return the average price of the clients orders

    Parameters:
    avg_price (int): The average price of the clients orders
    """
    # if avg_price < 0:
    #     return ValueError("Average price is invalid")

    return avg_price


def get_most_frequent_order_type(most_frequent_order_type: str = "Sell") -> str:
    """
    Will return the most frequent order type of the client

    Parameters:
    most_frequent_order_type (str): The most frequent order type of the client. Can only be one of the following: 'Buy', 'Sell'
    """
    # if most_frequent_order_type not in ['Buy', 'Sell']:
    #     return ValueError("Most frequent order type is invalid")

    return most_frequent_order_type


def get_most_frequent_execution_status(
    most_frequent_execution_status: str = "Executed",
) -> str:
    """
    Will return the most frequent execution status of the clients orders

    Parameters:
    most_frequent_execution_status (str): The most frequent execution status of the clients orders. Can only be one of the following: 'Executed', 'Not Executed', 'Partially Executed'
    """
    # if most_frequent_execution_status not in ['Executed', 'Not Executed', 'Partially Executed']:
    #     return ValueError("Most frequent execution status is invalid")

    return most_frequent_execution_status


def get_most_frequent_sector_name(most_frequent_sector_name: str = "Financials") -> str:
    """
    Will return the most frequent sector that the clients orders are in

    Parameters:
    most_frequent_sector_name (str): The most frequent sector that the clients orders are in
    """
    return most_frequent_sector_name


def get_avg_order_rate__difference(avg_order_rate__difference: str = "constant") -> str:
    """
    Will return the change in the client's order activity

    Parameters:
    get_avg_order_rate__difference (str): The change in the client's order activity. Can only be one of the following: 'increased', 'decreased', 'constant'
    """
    return avg_order_rate__difference


def get_avg_order_quantity_rate__difference(
    avg_order_quantity_rate__difference: str = "constant",
) -> str:
    """
    Will return the change in the client's order quantity

    Parameters:
    get_avg_order_quantity_rate__difference (str): The change in the client's order quantity. Can only be one of the following: 'increased', 'decreased', 'constant'
    """
    return avg_order_quantity_rate__difference


import inspect


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


print(build_raven_prompt([get_age], "53 year old"))
