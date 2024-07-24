# Functions to be called by llm so that the features can be extracted from natural language prompt

def get_age(age: int = 40) -> int:
    """
    Validates and returns the age value.

    Args:
        age (int, optional): The age of the client. Must be a non-negative integer. If not provided, defaults to None.

    Returns:
        int: The validated age. If no age is provided, None is returned.

    Raises:
        ValueError: If `age` is provided and is a negative integer.
    """
    if age is not None and age < 0:
        raise ValueError("age must be a non-negative integer.")

    return age


def get_risk_rate(risk_rate: str = "Not Assigned") -> str:
    """
    Will return the risk rate of the client

    Args:
        risk_rate (str): The risk rate of the client. Can only be one of the following: 'High', 'Low', 'Medium', 'Not Assigned'

    Returns:
        str: A valid category for the risk rate of the client 
    
    Raises:
        ValueError: If `risk_rate` is not one of the valid options.

    """
    if risk_rate not in ['High', 'Low', 'Medium', 'Not Assigned']:
        raise ValueError("Risk rate is invalid")

    return risk_rate


def get_gender(gender: str = "Male") -> int:
    """
    Will return if the client is male or not

    Args:
        gender: the gender of the client. Can only be one of the following: 'Male', 'Female'
    
    Returns:
        int: 1 if the client is male, 0 if the client is female
    
    Raises:
        ValueError: If `gender` is not one of the valid options.
    """

    if gender not in ['Male', 'Female']:
        raise ValueError("Gender not given")
    elif gender == "Male":
        return 1
    else:
        return 0


def get_completed_orders_ratio(completed_orders_ratio: str = "Most") -> str:
    """
    Will return the completed orders ratio of the client

    Args:
        completed_orders_ratio (str): The completed orders ratio of the client. can only be one of the following: 
        'All', 'More Than Half', 'Less Than Half', 'None'
    
    Returns:
        str: A valid category that shows the trend of the clients completed orders
    
    Raises:
        ValueError: If `completed_orders_ratio` is not one of the valid options.
    """
    if completed_orders_ratio not in ['All', 'More Than Half', 'Less Than Half', 'None']:
        raise ValueError("Completed orders ratio is invalid")

    return completed_orders_ratio


def get_canceled_orders_ratio(canceled_orders_ratio: str = "Moderate") -> str:
    """
    Will return the canceled orders ratio of the client

    Args:
        canceled_orders_ratio (str): The canceled orders ratio of the client. Can only be one of the following:
        'All', 'Most', 'Moderate', 'Little', 'None'

    Returns:    
        str: A valid category that shows the trend of the clients canceled orders

    Raises:
        ValueError: If `canceled_orders_ratio` is not one of the valid options.
    """
    if canceled_orders_ratio not in ['All', 'Most', 'Moderate', 'Little', 'None']:
        raise ValueError("Canceled orders ratio is invalid")

    return canceled_orders_ratio


def get_avg_price(avg_price: float = 9.56) -> float:
    """
    Will return the average price of the clients orders

    Args:
        avg_price (int): The average price of the clients orders
    
    Returns:
        float: The validated average price of the clients orders

    Raises: 
        ValueError: If `avg_price` is a negative value
    """
    if avg_price < 0:
        raise ValueError("Average price cannot be negative")

    return avg_price


def get_most_frequent_order_type(most_frequent_order_type: str = "Sell") -> str:
    """
    Will return the most frequent order type of the client

    Args:
        most_frequent_order_type (str): The most frequent order type of the client. Can only be one of the following: 
        'Buy', 'Sell'

    Returns:
        str: A valid category for the most frequent order type of the client

    Raises:
        ValueError: If `most_frequent_order_type` is not one of the valid options.
    """
    if most_frequent_order_type not in ['Buy', 'Sell']:
        raise ValueError("Most frequent order type is invalid")

    return most_frequent_order_type


def get_most_frequent_execution_status(most_frequent_execution_status: str = "Executed") -> str:
    """
    Will return the most frequent execution status of the clients orders

    Args:
        most_frequent_execution_status (str): The most frequent execution status of the clients orders. 
        Can only be one of the following: 'Executed', 'Not Executed', 'Partially Executed'

    Returns:
        str: A valid category for the most frequent execution status of the clients orders

    Raises:
        ValueError: If `most_frequent_execution_status` is not one of the valid options.
    """
    if most_frequent_execution_status not in ['Executed', 'Not Executed', 'Partially Executed']:
        raise ValueError("Most frequent execution status is invalid")

    return most_frequent_execution_status


def get_most_frequent_sector_name(most_frequent_sector_name: str = "Financials") -> str:
    """
    Will return the most frequent sector that the clients orders are in

    Args:
        most_frequent_sector_name (str): The most frequent sector that the clients orders are in
    """
    return most_frequent_sector_name


def get_avg_order_rate__difference(avg_order_rate__difference: str = "constant") -> str:
    """
    Will return the change in the client's order activity

    Args:
        get_avg_order_rate__difference (str): The change in the client's order activity. 
        Can only be one of the following: 'increased', 'decreased', 'constant'

    Returns:
        str: A valid category for the change in the client's order activity

    Raises:
        ValueError: If `avg_order_rate__difference` is not one of the valid options.
    """
    if avg_order_rate__difference not in ['increased', 'decreased', 'constant']:
        raise ValueError("Average order rate difference is invalid")
    
    return avg_order_rate__difference


def get_avg_order_quantity_rate__difference(avg_order_quantity_rate__difference: str = "constant") -> str:
    """
    Will return the change in the client's order quantity

    Args:
        get_avg_order_quantity_rate__difference (str): The change in the client's order quantity. 
        Can only be one of the following: 'increased', 'decreased', 'constant'

    Returns:        
        str: A valid category for the change in the client's order quantity

    Raises:
        ValueError: If `avg_order_quantity_rate__difference` is not one of the valid options.
    """
    if avg_order_quantity_rate__difference not in ['increased', 'decreased', 'constant']:
        raise ValueError("Average order quantity rate difference is invalid")
    return avg_order_quantity_rate__difference


def combine_features(age: int, risk_rate: str, is_male: int, 
                     completed_orders_ratio: str, canceled_orders_ratio: str, 
                     avg_price: float, most_frequent_order_type: str, 
                     most_frequent_execution_status: str, 
                     most_frequent_sector_name: str, 
                     avg_order_rate__difference: str, 
                     avg_order_quantity_rate__difference: str) -> dict:
    """
    Combines all the features into a single dictionary

    Args:
        age (int): The age of the client
        risk_rate (str): The risk rate of the client
        is_male (int): If the client is male (1) or not (0)
        completed_orders_ratio (str): The completed orders ratio of the client
        canceled_orders_ratio (str): The canceled orders ratio of the client
        avg_price (float): The average price of the clients orders
        most_frequent_order_type (str): The most frequent order type of the client
        most_frequent_execution_status (str): The most frequent execution status of the clients orders
        most_frequent_sector_name (str): The most frequent sector that the clients orders are in
        avg_order_rate__difference (str): The change in the client's order activity
        avg_order_quantity_rate__difference (str): The change in the client's order quantity
    
    Returns:
        dict: A dictionary containing all the features
    """
    return {
        "age": age,
        "risk_rate": risk_rate, 
        "is_male": is_male,
        "completed_orders_ratio": completed_orders_ratio,
        "canceled_orders_ratio": canceled_orders_ratio,
        "avg_price": avg_price,
        "most_frequent_order_type": most_frequent_order_type,
        "most_frequent_execution_status": most_frequent_execution_status,
        "most_frequent_sector_name": most_frequent_sector_name,
        "avg_order_rate__difference": avg_order_rate__difference,
        "avg_order_quantity_rate__difference": avg_order_quantity_rate__difference
    }