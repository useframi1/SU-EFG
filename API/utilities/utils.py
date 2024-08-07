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
            url=os.getenv("MODEL_API_BASE_URL") + "predict", json=provided_params
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
