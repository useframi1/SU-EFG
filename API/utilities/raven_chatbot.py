from utilities.utils import *
import requests
from dotenv import load_dotenv
import os

_ = load_dotenv()


class Chatbot:
    def __init__(self):
        self._history = []

    def _add_to_history(self, bot_response: dict, user_input: str):
        if type(bot_response) == dict and "client_data" in bot_response.keys():
            self._history.append({"response": bot_response, "input": user_input})

    def _get_conversation(self) -> str:
        conversation = ""
        for hist in self._history:
            conversation += f"""
Available client data: {hist["response"]["client_data"]}
User input: {hist["input"]}
Bot: {hist["response"]["message"]}
"""
        return conversation

    def _generate_prompt(self, user_input: str) -> str:
        return build_raven_prompt(
            [
                no_relevant_function,
                construct_client_dict,
                unknown_arguments,
                get_age,
                get_risk_rate,
                get_gender,
                get_completed_orders,
                get_canceled_orders,
                get_average_price,
                get_most_frequent_order_type,
                get_most_frequent_execution_status,
                get_most_frequent_sector_name,
                get_avg_order_rate_difference,
                get_avg_order_quantity_rate_difference,
            ],
            self._get_conversation(),
            user_input,
        )

    def _get_response(self, prompt: str) -> dict:
        func = query_raven(prompt)
        if func is None:
            return {"message": "Sorry, I couldn't understand your request."}

        try:
            response = eval(func)
            if type(response) != dict:
                return {"message": "Sorry, I couldn't understand your request."}
            return response
        except Exception as e:
            return {"message": e}

    def _prepare_client_data(self, client_data: dict) -> dict:
        temp_client_data = client_data.copy()
        temp_client_data["gender"] = 1 if temp_client_data["gender"] == "Male" else 0
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
            if old_key in temp_client_data:
                temp_client_data[new_key] = temp_client_data.pop(old_key)

        return temp_client_data

    def _predict_churn(self, client_data: dict) -> str:
        cleaned_client_data = self._prepare_client_data(client_data=client_data)

        try:
            response = requests.post(
                url=os.getenv("PREDICT_CHURN_URL"), json=cleaned_client_data
            )
            if response.status_code == 200:
                data = response.json()
                if "prediction" in data.keys():
                    if data["prediction"] == 1:
                        return "The client will most likely churn"
                    elif data["prediction"] == 0:
                        return "The client will not churn"
        except Exception as e:
            return f"Request failed: {e}"

    def handle_input(self, user_input: str) -> dict:
        prompt = self._generate_prompt(user_input)
        response = self._get_response(prompt)
        self._add_to_history(response, user_input)
        if response["message"] == "":
            response["message"] = self._predict_churn(response["client_data"])
        return response
