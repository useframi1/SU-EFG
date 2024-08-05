from utilities.utils import get_client_data
from groq import Groq
import json
from dotenv import load_dotenv
import os

_ = load_dotenv()


class Chatbot:
    def __init__(self):
        self.model: str = "llama3-groq-70b-8192-tool-use-preview"
        self.client: Groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.messages: list[dict] = [
            {
                "role": "system",
                "content": "Your name is Raven. You are a client churn prediction assistant. Use the get_client_data function to get all the client data and provide the results of the prediction. If any client detail is missing, you are required to notify the user, do not guess the value.",
            },
        ]

    def run_conversation(self, user_prompt):
        self.messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_client_data",
                    "description": "Gathers the client's data and returns a prediction from the model about whether the client will churn or not.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "age": {
                                "type": "int",
                                "description": "The age of the client. Must be a non-negative integer. If the age is unknown use 40 as the default value.",
                            },
                            "risk_rate": {
                                "type": "str",
                                "description": "The risk rate of the client. Can only be one of the following with case sensitivity; 'High', 'Low', 'Medium', 'Not Assigned'. If the risk rate is unknown use 'Not Assigned' as the default value.",
                            },
                            "gender": {
                                "type": "str",
                                "description": "the gender of the client. Can only be one of the following with case sensitivity; 'Male', 'Female'. If the gender is unknown use 'Male' as the default value.",
                            },
                            "completed_orders": {
                                "type": "str",
                                "description": "The completed orders of the client. can only be one of the following with case sensitivity; 'All', 'More Than Half', 'Less Than Half', 'None'. If the completed orders is unknown use 'More Than Half' as the default value.",
                            },
                            "canceled_orders": {
                                "type": "str",
                                "description": "The canceled orders of the client. Can only be one of the following with case sensitivity;'All', 'Most', 'Moderate', 'Little', 'None'. If the canceled orders is unknown use 'Moderate' as the default value.",
                            },
                            "average_price": {
                                "type": "float",
                                "description": "The average price of the clients orders. If the average price is not known use 9.5 as the default value.",
                            },
                            "most_frequent_order_type": {
                                "type": "str",
                                "description": "The most frequent order type means if the client mainly buys or sells stocks. Can only be one of the following with case sensitivity;'Buy', 'Sell'. If the order type is unknown use 'Sell' as the default value.",
                            },
                            "most_frequent_execution_status": {
                                "type": "str",
                                "description": "The most frequent execution status of the clients orders. Can only be one of the following with case sensitivity; 'Executed', 'Not Executed', 'Partially Executed'. If the execution status is unknown use 'Executed' as the default value",
                            },
                            "most_frequent_sector_name": {
                                "type": "str",
                                "description": "The most frequent sector that the clients orders are in. Can only be one of the following with case sensitivity;'Industries', 'Financials', 'Real Estate', 'Materials', 'Energy','INVESTMENT', 'Consumer Discretionary', 'INDUSTRIAL','Information Technology', 'Health Care', 'Consumer Staples','REAL ESTATE', 'Telecommunication Services', 'Basic Materials','Others', 'FOOD', 'Tourism', 'Telecommunications', 'SERVICES'. If the sector is unknown use 'Financials' as the default value.",
                            },
                            "avg_order_rate_difference": {
                                "type": "str",
                                "description": "The change in the client's order activity. Can only be one of the following with case sensitivity; 'Increased', 'Decreased', 'Constant'. If the change in order rate is unknown use 'Constant' as the default value.",
                            },
                            "avg_order_quantity_rate_difference": {
                                "type": "str",
                                "description": "The change in the client's order quantity. Can only be one of the following with case sensitivity; 'Increased', 'Decreased', 'Constant'. If the change in order quantity rate is unknown use 'Constant' as the default value.",
                            },
                        },
                        "required": [
                            "age",
                            "risk_rate",
                            "gender",
                            "completed_orders",
                            "canceled_orders",
                            "average_price",
                            "most_frequent_order_type",
                            "most_frequent_execution_status",
                            "most_frequent_sector_name",
                            "avg_order_rate_difference",
                            "avg_order_quantity_rate_difference",
                        ],
                    },
                },
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096,
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        if tool_calls:
            available_functions = {
                "get_client_data": get_client_data,
            }
            self.messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)

                function_response = function_to_call(
                    age=function_args.get("age"),
                    risk_rate=function_args.get("risk_rate"),
                    gender=function_args.get("gender"),
                    completed_orders=function_args.get("completed_orders"),
                    canceled_orders=function_args.get("canceled_orders"),
                    average_price=function_args.get("average_price"),
                    most_frequent_order_type=function_args.get(
                        "most_frequent_order_type"
                    ),
                    most_frequent_execution_status=function_args.get(
                        "most_frequent_execution_status"
                    ),
                    most_frequent_sector_name=function_args.get(
                        "most_frequent_sector_name"
                    ),
                    avg_order_rate_difference=function_args.get(
                        "avg_order_rate_difference"
                    ),
                    avg_order_quantity_rate_difference=function_args.get(
                        "avg_order_quantity_rate_difference"
                    ),
                )
                self.messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )

            second_response = self.client.chat.completions.create(
                model=self.model, messages=self.messages
            )

            return second_response.choices[0].message.content

        # self.messages.append(response_message.content)

        return response_message.content
