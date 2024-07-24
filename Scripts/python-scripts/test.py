from utils import query_raven, build_raven_prompt
from functions import get_age, get_risk_rate, get_gender, get_completed_orders_ratio, get_canceled_orders_ratio, get_avg_price, get_most_frequent_order_type, get_most_frequent_execution_status, get_most_frequent_sector_name, get_avg_order_rate__difference, get_avg_order_quantity_rate__difference, combine_features

USER_QUERY = "The client is a 53 year old man with a low risk rate. The client mainly buys stocks in the Industries sector with an average price of 7.65 and the orders are usually fully  executed. They have increased order activity but decreased quantity ordered. Most of the client's orders are completed and little of the client's orders are canceled."

prompt = build_raven_prompt(
    [get_age, 
     get_risk_rate, 
     get_gender, 
     get_completed_orders_ratio, 
     get_canceled_orders_ratio, 
     get_avg_price, 
     get_most_frequent_order_type, 
     get_most_frequent_execution_status, 
     get_most_frequent_sector_name, 
     get_avg_order_rate__difference, 
     get_avg_order_quantity_rate__difference, 
     combine_features
    ],
    USER_QUERY
)

result = query_raven(prompt)
print(result)