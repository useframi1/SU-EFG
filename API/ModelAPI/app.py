import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "API"))

if project_root not in sys.path:
    sys.path.append(project_root)

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utilities.utils import log_transform

scalers = joblib.load("../pickle_files/scalers.pkl")
encoder = joblib.load("../pickle_files/encoder.pkl")
model = joblib.load("../pickle_files/model.pkl")

app = FastAPI()


class DataInput(BaseModel):
    Age: float
    IsMale: int
    AvgPrice: float
    RiskRate: str
    AvgOrderRate_Difference: str
    AvgQuantityOrderedRate_Difference: str
    CompletedOrdersRatio: str
    CanceledOrdersRatio: str
    Most_Frequent_OrderType: str
    Most_Frequent_ExecutionStatus: str
    Most_Frequent_SectorName: str


@app.post("/predict")
def predict(data: DataInput):
    input_df = pd.DataFrame([data.model_dump()])

    categorical_columns = [
        "RiskRate",
        "AvgOrderRate_Difference",
        "AvgQuantityOrderedRate_Difference",
        "CompletedOrdersRatio",
        "CanceledOrdersRatio",
        "Most_Frequent_OrderType",
        "Most_Frequent_ExecutionStatus",
        "Most_Frequent_SectorName",
    ]

    numerical_columns = ["Age", "IsMale", "AvgPrice"]

    try:
        categorical_data = input_df[categorical_columns]
        numerical_data = input_df[numerical_columns]

        encoded_data = encoder.transform(categorical_data)

        for col in numerical_columns:
            if col in scalers:
                scaler = scalers[col]
                if isinstance(scaler, (MinMaxScaler, StandardScaler)):
                    numerical_data.loc[:, col] = scaler.transform(numerical_data[[col]])
                elif callable(scaler):
                    numerical_data.loc[:, col] = scaler(
                        numerical_data[col].values.flatten()
                    )
                else:
                    raise ValueError(f"Unknown scaler type for column {col}")

        transformed_data = np.hstack((numerical_data.values, encoded_data))

        prediction = model.predict(transformed_data)
        prediction = prediction.astype(int).tolist()

        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
