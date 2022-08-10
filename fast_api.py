# -*- coding: utf-8 -*-

# Import Needed Libraries
from fastapi import FastAPI
import joblib
import uvicorn
import pandas as pd
import numpy as np
from pydantic import BaseModel
from lime import lime_tabular
from colorama import init

# Initiate app instance
app = FastAPI()

# Load data


# Initialize model artifacte files. This will be loaded at the start of FastAPI model server.
lgbm = joblib.load("model_loans.joblib","rb")
features = joblib.load("features.joblib", "rb")

# This struture will be used for Json validation.
# With just that Python type declaration, FastAPI will perform below operations on the request data
## 1) Read the body of the request as JSON.
## 2) Convert the corresponding types (if needed).
## 3) Validate the data.If the data is invalid, it will return a nice and clear error,
##    indicating exactly where and what was the incorrect data.
class Data(BaseModel):
    ACTIVE_DAYS_CREDIT_MAX: int
    AMT_ANNUITY: float
    APPROVED_CNT_PAYMENT_MEAN: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    INSTAL_AMT_PAYMENT_SUM: float
    INSTAL_DPD_MEAN: float
    PAYMENT_RATE: float
    PREV_CNT_PAYMENT_MEAN: float
    SK_ID_CURR: int
    YEARS_BIRTH: int
    YEARS_EMPLOYED: int




# ML API endpoint for making prediction aganist the request received from client
@app.post("/predict")
def predict(data: Data):
    # Extract data in correct order
    data_dict = data.dict()
    data_df = pd.DataFrame.from_dict([data_dict])
    # Select features required for making prediction
    data_df = data_df[features]
    data_array = data_df.values.ravel()
    # Create prediction
    prediction = lgbm.predict(data_df)
    # Map prediction to appropriate label
    prediction_label = ['accordé' if label == 0 else 'non accordé' for label in prediction]
    str_prediction = " ".join(prediction_label)

    # Explain/Interpretability
    interpretor = lime_tabular.LimeTabularExplainer(
        training_data=np.array(df),
        feature_names=df.columns.values.tolist(),
        mode='classification')

    exp = interpretor.explain_instance(data_row=data_array,
                                       predict_fn=lgbm.predict_proba,
                                       num_samples=10000,
                                       num_features=6)
    exp.save_to_file('lime_fig.html')
    # Return response back to client
    return f"Le prêt est {str_prediction}"
    
# Determinate the host with precession else it is 8000
if __name__ == '__main__':
    init()
    uvicorn.run(app, host="127.0.0.1", port=8000)