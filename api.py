import pandas as pd
import numpy as np
import sklearn
import joblib

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin

sklearn.set_config(transform_output='pandas')

app = FastAPI()

model_loaded = joblib.load('./best_model.pkl')

class Dataframe(BaseModel):
    data: str


@app.post("/mod")
async def best_model(one_var: Dataframe):
    df = pd.read_json(one_var.data, orient='split')
    pred = model_loaded.predict(df)
    return {"pred": pred.to_json(orient='split')}