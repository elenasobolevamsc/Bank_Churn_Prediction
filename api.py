import pandas as pd
import sklearn
import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

sklearn.set_config(transform_output='pandas')

app = FastAPI()

model_loaded = joblib.load('./best_model.pkl')

class Dataframe(BaseModel):
    data: str


@app.post("/best_model")
async def best_model(one_var: Dataframe):
    df = pd.read_json(one_var.data, orient='split')
    pred = model_loaded.predict(df)
    pred = pd.DataFrame(pred)
    return {"pred": pred.to_json(orient='split')}

#%%
