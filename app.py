from fastapi import FastAPI
from Schema.UserInput import UserInput
from Schema.UserInput import  preprocess_features
from fastapi.middleware.cors import CORSMiddleware
import streamlit as st
import requests 
import pandas as pd
from fastapi.responses import JSONResponse
from Model.predict import predict_output,model,MODEL_VERSION


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.get('/')
def home():
    return {"message":"Diabetes Prediction App"}

# this is made for AWS services like kubernetes to know that our api is live and its health info
@app.get('/health')
def health_check():
    return{
        'status':"Ok",
        "version":MODEL_VERSION,
        "model_loaded":False if model is None else True # show that model is loaded
    }

@app.post("/predict")
def predict_diabetes(user_input:UserInput):
    raw_data_dict = user_input.model_dump() # model_dump will convert the pydantic object to dictionary and calculates the computed field
    try:
        processed_data = preprocess_features(raw_data_dict)
        probability = predict_output(processed_data)

        result = "" 
        if probability*100 > 30.0:
            result = "High risk of Diabetes"
        else:
            result = "No risk of diabetes,maintain a health lifestyle"
        return JSONResponse(
            status_code=200,
            content = {
            "status":"success","processed_data":processed_data,"probability_score":float(probability),"result":result
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500,content=str(e))


