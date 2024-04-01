import os

import pandas as pd
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pickle
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Add CORS support to allow requests from specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*", "https://www.e-hospital.ca"],
    allow_methods=["POST", "*"],
    allow_credentials=True,
    allow_headers=["*"],
)

# Load the trained decision tree model
with open('decision_tree_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Define Pydantic model for input data
class InputData(BaseModel):
    # Define 178 integer inputs
    X1: int
    X2: int
    # Add more features X3, X4, ..., X178 as needed

# Function to make prediction
def predict(input_data):
    try:
        # Convert input data to a numpy array
        input_array = np.array([list(input_data.dict().values())])

        # Standardize features
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_array)

        # Make prediction using the loaded model
        prediction = loaded_model.predict(input_scaled)

        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define endpoint to make predictions
@app.post("/predict")
async def predict_background(data: InputData, background_tasks: BackgroundTasks):
    # Run prediction in the background
    background_tasks.add_task(predict, data)

    # Return response
    return JSONResponse(content={"message": "Prediction started. Please wait for the result."})

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Epileptic Seizure Prediction API!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
