from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pickle
import numpy as np
from pydantic import BaseModel

# Load the trained decision tree model
with open('decision_tree_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

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


# Custom exception class
class CustomException(Exception):
    def __init__(self, detail: str):
        self.detail = detail


# Custom exception handler
@app.exception_handler(CustomException)
async def custom_exception_handler(request, exc):
    return JSONResponse(status_code=400, content={"message": exc.detail})


# Define Pydantic model for input data
class DataInput(BaseModel):
    # Define 178 integer inputs
    [f"X{i}: int" for i in range(1, 179)]


# Define endpoint to make predictions
@app.post("/predict")
async def predict(data: DataInput):
    try:
        # Convert input data to a numpy array
        input_data = np.array([[getattr(data, f"X{i}") for i in range(1, 179)]])

        # Make prediction using the loaded model
        prediction = loaded_model.predict(input_data)

        # Check if the prediction indicates epilepsy (1) or not (0)
        has_epilepsy = bool(prediction[0])

        # Return the result
        return {"has_epilepsy": has_epilepsy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
