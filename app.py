import asyncio
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
    X1: int
    X2: int
    X3: int
    X4: int
    X5: int
    X6: int
    X7: int
    X8: int
    X9: int
    X10: int
    X11: int
    X12: int
    X13: int
    X14: int
    X15: int
    X16: int
    X17: int
    X18: int
    X19: int
    X20: int
    X21: int
    X22: int
    X23: int
    X24: int
    X25: int
    X26: int
    X27: int
    X28: int
    X29: int
    X30: int
    X31: int
    X32: int
    X33: int
    X34: int
    X35: int
    X36: int
    X37: int
    X38: int
    X39: int
    X40: int
    X41: int
    X42: int
    X43: int
    X44: int
    X45: int
    X46: int
    X47: int
    X48: int
    X49: int
    X50: int
    X51: int
    X52: int
    X53: int
    X54: int
    X55: int
    X56: int
    X57: int
    X58: int
    X59: int
    X60: int
    X61: int
    X62: int
    X63: int
    X64: int
    X65: int
    X66: int
    X67: int
    X68: int
    X69: int
    X70: int
    X71: int
    X72: int
    X73: int
    X74: int
    X75: int
    X76: int
    X77: int
    X78: int
    X79: int
    X80: int
    X81: int
    X82: int
    X83: int
    X84: int
    X85: int
    X86: int
    X87: int
    X88: int
    X89: int
    X90: int
    X91: int
    X92: int
    X93: int
    X94: int
    X95: int
    X96: int
    X97: int
    X98: int
    X99: int
    X100: int
    X101: int
    X102: int
    X103: int
    X104: int
    X105: int
    X106: int
    X107: int
    X108: int
    X109: int
    X110: int
    X111: int
    X112: int
    X113: int
    X114: int
    X115: int
    X116: int
    X117: int
    X118: int
    X119: int
    X120: int
    X121: int
    X122: int
    X123: int
    X124: int
    X125: int
    X126: int
    X127: int
    X128: int
    X129: int
    X130: int
    X131: int
    X132: int
    X133: int
    X134: int
    X135: int
    X136: int
    X137: int
    X138: int
    X139: int
    X140: int
    X141: int
    X142: int
    X143: int
    X144: int
    X145: int
    X146: int
    X147: int
    X148: int
    X149: int
    X150: int
    X151: int
    X152: int
    X153: int
    X154: int
    X155: int
    X156: int
    X157: int
    X158: int
    X159: int
    X160: int
    X161: int
    X162: int
    X163: int
    X164: int
    X165: int
    X166: int
    X167: int
    X168: int
    X169: int
    X170: int
    X171: int
    X172: int
    X173: int
    X174: int
    X175: int
    X176: int
    X177: int
    X178: int
    # Add more features X3, X4, ..., X178 as needed


# Define function to make prediction asynchronously
async def predict_async(input_data):
    # Make prediction using the loaded model
    prediction = loaded_model.predict(input_data)

    # Check if the prediction indicates epilepsy (1) or not (0)
    has_epilepsy = bool(prediction[0])

    return {"has_epilepsy": has_epilepsy}


# Define endpoint to make predictions
@app.post("/predict")
async def predict(data: DataInput):
    try:
        # Convert input data to a numpy array
        input_data = np.array([[getattr(data, f"X{i}") for i in range(1, 179)]])

        # Execute the prediction asynchronously
        prediction_task = asyncio.create_task(predict_async(input_data))

        # Wait for the prediction result with a timeout of 5 seconds
        prediction_result = await asyncio.wait_for(prediction_task, timeout=5)

        # Return the prediction result
        return prediction_result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timeout: The prediction took too long to process.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
