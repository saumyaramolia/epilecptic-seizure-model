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
    X1: float
    X2: float
    X3: float
    X4: float
    X5: float
    X6: float
    X7: float
    X8: float
    X9: float
    X10: float
    X11: float
    X12: float
    X13: float
    X14: float
    X15: float
    X16: float
    X17: float
    X18: float
    X19: float
    X20: float
    X21: float
    X22: float
    X23: float
    X24: float
    X25: float
    X26: float
    X27: float
    X28: float
    X29: float
    X30: float
    X31: float
    X32: float
    X33: float
    X34: float
    X35: float
    X36: float
    X37: float
    X38: float
    X39: float
    X40: float
    X41: float
    X42: float
    X43: float
    X44: float
    X45: float
    X46: float
    X47: float
    X48: float
    X49: float
    X50: float
    X51: float
    X52: float
    X53: float
    X54: float
    X55: float
    X56: float
    X57: float
    X58: float
    X59: float
    X60: float
    X61: float
    X62: float
    X63: float
    X64: float
    X65: float
    X66: float
    X67: float
    X68: float
    X69: float
    X70: float
    X71: float
    X72: float
    X73: float
    X74: float
    X75: float
    X76: float
    X77: float
    X78: float
    X79: float
    X80: float
    X81: float
    X82: float
    X83: float
    X84: float
    X85: float
    X86: float
    X87: float
    X88: float
    X89: float
    X90: float
    X91: float
    X92: float
    X93: float
    X94: float
    X95: float
    X96: float
    X97: float
    X98: float
    X99: float
    X100: float
    X101: float
    X102: float
    X103: float
    X104: float
    X105: float
    X106: float
    X107: float
    X108: float
    X109: float
    X110: float
    X111: float
    X112: float
    X113: float
    X114: float
    X115: float
    X116: float
    X117: float
    X118: float
    X119: float
    X120: float
    X121: float
    X122: float
    X123: float
    X124: float
    X125: float
    X126: float
    X127: float
    X128: float
    X129: float
    X130: float
    X131: float
    X132: float
    X133: float
    X134: float
    X135: float
    X136: float
    X137: float
    X138: float
    X139: float
    X140: float
    X141: float
    X142: float
    X143: float
    X144: float
    X145: float
    X146: float
    X147: float
    X148: float
    X149: float
    X150: float
    X151: float
    X152: float
    X153: float
    X154: float
    X155: float
    X156: float
    X157: float
    X158: float
    X159: float
    X160: float
    X161: float
    X162: float
    X163: float
    X164: float
    X165: float
    X166: float
    X167: float
    X168: float
    X169: float
    X170: float
    X171: float
    X172: float
    X173: float
    X174: float
    X175: float
    X176: float
    X177: float
    X178: float
    # Add more features as needed


# Define endpoint to make predictions
@app.post("/predict/")
async def predict(data: DataInput):
    try:
        # Convert input data to a numpy array
        input_data = np.array([[data.feature_1, data.feature_2]])  # Add more features as needed

        # Make prediction using the loaded model
        prediction = loaded_model.predict(input_data)

        # Return the prediction
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
