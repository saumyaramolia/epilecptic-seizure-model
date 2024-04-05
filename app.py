import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Load the trained model
try:
    with open('decision_tree_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    raise RuntimeError("Failed to load the model: {}".format(str(e)))


# Define input data model
class InputData(BaseModel):
    features: list


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace ["*"] with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(data: InputData):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)

        # Interpret the prediction
        if prediction[0] == 1:
            result = "Going to have a seizure"
        else:
            result = "Not having a seizure"

        return {"prediction": int(prediction[0]), "result": result}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail="Bad request: {}".format(str(ve)))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: {}".format(str(e)))
