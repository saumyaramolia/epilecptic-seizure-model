import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Load the trained model
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Define input data model
class InputData(BaseModel):
    features: list


app = FastAPI()


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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))