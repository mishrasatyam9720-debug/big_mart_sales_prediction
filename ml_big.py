
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Input data structure
class ModelInput(BaseModel):
    Item_Identifier: str
    Item_Weight: float
    Item_Fat_Content: str
    Item_Visibility: float
    Item_Type: str
    Item_MRP: float
    Outlet_Identifier: str
    Outlet_Establishment_Year: float
    Outlet_Size: str
    Outlet_Location_Type: str
    Outlet_Type: str

# Load trained model
with open("random_forest_model.sav", "rb") as file:
    sales_model_classifier = pickle.load(file)


@app.post("/sales_prediction")
def sales_prediction(input_parameters: ModelInput):

    input_dictionary = input_parameters.model_dump()

    input_list = [
        input_dictionary["Item_Identifier"],
        input_dictionary["Item_Weight"],
        input_dictionary["Item_Fat_Content"],
        input_dictionary["Item_Visibility"],
        input_dictionary["Item_Type"],
        input_dictionary["Item_MRP"],
        input_dictionary["Outlet_Identifier"],
        input_dictionary["Outlet_Establishment_Year"],
        input_dictionary["Outlet_Size"],
        input_dictionary["Outlet_Location_Type"],
        input_dictionary["Outlet_Type"]
    ]

    input_array = np.array(input_list).reshape(1, -1)

    prediction = sales_model_classifier.predict(input_array)

    return {
        "predicted_sales": float(prediction[0])
    }
