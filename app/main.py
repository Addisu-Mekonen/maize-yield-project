import shutil
import tempfile
import os
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.schemas import TabularInput, PredictionResponse
from app.model_utils import fuse_prediction

app = FastAPI(title="Maize Yield Prediction API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Maize Yield Prediction API is running."}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    tabular_data: str = Form(...),
):
    # Parse and validate the JSON string against our Pydantic schema.
    # Raises a proper 400 error instead of letting a malformed request
    # fall through and mismatch our response_model (which caused a 500
    # earlier — response_model validation failures surface as 500s, not
    # clean 4xx errors, so input validation must be handled explicitly).
    try:
        parsed = json.loads(tabular_data)
        tabular_input = TabularInput(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid tabular_data: {str(e)}")

    # Save uploaded image to a temporary file, since our pipeline expects a file path
    suffix = os.path.splitext(image.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        result = fuse_prediction(tmp_path, tabular_input.dict())
    finally:
        os.remove(tmp_path)  # always clean up the temp file, even if prediction fails

    return result
