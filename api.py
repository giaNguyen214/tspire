"""
File: api.py
FastAPI server: Training + Inference + SHAP
"""

import logging
import os
import shutil
import io
import zipfile
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd

# ====== TRAINING IMPORT ======
from utils.config import Config
from utils.utils import setup_logging
from train import run_training_pipeline

# ====== INFERENCE IMPORT ======
from inference import (
    load_all_models,
    predict_single,
    predict_batch,
    compute_shap_outputs,
)

# Ensure SHAP output dir exists
os.makedirs(Config.SHAP_OUTPUT_DIR, exist_ok=True)

app = FastAPI(
    title="Churn Model API",
    description="Train pipeline + Inference + SHAP",
)

@app.on_event("startup")
async def startup_event():
    setup_logging()
    load_all_models()  # load models once at startup


# ==============================================================
#                 TRAIN PIPELINE ENDPOINT
# ==============================================================

@app.post("/train-pipeline/")
async def train_pipeline_endpoint(file: UploadFile = File(...)):
    """
    Upload CSV/XLSX → train pipeline → return ZIP of all models + metrics.
    """
    logger = logging.getLogger(__name__)

    # Validate extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload .csv, .xlsx, or .xls"
        )

    # Create temp file inside data dir
    temp_file_name = f"temp_upload_{uuid.uuid4()}{file_ext}"
    temp_file_path = os.path.join(Config.data_dir, temp_file_name)

    logger.info(f"Saving uploaded file to: {temp_file_path}")

    try:
        # Save upload
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info("Running training pipeline...")
        output_files = run_training_pipeline(temp_file_path)

        if not output_files:
            raise HTTPException(500, "Pipeline failed. Check logs.")

        # Build ZIP in-memory
        zip_io = io.BytesIO()
        with zipfile.ZipFile(zip_io, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for f_path in output_files:
                if os.path.exists(f_path):
                    zipf.write(f_path, arcname=os.path.basename(f_path))

        # Remove temp file
        os.remove(temp_file_path)

        zip_io.seek(0)
        return StreamingResponse(
            zip_io,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=artifacts.zip"}
        )

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(500, str(e))


# ==============================================================
#                  INFERENCE: SINGLE SAMPLE
# ==============================================================

@app.post("/predict-single/")
async def api_predict_single(data: dict, shap: bool = True, waterfall_index: int = 0):
    """
    Predict for a single record.
    - data: dict (1 sample)
    - shap: bool (default ON)
    - waterfall_index: used for SHAP waterfall
    """
    try:
        df = pd.DataFrame([data])
        result = predict_single(df)

        shap_paths = None
        if shap:
            shap_paths = compute_shap_outputs(df, waterfall_index=waterfall_index)

        return JSONResponse({
            "prediction": result,
            "shap_files": shap_paths if shap else None
        })

    except Exception as e:
        raise HTTPException(500, f"Predict-single failed: {e}")


# ==============================================================
#                  INFERENCE: BATCH PREDICT
# ==============================================================

@app.post("/predict-batch/")
async def api_predict_batch(
    file: UploadFile = File(...),
    shap: bool = True,
    max_samples: int = 100,
    waterfall_index: int = 0
):
    """
    Batch prediction:
    - Upload CSV/XLSX
    - SHAP summary/bar/waterfall (waterfall only first sample or chosen index)
    """
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".csv", ".xlsx"]:
            raise HTTPException(400, "File must be CSV or XLSX")

        # Load file
        if ext == ".csv":
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)

        df = df.head(max_samples)

        # Predict
        pred_df = predict_batch(df)

        # SHAP
        shap_paths = None
        if shap:
            shap_paths = compute_shap_outputs(df, waterfall_index=waterfall_index)

        # Return JSON only (predictions + shap paths)
        return JSONResponse({
            "num_samples": len(df),
            "results": pred_df.to_dict(orient="records"),
            "shap_files": shap_paths
        })

    except Exception as e:
        raise HTTPException(500, f"Predict-batch failed: {e}")


# ==============================================================
# ROOT
# ==============================================================

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Churn Model API.",
        "endpoints": [
            "/train-pipeline/",
            "/predict-single/",
            "/predict-batch/"
        ]
    }
