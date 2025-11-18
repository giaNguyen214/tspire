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
from fastapi.responses import StreamingResponse, FileResponse
import pandas as pd

# ====== TRAINING IMPORT ======
from utils.config import Config
from utils.utils import setup_logging
from train import run_training_pipeline

# ====== INFERENCE IMPORT ======
from inference import (
    load_all_models,
    generate_zip_results,   # dùng đúng 1 hàm duy nhất
)

# Ensure SHAP output dir exists
os.makedirs(Config.shap_api_dir, exist_ok=True)

app = FastAPI(
    title="Churn Model API",
    description="Train pipeline + Inference + SHAP",
)

@app.on_event("startup")
async def startup_event():
    setup_logging()
    load_all_models()


# ==============================================================
#                 TRAIN PIPELINE ENDPOINT
# ==============================================================

@app.post("/train-pipeline/")
async def train_pipeline_endpoint(file: UploadFile = File(...)):
    logger = logging.getLogger(__name__)

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(400, "Invalid file format")

    temp_file = os.path.join(Config.data_dir, f"upload_{uuid.uuid4()}{file_ext}")

    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_files = run_training_pipeline(temp_file)

        if not output_files:
            raise HTTPException(500, "Training pipeline failed")

        zip_io = io.BytesIO()
        with zipfile.ZipFile(zip_io, "w", zipfile.ZIP_DEFLATED) as zipf:
            for f in output_files:
                if os.path.exists(f):
                    zipf.write(f, arcname=os.path.basename(f))

        os.remove(temp_file)
        zip_io.seek(0)

        return StreamingResponse(
            zip_io,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=artifacts.zip"}
        )

    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

@app.get("/download/{filename}")
async def download_zip(filename: str):
    file_path = os.path.join(Config.shap_api_dir, filename)

    print("DOWNLOAD ROUTE → filename:", filename)
    print("ABS PATH:", file_path)
    print("EXISTS:", os.path.exists(file_path))

    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    return FileResponse(
        file_path,
        media_type="application/zip",
        filename=filename
    )


# ==============================================================
#                INFERENCE: SINGLE PREDICT (ZIP)
# ==============================================================

@app.post("/predict-single/")
async def api_predict_single(data: dict, shap: bool = True):
    """
    Predict 1 record:
    - return JSON: prediction + download_url
    """
    try:
        df = pd.DataFrame([data])

        zip_path, pred_df = generate_zip_results(
            df_raw=df,
            explain=shap,
            max_rows=1
        )

        filename = os.path.basename(zip_path)

        return {
            "prediction": pred_df.to_dict(orient="records")[0],
            "download_url": f"/download/{filename}"
        }

    except Exception as e:
        raise HTTPException(500, f"Predict-single failed: {e}")



# ==============================================================
#                INFERENCE: BATCH PREDICT (ZIP)
# ==============================================================

@app.post("/predict-batch/")
async def api_predict_batch(
    file: UploadFile = File(...),
    shap: bool = True,
    max_samples: int = 100
):
    """
    Batch JSON + link download.
    """
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".csv", ".xlsx"]:
            raise HTTPException(400, "Must upload CSV or XLSX")

        df = pd.read_csv(file.file) if ext == ".csv" else pd.read_excel(file.file)
        df = df.head(max_samples)

        zip_path, pred_df = generate_zip_results(
            df_raw=df,
            explain=shap,
            max_rows=max_samples
        )

        filename = os.path.basename(zip_path)

        return {
            "num_samples": len(pred_df),
            "predictions": pred_df.to_dict(orient="records"),
            "download_url": f"/download/{filename}"
        }

    except Exception as e:
        raise HTTPException(500, f"Predict-batch failed: {e}")



# ==============================================================
# ROOT
# ==============================================================

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Churn Model API",
        "endpoints": [
            "/train-pipeline/",
            "/predict-single/",
            "/predict-batch/"
        ]
    }
