"""
File: api.py
Entrypoint của FastAPI server.
"""
import logging
import os
import shutil
import io
import zipfile
import uuid 
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from .config import Config
from .pipeline_runner import run_training_pipeline
from .utils import setup_logging 

app = FastAPI(
    title="Churn Model Training Pipeline API",
    description="Upload a CSV or XLSX file to train a full churn model pipeline.",
)

@app.on_event("startup")
async def startup_event():
    setup_logging()

@app.post("/train-pipeline/")
async def train_pipeline_endpoint(file: UploadFile = File(...)):
    """
    Nhận file data (CSV/XLSX), chạy toàn bộ pipeline huấn luyện,
    và trả về 1 file ZIP chứa 5 model PKL và 1 file JSON metrics.
    """
    logger = logging.getLogger(__name__)
    
    # --- SỬA LỖI (Vấn đề 2) ---
    
    # 1. Kiểm tra đuôi file gốc
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload .csv, .xlsx, or .xls"
        )
    
    # 2. Tạo đường dẫn file tạm ĐỘNG (trong thư mục /data)
    temp_file_name = f"temp_upload_{uuid.uuid4()}{file_ext}"
    # SỬA: Dùng Config.data_dir (thay vì Config.BASE_DIR)
    temp_file_path = os.path.join(Config.data_dir, temp_file_name) 
    
    # --- Hết phần sửa ---

    logger.info(f"Receiving uploaded file: {file.filename}")

    try:
        # 3. Lưu file upload vào file tạm (với đường dẫn mới)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved temporarily to {temp_file_path}")

        # 4. Chạy pipeline (với đường dẫn mới)
        logger.info("Starting synchronous training pipeline...")
        output_files = run_training_pipeline(temp_file_path)
        
        if not output_files:
            raise HTTPException(
                status_code=500,
                detail="Pipeline failed to run. Check server logs."
            )

        logger.info("Pipeline finished. Zipping output files...")
        
        # 5. Tạo file Zip trong bộ nhớ (in-memory)
        zip_io = io.BytesIO()
        with zipfile.ZipFile(zip_io, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for f_path in output_files:
                if os.path.exists(f_path):
                    zipf.write(f_path, arcname=os.path.basename(f_path))
                    logger.info(f"Added to zip: {f_path}")
                else:
                    logger.warning(f"File not found, skipping: {f_path}")

        # 6. Xóa file tạm
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Removed temp file: {temp_file_path}")
            
        # 7. Trả về file Zip
        zip_io.seek(0)
        return StreamingResponse(
            content=zip_io,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=churn_model_artifacts.zip"
            }
        )

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        # Xóa file tạm nếu có lỗi
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Model Training API. POST to /train-pipeline/ to start."}