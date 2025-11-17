"""
File: api.py
Entrypoint của FastAPI server.
"""
import logging
import os
import shutil
import io
import zipfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from config import Config
from pipeline_runner import run_training_pipeline, initialize_logging
from utils import setup_logging # Import setup_logging

app = FastAPI(
    title="Churn Model Training Pipeline API",
    description="Upload a CSV or XLSX file to train a full churn model pipeline.",
)

# Khởi tạo logging ngay khi app khởi động
@app.on_event("startup")
async def startup_event():
    setup_logging()

@app.post("/train-pipeline/")
async def train_pipeline_endpoint(file: UploadFile = File(...)):
    """
    Nhận file data (CSV/XLSX), chạy toàn bộ pipeline huấn luyện,
    và trả về 1 file ZIP chứa 5 model PKL và 1 file JSON metrics.
    
    CẢNH BÁO: Đây là một request đồng bộ (synchronous) và sẽ mất
    RẤT NHIỀU THỜI GIAN (vài phút đến vài giờ) để hoàn thành.
    Hãy đảm bảo client của bạn có timeout đủ dài.
    """
    logger = logging.getLogger(__name__)
    config = Config()
    
    # Xác định đường dẫn file tạm
    temp_file_path = config.temp_upload_path
    
    # Kiểm tra đuôi file
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload .csv, .xlsx, or .xls"
        )
    
    logger.info(f"Receiving uploaded file: {file.filename}")

    try:
        # 1. Lưu file upload vào file tạm
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved temporarily to {temp_file_path}")

        # 2. Chạy pipeline (Đây là bước tốn thời gian nhất)
        logger.info("Starting synchronous training pipeline...")
        # Hàm này sẽ chạy (vài phút) và trả về list các file path
        output_files = run_training_pipeline(temp_file_path)
        
        if not output_files:
            raise HTTPException(
                status_code=500,
                detail="Pipeline failed to run. Check server logs."
            )

        logger.info("Pipeline finished. Zipping output files...")
        
        # 3. Tạo file Zip trong bộ nhớ (in-memory)
        zip_io = io.BytesIO()
        with zipfile.ZipFile(zip_io, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for f_path in output_files:
                if os.path.exists(f_path):
                    # Ghi file vào zip, dùng tên file (basename)
                    zipf.write(f_path, arcname=os.path.basename(f_path))
                    logger.info(f"Added to zip: {f_path}")
                else:
                    logger.warning(f"File not found, skipping: {f_path}")

        # 4. Xóa file tạm
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Removed temp file: {temp_file_path}")
            
        # 5. Trả về file Zip
        zip_io.seek(0) # Đưa con trỏ về đầu file
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