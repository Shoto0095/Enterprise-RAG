import asyncio
import json
import uuid
import os
import shutil
import time
from pathlib import Path
from .logger import get_logger
from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .chatbot import invoke as _invoke
from fastapi.templating import Jinja2Templates
from .helper_folder.helper_function import process_video_pipeline, ingest_pdf
from .helper_folder.job_status import JOB_STATUS, JOB_TIMEOUT

_logger = get_logger("api")
app = FastAPI(title="Video to PDF Transcription")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates_dir = Path(__file__).parent.parent/ "templates" 
templates = Jinja2Templates(directory=str(templates_dir))
# Serve React assets

# # Serve React app
# @app.get("/")
# async def serve_frontend():
#     return FileResponse(templates_dir / "index.html")

# @app.get("/html", response_class=HTMLResponse)
# async def chat_ui(request: Request):
#     return templates.TemplateResponse("chatbot.html", {
#         "request": request,
#         "title": "Video Analyzer Chatbot"
#     })
# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'Videos')
PDF_FOLDER = os.path.join(os.getcwd(), 'PDFs')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', "pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)


@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("templates/index.html")



@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Handle video file upload and start processing pipeline
    """
    # 🔥 FORCE CLEANUP ON RESTART
    for job_id, job in JOB_STATUS.items():
        if job.get("status") == "processing":
            job["status"] = "failed"
            job["message"] = "Job force-stopped due to server restart"
    _logger.info("✅ Cleaned up ongoing jobs on server restart")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    file_ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if file_ext not in ALLOWED_EXTENSIONS:
        _logger.error(f"File type not allowed: {file_ext}")
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
         # Create job entry
        job_id = str(uuid.uuid4())
        JOB_STATUS[job_id] = {
            "status": "processing",
            "started_at": time.time(),
            "message": "Processing started"
        }
        filename = file.filename
        content_type = (file.content_type or "").lower()
        _logger.info(f"Received file: {filename} of type: {content_type}")
        if content_type.startswith("video/"):
            video_path = os.path.join(UPLOAD_FOLDER, filename)

            # Save file
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)


            # Run pipeline in background
            background_tasks.add_task(
                process_video_pipeline,
                video_path,
                filename,
                job_id
            )

            return {
                "success": True,
                "job_id": job_id,
                "message": "Video uploaded. Processing started."
            }
        elif type == "application/pdf":
            pdf_path = os.path.join(PDF_FOLDER, filename)

            # Save file
            with open(pdf_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            success = ingest_pdf(pdf_path)

            if not success:
                _logger.error(f"PDF ingestion failed for {filename}")
                JOB_STATUS[job_id]["status"] = "failed"
                JOB_STATUS[job_id]["message"] = "Error occurred please try again later"
                raise HTTPException(status_code=500, detail="Error occurred please try again later")

            JOB_STATUS[job_id]["status"] = "success"
            JOB_STATUS[job_id]["message"] = "Processing completed successfully"

            return {
                "success": True,
                "job_id": job_id,
                "message": "PDF uploaded successfully. Processing completed."
            }
        else:
            _logger.error(f"Unsupported file type: {type}")
            raise HTTPException(status_code=400, detail="Unsupported file type")    

    except Exception as e:
        _logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = JOB_STATUS.get(job_id)

    if not job:
        return {"status": "unknown"}

    if job["status"] == "processing":
        if time.time() - job["started_at"] > JOB_TIMEOUT:
            job["status"] = "failed"
            job["message"] = "Processing timed out"

    return job

@app.post("/chatting")
async def chat(request: Request):
    user_query = None
    session_id = None
    try:
        content_type = (request.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            data = await request.json()
            if isinstance(data, dict):
                user_query = data.get("message")
                session_id = (data.get("session_id", "") or "").strip() or None
        elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            form = await request.form()
            user_query = form.get("message") if form else None
        else:
            user_query = request.query_params.get("message")
            if not user_query:
                try:
                    data = await request.json()
                    if isinstance(data, dict):
                        user_query = data.get("message")
                        session_id = (data.get("session_id", "") or "").strip() or None
                except Exception:
                    try:
                        form = await request.form()
                        user_query = form.get("message") if form else None
                    except Exception:
                        body = await request.body()
                        user_query = body.decode(errors="ignore").strip() if body else None
    except Exception:
        pass

    if user_query:
        user_query = str(user_query).strip()

    if not user_query:
        return JSONResponse({"error": "Please enter a message."}, status_code=400)

    if not session_id:
        session_id = request.headers.get("X-Session-ID", "").strip() or None

    if not session_id:
        session_id = str(uuid.uuid4())

    async def event_generator():
        yield json.dumps({"event": "status", "message": "AI Buzz is thinking...", "session_id": session_id}) + "\n"
        try:
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(None, lambda: _invoke(user_query, session_id))
            yield json.dumps({"event": "answer", "reply": answer, "session_id": session_id}) + "\n"
        except Exception as e:
            yield json.dumps({"event": "error", "message": str(e), "session_id": session_id}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
        
# # (optional) React Router support
# @app.get("/{path:path}")
# async def react_router(path: str):
#     return FileResponse(STATIC_DIR / "index.html")

        