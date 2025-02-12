from enum import StrEnum
import uuid
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from auth import APIKeyMiddleware
from pydantic import BaseModel
from typing import Optional, Union, Dict

from pdf_to_markdown import PdfToMarkdownConverter


class Status(StrEnum):
    RUNNING = 'running'
    FINISHED = 'finished'
    FAILED = 'failed'

class ResponseData(BaseModel):
    status: Status
    output: Optional[str] = None
    error: Optional[str] = None

store: Dict[str, ResponseData] = {}

app = FastAPI()
app.add_middleware(APIKeyMiddleware)

def run_kickoff(pdf_content: bytes, job_id: str):
    try:
        converter = PdfToMarkdownConverter(job_id)
        output = converter.convert_pdf(pdf_content=pdf_content)
        store[job_id] = ResponseData(status=Status.FINISHED, output=output)
    except Exception as e:
        store[job_id] = ResponseData(status=Status.FAILED, error=str(e))

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/kickoff")
async def convert_pdf_to_markdown(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        job_id = str(uuid.uuid4())
        pdf_content = await file.read()

        background_tasks.add_task(run_kickoff, pdf_content, job_id)
        store[job_id] = ResponseData(status=Status.RUNNING)
        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    try:
        if job_id not in store:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        response = store[job_id]
        if not response.status == Status.RUNNING:
            del store[job_id]

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
