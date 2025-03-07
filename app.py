from enum import StrEnum
import uuid
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, Form
import requests
from auth import APIKeyMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor

from pdf_to_markdown import ConverterByGPT, ConverterByDocumentIntelligence


class Status(StrEnum):
    RUNNING = 'running'
    FINISHED = 'finished'
    FAILED = 'failed'

class ResponseData(BaseModel):
    status: Status
    output_gpt: Optional[str] = None
    output_document: Optional[str] = None
    error: Optional[str] = None

store: Dict[str, ResponseData] = {}

app = FastAPI()
app.add_middleware(APIKeyMiddleware)

def run_kickoff(pdf_content: bytes, job_id: str, hook_url: str):
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both converter tasks to the executor
            future_gpt = executor.submit(ConverterByGPT(job_id).convert_pdf, pdf_content=pdf_content)
            future_document = executor.submit(ConverterByDocumentIntelligence().convert_pdf, pdf_content=pdf_content)

            # Wait for both futures to complete and get their results
            output_gpt = future_gpt.result()
            output_document = future_document.result()

        # Store the results in the shared store
        # store[job_id] = ResponseData(status=Status.FINISHED, output_gpt=output_gpt, output_document=output_document)

        # call hook_url with parsing result
        requests.post(hook_url, json={
            "status": Status.FINISHED,
            "output_gpt": output_gpt,
            "output_document": output_document
        })

        # Free the variables
        del output_gpt
        del output_document
    except Exception as e:
        # store[job_id] = ResponseData(status=Status.FAILED, error=str(e))

        # call hook_url with parsing result
        requests.post(hook_url, json={
            "status": Status.FAILED,
            "error": str(e)
        })

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/kickoff")
async def convert_pdf_to_markdown(background_tasks: BackgroundTasks, hook_url: str= Form(...), file: UploadFile = File(...)):
    try:
        job_id = str(uuid.uuid4())
        pdf_content = await file.read()

        background_tasks.add_task(run_kickoff, pdf_content, job_id, hook_url)
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
