from typing import Dict
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
import uvicorn

from agent import call_agent, call_manager
from helper import extract_pdf_text
app = FastAPI()

class ApplicationInformation(BaseModel):
    name: str
    

@app.get("/health")
def read_root():
    return {"we are healthy"}

@app.post("/upload/")
def upload_and_submit(jd: str = Form(...), file: UploadFile = File(...)):
    # file_bytes = file.file.read()
    # UploadFile returns bytes, decode to text before passing to the agent.
    file_text = extract_pdf_text(file.file)

    print('calling the agent.')
    call_agent("mistralai/mistral-small-latest", file_text, jd)
    # result: Dict = call_manager(file_text, jd)
    print('agent calling is finished.')
    return {"CV": result['cv'], "Cover Letter": result['cover_letter'], "Gap Analysis": result['gap_analysis']}

@app.get("/")
def index():

    return {"Hello": "World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
