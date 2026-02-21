from fastapi import FastAPI, HTTPException
from ..constants import HEALTHY_STATUS, SERVICE_NAME, API_VERSION


app = FastAPI()

@app.get("/health")
def heath_check():
    return {
        "status": HEALTHY_STATUS,
        "service": SERVICE_NAME,
        "version": API_VERSION,
    }