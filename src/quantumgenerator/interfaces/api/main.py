from fastapi import FastAPI
from quantumgenerator.interfaces.api.routes.generation import router


app = FastAPI()
app.include_router(router)
