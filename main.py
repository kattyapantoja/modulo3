from fastapi import FastAPI, Request
from pydantic import BaseModel
from agente import agente_controlador
import dotenv
import os

dotenv.load_dotenv()

app = FastAPI()

class Query(BaseModel):
    input: str

@app.post("/ask")
async def ask(query: Query):
    respuesta = agente_controlador.invoke({"input": query.input})
    return {"respuesta": respuesta.get("output", "Sin respuesta.")}
