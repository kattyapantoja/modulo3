from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agente import agente_controlador

app = FastAPI()

# Permitir frontend en local y producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # puedes restringir a tu dominio si lo deseas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    input: str

@app.post("/ask")
async def ask(query: Query):
    respuesta = agente_controlador.invoke({"input": query.input})
    return {"respuesta": respuesta.get("output", "Sin respuesta.")}
