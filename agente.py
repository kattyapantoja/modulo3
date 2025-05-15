# Este es un ejemplo mínimo para corregir el string
# en la función calcular_costos dentro de agente.py

def calcular_costos(params):
    faltantes = ["azúcar", "harina"]
    return f"Faltan precios para:\n- " + "\n- ".join(faltantes)

# Dummy controlador para import correcto
from langchain_core.runnables import RunnableLambda
agente_controlador = RunnableLambda(lambda x: {"output": calcular_costos(None)})

__all__ = ["agente_controlador"]
