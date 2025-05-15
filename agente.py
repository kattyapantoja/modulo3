from langchain_core.runnables import RunnableLambda

# Este es un agente controlador de prueba
# Sustituye esto por tu agente real cuando esté listo

def dummy_response(input):
    pregunta = input.get("input", "")
    return {"output": f"Simulación de respuesta para: {pregunta}"}

agente_controlador = RunnableLambda(dummy_response)
