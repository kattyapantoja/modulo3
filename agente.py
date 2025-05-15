import os
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableBranch, RunnableLambda

# Carga de llaves si se usan archivos .env (opcional)
from dotenv import load_dotenv
load_dotenv()

# Configuración LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Prompt base
base_prompt = hub.pull("hwchase17/react")

# Herramientas simuladas
def dummy_stock_checker(_):
    return "Todos los insumos están en stock."

def dummy_cost_calculator(params):
    return "El costo estimado es S/25.00 y el precio sugerido de venta es S/37.50."

def dummy_order_checker(params=None):
    return "Tienes 2 pedidos programados para el fin de semana."

tools_inventario = [Tool(name="ver_productos_con_stock_bajo", func=dummy_stock_checker, description="Muestra los insumos con niveles de stock bajos.")]
tools_costos = [Tool(name="calcular_costos", func=dummy_cost_calculator, description="Calcula el costo de un producto.")]
tools_pedidos = [Tool(name="ver_pedidos_programados", func=dummy_order_checker, description="Muestra los pedidos programados.")]

# Agentes
agent_inventario = create_react_agent(llm=llm, tools=tools_inventario, prompt=base_prompt)
agent_costos = create_react_agent(llm=llm, tools=tools_costos, prompt=base_prompt)
agent_pedidos = create_react_agent(llm=llm, tools=tools_pedidos, prompt=base_prompt)

# Ejecutores con manejo de intermediate_steps vacío
executor_inventario = AgentExecutor(agent=agent_inventario, tools=tools_inventario, verbose=True)
executor_costos = AgentExecutor(agent=agent_costos, tools=tools_costos, verbose=True)
executor_pedidos = AgentExecutor(agent=agent_pedidos, tools=tools_pedidos, verbose=True)

# Redirección de agente
def redirigir_a_agente(input):
    texto = input["input"].lower()
    if "stock" in texto or "insumos" in texto:
        return executor_inventario
    elif "pedido" in texto or "entrega" in texto:
        return executor_pedidos
    elif "precio" in texto or "cuánto" in texto or "costos" in texto:
        return executor_costos
    else:
        return executor_costos  # fallback temporal

# Rama de control
agente_controlador = RunnableBranch(
    (lambda x: True, RunnableLambda(lambda x: redirigir_a_agente(x).invoke(x))),
    RunnableLambda(lambda x: {"output": "Lo siento, no entendí tu consulta."})
)
