import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import YouTubeSearchTool
from langchain_elasticsearch import ElasticsearchStore
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import fitz  # PyMuPDF

# Cargar claves y datos
load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

productos_df = pd.read_csv("productos_kattys.csv")
recetas_df = pd.read_csv("recetas_katty.csv")
inventario_df = pd.read_csv("inventario_katty.csv")
precios_df = pd.read_csv("insumos_con_precios.csv")
pedidos_df = pd.read_csv("pedidos_programados.csv")

with open("reglas_empleados.txt") as f:
    reglas = f.read()

# Herramientas CSV
def calcular_costos(params):
    if isinstance(params, str):
        productos_disponibles = recetas_df["producto"].unique()
        producto = next((p for p in productos_disponibles if p.lower() in params.lower()), None)
        if not producto:
            return "No pude identificar el producto."
    elif isinstance(params, dict):
        producto = params.get("producto")
        if not producto:
            return "Debes especificar el nombre del producto."
    else:
        return "Formato inválido."

    ingredientes = recetas_df[recetas_df['producto'] == producto]
    if ingredientes.empty:
        return f"No se encontró receta para '{producto}'."

    total, detalles, faltantes = 0, [], []
    for _, fila in ingredientes.iterrows():
        insumo = fila["insumo"]
        cantidad = fila["cantidad"]
        fila_precio = precios_df[precios_df['insumo'] == insumo]
        if fila_precio.empty:
            faltantes.append(insumo)
            continue
        precio_unitario = fila_precio.iloc[0]["precio_unitario"]
        costo = cantidad * precio_unitario
        total += costo
        detalles.append(f"{insumo}: {cantidad} x S/{precio_unitario:.2f} = S/{costo:.2f}")

    if faltantes:
        return f"Faltan precios para:
- " + "\n- ".join(faltantes)
    precio_sugerido = round(total * 1.5, 2)
    return f"""Costo estimado: S/{total:.2f}
Precio sugerido: S/{precio_sugerido:.2f}
Desglose:
""" + "\n".join(detalles)

def ver_productos_con_stock_bajo(_input=None):
    bajo_stock = inventario_df[inventario_df['cantidad_actual'] < inventario_df['cantidad_minima']]
    if bajo_stock.empty:
        return "Todos los insumos están en niveles adecuados."
    return "Stock bajo:
" + "\n".join([
        f"- {row['insumo']}: {row['cantidad_actual']} (mínimo: {row['cantidad_minima']})"
        for _, row in bajo_stock.iterrows()
    ])

def ver_pedidos_programados(fecha_str=None):
    if isinstance(fecha_str, dict) and "fecha" in fecha_str:
        fecha_str = fecha_str["fecha"]
    try:
        fecha = datetime.strptime(fecha_str, "%Y-%m-%d").date()
        filtrado = pedidos_df[pedidos_df['fecha_entrega'] == fecha_str]
    except:
        filtrado = pedidos_df
    if filtrado.empty:
        return "No hay pedidos para esa fecha."
    return "\n".join([
        f"- {row['cliente']} - {row['producto']} ({row['fecha_entrega']}) - {row['estado']}"
        for _, row in filtrado.iterrows()
    ])

# PDFs como documentos LangChain
def load_pdf_as_documents(filepath, titulo, categoria, origen, idioma):
    pdf = fitz.open(filepath)
    docs = []
    for i, page in enumerate(pdf):
        text = page.get_text()
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={
                    "page": i + 1,
                    "categoria": categoria,
                    "idioma": idioma,
                    "origen": origen,
                    "titulo": titulo
                }
            ))
    return docs

docs_horno = load_pdf_as_documents("Manual_uso_de_horno.pdf", "Manual de Uso del Horno", "uso_horno", "miele", "es")
docs_practicas = load_pdf_as_documents("Guia_de_buenas_practicas_de_higiene_en_establecimientos.pdf", "Guía de Higiene", "buenas_practicas", "gob.mx", "es")

vector_store = ElasticsearchStore.from_documents(
    documents=docs_horno + docs_practicas,
    embedding=embeddings,
    es_url=os.environ["ES_URL"],
    es_user=os.environ["ES_USER"],
    es_password=os.environ["ES_PASSWORD"],
    index_name="indx_02"
)
vector_store.client.indices.refresh(index="indx_02")

retriever_horno = vector_store.as_retriever(search_kwargs={"filter": [{"term": {"categoria": "uso_horno"}}]})
retriever_buenas = vector_store.as_retriever(search_kwargs={"filter": [{"term": {"categoria": "buenas_practicas"}}]})

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Eres experto en normas de higiene y uso de hornos profesionales. Responde SIEMPRE en español.

=== CONTEXTO ===
{context}

=== PREGUNTA ===
{question}

=== RESPUESTA EN ESPAÑOL ==="""
)

qa_chain_horno = RetrievalQA.from_chain_type(llm=llm, retriever=retriever_horno, return_source_documents=True, chain_type_kwargs={"prompt": qa_prompt})
qa_chain_buenas = RetrievalQA.from_chain_type(llm=llm, retriever=retriever_buenas, return_source_documents=True, chain_type_kwargs={"prompt": qa_prompt})

def consultar_manual_horno(pregunta):
    return qa_chain_horno.invoke({"query": pregunta})["result"]

def consultar_buenas_practicas(pregunta):
    return qa_chain_buenas.invoke({"query": pregunta})["result"]

# Herramientas
tools_costos = [Tool(name="calcular_costos", func=calcular_costos, description="Calcula el costo de un producto")]
tools_inventario = [Tool(name="ver_productos_con_stock_bajo", func=ver_productos_con_stock_bajo, description="Lista los insumos con bajo stock.")]
tools_pedidos = [Tool(name="ver_pedidos_programados", func=ver_pedidos_programados, description="Muestra pedidos programados.")]
tools_consultas = [
    Tool(name="consultar_manual_horno", func=consultar_manual_horno, description="Consulta el manual de uso del horno."),
    Tool(name="consultar_buenas_practicas", func=consultar_buenas_practicas, description="Consulta normas de higiene.")
]
tools_web = [
    Tool(name="buscar_web", func=TavilySearchResults().run, description="Busca en internet sobre ingredientes y recetas."),
    Tool(name="buscar_youtube", func=YouTubeSearchTool().run, description="Busca videos de recetas y técnicas.")
]

prompt = PromptTemplate(
    template="Siempre responde en español.\n\n" + reglas + "\n\n" + hub.pull("hwchase17/react").template,
    input_variables=hub.pull("hwchase17/react").input_variables
)

# Agentes
agent_costos = create_react_agent(llm=llm, tools=tools_costos, prompt=prompt)
agent_inventario = create_react_agent(llm=llm, tools=tools_inventario, prompt=prompt)
agent_pedidos = create_react_agent(llm=llm, tools=tools_pedidos, prompt=prompt)
agent_consultas = create_react_agent(llm=llm, tools=tools_consultas, prompt=prompt)
agent_web = create_react_agent(llm=llm, tools=tools_web, prompt=prompt)

executor_costos = AgentExecutor(agent=agent_costos, tools=tools_costos, verbose=True)
executor_inventario = AgentExecutor(agent=agent_inventario, tools=tools_inventario, verbose=True)
executor_pedidos = AgentExecutor(agent=agent_pedidos, tools=tools_pedidos, verbose=True)
executor_consultas = AgentExecutor(agent=agent_consultas, tools=tools_consultas, verbose=True)
executor_web = AgentExecutor(agent=agent_web, tools=tools_web, verbose=True)

# Redirección automática
def redirigir_a_agente(input):
    texto = input["input"].lower()
    if "stock" in texto or "insumo" in texto:
        return executor_inventario
    elif "pedido" in texto or "entrega" in texto:
        return executor_pedidos
    elif "costo" in texto or "precio" in texto or "cuánto" in texto:
        return executor_costos
    elif "horno" in texto or "manual" in texto or "norma" in texto or "higiene" in texto:
        return executor_consultas
    else:
        return executor_web

agente_controlador = RunnableBranch(
    (lambda x: True, RunnableLambda(lambda x: redirigir_a_agente(x).invoke(x))),
    RunnableLambda(lambda x: {"output": "Lo siento, no entendí tu consulta."})
)

__all__ = ["agente_controlador"]
