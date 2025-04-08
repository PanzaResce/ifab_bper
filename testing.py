import functools, argparse, kagglehub, logging
import pandas as pd
import json
from typing import Annotated
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display

from utils.tools import python_repl
from utils.state import GeneratorSubgraphState
from agents.schema_analyzer import SchemaAnalyzer
from agents.generator import Generator
from agents.feedback import Feedback
from agents.data_profiler import DataProfiler
from utils.nodes import SchemaDescriptor, ValidityChecker

def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    args = parser.parse_args()
    return args
    
def to_serializable(obj):
        if hasattr(obj, "dict"):
            return obj.model_dump()
        # Fallback for any other non-serializable object
        return str(obj)

def import_dataframe(endpoint):
    path = kagglehub.dataset_download(endpoint)
    df = pd.read_csv(f"{path}/Base.csv")
    return df

def create_graph(llm, df):
    builder = StateGraph(GeneratorSubgraphState)
    schema_descriptor = SchemaDescriptor(df)
    schema_analyzer = SchemaAnalyzer(llm)
    data_profiler = DataProfiler(llm, df)
    generator = Generator(llm)
    # Define nodes: these do the work
    builder.add_node("schema_descriptor", schema_descriptor)
    #builder.add_node("schema_analyzer", schema_analyzer)
    builder.add_node("data_profiler", data_profiler)
    builder.add_node("generator", generator)

    '''builder.set_entry_point("schema_analyzer")
    builder.add_edge("schema_analyzer", "data_profiler")
    builder.add_edge("data_profiler", "generator")'''
    builder.set_entry_point("schema_descriptor")
    builder.add_edge("schema_descriptor", "data_profiler")
    builder.add_edge("data_profiler", "generator")
    # builder.add_edge("schema_descriptor", "generator")

    graph = builder.compile()
    img_data = graph.get_graph(xray=True).draw_mermaid_png()

    # Save the image to a file
    with open("output.png", "wb") as f:
        f.write(img_data)
    return graph

if __name__ == "__main__":
    args = parse_arguments()

    df = import_dataframe("sgpjesus/bank-account-fraud-dataset-neurips-2022")
    # tools = get_tools(df)
    # llm = ChatOllama(model="llama3.2")
    llm = ChatOllama(model="llama3.2", base_url="http://172.30.96.1:11434")

    graph = create_graph(llm, df)
    output = graph.invoke({})
    with open("graph_output.log", "w") as f:
        json_output = json.dumps(output, indent=4, default=to_serializable)
        f.write(json_output)