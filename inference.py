import functools, argparse, kagglehub
import pandas as pd
from typing import Annotated
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display

from utils.tools import python_repl
from utils.state import GlobalState
from agents.schema_analyzer import SchemaAnalyzer


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    args = parser.parse_args()
    return args

def import_dataframe(endpoint):
    path = kagglehub.dataset_download(endpoint)
    df = pd.read_csv(f"{path}/Base.csv")
    return df

def create_graph(llm, df):
    builder = StateGraph(GlobalState)
    analyzer = SchemaAnalyzer(llm, tools=[python_repl(df)])
    # schema_agent = create_agent_node(schema_analyzer, llm, [repl_tool], "Schema Analyzer")
    # schema_agent = create_agent_node(schema_analyzer, llm, [], "Schema Analyzer")

    # Define nodes: these do the work
    builder.add_node("schema_analyzer", analyzer)
    builder.set_entry_point("schema_analyzer")

    graph = builder.compile()
    return graph

if __name__ == "__main__":
    args = parse_arguments()

    df = import_dataframe("sgpjesus/bank-account-fraud-dataset-neurips-2022")
    # tools = get_tools(df)
    llm = ChatOllama(model="llama3.2")

    graph = create_graph(llm, df)
    messages = [HumanMessage(content="Generate a new row for the input dataset.")]
    # inputs = {"messages": [("user", "Generate a new row for the input dataset.")]}

    print(graph.invoke({}))