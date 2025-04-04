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
from agents.generator import Generator
from agents.feedback import Feedback


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    args = parser.parse_args()
    return args

def import_dataframe(endpoint):
    path = kagglehub.dataset_download(endpoint)
    df = pd.read_csv(f"{path}/Base.csv")
    return df

def is_valid_record(record: dict, reference_df: pd.DataFrame, check_values: bool = False) -> bool:
    """
    Checks if a given record (as a dictionary) is valid with respect to a reference DataFrame.
    
    Parameters:
    - record (dict): The record to validate.
    - reference_df (pd.DataFrame): The DataFrame that defines the schema and value constraints.
    - check_values (bool): If True, also checks if the record values are consistent with known values in the DataFrame.

    Returns:
    - bool: True if the record is valid, False otherwise.
    """
    # Check if keys match columns
    if set(record.keys()) != set(reference_df.columns):
        return False

    # Check data types
    for key, value in record.items():
        expected_type = reference_df[key].dtype
        try:
            pd.Series([value], dtype=expected_type)
        except ValueError:
            return False

def validity_checker(state: GlobalState, df):
    # print(state.generated_row.generated_row)
    print(f"NUM ITER: {state.iteration_count}")
    # print(df)
    generated_row = state.generated_row
    if is_valid_record(generated_row.generated_row, df) or state.iteration_count > 2:
        return END
    else:
        return "validation_feedback_agent"

def create_graph(llm, df):
    builder = StateGraph(GlobalState)
    # analyzer = SchemaAnalyzer(llm, tools=[python_repl(df)])
    analyzer = SchemaAnalyzer(llm)
    generator = Generator(llm)
    validation_feedback = Feedback(llm)

    # Define nodes: these do the work
    builder.add_node("schema_analyzer", analyzer)
    builder.add_node("generator", generator)
    builder.add_node("validation_feedback_agent", validation_feedback)
    builder.set_entry_point("schema_analyzer")

    builder.add_edge("schema_analyzer", "generator")
    builder.add_edge("validation_feedback_agent", "generator")
    builder.add_conditional_edges("generator", lambda state: validity_checker(state, df))

    # builder.add_edge("generator", END)

    graph = builder.compile()
    return graph

if __name__ == "__main__":
    args = parse_arguments()

    df = import_dataframe("sgpjesus/bank-account-fraud-dataset-neurips-2022")
    # tools = get_tools(df)
    llm = ChatOllama(model="llama3.2")

    graph = create_graph(llm, df)
    # messages = [HumanMessage(content="Generate a new row for the input dataset.")]
    # inputs = {"messages": [("user", "Generate a new row for the input dataset.")]}

    print(graph.invoke({}))