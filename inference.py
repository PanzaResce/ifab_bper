import functools, argparse, kagglehub
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
from utils.state import GlobalState
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
        return "The record keys does not match with the input schema"

    # Check data types
    for key, value in record.items():
        expected_type = reference_df[key].dtype
        try:
            pd.Series([value], dtype=expected_type)
        except ValueError as e:
            return e
    
    return ""

# def validity_checker(state: GlobalState, df):
#     # print(state.generated_row.generated_row)
#     print(f"NUM ITER: {state.iteration_count}")
#     # print(df)
#     generated_row = state.generated_row
#     error = is_valid_record(generated_row.generated_row, df)
#     if error == "" or state.iteration_count > 2:
#         return END
#     else:
#         state.validation_errors = error
#         print(f"ERROR: {state.validation_errors}")
#         return "validation_feedback_agent"

def create_graph(llm, df):
    builder = StateGraph(GlobalState)
    # analyzer = SchemaAnalyzer(llm, tools=[python_repl(df)])
    # analyzer = SchemaAnalyzer(llm)
    profiler = DataProfiler(llm, df)
    generator = Generator(llm)
    validation_feedback = Feedback(llm)
    schema_descriptor = SchemaDescriptor(df)
    validity_checker = ValidityChecker(df, goto_if_valid="__end__", goto_if_notvalid="validation_feedback_agent", max_iterations=5)

    # Define nodes: these do the work
    # builder.add_node("schema_analyzer", analyzer)
    builder.add_node("schema_descriptor", schema_descriptor)
    # builder.add_node("data_profiler", profiler)
    builder.add_node("generator", generator)
    builder.add_node("validation_feedback_agent", validation_feedback)
    builder.add_node("validity_checker", validity_checker)

    builder.set_entry_point("schema_descriptor")
    # builder.add_edge("schema_descriptor", "data_profiler")
    # builder.add_edge("data_profiler", "generator")
    builder.add_edge("schema_descriptor", "generator")
    builder.add_edge("generator", "validity_checker")
    # builder.add_conditional_edges("generator", validity_checker, [END, "validation_feedback_agent"])
    builder.add_edge("validation_feedback_agent", "generator")

    # builder.add_edge("generator", END)

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
    llm = ChatOllama(model="llama3.2")
    # llm = ChatOllama(model="llama3.2", base_url="http://172.30.96.1:11434")

    graph = create_graph(llm, df)
    # messages = [HumanMessage(content="Generate a new row for the input dataset.")]
    # inputs = {"messages": [("user", "Generate a new row for the input dataset.")]}
    
    output = graph.invoke({})
    
    with open("graph_output.log", "w") as f:
        json_output = json.dumps(output, indent=4, default=to_serializable)
        f.write(json_output)
