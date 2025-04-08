import argparse, kagglehub
import pandas as pd
import json
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from utils.tools import python_repl
from utils.state import GeneratorSubgraphState
from agents.generator import Generator
from agents.feedback import Feedback
from utils.nodes import SchemaDescriptor, ValidityChecker

def parse_arguments():
    parser = argparse.ArgumentParser(description="")
        
    parser.add_argument(
        "--log-level",
        default="ERROR",
        choices=["DEBUG", "INFO", "ERROR"],
        help="Set the logging level (default: ERROR)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="The maximum number of times the generator can produce erroneous records"
    )

    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="How many new rows to generate "
    )

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

def create_graph(llm, df, max_iterations):
    builder = StateGraph(GeneratorSubgraphState)
    # analyzer = SchemaAnalyzer(llm)
    # profiler = DataProfiler(llm, df)
    generator = Generator(llm)
    validation_feedback = Feedback(llm)
    schema_descriptor = SchemaDescriptor(df)
    validity_checker = ValidityChecker(df, goto_if_valid="__end__", goto_if_maxiter="__end__", goto_if_notvalid="validation_feedback_agent", max_iterations=max_iterations)

    # Define nodes: these do the work
    # builder.add_node("schema_analyzer", analyzer)
    builder.add_node("schema_descriptor", schema_descriptor)
    # builder.add_node("data_profiler", profiler)
    builder.add_node("generator", generator)
    builder.add_node("validation_feedback_agent", validation_feedback)
    builder.add_node("validity_checker", validity_checker)

    builder.set_entry_point("schema_descriptor")
    builder.add_edge("schema_descriptor", "generator")
    builder.add_edge("generator", "validity_checker")
    builder.add_edge("validation_feedback_agent", "generator")


    graph = builder.compile()
    img_data = graph.get_graph(xray=True).draw_mermaid_png()

    # Save the image to a file
    with open("output.png", "wb") as f:
        f.write(img_data)
    return graph

if __name__ == "__main__":
    args = parse_arguments()
    # model_name = "llama3.2"
    model_name = "gemma3"

    df = import_dataframe("sgpjesus/bank-account-fraud-dataset-neurips-2022")
    llm = ChatOllama(model=model_name)

    graph = create_graph(llm, df, args.max_iterations)
    
    for _ in range(args.num):
        output = graph.invoke({})
    
    with open("graph_output.log", "w") as f:
        json_output = json.dumps(output, indent=4, default=to_serializable)
        f.write(json_output)
