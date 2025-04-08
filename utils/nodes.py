import pandas as pd
import os, csv
from utils.state import GeneratorSubgraphState
from langgraph.types import Command
from langgraph.graph import END
from typing_extensions import Literal

# GotoType = TypeVar("GotoType", bound=str)

class SchemaDescriptor():
    def __init__(self, data, name="Boh"):
        self.name = name
        self.data = data
    
    def __call__(self, state: GeneratorSubgraphState) -> pd.DataFrame:
        
        dtype_mapping = {
            'object': 'str',
            'int64': 'int',
            'int32': 'int',
            'float64': 'float',
            'float32': 'float',
            'bool': 'bool',
            'datetime64[ns]': 'datetime',
            'timedelta64[ns]': 'timedelta',
            'category': 'category',
            'complex128': 'complex',
            'complex64': 'complex',
            'UInt8': 'int',
            'UInt16': 'int',
            'UInt32': 'int',
            'UInt64': 'int',
            'Int8': 'int',
            'Int16': 'int',
            'Int32': 'int',
            'Int64': 'int',
            'string': 'str',
            'datetime64[ns, UTC]': 'datetime'}

        if isinstance(self.data, pd.DataFrame):
            pass
        else:
            self.data = self.castToDataFrame(self.data)
            
        friendly_dtypes = self.data.dtypes.replace(dtype_mapping)
        # friendly_dtypes_str = friendly_dtypes.to_string(header=False, index=True)
        friendly_dtypes_dict = friendly_dtypes.to_dict()
        random_record = self.data.sample().to_dict("records")[0]

        return {"df_row_schema": friendly_dtypes_dict, "example": random_record}
    
class ValidityChecker():
    def __init__(self, data, goto_if_valid, goto_if_notvalid, goto_if_maxiter, max_iterations=2):
        self.data = data
        self.goto_if_valid = goto_if_valid
        self.goto_if_notvalid = goto_if_notvalid
        self.goto_if_maxiter = goto_if_maxiter
        self.max_iterations = max_iterations
    
    def __call__(self, state: GeneratorSubgraphState) -> Command[Literal["__end__", "validation_feedback_agent"]]:
        # print(state.generated_row.generated_row)
        # print(df)
        print(f"Iter: {state.iteration_count}/{self.max_iterations}")
        new_row = state.generated_row.row
        error = self.is_valid_record(new_row, self.data)

        if error == "" and state.iteration_count <= self.max_iterations:
            self.save_to_file(new_row)
            return Command(
                goto=self.goto_if_valid
            )
        elif state.iteration_count > self.max_iterations:
            print(f"Max iteration reached: {state.iteration_count}/{self.max_iterations}")
            return Command(
                goto=self.goto_if_maxiter
            )
        else:
            return Command(
                update={"validation_errors": error},
                goto=self.goto_if_notvalid
            )

    def save_to_file(self, new_row):
        filename = 'synth_dataframe.csv'

        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=new_row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(new_row)

    def is_valid_record(self, record: dict, reference_df: pd.DataFrame, check_values: bool = False) -> bool:
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
