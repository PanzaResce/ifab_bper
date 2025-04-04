import pandas as pd
from utils.state import GlobalState
from langgraph.types import Command
from langgraph.graph import END
from typing_extensions import Literal

class SchemaDescriptorNode():
    def __init__(self, data, name="Boh"):
        self.name = name
        self.data = data
    
    def __call__(self, state) -> pd.DataFrame:
        
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
        friendly_dtypes_str = friendly_dtypes.to_string(header=False, index=True)
        return {"df_row_schema": friendly_dtypes_str}
    
class ValidityChecker():
    def __init__(self, data, goto_if_false):
        self.data = data
        self.goto_if_false = goto_if_false

    def __call__(self, state: GlobalState) -> Command[Literal["__end__", "validation_feedback_agent"]]:
        # print(state.generated_row.generated_row)
        print(f"NUM ITER: {state.iteration_count}")
        # print(df)
        generated_row = state.generated_row
        error = self.is_valid_record(generated_row.generated_row, self.data)

        if error == "" or state.iteration_count > 2:
            return Command(
                goto=END
            )
        else:
            return Command(
                update={"validation_errors": error},
                goto=self.goto_if_false
            )
        

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
