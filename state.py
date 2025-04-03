from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

class DataframeCol(BaseModel):
    column_name: str = Field(description="The name of the column")
    column_descr: str = Field(description="The description of the column")
    column_type: str = Field(description="The data type of the column")

class OverallState(BaseModel):
    df_row_schema: Optional [List[DataframeCol]] = Field(
        description="The schema of the dataframe, including its column names, description and data",
        default=[]
        )
    stats: Optional [Dict[str, Any]] = Field(
        description="Statistical profile of the data to guide generation",
        default={})
    generated_row: Optional [List[DataframeCol]] = Field(
        description="The synthetic row generated.",
        default=[])
    is_row_valid: Optional [bool] = Field(
        description="Whether the synthetic row generetad is valid given the input schema",
        default=None)
    is_row_plausible: Optional [bool] = Field(
        description="Whether the synthetic row generetad is plausible given the input dataset",
        default=None)
    validation_errors: Optional[Dict[str, str]] = Field(
        description="Schema violations for each row during validation, if any",
        default={})
    plausibility_feedback: Optional[str] = Field(
        description="Natural language guidance from plausibility checker suggesting improvements to the synthetic row",
        default="")
    # messages: Annotated[List[AnyMessage], add_messages]
    # to prevent infinite loops we need iteration count and iteration limit
    iteration_count: Optional[int] = Field(
        description="The number of iterations the graph has run",
        default=0)