from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

class DataframeCol(BaseModel):
    column_name: str = Field(description="The name of the column")
    column_descr: str = Field(description="The description of the column")
    column_type: str = Field(description="The data type of the column")

class DataframeSchema(BaseModel):
    schema: List[DataframeCol] = Field(
        description="The schema of the dataframe.",
    )

class GeneratedRow(BaseModel):
    generated_row: Dict[str, str] = Field(description="The generated row, containing the column names alongside their corresponding values.")

class GlobalState(BaseModel):
    # df_row_schema: Optional [List[DataframeCol]] = Field(
    #     description="The schema of the dataframe, including its column names, description and data",
    #     default=[]
    #     )
    df_row_schema: Optional [DataframeSchema] = Field(
        description="The schema of the dataframe",
        default=""
    )
    stats: Optional [Any] = Field(
        description="Statistical profile of the data to guide generation",
        default={})
    # generated_row: Optional [List[DataframeCol]] = Field(
    #     description="The synthetic row generated.",
    #     default=[])
    generated_row: Optional [GeneratedRow] = Field(
        description="The synthetic row generated",
        default="")
    validation_errors: Optional[str] = Field(
        description="Schema violations for each row during validation, if any",
        default="")
    validation_feedback: Optional[str] = Field(
        description="Feedback to fix the generated row",
        default="")
    plausibility_feedback: Optional[str] = Field(
        description="Natural language guidance from plausibility checker suggesting improvements to the synthetic row",
        default="")
    # messages: Annotated[List[AnyMessage], add_messages]
    iteration_count: Optional[int] = Field(
        description="The number of iterations the graph has run",
        default=0)