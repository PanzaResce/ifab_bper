from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional
from langchain_core.messages import AnyMessage

class GeneratedRow(BaseModel):
    row: Dict[str, str] = Field(description="The generated row, containing the column names alongside their corresponding values.")

class ValidationFeedback(BaseModel):
    wrong_columns: Dict[str, str] = Field(description="Which fields were wrongly generated alongside why are they wrong.")

class GeneratorSubgraphState(BaseModel):
    df_row_schema: Optional [Dict[str, str]] = Field(
        description="The schema of the dataframe, containing the column names alongside their value type",
        default={})
    example: Optional [Dict[str, str]] = Field(
        description="An example of a randomly sampled record",
        default={})
    stats: Optional [Any] = Field(
        description="Statistical profile of the data to guide generation",
        default={})
    generated_row: Optional [GeneratedRow] = Field(
        description="The synthetic row generated",
        default="")
    validation_errors: Optional[str] = Field(
        description="Error for the generated row",
        default="")
    validation_feedback: Optional[ValidationFeedback] = Field(
        description="Feedback to fix the generated row",
        default="")
    conversation_history: Optional[List[AnyMessage]] = []
    iteration_count: Optional[int] = Field(
        description="The number of iterations the graph has run",
        default=0)