from typing import Dict, Any
from pydantic import BaseModel, Field
from utils.state import GeneratorSubgraphState
import pandas as pd

class OutputProfiler(BaseModel):
    patterns: str = Field(description="Patterns in the data")
    corr: str = Field(description="Correlations between columns")
    anomalies: str = Field(description="Anomalies in the data")
    warnings: str = Field(description="Warnings about data values limitations and ranges")

class DataProfiler:
    def __init__(self, llm, df: pd.DataFrame, name="DataProfiler"):
        self.name= name
        self.llm = llm
        self.df = df
        self.numeric_precision = 4
    
    def __call__(self, state: GeneratorSubgraphState):
        stats = self._calculate_basic_stats(self.df)
        llm_insights = self._llm_insights(self.df, self.llm, stats)
        return {"stats": llm_insights}

    def _calculate_basic_stats(self, df: pd.DataFrame):
        """
        Calculate basic statistics for each column in the DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to analyze.
        
        Returns:
            dict: A dictionary containing basic statistics for each column.
        """
        col_stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update(self._numeric_stats(df[col]))
            elif pd.api.types.is_string_dtype(df[col]):
                col_stats.update(self._string_stats(df[col]))
            else:
                col_stats.update({"note": "Unsupported data type"})
        return col_stats
    
    def _numeric_stats(self, series: pd.Series):
        """
        Calculate basic statistics for a numeric column.
        
        Args:
            series (pd.Series): The numeric column to analyze.
        
        Returns:
            dict: A dictionary containing basic statistics for the column.
        """
        return {
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "percentiles": {
                str(p): round(series.quantile(p/100), self.numeric_precision)
                for p in [5, 25, 50, 75, 95]
            }
        }
    
    def _string_stats(self, series: pd.Series) -> Dict[str, Any]:
        return {
            "max_length": series.str.len().max(),
            "min_length": series.str.len().min(),
            "sample_values": list(series.dropna().sample(5, random_state=42))
        }

    def _llm_insights(self, df: pd.DataFrame, llm, stats: Dict[str, Any]) -> OutputProfiler:
        """Generate enhanced insights using LLM"""
        prompt = f"""
        You are a senior data analyst. 
        I have a pandas dataframe with the following columns: {df.columns.tolist()}
        The first 5 rows of the dataframe are:
        {df.head().to_dict(orient='records')}
        These are basic stats about the columns: 
        {stats}
         Provide only concise insights about:
        1. Interesting patterns
        2. Correlations between columns
        3. Anomalies
        4. Warnings about data values limitations and ranges
        """
        response = llm.with_structured_output(OutputProfiler).invoke(prompt)
        return response