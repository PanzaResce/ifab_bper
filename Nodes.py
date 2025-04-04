from queryComp import queryGemini
import numpy as np
import pandas as pd 
import os 
from APIKeys import Kaggle_API
from kaggle.api.kaggle_api_extended import KaggleApi
from typing import List, Dict
import json
from io import StringIO



# Generic LLM 
class LLMNode:
    def __init__(self, name: str):
        self.name = name
        from APIKeys import Google_AI_Studio
        self.googleAPI = Google_AI_Studio

    def run(self, prompt: str) -> str:
        return queryGemini(prompt, self.googleAPI)

    def __call__(self, prompt: str) -> str:
        return self.run(prompt)

# Code execution
class CodeExecutionNode:
    def __init__(self, name="CodeExecutionNode"):
        self.name = name

    def run(self, code: str) -> dict:
        local_context = {}
        try:
            exec(code, {}, local_context)
            result = {k: v for k, v in local_context.items() if not k.startswith("_")}
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def __call__(self, code: str) -> dict:
        return self.run(code)

# Kaggle Data Import

class KaggleImport:
    def __init__(self, name: str):
        self.name = name
        self.api = KaggleApi()

    def authenticate(self):
        self.api.authenticate()
      
    def load_csv_as_np(self, identifier: str) -> np.ndarray:
        self.api.dataset_download_files(identifier, path='datasets/kaggle', unzip=True)
        extracted_files = os.listdir('datasets/kaggle')
        csv_files = [file for file in extracted_files if file.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file found.")
        file_name = f'datasets/kaggle/{csv_files[0]}'
        df = pd.read_csv(file_name)
        os.remove(file_name)
        return df.to_numpy()

    def run(self, dataType: str, identifier: str) -> np.ndarray:
        self.authenticate()
        if dataType == 'csv':
            return self.load_csv_as_np(identifier)
        else:
            raise ValueError("Unsupported data type. Use 'csv'.")

    def __call__(self, dataType: str, identifier: str) -> np.ndarray:
        return self.run(dataType, identifier)

# Schema Picker 
class SchemaDescriptorNode():
    def __init__(self, name):
        self.name = name

    

    def __call__(self, data) -> pd.DataFrame:
        
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

        if isinstance(data, pd.DataFrame):
            pass
        else:
            data = self.castToDataFrame(data)
            


        friendly_dtypes = data.dtypes.replace(dtype_mapping)
        friendly_dtypes_str = friendly_dtypes.to_string(header=False, index=True)
        return friendly_dtypes_str
    
    def castToDataFrame(self, data):
        if isinstance(data, pd.DataFrame):
            return data

        elif isinstance(data, pd.Series):
            return data.to_frame()

        elif isinstance(data, (np.ndarray, np.ma.MaskedArray)):
            return pd.DataFrame(data)

        elif isinstance(data, (list, dict)):
            return pd.DataFrame(data)

        elif isinstance(data, pd.Categorical):
            return pd.DataFrame(pd.Series(data))

       
        else: 
            raise ValueError("DataType non supportato")

'''
        elif isinstance(data, GeoDataFrame):
            return pd.DataFrame(data)
\ elif isinstance(data, dd.DataFrame):
            return data.compute()
        elif isinstance(data, PySparkDataFrame):
            return data.toPandas()
            
        elif isinstance(data, pa.Table):
            return data.to_pandas()
        
        elif isinstance(data, pa.Dataset):
            return pq.read_table(data).to_pandas()
'''

# Description Generator Node

class DescriptionGeneratorNode:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompt: str) -> List[Dict[str, str]]:
        description_prompt = f"""
        You are an agent that generates column names and descriptions for a dataset.
        Do not add reccomandations of any sort, do not ask for further instructions, only output what has been asked of you in the form column name and explaination.  
        Task: {prompt}
        """
        response = self.llm.run(description_prompt)
        #print(response.text)
        return response

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        return self.run(prompt)

# Table Generator Node
class TableGeneratorNode:
    def __init__(self, llm):
        self.llm = llm

    def run(self, columns: str) :
        prompt = f"""
        Generate structured tabular data based on the specifications, the data needs to be output in CSV like format, only produce the data and header, do not produce anything else. 
        Columns:
        {columns}
        
        """
        csv_data = self.llm.run(prompt)
        
        return pd.read_csv(StringIO(csv_data.text))

    def __call__(self, columns: str):
        return self.run(columns)


#DATA-GEN

class RandomDataNode():
    '''
    Generates random data for specified columns using uniform, normal, or custom distributions.
    '''
    def __init__(self, name: str):
        self.name = name  

    def run(self, num_samples: int, distribution: str = 'normal', params: dict = None):
            if params is None:
                params = {}

            if distribution == 'normal':
                data = np.random.normal(loc=params.get('mean', 0), scale=params.get('std', 1), size=num_samples)
            elif distribution == 'uniform':
                data = np.random.uniform(low=params.get('low', 0), high=params.get('high', 1), size=num_samples)
            else:
                raise ValueError("Dist non disponibile")
            
            return pd.Series(data, name=self.name)


class CorrelatedDataNode():
    '''
    Generates data that is correlated to other columns (e.g., age vs. income, height vs. weight).
    '''
    def __init__(self, name:str):
        self.name = name  

    def run(self, reference_data: pd.Series, correlation: float = 0.8, noise: float = 0.1):
        noise_data = np.random.normal(0, noise, size=len(reference_data))
        correlated_data = (reference_data * correlation) + noise_data * (1 - correlation)
        return pd.Series(correlated_data, name=self.name)
    
class CategoricalDataNode():
    '''
    Creates categorical variables with specified probabilities (e.g., gender, occupation, region).
    '''
    def __init__(self, name:str):
        self.name = name  

    def run(self, categories: list, probabilities: list, num_samples: int):
        if len(categories) != len(probabilities):
            raise ValueError("Categories and probabilities must have the same length.")
        
        data = np.random.choice(categories, size=num_samples, p=probabilities)
        return pd.Series(data, name=self.name)
    

class PatternedDataNode():
    '''
    Generates data according to specific rules or patterns (e.g., periodic, exponential growth).
    '''
    def __init__(self, name:str):
        self.name = name      
    def run(self, pattern_type: str = 'sin', num_samples: int = 100, amplitude: float = 1, frequency: float = 1):
        x = np.linspace(0, 2 * np.pi, num_samples)
        
        if pattern_type == 'sin':
            data = amplitude * np.sin(frequency * x)
        elif pattern_type == 'cos':
            data = amplitude * np.cos(frequency * x)
        elif pattern_type == 'linear':
            data = np.linspace(0, amplitude, num_samples)
        else:
            raise ValueError("Unsupported pattern type.")
        
        return pd.Series(data, name=self.name)


#DATA TRASFORM

class NoiseInjectionNode():

    '''
    Adds noise to numeric columns to simulate measurement errors or uncertainty.
    '''
    def __init__(self, name:str):
        self.name = name      
    def run(self):
        pass

class ScalingNode():
    '''
    Scales data using standardization, normalization, or custom scaling functions.
    '''
    def __init__(self, name:str):
        self.name = name      
    def run(self):
        pass

class MissingDataNode():
    '''
    Introduces missing values based on specified probabilities or patterns.

    '''
    def __init__(self, name:str):
        self.name = name      
    def run(self):
        pass
class EncodingNode():
    '''
    Converts categorical data to numerical representations (e.g., one-hot encoding, label encoding).
    '''
    def __init__(self, name:str):
        self.name = name      
    def run(self):
        pass



# Data Val 
class SchemaValidationNode():
    '''
    Ensures generated data adheres to a predefined schema (e.g., column types, ranges, etc.).
    '''
    def __init__(self, name:str):
        self.name = name      
    def run(self):
        pass
class DistributionComparisonNode():
    '''
    Compares generated data distributions to real-world data distributions for realism assessment.
    '''
    def __init__(self, name:str):
        self.name = name      
    def run(self):
        pass


class CorrelationAnalysisNode():
    '''
    Ensures specified correlations are maintained between generated columns.

    '''
    def __init__(self, name:str):
        self.name = name      
    def run(self):
        pass

class DataQualityNode():
    '''
    Checks for anomalies or inconsistencies in the generated data (e.g., duplicates, missing values).
    '''
    def __init__(self, name:str):
        self.name = name      
    def run(self):
        pass
# Data Output 
class CSVOutputNode():
    '''
    Saves generated data to a CSV file.
    '''
    def __init__(self, name:str):
        self.name = name      
    def run(self):
        pass