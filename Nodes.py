from langgraph import Node
from queryComp import queryGemini
import numpy as np
import pandas as pd 
from  kaggle import api
import os 
from APIKeys import Kaggle_API
from kaggle.api.kaggle_api_extended import KaggleApi

#Code 

class CodeExecutionNode(Node): #Python
    def __init__(self, name="CodeExecutionNode"):
        super().__init__(name=name)

    def run(self, code: str) -> dict:
        """
        Executes a given Python code string and returns the result or any error encountered.

        Args:
            code (str): The Python code to be executed.

        Returns:
            dict: A dictionary containing either the result or the error message.
        """
        local_context = {}
        try:
            exec(code, {}, local_context)
            result = {k: v for k, v in local_context.items() if not k.startswith("_")}
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}



#LLM 
class LLMNode(Node):
    '''
    Chiama un LLM, generica utilità  
    '''
    def __init__(self, name: str):
        super().__init__(name)

    def run(self, prompt: str) -> str:
        from APIKeys import googleAPI
        return queryGemini(prompt, googleAPI)




#Data Import 

class KaggleImport(Node):
    '''
    Ritorna oggetto numpy con tabella di kaggle
    '''
    def __init__(self, name: str):
        super().__init__(name)
        self.api = KaggleApi()
        self.api.authenticate()
    
    def authenticate(self):  # esempio identifier: 'sgpjesus/bank-account-fraud-dataset-neurips-2022'
        os.environ['KAGGLE_USERNAME'] = Kaggle_API["Username"]
        os.environ['KAGGLE_KEY'] = Kaggle_API["API_Key"]

        self.api.authenticate()

        del os.environ['KAGGLE_USERNAME']
        del os.environ['KAGGLE_KEY']

    def load_csv_as_np(self, identifier: str) -> np.ndarray: #assume ci sia solo un file nel target 
        """Load CSV file from Kaggle as a NumPy array."""
        
        # Scarica il dataset dalla fonte Kaggle
        self.api.dataset_download_files(identifier, path='datasets/kaggle', unzip=True)
        
        # Cerca il file CSV estratto nella cartella specificata
        extracted_files = os.listdir('datasets/kaggle')
        csv_files = [file for file in extracted_files if file.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError("Non è stato trovato nessun file CSV nella cartella specificata.")
        
        file_name = f'datasets/kaggle/{csv_files[0]}'
        
        # Carica il file CSV in un DataFrame pandas
        df = pd.read_csv(file_name)
        
        # Rimuove il file dopo l'utilizzo
        os.remove(file_name)
        
        # Restituisce il DataFrame come array NumPy
        return df.to_numpy()
    
    def run(self, dataType: str, identifier: str) -> np.ndarray:
        self.authenticate()

        if dataType == 'csv':
            return self.load_csv_as_np(identifier)
        else:
            raise ValueError("Tipo di dati non supportato. Usa 'csv'.")
#DATA-GEN


class RandomDataNode(Node):
    '''
    Generates random data for specified columns using uniform, normal, or custom distributions.
    '''
    def __init__(self, name: str):
        super().__init__(name)

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


class CorrelatedDataNode(Node):
    '''
    Generates data that is correlated to other columns (e.g., age vs. income, height vs. weight).
    '''
    def __init__(self, name:str):
        super().__init__(name)

    def run(self, reference_data: pd.Series, correlation: float = 0.8, noise: float = 0.1):
        noise_data = np.random.normal(0, noise, size=len(reference_data))
        correlated_data = (reference_data * correlation) + noise_data * (1 - correlation)
        return pd.Series(correlated_data, name=self.name)
    
class CategoricalDataNode(Node):
    '''
    Creates categorical variables with specified probabilities (e.g., gender, occupation, region).
    '''
    def __init__(self, name:str):
        super().__init__(name)
    
    def run(self, categories: list, probabilities: list, num_samples: int):
        if len(categories) != len(probabilities):
            raise ValueError("Categories and probabilities must have the same length.")
        
        data = np.random.choice(categories, size=num_samples, p=probabilities)
        return pd.Series(data, name=self.name)
    

class PatternedDataNode(Node):
    '''
    Generates data according to specific rules or patterns (e.g., periodic, exponential growth).
    '''
    def __init__(self, name:str):
        super().__init__(name)
    
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

class NoiseInjectionNode(Node):

    '''
    Adds noise to numeric columns to simulate measurement errors or uncertainty.
    '''
    def __init__(self, name:str):
        super().__init__(name)
    
    def run(self):
        pass

class ScalingNode(Node):
    '''
    Scales data using standardization, normalization, or custom scaling functions.
    '''
    def __init__(self, name:str):
        super().__init__(name)
    
    def run(self):
        pass

class MissingDataNode(Node):
    '''
    Introduces missing values based on specified probabilities or patterns.

    '''
    def __init__(self, name:str):
        super().__init__(name)
    
    def run(self):
        pass
class EncodingNode(Node):
    '''
    Converts categorical data to numerical representations (e.g., one-hot encoding, label encoding).
    '''
    def __init__(self, name:str):
        super().__init__(name)
    
    def run(self):
        pass



# Data Val 
class SchemaValidationNode(Node):
    '''
    Ensures generated data adheres to a predefined schema (e.g., column types, ranges, etc.).
    '''
    def __init__(self, name:str):
        super().__init__(name)
    
    def run(self):
        pass
class DistributionComparisonNode(Node):
    '''
    Compares generated data distributions to real-world data distributions for realism assessment.
    '''
    def __init__(self, name:str):
        super().__init__(name)
    
    def run(self):
        pass


class CorrelationAnalysisNode(Node):
    '''
    Ensures specified correlations are maintained between generated columns.

    '''
    def __init__(self, name:str):
        super().__init__(name)
    
    def run(self):
        pass

class DataQualityNode(Node):
    '''
    Checks for anomalies or inconsistencies in the generated data (e.g., duplicates, missing values).
    '''
    def __init__(self, name:str):
        super().__init__(name)
    
    def run(self):
        pass
# Data Output 
class CSVOutputNode(Node):
    '''
    Saves generated data to a CSV file.
    '''
    def __init__(self, name:str):
        super().__init__(name)
    
    def run(self):
        pass