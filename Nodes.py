from langgraph import Node
from queryComp import queryGemini

#LLM 
class LLMNode(Node):
    def __init__(self, name: str):
        super().__init__(name)

    def run(self, prompt: str) -> str:
        from APIKeys import googleAPI
        return queryGemini(prompt, googleAPI)


#DATA-GEN

class RandomDataNode(Node):
    '''
    Generates random data for specified columns using uniform, normal, or custom distributions.
    '''
    def __init__(self, name: str, dist = None):
        super().__init__(name)
        self.dist = dist

    def run(self):
        1+1
        #aggiungi coso per generazione dati casuali con varie distribuzioni



class CorrelatedDataNode(Node):
    '''
    Generates data that is correlated to other columns (e.g., age vs. income, height vs. weight).
    '''
    def __init__(self, name:str):
    
    def run(self):


class CategoricalDataNode(Node):
    '''
    Creates categorical variables with specified probabilities (e.g., gender, occupation, region).
    '''
    def __init__(self, name:str):
    
    def run(self):


class PatternedDataNode(Node):

    '''
    Generates data according to specific rules or patterns (e.g., periodic, exponential growth).
    '''
    def __init__(self, name:str):
    
    def run(self):


#DATA TRASFORM



class NoiseInjectionNode(Node):

    '''
    Adds noise to numeric columns to simulate measurement errors or uncertainty.
    '''
    def __init__(self, name:str):
    
    def run(self):

class ScalingNode(Node):
    '''
    Scales data using standardization, normalization, or custom scaling functions.
    '''
    def __init__(self, name:str):
    
    def run(self):


class MissingDataNode(Node):
    '''
    Introduces missing values based on specified probabilities or patterns.

    '''
    def __init__(self, name:str):
    
    def run(self):

class EncodingNode(Node):
    '''
    Converts categorical data to numerical representations (e.g., one-hot encoding, label encoding).
    '''
    def __init__(self, name:str):
    
    def run(self):




# Data Val 
class SchemaValidationNode(Node):
    '''
    Ensures generated data adheres to a predefined schema (e.g., column types, ranges, etc.).
    '''
    def __init__(self, name:str):
    
    def run(self):

class DistributionComparisonNode(Node):
    '''
    Compares generated data distributions to real-world data distributions for realism assessment.
    '''
    def __init__(self, name:str):
    
    def run(self):

class CorrelationAnalysisNode(Node):
    '''
    Ensures specified correlations are maintained between generated columns.

    '''
    def __init__(self, name:str):
    
    def run(self):

class DataQualityNode(Node):
    '''
    Checks for anomalies or inconsistencies in the generated data (e.g., duplicates, missing values).
    '''
    def __init__(self, name:str):
    
    def run(self):

# Data Output 
class CSVOutputNode(Node):
    '''
    Saves generated data to a CSV file.
    '''
    def __init__(self, name:str):
    
    def run(self):
