import langchain
from langgraph.graph import Graph, START, END
import numpy as np 
import pandas as pd 
import Nodes 



class SimpleAgent():
    def __init__(self, name: str, node: Nodes.LLMNode):
        super().__init__(name)
        self.node = node

    def handle_request(self, prompt: str) -> str:
        return self.node.run(prompt)

class DataIngestionAgent(): 
    def __init__(self, name: str):
        super().__init__(name)
        self.kaggle_reader = Nodes.Kaggle_API("KaggleAPI")

    def run(self, specifics: dict):
        if specifics["Name"] == "KaggleAPI":
            try:
                 return self.kaggle_reader.run(specifics["Params"]["Identifier"])
            except NameError:
                print("Kaggle Key non esist") 
            except KeyError:
                print("Kaggle Key mal strutturata") 
        
        elif specifics["Name"] == "":
            try:
                 return self.kaggle_reader.run(specifics["Params"]["Identifier"])
            except KeyError:
                print("yoMama")
        else: 
            print("fonte non disponibile")

class CodingAgent():
        
    def __init__(self, name:str):
        super().__init__(name)
        self.CodeWriter = Nodes.CodeWritingNode()
        self.CodeExecuter = Nodes.CodeExecutionNode()

    def run(self):
        pass



