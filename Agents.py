import langchain
from langgraph.graph import Graph, START, END
import numpy as np 
import pandas as pd 
import Nodes 



class SimpleAgent():
    def __init__(self, name: str, node: Nodes.LLMNode):
        self.name = name  
        self.node = node

    def handle_request(self, prompt: str) -> str:
        return self.node.run(prompt)

class DataIngestionAgent(): 
    def __init__(self, name: str):
        self.name = name  
        self.kaggle_reader = Nodes.Kaggle_API("KaggleAPI")
        #rimane da aggiungere nodi per altre fonti di input (DB reader, locale ecc)

    def run(self, specifics: dict):
        if specifics["Name"] == "KaggleAPI": #specifics è atteso essere un dizionario con le componenti necessarie ad eseguire la richiesta; 
            #sempre: Name -> il nome del tipo di richiesta, 
            #per kaggle ha anche bisogno di sezione params contenente a sua volta una kaggle key
            try:  #in caso bisogna fare che agente capisca il tipo di dato e cambi il parametro; per ora fatto solo codice per trattare csv 
                 return self.kaggle_reader.run("csv",specifics["Params"]["Identifier"])            
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
        self.name = name  
        self.CodeWriter = Nodes.CodeWritingNode()
        self.CodeExecuter = Nodes.CodeExecutionNode()

    def run(self):
        pass



