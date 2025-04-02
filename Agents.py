import langchain
from langgraph.graph import Graph, START, END
from langgraph import Agent, Node
import numpy as np 
import pandas as pd 


import os, getpass
import Nodes 
import queryComp



class SimpleAgent(Agent):
    def __init__(self, name: str, node: Nodes.LLMNode):
        super().__init__(name)
        self.node = node

    def handle_request(self, prompt: str) -> str:
        return self.node.run(prompt)


class DataIngestion(Agent): 
    def __init__(self, name: str, ):
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
        else: 
            print("formato non disponibile")







import functools
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


class SupervisorState(TypedDict):
    messages: MessagesState
    # call_agents: str

def agent_node(state, agent, name):
    agent_response = agent.invoke(state)
    return {
        "messages": [
            AIMessage(content=agent_response["messages"][-1].content, name=name)
        ]
    }

def supervisor_node(state: SupervisorState, agent, name):
    system_prompt = SystemMessage("""
    Your role is to generate a new row for the input dataset, the generated row has to be coherent with the original dataset.
    You have access to the following resources:
        - A coder agent that can write useful code to work on the input dataset
        - Access to a Python abstract REPL usable to execute python code, the dataframe is already stored as df.
    """)
    return agent.invoke([system_prompt] + state["messages"])

def create_coder_agent_node(llm, repl_tool):
    coder_prompt = """
    You are an agent that writes useful code to work on a pandas dataframe. 
    You have access to a Python abstract REPL, which you can use to execute the python code.
    The dataframe is stored as df. Encapsulate the generated code into a python function.
    As the final output return only the python function.
    """
    coder_agent = create_react_agent(llm, prompt=coder_prompt, tools=[repl_tool])
    coder_node = functools.partial(agent_node, agent=coder_agent, name="Coder")

    return coder_node


def create_dfinfo_agent_node(llm, search_tools: list):
    dfinfo_prompt="""
    You are given a webpage that describe a specific dataset.
    Extract all the relevant information concerning the following aspects:
        - Columns name and description
        - Scope of the dataset
        - The main features of the dataset
    
    Follow the links inside the webpage if they can help you to extract all the relevant information.
    """
    dfinfo_agent = create_react_agent(llm, prompt=dfinfo_prompt, tools=search_tools)
    dfinfo_node = functools.partial(agent_node, agent=dfinfo_agent, name="Dataframe information")

    return dfinfo_node


def create_generator_agent_node(llm, tools):
    generator_prompt="""
    Your role is to generate a new row for the input dataset, the generated row has to be coherent with the original dataset.
    You have access to the following resources:
        - A coder agent that can write useful code to work on the input dataset
        - Access to a Python abstract REPL usable to execute python code, assume that the dataframe is stored as df.
    """
    # generator_agent = create_react_agent(llm, prompt=generator_prompt, tools=tools)
    generator_agent = llm.bind_tools(tools)
    # generator_agent = generator_agent.with_structured_output(SupervisorState)
    generator_node = functools.partial(supervisor_node, agent=generator_agent, name="Generator")

    return generator_node


def supervisor_should_continue(state: SupervisorState):
    messages = state["messages"]
    # next_agent = state["call_agents"]
    last_message = messages[-1]


    # if next_agent
    if last_message.tool_calls:
        return "tools"
    else:
        return "end"