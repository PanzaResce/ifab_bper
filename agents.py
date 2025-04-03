import functools
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from typing import (
    Annotated,
    Sequence,
    TypedDict,
    Dict
)
from langgraph.graph import MessagesState
from state import *


def agent_node(state, agent, name):
    agent_response = agent.invoke(state)
    return {
        "messages": [
            AIMessage(content=agent_response["messages"][-1].content, name=name)
        ]
    }

def supervisor_node(state, agent, name):
    system_prompt = SystemMessage("""
    Your role is to generate a new row for the input dataset, the generated row has to be coherent with the original dataset.
    You have access to the following resources:
        - A coder agent that can write useful code to work on the input dataset, when using this agent provide a description of the code needed
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

def schema_analyzer(state: OverallState, llm, name):
    # print(state.df_row_schema)
    system_prompt = SystemMessage("""Generate a random name for a dataframe column""")
    
    fake_col1 = DataframeCol(column_name="Col 1", column_descr="Fake description1", column_type="int")
    fake_col2 = DataframeCol(column_name="Col 2", column_descr="Fake description2", column_type="int")

    agent_response = llm.invoke([system_prompt])
    return {"df_row_schema": [fake_col1, fake_col2]}

# def generator(state: OverallState, llm, name):
#     # print(state.df_row_schema)
#     system_prompt = SystemMessage(f"""Generate a random record given this input schema: {state.df_row_schema}""")
#     agent_response = llm.invoke([system_prompt])
#     return_val = {"df_row_schema": state.df_row_schema.append(fake_col)}
#     print(return_val)
#     return return_val

def create_agent_node(agent, llm, tools, name):
    llm_with_tools = llm.bind_tools(tools)
    agent_node = functools.partial(agent, llm=llm_with_tools, name=name)
    return agent_node

def supervisor_should_continue(state):
    messages = state["messages"]
    # next_agent = state["call_agents"]
    last_message = messages[-1]
    print(messages)


    # if next_agent
    if last_message.tool_calls:
        return "tools"
    else:
        return "end"
