from langchain_core.messages import SystemMessage, AIMessage
from utils.state import DataframeCol, GlobalState, ValidationFeedback
from utils.prompts import FEEDBACK_PROMPT

class Feedback:
    def __init__(self, llm, tools: list = [], name = "Feedback agent"):
        self.name = name
        self.tools = tools
        
        self.llm = llm.bind_tools(self.tools)
    
    def __call__(self, state: GlobalState):
        # print(f"ERROR STATE: {state}")
        # print(f"SCHEMA:\n{schema}")

        system_prompt = FEEDBACK_PROMPT.format(errors=state.validation_errors, schema=state.df_row_schema)
        print(f"FEEDBACK PROMPT: {system_prompt}")
        agent_response = self.llm.with_structured_output(ValidationFeedback).invoke(system_prompt)
        print(f"----------------FEEDBACK----------------\n{agent_response}")

        # random_schema = self.generate_random_schema()
        
        sys_msg = SystemMessage(system_prompt, name="Feedback")
        ai_resp = AIMessage(agent_response.model_dump_json())

        return {"validation_feedback": agent_response, "conversation_history": state.conversation_history + [sys_msg, ai_resp]}