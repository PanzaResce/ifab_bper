import yaml
from langchain_core.messages import SystemMessage, AIMessage
from utils.state import GeneratorSubgraphState, ValidationFeedback
from utils.prompts import FEEDBACK_PROMPT

class Feedback:
    def __init__(self, llm, tools: list = [], name = "Feedback agent"):
        self.name = name
        self.tools = tools
        
        self.llm = llm.bind_tools(self.tools)
    
    def __call__(self, state: GeneratorSubgraphState):
        # print(f"SCHEMA:\n{schema}")

        err_record = yaml.dump(state.generated_row.row, sort_keys=False, default_flow_style=False)
        schema = yaml.dump(state.df_row_schema, sort_keys=False, default_flow_style=False)
        
        system_prompt = FEEDBACK_PROMPT.format(record=err_record, schema=schema)
        agent_response = self.llm.with_structured_output(ValidationFeedback).invoke(system_prompt)
        print(f"----------------FEEDBACK----------------\n{agent_response}")

        # random_schema = self.generate_random_schema()
        
        sys_msg = SystemMessage(system_prompt, name="Feedback")
        ai_resp = AIMessage(agent_response.model_dump_json())

        return {"validation_feedback": agent_response, "conversation_history": state.conversation_history + [sys_msg, ai_resp]}