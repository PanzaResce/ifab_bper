import yaml, json
from utils.state import DataframeCol, GlobalState, GeneratedRow
from langchain_core.messages import SystemMessage, AIMessage
from utils.prompts import GENERATOR_PROMPT, GENERATOR_FEEDBACK

class Generator:
    def __init__(self, llm, tools: list = [], name = "Generator"):
        self.name = name
        self.tools = tools
        
        self.llm = llm.bind_tools(self.tools)
    
    def __call__(self, state: GlobalState):
        stats = state.stats

        if state.validation_errors != "":
            feedback = GENERATOR_FEEDBACK.format(error=state.validation_feedback.error, suggestion=state.validation_feedback.fix)
        else:
            feedback = ""
        
        schema = yaml.dump(state.df_row_schema, sort_keys=False, default_flow_style=False)

        system_prompt = GENERATOR_PROMPT.format(schema=schema, feedback=feedback)

        agent_response = self.llm.with_structured_output(GeneratedRow).invoke(system_prompt)
        print(f"----------------GENERATOR----------------\n{agent_response}")
        # print(type(agent_response))
        
        iterations = state.iteration_count+1

        sys_msg = SystemMessage(system_prompt, name="Generator")
        ai_resp = AIMessage(json.dumps(agent_response.generated_row), name="Generator")

        return {"generated_row": agent_response, "iteration_count": iterations, "conversation_history": state.conversation_history + [sys_msg, ai_resp]}