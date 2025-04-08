import yaml, json
from utils.state import GeneratorSubgraphState, GeneratedRow
from langchain_core.messages import SystemMessage, AIMessage
from utils.prompts import GENERATOR_PROMPT, GENERATOR_FEEDBACK

class Generator:
    def __init__(self, llm, tools: list = [], name = "Generator"):
        self.name = name
        self.tools = tools
        
        self.llm = llm.bind_tools(self.tools)
    
    def __call__(self, state: GeneratorSubgraphState):

        if state.validation_errors != "":
            wrong_cols = yaml.dump(state.validation_feedback.wrong_columns, sort_keys=False, default_flow_style=False)
            # print(f"----------------WRONG COLUMNS----------------\n{wrong_cols}")
            feedback = GENERATOR_FEEDBACK.format(wrong_columns=wrong_cols)
        else:
            feedback = ""
        
        schema = yaml.dump(state.df_row_schema, sort_keys=False, default_flow_style=False)
        ex = yaml.dump(state.example, sort_keys=False, default_flow_style=False)
        system_prompt = GENERATOR_PROMPT.format(schema=schema, example=ex, feedback=feedback)

        agent_response = self.llm.with_structured_output(GeneratedRow).invoke(system_prompt)
        print(f"----------------GENERATOR----------------\n{agent_response}")
        
        iterations = state.iteration_count+1
        sys_msg = SystemMessage(system_prompt, name="Generator")
        ai_resp = AIMessage(json.dumps(agent_response.row), name="Generator")

        return {"generated_row": agent_response, "iteration_count": iterations, "conversation_history": state.conversation_history + [sys_msg, ai_resp]}