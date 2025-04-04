from utils.state import DataframeCol, GlobalState

class Feedback:
    def __init__(self, llm, tools: list = [], name = "Feedback agent"):
        self.name = name
        self.tools = tools
        
        self.llm = llm.bind_tools(self.tools)
    
    def __call__(self, state: GlobalState):
        print(f"ERROR STATE: {state}")
        errors = state.validation_errors
        # print(f"SCHEMA:\n{schema}")

        system_prompt = f"""
        The previous generation gave this error:
        {errors}
        Give useful feedback to help the generator to generate a record consistent with the dataframe schema"""

        agent_response = self.llm.invoke(system_prompt)
        print(f"Model out\n{agent_response.content}")

        # random_schema = self.generate_random_schema()
        
        return {"validation_feedback": agent_response}