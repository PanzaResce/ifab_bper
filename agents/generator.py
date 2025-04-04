from utils.state import DataframeCol, GlobalState, GeneratedRow

class Generator:
    def __init__(self, llm, tools: list = [], name = "Generator"):
        self.name = name
        self.tools = tools
        
        self.llm = llm.bind_tools(self.tools)
    
    def __call__(self, state: GlobalState):
        schema = state.df_row_schema
        stats = state.stats
        if state.validation_errors != "":
            feedback = f"The previous record you tried to generate gave an error, use this suggestion to improve the generation: {state.validation_errors}"
        else:
            feedback = ""
        # print(f"SCHEMA:\n{schema}")

        system_prompt = f"""
        You are an agent that has to generate a new record for a dataframe.
        The dataframe has the following schema, with the column names alongside their value type:\n{schema}.
        Generate a new record complying with the given schema.
        Consider also the statistical profile of the data provided: 
        {stats}.
        {feedback}
        Just output the new record in csv format so it can be directly passed to another agent.
        Do not produce code but just the new record."""
        print(f"PROMPT\n{system_prompt}")

        agent_response = self.llm.with_structured_output(GeneratedRow).invoke(system_prompt)
        print(f"Generator out\n{agent_response}")
        # print(state)
        # random_schema = self.generate_random_schema()
        iterations = state.iteration_count+1

        return {"generated_row": agent_response, "iteration_count": iterations}