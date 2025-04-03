from state import GlobalState
class SchemaAnalyzer:
    def __init__(self, name = "SchemaAnalyzer"):
        self.name = name
    
    def __call__(self, state: GlobalState, llm):
        """
        This function is called by the graph to analyze the schema of the dataframe.
        Args:
            state (GlobalState): The global state of the graph.
            llm: Instance of the language model to use.
        Returns:
            dict: A dictionary containing the schema of the dataframe.
        """
        #system_prompt = SystemMessage("""Generate a random name for a dataframe column""")
        #agent_response = llm.invoke([system_prompt])
        random_schema = self.generate_random_schema()
        return state.model_copy(update={
            "df_row_schema": random_schema,
        })

    def generate_random_schema(self, num_columns: int = 2):
        """Generate random schema """
        from faker import Faker
        fake = Faker()
        
        mock_data = [
            DataframeCol(
                column_name=f"col_{i}",
                column_descr=fake.sentence(),
                column_type=fake.random_element(["int", "float", "str", "bool"]),
            ) for i in range(num_columns)
        ]
        return mock_data
    
'''
analyzer = SchemaAnalyzer(num_columns=3)
state = GlobalState()  # Fresh state
new_state = analyzer(state)

print(f"Generated {len(new_state.df_row_schema)} columns:")
for col in new_state.df_row_schema:
    print(f"  {col.column_name}: {col.column_type} | {col.column_descr}") 
'''