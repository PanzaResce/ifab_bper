from faker import Faker
from utils.state import DataframeCol, GlobalState
from langchain_core.messages import SystemMessage

class SchemaAnalyzer:
    def __init__(self, llm, tools: list, name = "SchemaAnalyzer"):
        self.name = name
        self.llm = llm
        self.tools = tools
    
    def __call__(self, state: GlobalState):
        """
        This function is called by the graph to analyze the schema of the dataframe.
        Args:
            state (GlobalState): The global state of the graph.
            llm: Instance of the language model to use.
        Returns:
            dict: A dictionary containing the schema of the dataframe.
        """
        # system_prompt = SystemMessage("""
        # You have to generate the structure of an input pandas dataframe. 
        # You have access to a Python abstract REPL, which you can use to execute the python code.
        # The dataframe is stored as df.
        # Return a list of columns where for each column the following properties are defined:
        #     - colum name
        #     - column description
        #     - column data type
        # Make sure to wrap the answer in json""")
        # system_prompt = SystemMessage("""
        # Generate random name, description and type for 3 pandas dataframe columns.
        # Return a list of columns where for each column the following properties are defined:
        #     - colum name
        #     - column description
        #     - column data type
        # Make sure to wrap the answer in json""")
        system_prompt = SystemMessage("""Hi, how are you ?""")
        agent_response = self.llm.invoke([system_prompt])
        print(f"Model out\n{agent_response}")

        # random_schema = self.generate_random_schema()
        
        return {"df_row_schema": [agent_response]}

    def generate_random_schema(self, num_columns: int = 2):
        """Generate random schema """
        fake = Faker()
        
        mock_data = [
            DataframeCol(
                column_name=f"col_{i}",
                column_descr=fake.sentence(),
                column_type=fake.random_element(["int", "float", "str", "bool"]),
            ) for i in range(num_columns)
        ]
        return mock_data
