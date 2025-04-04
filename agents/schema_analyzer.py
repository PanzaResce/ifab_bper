from faker import Faker
from utils.state import DataframeCol, GlobalState
from langchain_core.messages import SystemMessage

class SchemaAnalyzer:
    def __init__(self, llm, tools: list = [], name = "SchemaAnalyzer"):
        self.name = name
        self.tools = tools
        
        self.llm = llm.bind_tools(self.tools)
    
    def __call__(self, state: GlobalState):
        """
        This function is called by the graph to analyze the schema of the dataframe.
        Args:
            state (GlobalState): The global state of the graph.
            llm: Instance of the language model to use.
        Returns:
            dict: A dictionary containing the schema of the dataframe.
        """
        system_prompt = """
        You have to generate the structure of an input pandas dataframe. 
        You have access to a Python abstract REPL, which you can use to execute the python code.
        The dataframe is stored as df.
        Return a list of columns where for each column the following properties are defined:
            - colum name
            - column description
            - column data type
        Make sure to wrap the answer in json"""
        # system_prompt = SystemMessage("""
        # Generate random name, description and type for 3 pandas dataframe columns.
        # Return a list of columns where for each column the following properties are defined:
        #     - colum name
        #     - column description
        #     - column data type
        # Make sure to wrap the answer in json""")
        # agent_response = self.llm.invoke(system_prompt)
        # print(f"Model out\n{agent_response.content}")

        # random_schema = self.generate_random_schema()
        
        out = """
        fraud_bool: int64
        income: float64
        name_email_similarity: float64
        prev_address_months_count: int64
        current_address_months_count: int64
        customer_age: int64
        days_since_request: float64
        intended_balcon_amount: float64
        payment_type: object
        zip_count_4w: int64
        velocity_6h: float64
        velocity_24h: float64
        velocity_4w: float64
        """

        return {"df_row_schema": out}

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
