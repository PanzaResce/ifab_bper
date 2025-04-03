'''
import Nodes 
from langgraph.prebuilt import create_react_agent



# Create a new Graph
# Create Graph
llm_client = Nodes.LLMNode("GoogleGemini")
description_node = Nodes.DescriptionGeneratorNode(llm_client)
table_node = Nodes.TableGeneratorNode(llm_client)

# Initialize Agent
graph_agent = create_react_agent(name="TableGenerationAgent", nodes={ #alternativamente subclasse 
    "DescriptionGenerator": description_node,
    "TableGenerator": table_node
})

# Generate structured data
task_description = "Generate a dataset containing information about employees in a company."

descriptions = graph_agent.run("DescriptionGenerator", prompt=task_description)


# Step 2: Generate Tabular Data based on Descriptions
num_rows = 5
generated_table = graph_agent.run("TableGenerator", columns=descriptions, num_rows=num_rows)

# Output the generated table
print(generated_table)
'''

from langgraph.graph import Graph, START, END
import Nodes

llm_client = Nodes.LLMNode("GoogleGemini")
description_node = Nodes.DescriptionGeneratorNode(llm_client)
table_node = Nodes.TableGeneratorNode(llm_client)

graph_agent = Graph()

# Add nodes explicitly
graph_agent.add_node("DescriptionGenerator", description_node)
graph_agent.add_node("TableGenerator", table_node)

# Define edges to describe flow
graph_agent.add_edge(START, "DescriptionGenerator")
graph_agent.add_edge("DescriptionGenerator", "TableGenerator")
graph_agent.add_edge("TableGenerator", END)

#graph_agent.set_entry_point("DescriptionGenerator")

# Compile the graph
compiled_agent = graph_agent.compile()

# Define the input for the entire graph
task_description =  "Generate a dataset containing information about employees in a company."

# Run the graph
generated_table = compiled_agent.invoke(task_description)

print(generated_table)


