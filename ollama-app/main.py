import AgentNodes 
import APIKeys
from IPython.display import Image, display



# Create a new Graph
# Create Graph
llm_graph = Graph()

node_A = LLMNode('Node A - Ollama', call_ollama)
node_B = LLMNode('Node B - Gemini', lambda prompt: query_gemini(prompt, googleAPI))


llm_graph.connect(node_A, node_B)

node_A.process("Hello, how are you?")


display(Image(app.get_graph().draw_mermaid_png()))


response = app.invoke("hello")
print(response.content)