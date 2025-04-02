import langchain
from langgraph.graph import Graph, START, END
import queryComp

import os, getpass



class LLMNode(Node):
    def __init__(self, name: str, llm_function: Callable[[str], str]):
        super().__init__(name, self.process)
        self.llm_function = llm_function

    def process(self, input_text: str):
        output_text = self.llm_function(input_text)
        print(f'{self.name} Output: {output_text}')
        for node in self.connections:
            node.process(output_text)


