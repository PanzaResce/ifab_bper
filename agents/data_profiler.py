from state import GlobalState
class DataProfiler:
    def __init__(self, name="DataProfiler"):
        self.name= name
    
    def __call__(self, state: GlobalState, llm):
        """ Generate mock statistics """