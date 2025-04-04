from state import GlobalState
class DataProfiler:
    def __init__(self, llm, name="DataProfiler"):
        self.name= name
        self.llm = llm
    
    def __call__(self, state: GlobalState):
        # Mocked statistical profile output
        stats_mock = {
            "name": {"unique": True, "most_common": "John Doe"},
            "age": {"min": 18, "max": 90, "mean": 45},
            "email": {"unique": True, "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"},
        }
        print(state)
        return {"stats": stats_mock}