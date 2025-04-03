import requests
from langchain.tools import tool
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

@tool("internet_search_DDGO", return_direct=False)
def internet_search_DDGO(query: str) -> str:

  """Searches the internet using DuckDuckGo."""

  with DDGS() as ddgs:
    results = [r for r in ddgs.text(query, max_results=5)]
    return results if results else "No results found."

@tool("process_content", return_direct=False)
def process_page_content(url: str) -> str:

    """Processes content from a webpage given its url."""

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def python_repl(df):
    repl = PythonREPL()
    python_repl = PythonREPL()

    # Pass the dataframe into the globals dictionary of the PythonREPL instance
    repl.globals['df'] = df

    # You can create the tool to pass to an agent
    repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )
    
    return repl_tool


def get_tools(repl_df):
   return [internet_search_DDGO, process_page_content, python_repl(repl_df)]