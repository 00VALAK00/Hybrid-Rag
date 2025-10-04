from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated

class graphState(TypedDict):
    messages : Annotated[list, add_messages]
    context : list[str]
    answer : str 

 