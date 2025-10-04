from langgraph.graph import StateGraph, START, END
from .state import graphState as State

from pathlib import Path
import sys
root_directory = Path(__file__).parent.parent.resolve()
if root_directory not in sys.path:
    sys.path.insert(0, str(root_directory))

from src.utils.chat_model import llm
from src.database.vectordb import QdrantVectorDB
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint


graph_builder = StateGraph(state_schema=State)
vectorDBService  = QdrantVectorDB()

# define prompt templates for clarifying the question
refine_prompt = PromptTemplate(template=
        """
        Given the following user query, refine it to be more specific and clear. Do not change its intent or add any new information.
        User query: {question}
        Only provide the refined question without any additional text.
        """,
        input_variables=["question"]
)
refine_chain = refine_prompt | llm | StrOutputParser()

# define the prompt for the RAG
rag_template = PromptTemplate(template=
        """
        Given the following user query, use the provided context to generate a comprehensive and accurate answer. Note that the context may contain irrelevant pieces of information, so be sure to synthesize it effectively. 
        When formullating the answer, use only the information from the context and do not rely on prior knowledge and cite the sources (Page number).
        If the context does not contain information relevant to the user query, express you inability to provide an answer.
        ## USER QUERY: \n\n {question}
        ## CONTEXT : \n\n {context}    
        """,
        input_variables=["question","context"]
)
rag_chain = rag_template | llm | StrOutputParser()





def refine_query(state):

    question = state["messages"][-1]

    return {
        "messages": refine_chain.invoke({"question":question})
    }

def retrieve_context(state):

    question = state["messages"][-1].content
    context = vectorDBService.search(query=question, topk=5)

    return {   
        "context": context
    }


def generate_answer(state):

    question = state["messages"][-1].content
    context = state["context"]

    formated_context: str = "\n\n".join(context)
    
    result = rag_chain.invoke({"context":formated_context,"question":question})

    pprint("\n\n" + result)
    return {
        "answer": result
    }


# add nodes 
graph_builder.add_node("refine",refine_query)
graph_builder.add_node("retrieve_context",retrieve_context)
graph_builder.add_node("generate",generate_answer)

# add edges
graph_builder.add_edge(START,"refine")
graph_builder.add_edge("refine","retrieve_context")
graph_builder.add_edge("retrieve_context","generate")   
graph_builder.add_edge("generate",END)  

# compile the graph 
graph = graph_builder.compile()

