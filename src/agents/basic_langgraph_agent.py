from langgraph.graph import StateGraph, START
from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langchain_openai import ChatOpenAI


import os
from dotenv import load_dotenv

# from langgraph.checkpoint.memory import MemorySaver

load_dotenv()  # Load environment variables from a .env file

# Access API keys and credentials
OPENAI_API_KEY    = os.environ["OPENAI_API_KEY"]
# TIMESCALE_DB_URI  = os.environ["TIMESCALE_DB_URI"]
# TAVILY_API_KEY    = os.environ["TAVILY_API_KEY"]
# LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"] 
# MAIN_AGENT_DB_URI = os.environ["MAIN_AGENT_DB_URI"]

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"]    = "langchain-academy"




class AgentState(MessagesState, total=False):
    """`total=False` marks fields added here as optional (PEP 589),
    while inherited fields (like `messages` from MessagesState) keep their original requirements.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """
    # # Safety metadata from LlamaGuard (populated by guard nodes)
    # safety: LlamaGuardOutput

    # # LangGraph-managed remaining step budget for the current run
    # remaining_steps: RemainingSteps


MODEL_SYSTEM_MESSAGE = """ You are a helpful chatbot on the website of an accounting firm.
Your role is to provide users with accurate information about the firm’s services.
The details of services provided by the accounting firm are as follows:

<accounting_services>
{accounting_services}
</accounting_services>
"""


accounting_services = 'null'


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Main model node:
       - Selects the concrete model (from config or default),
       - Runs the tool-enabled chat model,
       - Post-checks the output with LlamaGuard,
       - Enforces step budget if tool calls remain."""
    
    # can later decide to place it outside the fn, if observed high latency in traces
    # Initialize OpenAI's GPT model
    model      = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0)
    
    system_msg = MODEL_SYSTEM_MESSAGE.format(accounting_services=accounting_services)
    response   = await model.ainvoke(
                                        [SystemMessage(content=system_msg)] + state["messages"],
                                        config=config
                                    )
    
    return {"messages": [response]}



# -------------------------
# BUILD THE GRAPH
# -------------------------


# from IPython.display import Image, display




builder = StateGraph(AgentState)

builder.add_node("acall_model", acall_model)

builder.set_entry_point("acall_model")
builder.add_edge('acall_model', END)


# memory = MemorySaver()

# Compile the graph with persistent checkpointer and in-memory store
basic_langgraph_agent = builder.compile()#checkpointer=memory, store=across_thread_memory) #checkpointer

    