"""
A minimal LangGraph agent that:
- Asks a fixed list of questions as **AI messages** (no LLMs involved)
- Waits for the user's **human** reply at each step using `interrupt`
- Fills a form on the fly
- When all questions are answered, returns the filled form as a final **AI** message in Markdown

Graph topology (no external loop):
START -> ask_question -> (loops back to ask_question until done) -> finalize -> END

To use at runtime, compile with a MemorySaver checkpointer and a thread_id. Each time the graph
interrupts, resume it by providing the user's answer for that thread. See the notes at the bottom.
"""
from __future__ import annotations
from typing import TypedDict, Annotated, Dict, List

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
# from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

# --- Static configuration (no LLMs involved) ---------------------------------
list_of_questions: List[str] = [
    "what is your name?",
    "what is your age?",
    "what is your hair color?",
]
# Keys for the form (aligned by index to the questions above)
form_fields: List[str] = ["name", "age", "hair_color"]


# --- State -------------------------------------------------------------------
class SurveyState(TypedDict):
    """Graph state.

    - messages accumulates chat history (AI + Human)
    - idx is the index of the next question to ask
    - answers is the partially filled form
    """

    messages: Annotated[List[BaseMessage], add_messages]
    idx: int
    answers: Dict[str, str]


# --- Nodes -------------------------------------------------------------------

def ask_question(state: SurveyState) -> SurveyState:
    """Ask the current question as an AI message, pause for a human reply, then
    record the answer and advance `idx`.

    This node loops on itself via conditional routing until all questions are answered.
    """
    i = state.get("idx", 0)

    # If we've exhausted the questions, do nothing here; routing will send us to finalize
    if i >= len(list_of_questions):
        return {}

    question = list_of_questions[i]

    # 1) Interrupt to yield control and **ask** the question. The value we pass to
    #    `interrupt(...)` is what the caller will see and should respond to.
    #    When the graph is resumed with the user's reply, `answer` will contain that value.
    answer: str = interrupt(question)

    # 2) After resume, we append both the AI question and the Human answer to history
    #    and update the form in state.
    new_answers = dict(state.get("answers", {}))
    new_answers[form_fields[i]] = answer

    return {
        "messages": [
            AIMessage(content=question),
            HumanMessage(content=answer),
        ],
        "answers": new_answers,
        "idx": i + 1,
    }


def finalize(state: SurveyState) -> SurveyState:
    """Emit a final AI message with the filled form in Markdown."""
    # Build Markdown summary deterministically in question order
    lines = ["# Filled Form", ""]
    for key in form_fields:
        val = state.get("answers", {}).get(key, "")
        # Pretty label
        label = key.replace("_", " ").title()
        lines.append(f"- **{label}:** {val}")
    md = "\n".join(lines)

    return {"messages": [AIMessage(content=md)]}


# --- Graph -------------------------------------------------------------------
workflow = StateGraph(SurveyState)
workflow.add_node("ask_question", ask_question)
workflow.add_node("finalize", finalize)

# Entry
workflow.add_edge(START, "ask_question")

# Conditional routing: keep asking until all questions are answered

def route_after_ask(state: SurveyState) -> str:
    return "finalize" if state.get("idx", 0) >= len(list_of_questions) else "ask_question"

workflow.add_conditional_edges(
    "ask_question",
    route_after_ask,
    {
        "ask_question": "ask_question",
        "finalize": "finalize",
    },
)

# Exit
workflow.add_edge("finalize", END)

# Compile with an in-memory checkpointer so we can resume after each interrupt
# checkpointer = MemorySaver()
app = workflow.compile()#checkpointer=checkpointer)
