"""
A minimal LangGraph survey agent WITHOUT `interrupt`.

Flow (no external loop inside graph logic):
START -> ask_question (asks or records) -> [END to wait for user OR ask again] -> ... -> finalize -> END

Behavior:
- Sends each question as an AI message.
- On the next invocation (after the user replies), it records the latest human message
  as the answer for the pending question, advances, and immediately asks the next one.
- After the last answer, it emits a final AI message containing the filled Markdown form.

You still resume the SAME thread per user turn (no busy loop). The pause is achieved by
conditionally routing to END when awaiting a human reply.
"""
from __future__ import annotations
from typing import TypedDict, Annotated, Dict, List

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

# --- Static configuration (no LLMs involved) ---------------------------------
list_of_questions: List[str] = [
    "what is your name?",
    "what is your age?",
    "what is your hair color?",
]
form_fields: List[str] = ["name", "age", "hair_color"]


# --- State -------------------------------------------------------------------
class SurveyState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    idx: int                    # index of the current/question-to-ask-or-record
    awaiting_answer: bool       # True if we've asked and are waiting for a human reply
    answers: Dict[str, str]     # the growing form


# --- Nodes -------------------------------------------------------------------

def ask_question(state: SurveyState) -> SurveyState:
    """One node handles BOTH asking and recording depending on state.

    - If awaiting_answer and the latest message is Human -> record answer, advance idx.
    - Else if not awaiting_answer and there are questions left -> ask next question (AI message) and set awaiting.
    - Else do nothing (routing will handle finalize / wait).
    """
    idx = state.get("idx", 0)
    awaiting = state.get("awaiting_answer", False)
    msgs = state.get("messages", [])

    # If done, nothing to do here; routing will send to finalize
    if idx >= len(list_of_questions):
        return {}

    # Case 1: We previously asked; now record the user's reply (latest message)
    if awaiting and msgs and isinstance(msgs[-1], HumanMessage):
        answer_text = str(msgs[-1].content[0]['text'])
        new_answers = dict(state.get("answers", {}))
        new_answers[form_fields[idx]] = answer_text
        return {
            "answers": new_answers,
            "idx": idx + 1,           # move to the next question index
            "awaiting_answer": False, # we just consumed the answer
        }

    # Case 2: Not awaiting and there are questions left -> ask the next question
    if not awaiting:
        question = list_of_questions[idx]
        return {
            "messages": [AIMessage(content=question)],
            "awaiting_answer": True,
        }

    # Case 3: Awaiting but no human yet -> nothing to update (we'll route to END)
    return {}


def finalize(state: SurveyState) -> SurveyState:
    lines = ["# Filled Form", ""]
    for key in form_fields:
        val = state.get("answers", {}).get(key, "")
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

# Conditional routing to either continue asking, wait for user (END), or finalize

def route(state: SurveyState) -> str:
    idx = state.get("idx", 0)
    if idx >= len(list_of_questions):
        return "finalize"
    if state.get("awaiting_answer", False):
        # Pause run; wait for a HumanMessage in the next invocation
        return "END"
    return "ask_question"

workflow.add_conditional_edges(
    "ask_question",
    route,
    {
        "ask_question": "ask_question",
        "finalize": "finalize",
        "END": END,
    },
)

# Exit
workflow.add_edge("finalize", END)

# Compile with an in-memory checkpointer; resume by calling invoke() on the same thread_id
#checkpointer = MemorySaver()
app = workflow.compile()#checkpointer=checkpointer)


# --- Usage (conceptual) ------------------------------------------------------
# from uuid import uuid4
# thread_id = f"survey-no-interrupt-{uuid4()}"
# cfg = {"configurable": {"thread_id": thread_id}}
#
# # Kick off the flow
# app.invoke({"messages": [], "idx": 0, "awaiting_answer": False, "answers": {}}, cfg)
#   -> emits AI: "what is your name?" and then stops (awaiting_answer=True)
#
# # When the user replies, resume the SAME thread with their HumanMessage
# app.invoke({"messages": [HumanMessage("Alice")]}, cfg)
#   -> records name, immediately asks next AI question, pauses
# app.invoke({"messages": [HumanMessage("30")]}, cfg)
#   -> records age, asks hair color, pauses
# result = app.invoke({"messages": [HumanMessage("Brown")]}, cfg)
#   -> records hair color, emits final Markdown summary as AI, ends
