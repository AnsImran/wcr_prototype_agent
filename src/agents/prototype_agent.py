from __future__ import annotations

from typing import List, Union, Annotated, Literal, Optional
from uuid import uuid4
from pydantic import BaseModel, Field
import json

# ---- Option variants ----
class Predefined_Option(BaseModel):
    option_kind:   Literal["predefined"] = "predefined"  # discriminator
    option_number: int  = Field(ge=1, description="Option number, 1, 2, 3 and so on ...")
    option_str:    str
    value:         bool = Field(default=False, description="Binary flag (True/False).")

class UserDefined_Option(BaseModel):
    option_kind:   Literal["user_defined"] = "user_defined"  # discriminator
    option_number: int  = 1  # In this case it is always 1
    option_str:    str  = Field(default="empty str", description="The user writes whatever they want as an answer")
    value:         bool = Field(default=False, description="Binary flag (True/False).")

# Discriminated union of option variants
Option = Annotated[
    Union[Predefined_Option, UserDefined_Option],
    Field(discriminator="option_kind"),
]

# ---- Higher-level models ----
class Question(BaseModel):
    question_topic: str  # main topic this question belongs to
    question_id:    str = Field(default_factory=lambda: str(uuid4()))
    question_str:   str
    options:        List[Option] = Field(default_factory=list)

class Topic(BaseModel):
    topic_name: str
    topic_id:   str = Field(default_factory=lambda: str(uuid4()))
    questions:  List[Question] = Field(default_factory=list)

    def question_count(self) -> int:
        """Return the number of questions in this topic."""
        return len(self.questions)

class All_Topics(BaseModel):
    all_topics: List[Topic] = Field(default_factory=list)

# ---- Load forms ----
with open("return_journey_planning_form.json", "r") as f:
    data = json.load(f)
return_journey_planning_form = Topic(**data)

with open("financial_recovery_aid_form.json", "r") as f:
    data = json.load(f)
financial_recovery_aid_form = Topic(**data)

# ---- LangGraph / LangChain imports ----
from typing import Annotated, List, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

FormName = Literal[return_journey_planning_form.topic_name, financial_recovery_aid_form.topic_name]

class AgentState(BaseModel):
    """
    messages:            conversation buffer handled by LangGraph (add_messages aggregator)
    form_filling_initiated: once a form has been chosen
    form_being_filled:   the name of the form currently in progress
    already_filled_forms: record of forms that were completed in this thread
    awaiting_initial:    waiting for the user to choose form 1 or 2
    awaiting_answer:     waiting for a numeric answer to the current question
    idx:                 question index within the current form
    form:                a working copy of the selected Topic
    question_limit:      OPTIONAL cap on how many questions to ask (e.g., 3 or 4). If None -> ask all.
    """
    messages:               Annotated[List[BaseMessage], add_messages]
    form_filling_initiated: bool = False
    form_being_filled:      Optional[FormName] = None
    already_filled_forms:   List[FormName]     = Field(default_factory=list)
    awaiting_initial:       bool               = False
    awaiting_answer:        Optional[bool]     = None
    idx:                    int                = 0
    form:                   Optional[Topic]    = None
    question_limit:         Optional[int]      = 3  # <---- NEW

INITIAL_MESSAGE = """Hi! We are currently offering help with:

1. `{form_name_1}`
2. `{form_name_2}`

Kindly reply with either `1` or `2`.
"""

QUESTION_MESSAGE = """{question}

{options}
Kindly enter a number between 1–{total_number_of_options}.
"""

INVALID_INPUT_MESSAGE = """Sorry, I didn't catch that.
Please enter a number between 1–{total_number_of_options}.
"""

def _extract_text_from_message(msg: BaseMessage) -> str:
    """
    Tries to robustly extract user text from a LangChain BaseMessage.
    Supports plain strings and LC's content-as-list-of-chunks (dicts with 'type'/'text').
    """
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content.strip()
    # content may be a list of dicts like [{'type': 'text', 'text': '...'}]
    try:
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"]).strip()
            # Fallback if it's a list of strings
            if isinstance(first, str):
                return first.strip()
    except Exception:
        pass
    return str(content).strip()

def _num_questions_to_ask(state: AgentState) -> int:
    """
    Determines how many questions to ask in this run.
    If question_limit is None -> ask all questions in the form.
    Otherwise ask min(limit, total).
    """
    if state.form is None:
        return 0
    total = state.form.question_count()
    if state.question_limit is None:
        return total
    # enforce sensible bounds
    limit = max(1, int(state.question_limit))
    return min(limit, total)

def _build_options_text(options: List[Option]) -> str:
    lines = []
    for opt in options:
        lines.append(f"{opt.option_number}. {opt.option_str}")
    return "\n".join(lines)

def ask_question(state: AgentState) -> AgentState:
    # --- Phase 1: Form selection ---
    if not state.form_filling_initiated:
        if not state.awaiting_initial:
            content = INITIAL_MESSAGE.format(
                form_name_1=return_journey_planning_form.topic_name,
                form_name_2=financial_recovery_aid_form.topic_name,
            )
            return {
                "messages": [AIMessage(content=content)],
                "awaiting_initial": True,
            }

        # Process user's choice
        msgs = state.messages or []
        if state.awaiting_initial and msgs and isinstance(msgs[-1], HumanMessage):
            answer_text = _extract_text_from_message(msgs[-1])
            answer_text_stripped = answer_text.strip().lower()

            if answer_text_stripped in {"1", "one"}:
                return {
                    "form_filling_initiated": True,
                    "form_being_filled":      return_journey_planning_form.topic_name,
                    "form":                   return_journey_planning_form.model_copy(deep=True),
                    "awaiting_initial":       False,
                    "idx":                    0,
                }
            elif answer_text_stripped in {"2", "two"}:
                return {
                    "form_filling_initiated": True,
                    "form_being_filled":      financial_recovery_aid_form.topic_name,
                    "form":                   financial_recovery_aid_form.model_copy(deep=True),
                    "awaiting_initial":       False,
                    "idx":                    0,
                }

        # If we got here, we either haven't received a valid HumanMessage yet, or it wasn't 1/2.
        return {}

    # --- Phase 2: Question/Answer flow ---
    if state.form_filling_initiated and state.form is not None:
        idx      = getattr(state, "idx", 0)
        awaiting = bool(getattr(state, "awaiting_answer", False))
        msgs     = getattr(state, "messages", [])

        # If done (respecting question_limit), finalize soon
        if idx >= _num_questions_to_ask(state):
            # Build a NEW list to avoid returning result of list.append(None)
            new_filled = list(state.already_filled_forms)
            if state.form.topic_name not in new_filled:
                new_filled.append(state.form.topic_name)

            return {
                "already_filled_forms":   new_filled,
                "form_filling_initiated": False,
                "form_being_filled":      None,
                "awaiting_initial":       False,
                "awaiting_answer":        False,
                "idx":                    0,
            }

        # Case 1: We previously asked; now record the user's reply (latest message)
        if awaiting and msgs and isinstance(msgs[-1], HumanMessage):
            answer_text = _extract_text_from_message(msgs[-1])

            # Validate numeric input
            options = state.form.questions[idx].options
            total_options = len(options)
            try:
                option_number = int(answer_text)
                if not (1 <= option_number <= total_options):
                    raise ValueError
            except ValueError:
                # Re-ask with an invalid-input nudge
                content = INVALID_INPUT_MESSAGE.format(total_number_of_options=total_options)
                return {"messages": [AIMessage(content=content)], "awaiting_answer": True}

            # Record the selected option
            selected_idx = option_number - 1
            form_copy = state.form.model_copy(deep=True)
            # Reset any previous selections on this question to avoid multiple True
            for opt in form_copy.questions[idx].options:
                opt.value = False
            form_copy.questions[idx].options[selected_idx].value = True

            return {
                "form":            form_copy,
                "idx":             idx + 1,   # move to the next question index
                "awaiting_answer": False,     # we just consumed the answer
            }

        # Case 2: Not awaiting and there are questions left -> ask the next question
        if not awaiting:
            q = state.form.questions[idx]
            options_text = _build_options_text(q.options)
            content = QUESTION_MESSAGE.format(
                question=q.question_str,
                options=options_text,
                total_number_of_options=len(q.options)
            )
            return {"messages": [AIMessage(content=content)], "awaiting_answer": True}

        # Case 3: Awaiting but no HumanMessage yet -> nothing to update (we'll route to END)
        return {}

    # Default no-op
    return {}

def finalize(state: AgentState) -> AgentState:
    """
    Summarizes only the questions answered in this interaction (respecting question_limit).
    """
    if state.form is None:
        return {"messages": [AIMessage(content="# Filled Form\n\n(No form data found.)")]}

    lines = ["# Filled Form", ""]
    to_summarize = min(state.form.question_count(), _num_questions_to_ask(state))

    for question_index in range(to_summarize):
        q = state.form.questions[question_index]
        # Find selected option (value=True). If none, skip to avoid leaking unanswered items.
        chosen = None
        for opt in q.options:
            if bool(getattr(opt, "value", False)):
                chosen = opt
                break
        if chosen is None:
            # Shouldn't happen if flow is followed, but guard anyway.
            continue
        if chosen.option_kind == 'predefined':
            answer_text = f"{chosen.option_number}. {chosen.option_str}"
        elif chosen.option_kind == 'user_defined':
            # change the following logic later, take an input from userthat should be the real answer_text in this scenario
            answer_text = f"{chosen.option_number}. {chosen.option_str}"
        lines.append(f"- **{q.question_str}** {answer_text}")

    md = "\n".join(lines) if len(lines) > 2 else "# Filled Form\n\n(No answers captured.)"
    return {"messages": [AIMessage(content=md)]}

# --- Graph -------------------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("ask_question", ask_question)
workflow.add_node("finalize", finalize)

# Entry
workflow.add_edge(START, "ask_question")

# Conditional routing to either continue asking, wait for user (END), or finalize
def route(state: AgentState) -> str:
    # Before a form is chosen, keep asking or wait for the user
    if not state.form_filling_initiated or state.form is None:
        return "END" if state.awaiting_initial else "ask_question"

    idx = state.idx or 0
    if idx >= _num_questions_to_ask(state):
        return "finalize"

    if state.awaiting_answer:
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

# Compile
graph = workflow.compile()

# --- Usage example (conceptual) ---
# cfg = {"configurable": {"thread_id": f"survey-{uuid4()}"}}
# # Ask *only 3* questions:
# graph.invoke({"messages": [], "question_limit": 3}, cfg)
# # Or original behavior (ask all questions):
# graph.invoke({"messages": [], "question_limit": None}, cfg)
