# graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict


class MessageState(TypedDict):
    sender: str
    message: str
    messageType: str
    reply: str

# Node 1: Message receive karo
def receive_message(state: MessageState) -> MessageState:
    return state

# Router function — messageType dekho
def route_message(state: MessageState) -> str:
    if state.get("messageType") == "audioMessage":
        return "voice"
    return "text"


# Node 2a: Text process karo
def process_message(state: MessageState):
    reply = "Hello! I am from LangGraph"
    return {"reply": reply}

# Node 2b: Voice handle karo
def process_voice(state: MessageState):
    reply = "Hey your voice msg received"
    return {"reply": reply}

# Node 3: Reply ready
def send_reply(state: MessageState) -> MessageState:
    return state


# Graph banao
builder = StateGraph(MessageState)

builder.add_node("receive_message", receive_message)
builder.add_node("process_message", process_message)
builder.add_node("process_voice", process_voice)
builder.add_node("send_reply", send_reply)

builder.set_entry_point("receive_message")

# Conditional edge — router decide karega kaunsa node jayega
builder.add_conditional_edges(
    "receive_message",
    route_message,
    {
        "text": "process_message",
        "voice": "process_voice",
    }
)

builder.add_edge("process_message", "send_reply")
builder.add_edge("process_voice", "send_reply")
builder.add_edge("send_reply", END)

graph = builder.compile()
graph