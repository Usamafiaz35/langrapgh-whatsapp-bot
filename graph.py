# graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Graph ka state — yahan message aur reply store hoti hai
class MessageState(TypedDict):
    sender: str
    message: str
    reply: str

# Node 1: Message receive karo
def receive_message(state: MessageState) -> MessageState:
    print(f"📩 Message aaya from {state['sender']}: {state['message']}")
    return state

# Node 2: Message process karo (yahan apni logic lagao)
def process_message(state: MessageState) -> MessageState:
    reply = "Hello! I am from LangGraph 🤖"
    return {**state, "reply": reply}

# Node 3: Reply ready (yahan log ya kuch aur kar sakte ho)
def send_reply(state: MessageState) -> MessageState:
    print(f"📤 Reply ready: {state['reply']}")
    return state

# Graph banao
builder = StateGraph(MessageState)

builder.add_node("receive_message", receive_message)
builder.add_node("process_message", process_message)
builder.add_node("send_reply", send_reply)

builder.set_entry_point("receive_message")
builder.add_edge("receive_message", "process_message")
builder.add_edge("process_message", "send_reply")
builder.add_edge("send_reply", END)

graph = builder.compile()