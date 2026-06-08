from langgraph.graph import StateGraph, END, START
from state import MessageState
from nodes import receive_message, process_message, transcribe_audio_node, process_message_audio, text_to_speech_node, send_reply
from nodes import route_message



# ------------State------------------------------
builder = StateGraph(MessageState)



# ------------Nodes------------------------------
builder.add_node("receive_message", receive_message)
builder.add_node("process_message", process_message) 

builder.add_node("transcribe_audio_node", transcribe_audio_node)
builder.add_node("process_message_audio", process_message_audio)
builder.add_node("text_to_speech_node", text_to_speech_node)
builder.add_node("send_reply", send_reply)
 


# ------------Edges------------------------------
builder.add_edge(START,"receive_message")

builder.add_conditional_edges(
    "receive_message",
    route_message,
    {
        "text": "process_message",
        "voice": "transcribe_audio_node",
    }
)

builder.add_edge("process_message", "send_reply")
builder.add_edge("send_reply", END)


builder.add_edge("transcribe_audio_node", "process_message_audio")
builder.add_edge("process_message_audio", "text_to_speech_node")
builder.add_edge("text_to_speech_node", "send_reply")
builder.add_edge("send_reply", END)

# ------------Build------------------------------
graph = builder.compile()