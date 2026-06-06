from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import List, TypedDict, Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
#--------------------------------------------------------

# Updated MessageState with history
class MessageState(TypedDict):
    sender: str
    message: str
    messageType: str
    reply: str
    history: List[Dict[str, str]]  # Added history field
#--------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#--------------------------------------------------------

# Global memory storage for different users
# Structure: { "user_phone_number": [list of messages] }
user_memories: Dict[str, List[Dict[str, str]]] = {}

def get_user_history(sender: str, limit: int = 10) -> List[Dict[str, str]]:
    """Get last N messages for a specific user"""
    if sender not in user_memories:
        user_memories[sender] = []
    return user_memories[sender][-limit:]

def update_user_history(sender: str, user_msg: str, ai_reply: str):
    """Update user's conversation history"""
    if sender not in user_memories:
        user_memories[sender] = []
    
    user_memories[sender].append({"role": "user", "content": user_msg})
    user_memories[sender].append({"role": "assistant", "content": ai_reply})
    
    # Keep only last 20 messages (10 exchanges)
    if len(user_memories[sender]) > 20:
        user_memories[sender] = user_memories[sender][-20:]
#--------------------------------------------------------

# Node 1: Message receive
def receive_message(state: MessageState) -> MessageState:
    return state


# Router function — To check messageType
def route_message(state: MessageState) -> str:
    if state.get("messageType") == "audioMessage":
        return "voice"
    return "text"
#--------------------------------------------------------

# Node 2a: Text process Ai Response
class Response(BaseModel):
    response: str = Field(  # Fixed typo: responce -> response
        ...,
        description="The response to the user's message."
    )

# Updated prompt with memory
Response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful and intelligent assistant.

            ## Language Rule (Most Important)
            - ALWAYS detect the language of the user's message.
            - ALWAYS respond in the EXACT same language the user used.
            - NEVER switch languages unless the user switches first.

            ## Conversation History (VERY IMPORTANT)
            Here is the recent conversation history between you and the user:
            {history_context}
            
            Use this history to:
            1. Maintain context and continuity
            2. Refer back to previous topics
            3. Answer follow-up questions correctly
            4. Remember user preferences or information shared earlier

            ## Examples

            User: "مرحبا، كيف حالك؟"
            Assistant: "مرحباً! أنا بخير، شكراً لك. كيف يمكنني مساعدتك اليوم؟"

            User: "Hello, what is the capital of France?"
            Assistant: "The capital of France is Paris."

            User: "Mujhe biryani ki recipe chahiye."
            Assistant: "Zaroor! Biryani banane ke liye in cheezon ki zaroorat hogi..."

            ## Behavior
            - Be helpful, accurate, and concise.
            - Match the user's tone — formal ya casual jo bhi user use kare.
            - If a message is unclear, ask for clarification in the same language they used.
            - If the conversation history is empty, just respond to the current message normally."""
        ),
        ("human", "Current message: {message}"),
    ]
)

llm_response = llm.with_structured_output(Response)

def process_message(state: MessageState):
    message = state.get("message", "")
    sender = state.get("sender", "")
    
    # Get user's conversation history
    user_history = get_user_history(sender, limit=10)
    
    # Create readable history context
    history_context = ""
    if user_history:
        history_context = "Previous conversation:\n"
        for msg in user_history:
            if msg["role"] == "user":
                history_context += f"User: {msg['content']}\n"
            else:
                history_context += f"Assistant: {msg['content']}\n"
        history_context += "\n"
    else:
        history_context = "No previous conversation. This is a new conversation.\n\n"
    
    # Format prompt with history
    formatted_prompt = Response_prompt.format_prompt(
        history_context=history_context,
        message=message
    )
    
    # Get response from LLM
    response: Response = llm_response.invoke(formatted_prompt)
    
    # Update user's history with this exchange
    update_user_history(sender, message, response.response)
    
    return {"reply": response.response, "history": get_user_history(sender)}

#--------------------------------------------------------

# Node 2b: Voice handle 
def process_voice(state: MessageState):
    reply = "Hey your voice msg received"
    return {"reply": reply, "history": state.get("history", [])}

#--------------------------------------------------------

# Node 3: Reply ready
def send_reply(state: MessageState) -> MessageState:
    return state

#--------------------------------------------------------
# Graph--------
builder = StateGraph(MessageState)

# Nodes -------
builder.add_node("receive_message", receive_message)
builder.add_node("process_message", process_message)
builder.add_node("process_voice", process_voice)
builder.add_node("send_reply", send_reply)

# Edges---------
builder.set_entry_point("receive_message")

# Conditional edge
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

# Build
graph = builder.compile()