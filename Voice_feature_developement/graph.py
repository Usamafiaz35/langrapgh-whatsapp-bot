from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import List, TypedDict, Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
import base64
import tempfile
import os

load_dotenv()
#--------------------------------------------------------

# Updated MessageState with history
class MessageState(TypedDict):
    sender: str
    message: str
    messageType: str
    

    audio_b64: str

    transcript: str

    reply: str

    audio_bytes: bytes
    response_mimetype: str
#--------------------------------------------------------
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


#--------------------------------------------------------

# Node 1: Message receive
def receive_message(state: MessageState) -> MessageState:
    return state

#--------------------------------------------------------


# Audio base 64 ko transcribe karne wala node

def transcribe_audio_node(state: MessageState):
    """
    LangGraph node: base64 audio ko transcribe karta hai.
    
    state mein expect karta hai:
        - audio_b64 (str): base64 encoded audio (WhatsApp se receive hua)
    
    state mein add karta hai:
        - transcript (str): transcribed text
    """


    audio_b64 = state["audio_b64"]
    
    # Base64 decode karo
    audio_bytes = base64.b64decode(audio_b64)
    
    # Temp file banao — no manual path management
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        with open(tmp_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )
        transcript = result.text
    finally:
        os.unlink(tmp_path)  # cleanup
    
    return {"transcript": transcript}


#--------------------------------------------------------
class Response(BaseModel):
    response: str = Field(
        ...,
        description="The response to the user's message."
    )

Response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful and intelligent assistant.

            ## Language Rule (Most Important)
            - ALWAYS detect the language of the user's message.
            - ALWAYS respond in the EXACT same language the user used.
            - NEVER switch languages unless the user switches first.

            ## Examples

            User: "مرحبا، كيف حالك؟"
            Assistant: "مرحباً! أنا بخير، شكراً لك. كيف يمكنني مساعدتك اليوم؟"

            User: "Hello, what is the capital of France?"
            Assistant: "The capital of France is Paris."

            User: "Mujhe biryani ki recipe chahiye."
            Assistant: "Zaroor! Biryani banane ke liye in cheezон ki zaroorat hogi..."

            User: "¿Cuál es la mejor manera de aprender inglés?"
            Assistant: "La mejor manera de aprender inglés es practicar todos los días..."

            User: "كيف أتعلم البرمجة؟"
            Assistant: "يمكنك تعلم البرمجة من خلال البدء بلغة Python لأنها سهلة للمبتدئين..."

            ## Behavior
            - Be helpful, accurate, and concise.
            - Match the user's tone — formal ya casual jo bhi user use kare.
            - If a message is unclear, ask for clarification in the same language they used."""
        ),
        ("human", "Message: {message}"),
    ]
)

llm_responce = llm.with_structured_output(Response)


# Node 2a: Text process karo
def process_message(state: MessageState):
    transcription = state["transcript"]

    response: Response = llm_responce.invoke(Response_prompt.format_prompt(message=transcription))
    
    return {"reply": response.response}

#--------------------------------------------------------
def text_to_speech_node(state: MessageState):
    """
    LangGraph node: text ko speech mein convert karta hai.
    
    state mein expect karta hai:
        - tts_input (str): jo text bولنا hai
    
    state mein add karta hai:
        - audio_bytes (bytes): mp3 file ka content
        - audio_path (str): temp file path (agar directly path chahiye)
    """
    text = state["reply"]

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    )

    # Option 1: bytes state mein store karo
    audio_bytes = response.read()

    # Option 2: temp file mein save karo aur path store karo
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    return {
        "audio_bytes": audio_bytes,   
        "response_mimetype": "audio/mpeg"         
    }
#--------------------------------------------------------

def send_reply(state: MessageState) -> MessageState:
    return state

#--------------------------------------------------------
# Graph--------
builder = StateGraph(MessageState)

# Nodes -------
builder.add_node("receive_message", receive_message) 
builder.add_node("transcribe_audio_node", transcribe_audio_node) 
builder.add_node("process_message", process_message) 
builder.add_node("text_to_speech_node", text_to_speech_node)
builder.add_node("send_reply", send_reply)

# Edges---------
builder.add_edge(START,"receive_message")
builder.add_edge("receive_message", "transcribe_audio_node")
builder.add_edge("transcribe_audio_node", "process_message")
builder.add_edge("process_message", "text_to_speech_node")
builder.add_edge("text_to_speech_node","send_reply")
builder.add_edge("send_reply",END)

# Build
graph = builder.compile()