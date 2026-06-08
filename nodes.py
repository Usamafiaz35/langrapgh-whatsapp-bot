from state import MessageState
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from utility_func import get_user_history, update_user_history
import base64
import tempfile
import os
from models import client, llm




#----------------------Node 1 (receive_message)----------------------------

# Message receive
def receive_message(state: MessageState) -> MessageState:
    return state


# Router function — To check messageType
def route_message(state: MessageState) -> str:
    if state.get("messageType") == "audioMessage":
        return "voice"
    return "text"
#--------------------------Node 2 (process_message) ---------------------------------------

# Text process Ai Response
class Response(BaseModel):
    response: str = Field(  
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
    
    return {"reply_msg": response.response, "history": get_user_history(sender)}

#---------------------------------Node 3 (transcribe_audio_node)------------------------------------------------

# Transcribe Audio_base_64 file

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


#--------------------------Node 4 (process_message_audio)---------------------------------------
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

def process_message_audio(state: MessageState):
    transcription = state.get("transcript", "")
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
        message=transcription
    )
    
    # Get response from LLM
    response: Response = llm_response.invoke(formatted_prompt)
    
    # Update user's history with this exchange
    update_user_history(sender, transcription, response.response)
    
    return {"reply_audio": response.response, "history": get_user_history(sender)}



#--------------------------Node 5 (text_to_speech_node)---------------------------------------

def text_to_speech_node(state: MessageState):
    """
    LangGraph node: text ko speech mein convert karta hai.
    
    state mein expect karta hai:
        - tts_input (str): jo text bولنا hai
    
    state mein add karta hai:
        - audio_bytes (bytes): mp3 file ka content
        - audio_path (str): temp file path (agar directly path chahiye)
    """
    text = state["reply_audio"]

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


#--------------------------Node 6 (send_reply)---------------------------------------

# Node 3: Reply ready
def send_reply(state: MessageState) -> MessageState:
    return state
