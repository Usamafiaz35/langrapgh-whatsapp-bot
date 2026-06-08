from typing import List, TypedDict, Literal, Dict, Any

class MessageState(TypedDict):
    sender: str
    message: str
    messageType: str
    reply_msg: str

    audio_b64: str

    transcript: str

    reply_audio: str

    audio_bytes: bytes
    response_mimetype: str

    history: List[Dict[str, str]]