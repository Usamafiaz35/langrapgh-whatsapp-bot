from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from graphmerge import graph
from typing import Optional

app = FastAPI()

class MediaPayload(BaseModel):
    type: str
    mimetype: Optional[str] = None
    fileName: Optional[str] = None
    caption: Optional[str] = None
    seconds: Optional[int] = None
    ptt: bool = False
    base64: str = ""

class WhatsAppPayload(BaseModel):
    sender: str
    senderName: str = ""
    replyToJid: str
    messageId: str = ""
    messageTimestamp: int = 0
    messageType: str = "text"
    message: str = ""
    media: Optional[MediaPayload] = None

@app.get("/")
def health():
    return {"status": "LangGraph server running ✅"}

@app.post("/webhook")
async def whatsapp_webhook(payload: WhatsAppPayload):
    if payload.messageType == "audioMessage":

        result = graph.invoke({
            "sender": payload.sender,
            "message": payload.message,
            "messageType": payload.messageType,
            "audio_b64": payload.media.base64 if payload.media else "",
            "audio_bytes": b"",       
            "transcript": "",
            "reply": "",
            "history": []
        })

    
        return Response(
            content=result["audio_bytes"],
            media_type="audio/mpeg"
        )
    
    else:
        
        result = graph.invoke({
        "sender": payload.sender,
        "message": payload.message,
        "messageType": payload.messageType,
        "reply": "",  
        "history": [] 
        })
    

        return {"reply": result['reply_msg']}