# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from graph import graph

app = FastAPI()

# Baileys se aane wala payload ka format
class WhatsAppPayload(BaseModel):
    sender: str
    senderName: str = ""
    replyToJid: str
    messageId: str = ""
    messageTimestamp: int = 0
    messageType: str = "text"
    message: str = ""

@app.post("/webhook")
async def whatsapp_webhook(payload: WhatsAppPayload):
    # LangGraph invoke karo
    result = graph.invoke({
        "sender": payload.sender,
        "message": payload.message,
        "reply": ""
    })

    # Reply return karo — Baileys ye padhega aur WhatsApp per bhejena
    return {"reply": result["reply"]}

@app.get("/")
def health():
    return {"status": "LangGraph server running ✅"}