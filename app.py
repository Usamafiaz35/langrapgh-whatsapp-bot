from fastapi import FastAPI
from fastapi.responses import Response
from schemas import WhatsAppPayload
from main import graph


app = FastAPI()


#-----------------Routes-------------------------

@app.get("/")
def health():
    return {"status": "LangGraph server running ✅"}


#----------------- Main Route----------------------
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