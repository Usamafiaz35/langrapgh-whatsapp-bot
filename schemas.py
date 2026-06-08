from pydantic import BaseModel
from typing import Optional


#-----------------Recieve msg from Bailey's-------------
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

#-----------------------------------------------------------