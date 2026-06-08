from typing import List, TypedDict, Literal, Dict, Any



#---------------------Utility Function Memory-------------------------------
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
