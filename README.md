# WhatsApp Server in LangGraph

This project connects a WhatsApp server (using Baileys) with a LangGraph backend to process incoming messages.

---

## 🚀 Steps to Run

### Step 1: Start Baileys WhatsApp Server

- Run the Baileys WhatsApp server on your local machine or production server.
- After starting the server, paste the following webhook URL in the Baileys server code to send requests to LangGraph:

**For Local PC:**
http://localhost:8000/webhook

**For Production Server:**
https://your-domain.com/webhook

- Connect your WhatsApp account with the Baileys server.

---

### Step 2: Start LangGraph Server

Open a **second terminal** and run the following command:

```bash
uvicorn app:app --reload
This will start the LangGraph server on http://localhost:8000

Step 3: Enjoy!
Once both servers are running, your WhatsApp messages will be forwarded to LangGraph automatically.

