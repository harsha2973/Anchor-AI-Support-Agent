"""
FastAPI web application for Anchor AI Customer Support Console.

Serves the Anchor evaluation interface and proxies queries to agent.pipeline.run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import agent

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Anchor Support Agent Web Console",
    description="Internal evaluation & demo interface for Anchor AI Support Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


class ChatMessageRequest(BaseModel):
    message: str
    thread_context: Optional[List[str]] = None


# Pre-populated test queries matching known golden evaluation scenarios
PRESET_TEST_CASES = [
    {
        "id": "order_status",
        "label": "Order Status Delay",
        "query": "Hi, I ordered a pair of headphones 5 days ago (order #112-9847291) and it still says 'preparing for shipment.' Can you tell me when it'll actually ship?",
        "intent": "order_status_delay",
        "expected_escalate": False,
    },
    {
        "id": "damaged_item",
        "label": "Damaged / Defective Item",
        "query": "The TV arrived completely shattered with a cracked screen. I took photos of the broken box and panel. How can I get a replacement?",
        "intent": "damaged_defective_wrong_item",
        "expected_escalate": False,
    },
    {
        "id": "account_security",
        "label": "Account Security & Fraud",
        "query": "Someone compromised my Amazon account and ordered a $500 gift card! Cancel it immediately, my email was changed without permission!",
        "intent": "account_security_access",
        "expected_escalate": True,
    },
    {
        "id": "refund_window",
        "label": "Return & Refund Window",
        "query": "How do I return a dress that doesn't fit? What is the return window and where can I print the prepaid return label?",
        "intent": "returns_refunds_inquiries",
        "expected_escalate": False,
    },
    {
        "id": "low_signal",
        "label": "Low-Signal Inquiry",
        "query": "link??? what is this",
        "intent": "other_unclear",
        "expected_escalate": True,
    },
]


@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "app": "Anchor AI Console"}


@app.get("/api/test-cases")
async def get_test_cases() -> List[Dict[str, Any]]:
    return PRESET_TEST_CASES


@app.post("/api/chat")
async def process_chat(req: ChatMessageRequest) -> Dict[str, Any]:
    cleaned_query = (req.message or "").strip()
    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query message cannot be empty")

    try:
        # Pass conversation history as thread_context to agent.run
        thread_history = req.thread_context if req.thread_context else None
        result = agent.run(message=cleaned_query, thread_context=thread_history)
        return result
    except Exception as e:
        logger.exception("Error executing agent.run: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent execution error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="Web UI template not found")
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=True)
