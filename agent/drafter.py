"""
Response drafting module conditioned on retrieved grounding documents.

Synthesizes helpful, empathetic, and strictly grounded customer support responses.
Enforces that facts, timelines, and policy statements are backed by retrieved context.
"""

from __future__ import annotations

import abc
import logging
from typing import List, Optional, Tuple

from retrieval.store import RetrievalResult

logger = logging.getLogger(__name__)


class BaseResponseDrafter(abc.ABC):
    """Abstract interface for drafting customer support replies."""

    @abc.abstractmethod
    def draft(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
        intent_category: Optional[str] = None,
        thread_context: Optional[str | List[str]] = None,
    ) -> Tuple[str, List[str]]:
        """
        Draft an answer conditioned on query, grounding context, and thread history.

        Args:
            query: Inbound customer inquiry.
            retrieved_docs: Relevant grounding documents.
            intent_category: Predicted intent category.
            thread_context: Conversational history prior to current turn.

        Returns:
            Tuple of (drafted_response_text, list_of_cited_doc_ids).
        """
        raise NotImplementedError


class GroundedResponseDrafter(BaseResponseDrafter):
    """
    Synthesizes grounded support responses.

    Injects retrieved historical brand replies and reference documentation into
    the LLM context with strict anti-hallucination instructions.
    """

    SYSTEM_PROMPT = """
You are an AI customer support specialist for Amazon.
Your objective is to craft a helpful, polite, and empathetic response addressing the customer's inquiry.

STRICT GROUNDING DIRECTIVE:
1. You MUST ground all specific policy claims (return windows, delivery buffers, refund timelines, replacement routes) EXCLUSIVELY in the provided retrieved precedents.
2. DO NOT invent specific timelines, policies, or refund promises that are not evidenced in the reference context.
3. If the precedents advise allowing a 24-36 hour buffer for premature carrier delivery scans, advise that.
4. If the precedents provide specific self-service paths (e.g., 'Your Orders' at amazon.com/orders or Online Returns Center at amazon.com/returns), provide those links.
5. Never ask the customer to post passwords, OTPs, or payment details publicly.
6. Write a direct, professional, and empathetic customer-facing reply. Do NOT output internal commentary or preambles.
"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        api_key: Optional[str] = None,
    ) -> None:
        import os
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def _format_context_block(
        self,
        retrieved_docs: List[RetrievalResult],
        thread_context: Optional[str | List[str]],
    ) -> str:
        """Format precedents and conversation thread history for the prompt."""
        sections = []

        if thread_context:
            if isinstance(thread_context, list):
                hist = "\n".join(f"- {t}" for t in thread_context)
            else:
                hist = str(thread_context).strip()
            sections.append(f"CONVERSATION HISTORY:\n{hist}")

        if retrieved_docs:
            precedents = []
            for i, res in enumerate(retrieved_docs, 1):
                doc = res.document
                precedents.append(
                    f"[{i}] Document ID: {doc.doc_id} | Category: {doc.category or 'general'} | Score: {res.score:.2f}\n"
                    f"    Verified Content: \"{doc.content.strip()}\""
                )
            sections.append("RETRIEVED PRECEDENTS & POLICY GUIDANCE:\n" + "\n".join(precedents))
        else:
            sections.append("RETRIEVED PRECEDENTS: No verified precedents found.")

        return "\n\n".join(sections)

    def draft(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
        intent_category: Optional[str] = None,
        thread_context: Optional[str | List[str]] = None,
    ) -> Tuple[str, List[str]]:
        logger.info("Drafting grounded response for query (docs count=%d)", len(retrieved_docs))

        if not retrieved_docs:
            return (
                "Thank you for contacting customer support. We are looking into your request. "
                "A support specialist will assist you shortly.",
                [],
            )

        citations = [r.document.doc_id for r in retrieved_docs]

        # Check if live LLM API is available
        if self.api_key and not self.api_key.startswith("your-"):
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                context_block = self._format_context_block(retrieved_docs, thread_context)
                user_content = (
                    f"Customer Inquiry: \"{query}\"\n"
                    f"Classified Intent: {intent_category or 'general'}\n\n"
                    f"{context_block}\n\n"
                    "Draft an empathetic, grounded response strictly adhering to the precedents above:"
                )

                response = client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                )
                draft_text = (response.choices[0].message.content or "").strip()
                if draft_text:
                    return draft_text, citations
            except Exception as e:
                logger.warning("LLM response generation failed (%s), falling back to grounded synthesis", e)

        # High-fidelity grounded synthesis over top retrieved precedent
        best_doc = retrieved_docs[0].document
        import re
        # Strip trailing agent signoffs like ^SD, ^HN, ^CB
        clean_content = re.sub(r"\s*\^[A-Z]{2,3}\s*$", "", best_doc.content.strip())
        # Clean Twitter handle mentions like @Customer
        clean_content = re.sub(r"^@\w+\s+", "", clean_content)

        draft = f"Hello! {clean_content}"
        return draft, citations
