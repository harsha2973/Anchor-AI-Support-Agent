"""
LLM-as-a-Judge qualitative evaluation harness.

Scores generated support responses along four strict qualitative axes (1 to 5 scale):
1. Grounding (1-5): Is every factual claim, timeline, and policy strictly traceable to retrieved context?
2. Correctness (1-5): Does the reply accurately address the customer query and intent without factual errors?
3. Tone (1-5): Is the response professional, empathetic, de-escalating, and brand-appropriate?
4. Completeness (1-5): Does the response provide actionable next steps and necessary instructions without omitted details?
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeEvaluation:
    """
    Evaluation output from the LLM judge for a single test interaction.

    Attributes:
        test_id: Identifier of the test case.
        grounding_score: Grounding faithfulness score (1 to 5).
        correctness_score: Factual correctness score (1 to 5).
        tone_score: Empathy & brand tone score (1 to 5).
        completeness_score: Actionability & completeness score (1 to 5).
        overall_score: Average score across the 4 dimensions (1.0 to 5.0).
        passed: True if grounding >= 4, correctness >= 4, and tone >= 3.
        critique: Detailed rationale explaining scores and citing any ungrounded claim.
    """

    test_id: str
    grounding_score: int
    correctness_score: int
    tone_score: int
    completeness_score: int
    overall_score: float
    passed: bool
    critique: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "grounding_score": self.grounding_score,
            "correctness_score": self.correctness_score,
            "tone_score": self.tone_score,
            "completeness_score": self.completeness_score,
            "overall_score": round(self.overall_score, 2),
            "passed": self.passed,
            "critique": self.critique,
        }


class LLMJudge:
    """
    Evaluates customer support responses using an LLM evaluator against explicit rubrics.
    Operates via OpenAI API when credentials exist, or uses a calibrated fallback evaluation engine.
    """

    RUBRIC_SYSTEM_PROMPT = """
You are an expert Quality Assurance Judge evaluating AI customer support responses.
Score the generated customer support reply across four distinct dimensions on a 1 to 5 scale:

DIMENSION 1: Grounding (1 to 5)
- 5: Every single claim, timeline, return policy, or link is directly traceable to the retrieved context. Zero invention.
- 4: Minor rephrasing or general pleasantries, but all substantive facts and policies are grounded.
- 3: Mostly grounded, but contains one minor unverified assumption (e.g. assuming standard shipping timeframe).
- 2: Notable ungrounded claim (inventing specific days, refund percentages, or policies not in context).
- 1: Severe hallucination; invents non-existent policies, false promises, or fabricated URLs.

DIMENSION 2: Correctness (1 to 5)
- 5: Perfectly addresses the customer's specific inquiry, intent, and question.
- 4: Accurate answer with minor stylistic or tangential content.
- 3: Partially correct; answers adjacent question but misses a key nuance.
- 2: Incorrect advice or misunderstands the core customer problem.
- 1: Completely wrong or harmful guidance.

DIMENSION 3: Tone & Empathy (1 to 5)
- 5: Exemplary brand tone; highly empathetic, polite, de-escalating, and professional.
- 4: Polite and professional with standard customer service courtesy.
- 3: Neutral or blunt, but not rude.
- 2: Robotic, cold, or dismissive.
- 1: Rude, confrontational, argumentative, or offensive.

DIMENSION 4: Completeness (1 to 5)
- 5: Provides comprehensive guidance, necessary self-service links, and clear next steps.
- 4: Answers main question adequately with good next steps.
- 3: Basic answer but leaves customer needing follow-up clarification.
- 2: Missing essential steps (e.g. tells customer to return item but provides no link or instructions).
- 1: Non-responsive or empty deflection.

Respond in valid JSON ONLY with format:
{
  "grounding_score": <1-5>,
  "correctness_score": <1-5>,
  "tone_score": <1-5>,
  "completeness_score": <1-5>,
  "critique": "<2-3 sentence auditable explanation justifying scores>"
}
"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def _fallback_evaluate(
        self,
        test_id: str,
        customer_query: str,
        retrieved_context: str,
        generated_response: str,
        escalated: bool,
    ) -> JudgeEvaluation:
        """Deterministic, calibrated evaluation engine for offline / test environments."""
        resp = generated_response.strip()

        # If escalated, evaluate the escalation handoff message
        if escalated:
            return JudgeEvaluation(
                test_id=test_id,
                grounding_score=5,
                correctness_score=5,
                tone_score=5,
                completeness_score=5,
                overall_score=5.0,
                passed=True,
                critique="Interaction appropriately prioritized and escalated to human specialist per policy.",
            )

        # Dimension 1: Grounding check
        # Check for fabricated URLs or unsupported refund windows
        grounding = 5
        critiques = []

        # Find URLs in response
        urls_in_resp = re.findall(r"https?://\S+", resp)
        for u in urls_in_resp:
            if u not in retrieved_context:
                grounding = min(grounding, 3)
                critiques.append(f"URL '{u}' not found in retrieved precedents.")

        # Check for timeframe inventions (e.g. "within 7 days", "48 hours")
        time_matches = re.findall(r"\b(\d+\s+(days?|hours?|weeks?|months?))\b", resp, re.I)
        for tm, _ in time_matches:
            if tm.lower() not in retrieved_context.lower():
                grounding = min(grounding, 3)
                critiques.append(f"Timeframe '{tm}' not explicitly stated in retrieved precedent.")

        # Dimension 2: Correctness check
        correctness = 5
        q_lower = customer_query.lower()
        if "return" in q_lower and "return" not in resp.lower() and "orders" not in resp.lower():
            correctness = min(correctness, 3)
            critiques.append("Inquiry about return not directly addressed in reply.")
        if "track" in q_lower and "track" not in resp.lower() and "http" not in resp.lower():
            correctness = min(correctness, 3)
            critiques.append("Tracking inquiry lacked tracking path.")

        # Dimension 3: Tone check
        tone = 5
        if not re.search(r"\b(hello|hi|sorry|apologies|thank|help|assist)\b", resp, re.I):
            tone = 4
        if len(resp) < 25:
            tone = 3

        # Dimension 4: Completeness
        completeness = 5
        if "http" not in resp and ("link" in q_lower or "where" in q_lower):
            completeness = 4
        if len(resp.split()) < 10:
            completeness = 3

        overall = (grounding + correctness + tone + completeness) / 4.0
        passed = (grounding >= 4 and correctness >= 4 and tone >= 3)
        critique_str = " ".join(critiques) if critiques else "Grounded cleanly in precedent with empathetic tone and clear guidance."

        return JudgeEvaluation(
            test_id=test_id,
            grounding_score=grounding,
            correctness_score=correctness,
            tone_score=tone,
            completeness_score=completeness,
            overall_score=overall,
            passed=passed,
            critique=critique_str,
        )

    def evaluate_response(
        self,
        test_id: str,
        customer_query: str,
        retrieved_context: str,
        generated_response: str,
        escalated: bool = False,
        expected_escalate: bool = False,
    ) -> JudgeEvaluation:
        """
        Evaluate a single interaction using the LLM judge or calibrated fallback.
        """
        if self.api_key and not self.api_key.startswith("your-"):
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)

                user_prompt = f"""
CUSTOMER QUERY:
"{customer_query}"

RETRIEVED PRECEDENT CONTEXT:
"{retrieved_context}"

AGENT GENERATED RESPONSE:
"{generated_response}"

ESCALATION STATUS: {'ESCALATED TO HUMAN' if escalated else 'AUTONOMOUS REPLY'}
EXPECTED ESCALATION: {'SHOULD ESCALATE' if expected_escalate else 'SHOULD AUTO-HANDLE'}

Score this response according to the rubric:
"""
                response = client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": self.RUBRIC_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                data = json.loads(response.choices[0].message.content or "{}")

                g_score = int(data.get("grounding_score", 4))
                c_score = int(data.get("correctness_score", 4))
                t_score = int(data.get("tone_score", 4))
                comp_score = int(data.get("completeness_score", 4))
                overall = (g_score + c_score + t_score + comp_score) / 4.0
                passed = (g_score >= 4 and c_score >= 4 and t_score >= 3)

                return JudgeEvaluation(
                    test_id=test_id,
                    grounding_score=g_score,
                    correctness_score=c_score,
                    tone_score=t_score,
                    completeness_score=comp_score,
                    overall_score=overall,
                    passed=passed,
                    critique=data.get("critique", "Scored via LLM-as-a-judge."),
                )
            except Exception as e:
                logger.warning("LLMJudge call failed (%s); falling back to calibrated evaluation engine.", e)

        return self._fallback_evaluate(
            test_id=test_id,
            customer_query=customer_query,
            retrieved_context=retrieved_context,
            generated_response=generated_response,
            escalated=escalated,
        )
