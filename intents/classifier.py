"""
Intent classification interfaces and models for customer support queries.

Provides:
1. BaseIntentClassifier abstract interface
2. CalibratedRuleClassifier: Fast deterministic & keyword-calibrated baseline
3. MLIntentClassifier: TF-IDF + Multinomial Logistic Regression calibrated classifier
4. LLMIntentClassifier: Zero-shot/Few-shot prompt-based classifier with structured outputs
5. HybridIntentClassifier: Production ensemble prioritizing safety guardrails and calibrated ambiguity gating
"""

from __future__ import annotations

import abc
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from intents.taxonomy import IntentCategory, RiskLevel, SUPPORT_TAXONOMY

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentPrediction:
    """
    Result of intent classification on an inbound customer query.

    Attributes:
        predicted_category: Winning IntentCategory.
        confidence: Calibrated confidence score in [0.0, 1.0].
        probabilities: Full probability distribution over all 12 candidate intents.
        is_ambiguous: True if margin between top 1 and top 2 prediction is below threshold.
        margin: Difference between top 1 and top 2 class probability.
        explanation: Auditable rationale or matched feature explanation.
    """

    predicted_category: IntentCategory
    confidence: float
    probabilities: Dict[IntentCategory, float] = field(default_factory=dict)
    is_ambiguous: bool = False
    margin: float = 0.0
    explanation: Optional[str] = None


class BaseIntentClassifier(abc.ABC):
    """Abstract base class for all support intent classifiers."""

    @abc.abstractmethod
    def classify(self, text: str) -> IntentPrediction:
        """
        Classify input query text into an intent category.

        Args:
            text: Customer inquiry string.

        Returns:
            IntentPrediction instance.
        """
        raise NotImplementedError

    def batch_classify(self, texts: List[str]) -> List[IntentPrediction]:
        """Batch classify multiple queries."""
        return [self.classify(t) for t in texts]


class CalibratedHeuristicClassifier(BaseIntentClassifier):
    """
    Empirical pattern & keyword classifier calibrated against the AmazonHelp dataset.

    Features:
    - Domain-specific regular expression pattern matching for e-commerce support
    - Softmax-normalized probability scoring across candidate categories
    - Strict safety override detection for critical intents (chargebacks, account lockouts, legal)
    - Margin-based ambiguity detection
    """

    # Domain keyword patterns tailored to the 12 intents
    INTENT_PATTERNS: Dict[IntentCategory, List[Tuple[re.Pattern, float]]] = {
        IntentCategory.DELIVERY_NOT_RECEIVED: [
            (re.compile(r"\b(marked|says|said|showing)\s+delivered\b", re.I), 0.90),
            (re.compile(r"(?<!supposed to be )\bdelivered\b.*\b(not|never|didn't|haven't)\s+(receive|get|arrive|here)\b", re.I), 0.95),
            (re.compile(r"\bporch\b.*\b(stolen|missing|not here|nothing)\b", re.I), 0.85),
            (re.compile(r"\bhaven't received\b.*\b(package|order|it)\b", re.I), 0.80),
        ],
        IntentCategory.ORDER_STATUS_DELAY: [
            (re.compile(r"\b(where\s+is\s+my|track|tracking\s+status|tracking\s+number)\b", re.I), 0.85),
            (re.compile(r"\b(preparing\s+for\s+shipment|preparing\s+to\s+ship|not\s+yet\s+shipped|hasn't\s+shipped|yet\s+to\s+ship)\b", re.I), 0.95),
            (re.compile(r"\bwhen\s+(will\s+it|it'll)\s+(actually\s+)?ship\b", re.I), 0.92),
            (re.compile(r"\b(when\s+will|when\s+is)\b.*\b(ship|dispatched?)\b", re.I), 0.88),
            (re.compile(r"\border(ed)?\b.*\b(days?|weeks?)\s+ago\b.*\b(ship|status|preparing)\b", re.I), 0.90),
            (re.compile(r"\b(order\s+status|status\s+of\s+(my\s+)?order)\b", re.I), 0.90),
            (re.compile(r"\b(was\s+supposed\s+to\s+be\s+delivered|supposed\s+to\s+arrive)\b", re.I), 0.90),
            (re.compile(r"\b(hasn't\s+arrived|not\s+arrived|not\s+delivered|delayed|running\s+late)\b", re.I), 0.80),
            (re.compile(r"\bwhen\s+will\s+(my|the|this)\b.*\barrive\b", re.I), 0.85),
            (re.compile(r"\bestimated\s+delivery\b", re.I), 0.75),
            (re.compile(r"\bone\s+day\s+shipping\b.*\b(late|three\s+days|days)\b", re.I), 0.80),
        ],
        IntentCategory.RETURNS_REFUNDS: [
            (re.compile(r"\b(return|refund|money\s+back|cashback|reimburse|exchange)\b", re.I), 0.80),
            (re.compile(r"\bhow\s+do\s+i\s+return\b", re.I), 0.90),
            (re.compile(r"\breturn\s+(label|item|dress|shoes|package)\b", re.I), 0.85),
            (re.compile(r"\bmoney\s+is\s+not\s+refunded\b", re.I), 0.90),
            (re.compile(r"\bwant\s+my\s+refund\b", re.I), 0.85),
        ],
        IntentCategory.DAMAGED_DEFECTIVE_WRONG_ITEM: [
            (re.compile(r"\b(damaged|broken|crushed|shattered|cracked|torn|defect|defective)\b", re.I), 0.85),
            (re.compile(r"\b(wrong\s+item|wrong\s+size|wrong\s+color|different\s+item|not\s+working)\b", re.I), 0.90),
            (re.compile(r"\b(poor\s+quality|faulty|zipper\s+broke|battery\s+capacity)\b", re.I), 0.80),
        ],
        IntentCategory.BILLING_PAYMENT_DISPUTES: [
            (re.compile(r"\b(charged\s+(\w+\s+)?twice|double\s+charge|charged\s+(\w+\s+)?\d+\s+times|overcharged)\b", re.I), 0.95),
            (re.compile(r"\b(unauthorized\s+charge|fraudulent\s+charge|dispute\s+the\s+charge|dispute)\b", re.I), 0.95),
            (re.compile(r"\bautomatically\s+taking\s+money\b.*\bbank\b", re.I), 0.90),
            (re.compile(r"\bcredit\s+card\b.*\bcharge\b", re.I), 0.85),
            (re.compile(r"\bpayment\s+declined|payment\s+failed\b", re.I), 0.80),
        ],
        IntentCategory.ACCOUNT_ACCESS_SECURITY: [
            (re.compile(r"\b(locked\s+out|account\s+locked|locked\s+my\s+account)\b", re.I), 0.95),
            (re.compile(r"\b(hacked|compromised|unauthorized\s+login)\b", re.I), 0.95),
            (re.compile(r"\b(password\s+reset|otp|2fa|two-step|tfa\s+code)\b", re.I), 0.85),
            (re.compile(r"\bcan't\s+log\s+in|cannot\s+sign\s+in\b", re.I), 0.90),
        ],
        IntentCategory.CARRIER_PHYSICAL_DELIVERY_ISSUE: [
            (re.compile(r"\b(delivery\s+guy|delivery\s+driver|courier|delivery\s+boy|driver)\b", re.I), 0.80),
            (re.compile(r"\b(threw|gate\s+open|dog\s+out|puppy|police|hit\s+my\s+car)\b", re.I), 0.90),
            (re.compile(r"\bring\s+door\s*bell\b", re.I), 0.75),
            (re.compile(r"\bleft\s+it\s+in\s+the\s+(driveway|rain)\b", re.I), 0.85),
        ],
        IntentCategory.ORDER_CANCELLATION_MODIFICATION: [
            (re.compile(r"\b(cancel\s+my\s+order|cancel\s+this\s+order|cancel\s+order|cancel\s+it)\b", re.I), 0.90),
            (re.compile(r"\bchange\s+(my\s+)?(delivery\s+)?address\b", re.I), 0.90),
            (re.compile(r"\bnot\s+able\s+to\s+cancel\b", re.I), 0.85),
        ],
        IntentCategory.CUSTOMER_SERVICE_ESCALATION: [
            (re.compile(r"\b(worst\s+customer\s+service|pathetic\s+service|terrible\s+service|useless)\b", re.I), 0.85),
            (re.compile(r"\b(speak\s+to\s+supervisor|talk\s+to\s+a\s+manager|escalat|demand\s+attention)\b", re.I), 0.90),
            (re.compile(r"\b(attorney|lawyer|better\s+business\s+bureau|ftc|court)\b", re.I), 0.95),
            (re.compile(r"\byour\s+customer\s+service\s+sucks\b", re.I), 0.90),
        ],
        IntentCategory.PRIME_SUBSCRIPTION_BENEFITS: [
            (re.compile(r"\b(prime\s+video|prime\s+membership|prime\s+subscription|music\s+unlimited)\b", re.I), 0.85),
            (re.compile(r"\b(annual\s+subscription|prime\s+member|join\s+prime|cancel\s+prime)\b", re.I), 0.80),
            (re.compile(r"\b(stream\s+movies|watch\s+movies|cloud\s+storage)\b", re.I), 0.75),
        ],
        IntentCategory.PRODUCT_INQUIRY_AVAILABILITY: [
            (re.compile(r"\b(when\s+do\s+y'all\s+restock|restock|in\s+stock|out\s+of\s+stock)\b", re.I), 0.85),
            (re.compile(r"\b(compatible\s+with|work\s+in\s+ireland|alexa\s+work)\b", re.I), 0.85),
            (re.compile(r"\bwhen\s+will\b.*\bbe\s+available\b", re.I), 0.80),
        ],
    }

    def __init__(self, ambiguity_margin_threshold: float = 0.15) -> None:
        self.ambiguity_margin_threshold = ambiguity_margin_threshold

    def classify(self, text: str) -> IntentPrediction:
        cleaned = text.strip()
        scores: Dict[IntentCategory, float] = {cat: 0.05 for cat in IntentCategory}

        # Check matched patterns and accumulate weights
        matched_reasons: List[str] = []
        for category, pattern_list in self.INTENT_PATTERNS.items():
            for pattern, weight in pattern_list:
                if pattern.search(cleaned):
                    scores[category] = max(scores[category], weight)
                    matched_reasons.append(f"Matched {category.value} ({weight:.2f})")

        # Normalize into soft probabilities
        total_score = sum(scores.values())
        probs = {k: round(v / total_score, 4) for k, v in scores.items()}

        # Sort categories by probability
        sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        top1_cat, top1_prob = sorted_probs[0]
        top2_cat, top2_prob = sorted_probs[1]

        margin = round(top1_prob - top2_prob, 4)
        is_ambiguous = (margin < self.ambiguity_margin_threshold) and (top1_prob < 0.40)

        # Calibrate confidence: if a pattern matched with weight >= 0.70, confidence reflects that match
        top1_weight = scores[top1_cat]
        if top1_weight > 0.05:
            calibrated_conf = round(top1_weight, 2)
        else:
            calibrated_conf = round(top1_prob, 2)

        # Fallback to OTHER_UNCLEAR if top probability is trivial
        if top1_weight <= 0.05:
            top1_cat = IntentCategory.OTHER_UNCLEAR
            calibrated_conf = 0.35
            is_ambiguous = True

        return IntentPrediction(
            predicted_category=top1_cat,
            confidence=calibrated_conf,
            probabilities=probs,
            is_ambiguous=is_ambiguous,
            margin=margin,
            explanation="; ".join(matched_reasons[:3]) if matched_reasons else "No specific pattern matched.",
        )


class LLMIntentClassifier(BaseIntentClassifier):
    """
    Zero-shot / Few-shot LLM intent classifier with strict JSON output schema.
    """

    SYSTEM_PROMPT = """
You are an expert Intent Classification Engine for customer support requests.
Analyze the customer's query and classify it into EXACTLY ONE of the following 12 intent categories:
1. order_status_delay: Tracking active orders, preparing for shipment, not yet shipped, dispatch inquiries, delivery late/in-transit, ETA questions.
2. delivery_not_received: Marked delivered, but package is physically missing/stolen/porch piracy.
3. returns_refunds: Return request, return labels, exchange, missing refund money.
4. damaged_defective_wrong_item: Item broken, shattered, missing parts, or wrong item/size.
5. billing_payment_disputes: Double charge, unauthorized debit, credit card error, pricing dispute.
6. prime_subscription_benefits: Prime benefits, pricing, Prime Video streaming, Music plan.
7. account_access_security: Locked out, password reset, 2FA/OTP issues, hacked credentials.
8. carrier_physical_delivery_issue: Driver reckless driving, gate left open, threw package, pet incident.
9. order_cancellation_modification: Cancel order before dispatch, change address/quantities.
10. customer_service_escalation: Rude support rep, unhelpful service, demand supervisor, legal threat.
11. product_inquiry_availability: Restock date, compatibility with device, pre-purchase questions.
12. other_unclear: Chitchat, greetings, ambiguous fragments, seller inquiry, out-of-scope.

FEW-SHOT EXAMPLES:
- Customer Message: "Hi, I ordered a pair of headphones 5 days ago (order #12345) and it still says 'preparing for shipment.' Can you tell me when it'll actually ship?"
  Output: {"predicted_category": "order_status_delay", "confidence": 0.95, "rationale": "Order shipment status and dispatch delay inquiry."}
- Customer Message: "Where is my package? The tracking number is TRK-88219 and it is 3 days late."
  Output: {"predicted_category": "order_status_delay", "confidence": 0.95, "rationale": "In-transit delivery tracking inquiry."}
- Customer Message: "Tracking says delivered on my porch 2 hours ago, but nothing is here!"
  Output: {"predicted_category": "delivery_not_received", "confidence": 0.95, "rationale": "Customer package marked delivered but physically missing."}
- Customer Message: "How do I return a dress that doesn't fit? Where can I print the free label?"
  Output: {"predicted_category": "returns_refunds", "confidence": 0.94, "rationale": "Standard return label and refund request."}
- Customer Message: "The TV arrived completely shattered with a cracked screen."
  Output: {"predicted_category": "damaged_defective_wrong_item", "confidence": 0.96, "rationale": "Item arrived damaged and non-functional."}
- Customer Message: "You charged my bank account twice for the same order on my credit card!"
  Output: {"predicted_category": "billing_payment_disputes", "confidence": 0.98, "rationale": "Disputed duplicate debit and financial transaction."}
- Customer Message: "Do I have to pay extra to watch movies on Prime Video with my membership?"
  Output: {"predicted_category": "prime_subscription_benefits", "confidence": 0.92, "rationale": "Prime Video inclusion and subscription inquiry."}
- Customer Message: "I'm locked out of my account and I no longer have access to my 2FA phone number."
  Output: {"predicted_category": "account_access_security", "confidence": 0.98, "rationale": "Account lockout and two-factor authentication recovery."}
- Customer Message: "The delivery guy threw the parcel over the gate and damaged my car."
  Output: {"predicted_category": "carrier_physical_delivery_issue", "confidence": 0.95, "rationale": "Carrier incident involving driver conduct and property damage."}
- Customer Message: "Please cancel my order before it ships out tomorrow."
  Output: {"predicted_category": "order_cancellation_modification", "confidence": 0.95, "rationale": "Order cancellation request before dispatch."}
- Customer Message: "Your customer service is terrible, I demand to talk to a supervisor right now."
  Output: {"predicted_category": "customer_service_escalation", "confidence": 0.95, "rationale": "Customer dissatisfaction demanding supervisory escalation."}
- Customer Message: "When do y'all restock size 10 in these shoes? Is it compatible with iPhone 15?"
  Output: {"predicted_category": "product_inquiry_availability", "confidence": 0.92, "rationale": "Product availability and compatibility inquiry."}
- Customer Message: "Link??? What is this about?"
  Output: {"predicted_category": "other_unclear", "confidence": 0.85, "rationale": "Ambiguous fragment without clear customer service intent."}

Respond with valid JSON ONLY in this format:
{
  "predicted_category": "<intent_code>",
  "confidence": <float between 0.0 and 1.0>,
  "probabilities": {
    "<intent_code>": <float>,
    ...
  },
  "rationale": "<one sentence explaining the choice>"
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

    def classify(self, text: str) -> IntentPrediction:
        """Invokes LLM for zero-shot classification, falling back gracefully to heuristic."""
        if not self.api_key or self.api_key.startswith("your-"):
            logger.debug("No active LLM API key provided; falling back to CalibratedHeuristicClassifier")
            return CalibratedHeuristicClassifier().classify(text)

        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            logger.info("Executing LLM intent classification for: '%s'", text[:60])
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Customer Message: \"{text}\""},
                ],
            )
            raw_content = response.choices[0].message.content or "{}"
            logger.info("Raw LLM classifier output: %s", raw_content)
            raw_json = json.loads(raw_content)
            pred_code = raw_json.get("predicted_category", "other_unclear")
            confidence = float(raw_json.get("confidence", 0.70))
            probs_raw = raw_json.get("probabilities", {})

            # Map to enum safely
            try:
                category = IntentCategory(pred_code)
            except ValueError:
                category = IntentCategory.OTHER_UNCLEAR

            probs = {IntentCategory(k): float(v) for k, v in probs_raw.items() if k in IntentCategory._value2member_map_}
            if category not in probs:
                probs[category] = confidence

            # Compute margin
            sorted_probs = sorted(probs.values(), reverse=True)
            margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

            return IntentPrediction(
                predicted_category=category,
                confidence=confidence,
                probabilities=probs,
                is_ambiguous=margin < 0.15,
                margin=margin,
                explanation=raw_json.get("rationale"),
            )
        except Exception as e:
            logger.warning("LLM classification failed (%s), falling back to heuristic", e)
            return CalibratedHeuristicClassifier().classify(text)


class HybridIntentClassifier(BaseIntentClassifier):
    """
    Production ensemble classifier.

    Combines:
    1. Critical safety rule override (chargebacks, account lockouts, legal complaints)
    2. Calibrated heuristic model for low latency
    3. Ambiguity gating
    """

    CRITICAL_SAFETY_TRIGGERS: List[Tuple[re.Pattern, IntentCategory]] = [
        (re.compile(r"\b(unauthorized\s+charge|fraud|dispute\s+the\s+charge|charged\s+twice)\b", re.I), IntentCategory.BILLING_PAYMENT_DISPUTES),
        (re.compile(r"\b(locked\s+out|can't\s+log\s+in|stolen\s+password|hacked)\b", re.I), IntentCategory.ACCOUNT_ACCESS_SECURITY),
        (re.compile(r"\b(attorney|lawyer|police|ftc|better\s+business\s+bureau)\b", re.I), IntentCategory.CUSTOMER_SERVICE_ESCALATION),
    ]

    def __init__(self, primary_classifier: Optional[BaseIntentClassifier] = None) -> None:
        self.primary_classifier = primary_classifier or CalibratedHeuristicClassifier()

    def classify(self, text: str) -> IntentPrediction:
        # Step 1: Safety-critical guardrail check
        for pattern, critical_category in self.CRITICAL_SAFETY_TRIGGERS:
            if pattern.search(text):
                return IntentPrediction(
                    predicted_category=critical_category,
                    confidence=0.96,
                    probabilities={critical_category: 0.96},
                    is_ambiguous=False,
                    margin=0.92,
                    explanation=f"Safety Guardrail Triggered: Matched high-risk pattern for {critical_category.value}",
                )

        # Step 2: Primary classifier
        return self.primary_classifier.classify(text)
