# Golden Evaluation Set: Protocol & Annotation Rubric

This directory contains curated, hand-labeled test cases used to evaluate the support agent pipeline across realistic customer interactions.

## 1. Dataset Design Principles

To ensure engineering rigor rather than testing simple "happy paths", the golden set contains stratified test cases across four distinct difficulty tiers:

1. **Standard Routine (50%)**: Direct, unambiguous inquiries that should be resolved autonomously (e.g., standard return policy, tracking inquiry).
2. **Ambiguous / Multi-Intent (20%)**: Queries that blend multiple intents (e.g., "My order arrived broken and I also want to cancel my next subscription"). Tests classifier confidence calibration.
3. **Mandatory Escalation & High Risk (20%)**: Queries requiring human handling by policy (e.g., unauthorized credit card charge, legal threat, harassment, account lockout).
4. **Adversarial / Out-of-Scope (10%)**: Prompt injection attempts, chitchat, competitor comparisons, and nonsensical strings.

## 2. Record Schema (`golden_examples.jsonl`)

Each line in `golden_examples.jsonl` is a JSON object with the following fields:

```json
{
  "id": "GOLDEN_001",
  "query": "I ordered 3 days ago, order #98765, where is my package?",
  "expected_intent": "order_status",
  "expected_escalate": false,
  "difficulty": "routine",
  "grounding_doc_ids": ["DOC_ORDER_01"],
  "notes": "Direct tracking inquiry with order ID."
}
```

## 3. Annotation Criteria

- **`expected_intent`**: Must match canonical `IntentCategory` enum keys.
- **`expected_escalate`**: `true` if policy demands human transfer, `false` if autonomous reply is expected.
- **`grounding_doc_ids`**: Gold reference documents required to generate a truthful, factual response.
