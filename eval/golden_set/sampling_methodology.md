# Golden Evaluation Set: Sampling & Labeling Methodology

This document outlines the rigorous protocol for sampling, labeling, and verifying the **200-example Golden Benchmark** in `eval/golden_set/golden_examples.jsonl`.

---

## 1. Objectives & Evaluation Philosophy

A machine learning system cannot be honestly evaluated on clean, synthetically simplified queries. In customer support:
- Real queries are frequently grammatically messy, emotionally charged, multi-clause, or underspecified.
- High-risk failure modes (e.g., missed fraud or legal complaints) carry severe real-world operational liability.
- False escalations create costly human agent queue backlogs.

Therefore, this golden set **deliberately oversamples difficult edge cases** while maintaining broad representation across all 12 empirical taxonomy categories.

---

## 2. Target Composition & Stratification Plan

Target total dataset size: **200 curated examples**.

### A. Difficulty & Edge Case Stratification

| Difficulty Tier | Target Share | Target Count | Sampling Criteria / Definition |
| :--- | :--- | :--- | :--- |
| **Routine / Clear** | 40% | 80 | Unambiguous single-intent queries (order tracking, standard 30-day return, prime benefits). |
| **Ambiguous / Multi-Issue** | 20% | 40 | Queries spanning 2+ intents (e.g., late delivery + demand for refund, or damaged item + cancellation). |
| **Angry / Profane / Escalated** | 15% | 30 | Queries containing profanity, hostile sentiment, exclamation marks, or agent complaints. |
| **High Risk (Safety Critical)** | 15% | 30 | Unauthorized credit card charges, 2FA/account lockouts, driver property damage, legal/FTC threats. |
| **Low Signal / Fragmentary / Out-of-Scope** | 10% | 20 | Messages < 25 characters, single-word fragments ("Link???"), social chitchat, or seller central queries. |

### B. Target Intent Distribution Across 200 Examples

| Intent Category | Target Examples | Key Sub-Types / Edge Cases Included |
| :--- | :--- | :--- |
| `ORDER_STATUS_DELAY` | ~24 | Routine tracking, multi-day transit delay, missed Prime promise. |
| `DELIVERY_NOT_RECEIVED` | ~20 | Scan code error, porch theft suspicion, delivered to wrong building. |
| `RETURNS_REFUNDS` | ~20 | Standard return label, missing refund after 30 days, cashback dispute. |
| `DAMAGED_DEFECTIVE_WRONG_ITEM` | ~20 | Shattered goods, wrong size sent by 3rd-party seller, manufacturer warranty past 30 days. |
| `BILLING_PAYMENT_DISPUTES` | ~22 | Double debit, fraudulent charges, bank dispute threats, unannounced renewal. |
| `PRIME_SUBSCRIPTION_BENEFITS` | ~16 | Streaming inclusion, geo-licensing restrictions, multi-device Echo music. |
| `ACCOUNT_ACCESS_SECURITY` | ~18 | 2FA phone loss, locked account, suspicious OTP alerts, account takeover. |
| `CARRIER_PHYSICAL_DELIVERY_ISSUE` | ~16 | Driver property damage, gate left open/dog escape, package thrown in rain. |
| `ORDER_CANCELLATION_MODIFICATION` | ~14 | Pre-dispatch cancellation, in-transit cancellation attempts, address corrections. |
| `CUSTOMER_SERVICE_ESCALATION` | ~16 | Demands for supervisor, rude phone rep complaints, regulatory/BBB threats. |
| `PRODUCT_INQUIRY_AVAILABILITY` | ~12 | Out-of-stock restock inquiries, hardware compatibility, regional availability. |
| `OTHER_UNCLEAR` | ~12 | Social media casual chitchat, unanswerable fragments, Seller Central requests. |

---

## 3. Data Schema (`golden_examples.jsonl`)

Each line is a JSON object with the following schema:

```json
{
  "id": "GOLDEN_001",
  "message_text": "Cleaned customer query text without @mentions",
  "thread_context": "None (opening turn) OR prior chronological turns: [Customer]: ... | [AmazonHelp]: ...",
  "turn_index": 1,
  "true_intent": "order_status_delay",
  "true_escalate": false,
  "escalate_reason": "Clear tracking request with no safety risk; resolvable autonomously with tracking link and EDD guidance.",
  "difficulty": "routine",
  "review_status": "manually_verified",
  "reference_brand_reply": "Original AmazonHelp response from twcs dataset used for qualitative benchmark comparison."
}
```

---

## 4. LLM-Assisted Labeling Prompt & Review Protocol

### First-Pass Labeling System Prompt
```text
You are an expert annotator for an enterprise customer support evaluation benchmark.
You will be provided with a customer message, its preceding thread context, and the real historical brand reply.
Classify the interaction against the 12-class taxonomy:
[order_status_delay, delivery_not_received, returns_refunds, damaged_defective_wrong_item,
billing_payment_disputes, prime_subscription_benefits, account_access_security,
carrier_physical_delivery_issue, order_cancellation_modification, customer_service_escalation,
product_inquiry_availability, other_unclear]

Rules:
1. Multi-issue priority: Assign the primary intent causing the contact. If the customer reports a damaged item and asks to return it, assign damaged_defective_wrong_item as the root cause.
2. Escalation decision:
   - true if intent is billing_payment_disputes, account_access_security, or customer_service_escalation (per policy).
   - true if query involves property damage, safety incidents, or legal threats.
   - false for standard tracking, return instructions, Prime specs, or clear autonomous requests.
3. Provide a crisp, defensible escalate_reason.
4. Output valid JSON matching the schema.
```

### Human Review & Quality Assurance Protocol
All records in the initial pass are tagged with `"review_status": "auto_labeled_pending_review"`.

To convert this into a verified ground truth benchmark:
1. **100% Mandatory Review** of High-Risk & Edge Categories:
   - All `BILLING_PAYMENT_DISPUTES` (~22)
   - All `ACCOUNT_ACCESS_SECURITY` (~18)
   - All `CARRIER_PHYSICAL_DELIVERY_ISSUE` (~16)
   - All `CUSTOMER_SERVICE_ESCALATION` (~16)
   - All `ambiguous` and `multi_issue` difficulty tiers (~40)
2. **20% Random Audit** of Routine Categories (~16 examples).
3. **Total Minimum Human Review Target**: **~85 to 110 examples** (42% to 55% of the total dataset).
4. Once verified by a human engineer, the field is updated to `"review_status": "manually_verified"`.
