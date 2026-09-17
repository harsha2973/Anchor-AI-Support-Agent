"""
Unit tests for the 12-class intent classification engine and safety guardrails.
"""

import pytest
from intents.classifier import CalibratedHeuristicClassifier, HybridIntentClassifier
from intents.taxonomy import IntentCategory, SUPPORT_TAXONOMY


def test_taxonomy_completeness():
    """Verify all 12 intent categories are defined with non-empty metadata."""
    assert len(SUPPORT_TAXONOMY) == 12
    for category in IntentCategory:
        assert category in SUPPORT_TAXONOMY
        definition = SUPPORT_TAXONOMY[category]
        assert definition.name
        assert definition.description
        assert len(definition.example_queries) >= 3
        assert definition.resolution_guidance


@pytest.mark.parametrize(
    "query,expected_intent",
    [
        ("Where is my package? The tracking number is TRK-88219, it's late.", IntentCategory.ORDER_STATUS_DELAY),
        ("Tracking says delivered on my porch 2 hours ago, but nothing is here.", IntentCategory.DELIVERY_NOT_RECEIVED),
        ("How do I return a dress that doesn't fit? Where is the return label?", IntentCategory.RETURNS_REFUNDS),
        ("The item arrived completely shattered and broken in the box.", IntentCategory.DAMAGED_DEFECTIVE_WRONG_ITEM),
        ("You charged me twice for the same transaction on my credit card!", IntentCategory.BILLING_PAYMENT_DISPUTES),
        ("Do I have to pay extra to watch movies on Prime Video with my membership?", IntentCategory.PRIME_SUBSCRIPTION_BENEFITS),
        ("I'm locked out of my account and I no longer have access to my 2FA phone number.", IntentCategory.ACCOUNT_ACCESS_SECURITY),
        ("The delivery guy threw the parcel over the gate and damaged my car.", IntentCategory.CARRIER_PHYSICAL_DELIVERY_ISSUE),
        ("Please cancel my order before it ships out tomorrow.", IntentCategory.ORDER_CANCELLATION_MODIFICATION),
        ("Your customer service is terrible, I want to talk to a manager or supervisor.", IntentCategory.CUSTOMER_SERVICE_ESCALATION),
        ("When do y'all restock size 10 in these shoes?", IntentCategory.PRODUCT_INQUIRY_AVAILABILITY),
    ],
)
def test_classifier_accuracy_across_core_intents(query, expected_intent):
    classifier = HybridIntentClassifier()
    prediction = classifier.classify(query)
    assert prediction.predicted_category == expected_intent
    assert 0.0 <= prediction.confidence <= 1.0


def test_critical_safety_guardrails():
    """Ensure critical fraud and security queries trigger immediate safety confidence."""
    classifier = HybridIntentClassifier()

    fraud_pred = classifier.classify("There is an unauthorized charge of $500 on my bank account, dispute the charge!")
    assert fraud_pred.predicted_category == IntentCategory.BILLING_PAYMENT_DISPUTES
    assert fraud_pred.confidence >= 0.90
    assert not fraud_pred.is_ambiguous

    security_pred = classifier.classify("My account was hacked and someone changed my password, I'm locked out!")
    assert security_pred.predicted_category == IntentCategory.ACCOUNT_ACCESS_SECURITY
    assert security_pred.confidence >= 0.90

    legal_pred = classifier.classify("I am contacting my attorney and filing a complaint with the FTC!")
    assert legal_pred.predicted_category == IntentCategory.CUSTOMER_SERVICE_ESCALATION
    assert legal_pred.confidence >= 0.90


def test_ambiguous_or_unclear_fallback():
    """Verify that vague fragments fallback to OTHER_UNCLEAR with ambiguity flag."""
    classifier = CalibratedHeuristicClassifier()
    prediction = classifier.classify("asdf jkl; 1234 random gibberish")
    assert prediction.predicted_category == IntentCategory.OTHER_UNCLEAR
    assert prediction.is_ambiguous


def test_order_status_preparing_for_shipment():
    """Verify clean order-status inquiry with preparing for shipment classifies with high confidence."""
    query = (
        "Hi, I ordered a pair of headphones 5 days ago (order #12345) and it still says "
        "'preparing for shipment.' Can you tell me when it'll actually ship?"
    )
    classifier = HybridIntentClassifier()
    prediction = classifier.classify(query)
    assert prediction.predicted_category == IntentCategory.ORDER_STATUS_DELAY
    assert prediction.confidence >= 0.85
    assert not prediction.is_ambiguous

