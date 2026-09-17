import json
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import agent

samples = [
    (
        "GOLDEN_001",
        "where is my package? Last time you deliver to my neighbors and this time it's no where. :'(",
        "None (opening turn)",
        "order_status_delay",
        False,
    ),
    (
        "GOLDEN_014",
        "Tracking number Q34005162743",
        "[Customer]: I ordered with one day delivery and still not arrived | [AmazonHelp]: Have we missed the delivery date? | [Customer]: Yes | [AmazonHelp]: Will you confirm tracking details?",
        "order_status_delay",
        False,
    ),
    (
        "GOLDEN_030",
        "22 hours since order was marked delivered & it's still not here. Where's my stuff Pathetic customer service.",
        "[Customer]: Hey when your drivers mark something delivered, they should actually deliver it first.",
        "delivery_not_received",
        False,
    ),
    (
        "GOLDEN_042",
        "Amazon can you fucking not? 'Delivered' does not mean leave it in the sodding driveway where anyone can steal it!",
        "None (opening turn)",
        "delivery_not_received",
        True,
    ),
    (
        "GOLDEN_071",
        "Amazon Marketplace seller shipped me the wrong item. Initial response to my return/refund request: I was confused or ripping them off.",
        "None (opening turn)",
        "damaged_defective_wrong_item",
        False,
    ),
    (
        "GOLDEN_088",
        "Why u guys r duping loyal customers & doing fraud by selling ipad 2017 but showing images of ipad pro?",
        "None (opening turn)",
        "billing_payment_disputes",
        True,
    ),
]

for sid, msg, ctx, exp_intent, exp_esc in samples:
    out = agent.run(msg, thread_context=ctx)
    print(f"\n=================================================================")
    print(f"CASE: {sid} | Expected: {exp_intent} (Escalate: {exp_esc})")
    print(f"Customer Query  : {msg}")
    print(f"Thread Context  : {ctx}")
    print(f"=================================================================")
    print(json.dumps(out, indent=2))
