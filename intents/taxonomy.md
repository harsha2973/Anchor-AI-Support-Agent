# Customer Support Intent Taxonomy (AmazonHelp Benchmark)

Based on empirical analysis of **168,065 customer<->brand dialogue turns** from the Kaggle Customer Support on Twitter dataset (`@AmazonHelp`), this taxonomy defines **11 specific domain intents** plus **1 catch-all category** (`OTHER_UNCLEAR`).

---

## Taxonomy Overview

| Intent Code | Category Name | Core Customer Problem | Escalation Risk |
| :--- | :--- | :--- | :--- |
| `ORDER_STATUS_DELAY` | Order Tracking & In-Transit Delays | Where is my order? Delivery is late/delayed. | Medium |
| `DELIVERY_NOT_RECEIVED` | Marked Delivered But Not Received | Tracking says delivered, but package is missing. | High |
| `RETURNS_REFUNDS` | Returns, Exchanges & Refund Processing | How to return item, get return label, missing refund. | Medium |
| `DAMAGED_DEFECTIVE_WRONG_ITEM` | Damaged, Defective or Wrong Item | Item arrived broken, wrong size/product received. | Medium |
| `BILLING_PAYMENT_DISPUTES` | Unauthorized Charges & Billing Issues | Double charge, unrecognized transaction, refund dispute. | Critical |
| `PRIME_SUBSCRIPTION_BENEFITS` | Prime Membership & Digital Services | Prime Video catalog, Music Unlimited plans, Prime perks. | Low |
| `ACCOUNT_ACCESS_SECURITY` | Account Lockout & Security Issues | Locked out, 2FA/OTP problems, suspicious login. | Critical |
| `CARRIER_PHYSICAL_DELIVERY_ISSUE` | Driver Conduct & Delivery Incidents | Reckless driving, property damage, threw package. | High |
| `ORDER_CANCELLATION_MODIFICATION` | Cancellation & Address Changes | Cancel order before shipment, modify address. | Medium |
| `CUSTOMER_SERVICE_ESCALATION` | Service Dissatisfaction & Supervisor Demands | Terrible service, rude agent, demands human manager. | High |
| `PRODUCT_INQUIRY_AVAILABILITY` | Stock Availability & Product Compatibility | When will it restock? Compatibility with device/region. | Low |
| `OTHER_UNCLEAR` | Catch-All: Chitchat, Fragments, Spam & Seller | Social pleasantries, vague query, Seller Central issue. | Low |

---

## Detailed Intent Specifications

### 1. `ORDER_STATUS_DELAY`
- **Definition**: Customer inquiring about the current transit status, tracking number, or estimated arrival date of an active order, or complaining about delays past the promised delivery window.
- **Real Examples from Data**:
  1. *"I have a package from due to be delivered today, last tracking update is 'arrived at Norwich Mail Centre...'"*
  2. *"if I’m paying for one day shipping why is estimated delivery not for three days?"*
  3. *"Still not get product order 405-4841727-8482714, which used to deliver on 9th. Nor update the delivery time."*
  4. *"wrst service by AmazonIN. Ordr 402-0028870-5331508 not delivered & Ordr 408-2570476-7639509 shows it will be delayed"*
- **What a "Correctly Resolved" Reply Contains**:
  - Inquires about or verifies the original Estimated Delivery Date (EDD) provided on order confirmation.
  - Guides the customer to real-time carrier tracking via `Your Orders` portal.
  - Explains the distinction between shipping dispatch time vs. transit time.
  - Warns the customer not to post private order numbers or PII on public social channels.

---

### 2. `DELIVERY_NOT_RECEIVED`
- **Definition**: Carrier tracking indicates the package was delivered, but the customer cannot locate it on their porch, mailbox, lobby, or suspects theft/premature scanning.
- **Real Examples from Data**:
  1. *"how am I a prime member but I can't have any information on a package I paid for that says delivered and I never received"*
  2. *"406-2385037-2501155 showing they delivered to customer yesterday. But I am the customer and didn't got delivery"*
  3. *"Delivery standards falling. Already delayed by 2 days even after purchasing Prime today order marked delivered but haven't rcvd it"*
  4. *"Tracking says delivered on my porch 2 hours ago, but nothing is here."*
- **What a "Correctly Resolved" Reply Contains**:
  - Clarifies that delivery carriers occasionally scan packages prematurely as "delivered" up to 24-36 hours before physical arrival ("scan code error").
  - Suggests checking common safe drop areas (porch, garage, side door, mailroom) and verifying with neighbors/household members.
  - Directs customer to the `Find a Missing Package` help page.
  - Provides a direct link to initiate a missing package investigation or replacement if 36 hours have passed.

---

### 3. `RETURNS_REFUNDS`
- **Definition**: Inquiries regarding how to return an item, obtain return shipping labels, check exchange eligibility, or resolve delays in receiving an issued refund or cashback.
- **Real Examples from Data**:
  1. *"could you help me with a return ?"*
  2. *"till now my money is not refunded it’s been a month. Unable to contact customer care. Order no-408-9697378-6505948"*
  3. *"Still not received my Cashback after 2 months. Contacted 13 times to customer support."*
  4. *"Highly disappointed with ur services, a product bought can not be claimed for a refund but only we can get it replaced wd d same"*
- **What a "Correctly Resolved" Reply Contains**:
  - Outlines the 30-day return window and directs to the Online Returns Center (`amazon.com/returns`).
  - Differentiates return processing for items sold by Amazon vs. third-party Marketplace sellers.
  - States standard refund timelines (typically 3–5 business days after carrier drop-off).
  - Offers a secure routing form if the refund window has elapsed without credit.

---

### 4. `DAMAGED_DEFECTIVE_WRONG_ITEM`
- **Definition**: Inbound issue stating that the product arrived physically broken, shattered, missing components, non-functional, or is the incorrect item/size compared to the purchase invoice.
- **Real Examples from Data**:
  1. *"Three time I order in last 15days for size 10 but every time seller is giving 9 size. Poor experience (1/5) 404-8216469-8917902"*
  2. *"The real battery capacity of Sony xz1 is 2700mah,but amazon fooled me into buying this as i saw it has 3430mah battery. Order id:407-3621368-7052332"*
  3. *"Before 2 month I have purchased 32 inches TCL led tv And now there is a problem in that tv And TCL is not helping me"*
  4. *"where is the need?? this was a bulk order of cotton wool, hardly at risk of being damaged.. packaging was completely torn"*
- **What a "Correctly Resolved" Reply Contains**:
  - Expresses immediate empathy for the compromised condition of the item.
  - Directs customer to select "Defective/Damaged" or "Wrong item received" in the Returns Center to generate a free return label or immediate replacement dispatch.
  - Clarifies manufacturer warranty policies for items beyond the standard 30-day return window.
  - Provides a link for packaging feedback (`amazon.com/packaging`).

---

### 5. `BILLING_PAYMENT_DISPUTES`
- **Definition**: Customer complaining of double charges, unexpected automated subscription debits, payment processing errors, card declines, or demanding an immediate charge dispute.
- **Real Examples from Data**:
  1. *"said they're not going to reimburse me my package I never received DISPUTE THE WHOLE CHARGE PLEASE !!!!"*
  2. *"cheers for automatically taking money from my bank account without any emails to confirm that I want a prime membership, this money was set aside for my energy bill"*
  3. *"Hey, you’ve charged me twice for a purchase I’ve made, can I DM you a screenshot?"*
  4. *"not only do I get charged 3 times for an order, I now get informed it’s going to be late getting delivered"*
- **What a "Correctly Resolved" Reply Contains**:
  - **Mandatory Safety Rule**: Reminds customer never to post card digits, billing receipts, or personal credentials publicly.
  - Explains the difference between temporary bank pre-authorization holds vs. final settled debits.
  - Informs customer of auto-refund policies for unused Prime memberships.
  - Provides direct secure authentication link to connect with an account billing specialist.

---

### 6. `PRIME_SUBSCRIPTION_BENEFITS`
- **Definition**: General inquiries regarding Amazon Prime membership perks, plan pricing, regional streaming availability on Prime Video, or Amazon Music Unlimited device limits.
- **Real Examples from Data**:
  1. *"do u i need to pay extra amount to watch movies are annual subscription is enough"*
  2. *"why does prime member ship in india not give access to unlimited cloud storage ? And music"*
  3. *"I guess if I own two Echo devices I can stream different music to each device simultaneously with Music Unlimited?"*
  4. *"Will Arjun Reddy be available for users in America with Amazon Prime?"*
- **What a "Correctly Resolved" Reply Contains**:
  - Details specific subscription inclusions (e.g. Prime Video included with Prime; Amazon Music Unlimited Family Plan needed for simultaneous multi-device streams).
  - Clarifies geographical/regional licensing constraints for media catalogs.
  - Directs to the Prime Management hub (`amazon.com/mc`).

---

### 7. `ACCOUNT_ACCESS_SECURITY`
- **Definition**: Issues logging in, account lockouts, failure to receive 2FA/OTP codes, or alerts regarding compromised passwords or unauthorized login attempts.
- **Real Examples from Data**:
  1. *"Got multiple password assistance and OTP messages on mail. What is going on??"*
  2. *"please help. I can't get into the live chat or any contact us area as your site has locked my account. Who do I contact in UK?"*
  3. *"Change your password for Amazon constantly!! If you don't and you get hacked is useless. Your account will be locked And be asked to pay for fraudulent charges"*
  4. *"accepts my TFA code to sign in but not to access the 'Login & Security' page of my account... can't disable TFA"*
- **What a "Correctly Resolved" Reply Contains**:
  - Strongly cautions against sharing email addresses, phone numbers, or passwords on public forums.
  - Instructs the user to initiate a password reset or security review if unauthorized access is suspected.
  - Directs user to the Two-Step Verification Account Recovery pathway (which allows identity verification without signing in).
  - Escalates immediately to the Account Security team.

---

### 8. `CARRIER_PHYSICAL_DELIVERY_ISSUE`
- **Definition**: Complaints regarding physical driver behavior, damaged property (gates, lawns, vehicles), unsafe parcel placement, pet-related incidents, or failure to follow delivery notes.
- **Real Examples from Data**:
  1. *"your delivery guy in Lincoln park NJ took my friends puppy. Need help now!!! Police next call... Thank you!"*
  2. *"When the driver leaves the gate open and let's the dog out #noiwonttakeyourparcel"*
  3. *"Ring Door bell - some Amazon delivery guy deliver parcel like this: [video link]"*
  4. *"didn’t get a call from delivery guy & I see a message that the delivery guy was not able to contact me. Order: 402-4119868-0140340"*
- **What a "Correctly Resolved" Reply Contains**:
  - Treats delivery incidents with urgency and formal corporate apology.
  - Identifies the delivery carrier (Amazon Logistics vs. USPS/UPS/courier).
  - Guides customer to update "Delivery Instructions" and designate a verified "Safe Place" on their account profile.
  - Routes the report to carrier operations for internal driver investigation.

---

### 9. `ORDER_CANCELLATION_MODIFICATION`
- **Definition**: Requests to cancel an order prior to shipment, alter order quantity/size, or update shipping addresses for pending shipments.
- **Real Examples from Data**:
  1. *"I am not able to cancel this order. Do help me out with this."*
  2. *"how i cancel my order and amazon still snatch the money out my acc"*
  3. *"I need to cancel my order #55443 immediately before it ships out tomorrow"*
  4. *"I placed an order by mistake, can I change the delivery address?"*
- **What a "Correctly Resolved" Reply Contains**:
  - Explains that cancellation is only possible before the order enters dispatch (`Your Orders` -> `Cancel Items`).
  - Clarifies that once marked "Shipped", the order cannot be intercepted in transit; the customer must refuse delivery or return upon arrival.
  - Clarifies that funds for canceled orders are released as pending authorizations per bank timelines.

---

### 10. `CUSTOMER_SERVICE_ESCALATION`
- **Definition**: High-friction customer interactions involving extreme frustration with previous support agents, broken promises, rude representative conduct, or demands for immediate supervisory intervention.
- **Real Examples from Data**:
  1. *"(2) worst customer service, evn after 15 days, no resolution. my loyalty shattered. i demand attention to my cause."*
  2. *"Dissapointed with terrible service from . Not like them. Let down big time."*
  3. *"what a joke. The most #unhelpful #rude #unprofessional #uncommunicative team I’ve come across. Who taught you #CustomerService?"*
  4. *"Today I sent an email and was offered 1 month of prime after explaining how much of a hassle this has been, replied asking for somebody who could actually do something to get a hold of me and they told me to call in.... I want to talk to somebody that can actually correct this."*
- **What a "Correctly Resolved" Reply Contains**:
  - Validates customer frustration with respectful de-escalation language and avoids defensive argument.
  - Refrains from requesting or displaying customer order numbers publicly.
  - Provides a high-priority contact link (`amazon.com/gp/help/contact-us`) for phone callback or supervisory review.
  - Reassures customer that agent conduct feedback is logged internally.

---

### 11. `PRODUCT_INQUIRY_AVAILABILITY`
- **Definition**: Pre-purchase questions regarding stock availability, restock timing, hardware specifications, or regional/device compatibility.
- **Real Examples from Data**:
  1. *"when do y'all restock????????"*
  2. *"Anyone know does Amazon Alexa work in Ireland? I can't download the app, says not available in your country."*
  3. *"Is this phone case compatible with the iPhone 15 Pro Max specifically?"*
  4. *"what's up? Need Stupid Watergate updates. Why is only this episode of not available in my location"*
- **What a "Correctly Resolved" Reply Contains**:
  - Advises checking product detail pages or adding items to a Wishlist to receive automated in-stock alerts.
  - Clarifies geographical/hardware compatibility for Amazon devices (e.g. Echo/Alexa).
  - Encourages reaching out directly to the manufacturer or third-party seller for unlisted technical dimensions.

---

### 12. `OTHER_UNCLEAR` (Catch-All)
- **Definition**: Messages that cannot be categorized into core support intents: casual social chitchat, unintelligible fragments, spam/bot mentions, third-party seller central queries, or affiliate program questions.
- **Real Examples from Data**:
  1. *"So, Couch, Tässchen Blasen- und Nierentee und TAAHM in Dauerschleife auf Amazon Prime Video. #Sonntag"* (Social chitchat)
  2. *"Hey do you still monitor the Amazon Cares twitter account?"* (Meta-query)
  3. *"team im facing issue on my amazon seller account ?? Can u help i already droped mail also still didn't received any response.."* (Seller inquiry)
  4. *"Link???"* (Ambiguous fragment)
- **What a "Correctly Resolved" Reply Contains**:
  - For social chitchat: Friendly, lighthearted brand engagement.
  - For vague fragments: Politely requests clarification and issue description without sharing private data.
  - For Seller/Merchant inquiries: Clarifies that customer care only handles consumer orders and provides the link to Amazon Seller Central (`sellercentral.amazon.com`).
