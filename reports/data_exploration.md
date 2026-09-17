# Exploratory Data Analysis: AmazonHelp Customer Support Corpus

- **Total Interaction Pairs**: 168,065
- **Unique Conversation Threads**: 114,197
- **Date Range**: 2015-06-13 to 2017-12-03

## 1. Message Length Distributions

| Metric                        | Customer Inquiry   | Brand Reply (AmazonHelp)   |
|-------------------------------|--------------------|----------------------------|
| Char Count: Mean +/- Std      | 105.2 +/- 58.0     | 115.9 +/- 46.1             |
| Char Count: Median [p25, p75] | 105 [62, 133]      | 115 [90, 127]              |
| Char Count: 95th Percentile   | 227                | 215                        |
| Word Count: Mean +/- Std      | 18.2 +/- 11.1      | 18.8 +/- 8.6               |
| Word Count: Median [p25, p75] | 18 [10, 24]        | 19 [14, 22]                |

## 2. Monthly Volume Distribution

| Month | Interaction Count |
| :--- | :--- |
| 2015-06 | 4 |
| 2015-07 | 2 |
| 2015-12 | 1 |
| 2016-01 | 1 |
| 2016-03 | 3 |
| 2016-04 | 1 |
| 2016-05 | 4 |
| 2016-06 | 1 |
| 2016-08 | 1 |
| 2016-09 | 1 |
| 2016-10 | 3 |
| 2016-11 | 9 |
| 2016-12 | 4 |
| 2017-01 | 8 |
| 2017-02 | 6 |
| 2017-03 | 3 |
| 2017-04 | 11 |
| 2017-05 | 25 |
| 2017-06 | 45 |
| 2017-07 | 50 |
| 2017-08 | 67 |
| 2017-09 | 589 |
| 2017-10 | 79,338 |
| 2017-11 | 78,104 |
| 2017-12 | 9,784 |

## 3. Frequent Opening Phrases & Top Keywords

### Top Opening Trigrams
- `where is my` (253)
- `your customer service` (234)
- `was supposed to` (229)
- `the point of` (223)
- `this is the` (202)
- `what is the` (202)
- `placed an order` (200)
- `what the point` (193)
- `thank you for` (192)
- `can you please` (184)
- `pay for prime` (172)
- `not able to` (153)
- `still waiting for` (151)
- `customer service is` (147)
- `my order is` (146)

### Top Content Bigrams (Domain Specific)
- `https co` (30,171)
- `customer service` (4,011)
- `day delivery` (2,077)
- `customer care` (1,744)
- `day shipping` (1,569)
- `next day` (1,502)
- `delivery date` (1,261)
- `prime membership` (1,176)
- `delivered today` (991)
- `order id` (892)
- `one day` (891)
- `prime member` (875)
- `still waiting` (872)
- `cancel order` (689)
- `placed order` (680)

## 4. Boilerplate & Automation Insights

- **Brand Uniqueness Ratio**: 92.19%
- **Top 10 Brand Macros Share**: 0.57% of total volume.
- **Customer Message Uniqueness**: 90.72%

## 5. Sampled Real Customer Inquiries

**Sample 01** (Turn 1)
- **Customer**: So, Couch, Tässchen Blasen- und Nierentee und TAAHM in Dauerschleife auf Amazon Prime Video. #Sonntag
- **AmazonHelp Reply**: Na, dann mal gute Unterhaltung und gute Besserung :) ^NW

**Sample 02** (Turn 1)
- **Customer**: I have never performed this amazon prime member transaction .. Already complained in the customer service but no reply https://t.co/el9v6w9Jwq
- **AmazonHelp Reply**: You can report this to their support team directly here: https://t.co/O1IhqmoJ8o (2/2) ^CB

**Sample 03** (Turn 1)
- **Customer**: Muy contenta siempre con pero hoy el mensajero me ha despierto a las 10:30 y es sábado, no trabajo y ayer salí 😭😭😩😩
- **AmazonHelp Reply**: Nos alegra que tengas tu pedido Berta pero nos disculpamos por despertarte 😯 Siempre puedes intentar dormir de nuevo 😴 ^AV https://t.co/mZONQTBlMl

**Sample 04** (Turn 5)
- **Customer**: 分かりました！ 恐らく交換と言う形になると思いますが、その際の発送料などは着払いで良いのでしょうか？ それとも出品者との間で決める形になるのでしょうか？
- **AmazonHelp Reply**: 出品者により、対応が異なりますので、送料などの詳細も出品者にご確認くださいませ。TY
- **Prior Context**: `[Customer]: Amazonの配送状況さ、昨日で配達完了になってるのに荷物ないんだけどこれは連絡した方がいいのかな？もう数日待っても来なかったらしてみようかな？ | [AmazonHelp]: 配達完了になっているにも関わらず、未着ですか？ ポストなどご確認いただき、未着時は、配送業者にご確認ください。https://t.co/xIFJYmjA1B 解決されない際は、下記よりご連絡ください。https://t.co/bwBU0NvYIn TY | [Customer]: 昨日で配達完了になってるのですが、まだ商品は未着ですね。 配送業者の方にも連絡入れてみます！ ありがとうございます！ | [AmazonHelp]: ご迷惑をおかけしております。 配送業者にご連絡後も問題が解決しない場合には、いつでも先ほどのURLよりカスタマーサービスへご連絡くださいませ。 お手数おかけしますが、よろしくお願いします。TY | [Customer]: 荷物ですが、奥の方に落ちててきちんと届いてました 自分の確認ミスでしたすみません！ それで、届いたのは良いのですが、商品が注文画面のと違い通常盤じゃない方が届いたのですがこれは出品者に連絡って形でいいのでしょうか？ | [AmazonHelp]: とんでもないことでございます。到着されてたとのことで、何よりでした。 しかしながら、ご注文商品と違う商品だったとの事ですので、出品者が発送している商品は、出品者にご連絡を行っていただければと存じます。https://t.co/ZXCXcbN3Tv TY | [Customer]: こちらこそご迷惑をおかけしました！ 早速、出品者の方に連絡を入れてみようとおもいます。 ありがとうございます。 | [AmazonHelp]: お手数おかけいたしますが、よろしくお願いします。 なお、お客様からのご連絡につきましては、2営業日以内に出品者よりご返答することとなっております。ご返答までお待ちくださいませ。TY`

**Sample 05** (Turn 1)
- **Customer**: When I pay for 1 day shipping usually means I need it ASAP. Looks like I paid for 2 day shipping which is free. Thanks
- **AmazonHelp Reply**: I'm sorry for this delay. Please reach out to us by phone or chat here: https://t.co/hApLpMlfHN ^RR

**Sample 06** (Turn 1)
- **Customer**: I recharged my jio no for the first time using amazon pay. But I received a mail saying I’ll get cashback of ₹20 instead of ₹99.
- **AmazonHelp Reply**: We have responded to your concern here: https://t.co/qglH2ZWSnN. Request you to check. ^MP

**Sample 07** (Turn 1)
- **Customer**: this ORDER # 113-4343877-9085812 not arrive to my address I called the customer service but nothing happen i hope you help me .
- **AmazonHelp Reply**: Sorry for the problems w/ your order, Abdullah! What does the tracking info state for your order here: https://t.co/Y5jpI9gRhE ^JZ

**Sample 08** (Turn 1)
- **Customer**: ¿Cuándo va a salir la app para Xbox One?
- **AmazonHelp Reply**: Hola, de momento no tenemos esa información, nuestras novedades son anunciadas en nuestro sitio web, sugiero visitarlo.^AZ

**Sample 09** (Turn 1)
- **Customer**: I guess if I own two Echo devices I can stream different music to each device simultaneously with Music Unlimited?
- **AmazonHelp Reply**: Hi, Andy! You can stream different music to different devices with Amazon Music Unlimited (AMU) as long as you have the AMU Family Plan, which allows streaming on up to 6 devices at once. More info here: https://t.co/spRjSPxIZJ I hope this helps! ^LB

**Sample 10** (Turn 1)
- **Customer**: product nahi mila, lekin delivered sms aya 😣 #IndiaPost https://t.co/cBLp2rd0TV
- **AmazonHelp Reply**: we'll be happy to help. 3/3 ^SH

**Sample 11** (Turn 1)
- **Customer**: Will Arjun Reddy be available for users in America with Amazon Prime?
- **AmazonHelp Reply**: I'm sorry, the selection of movies and TV shows may vary based on your location. Due to geographical limitations, certain ^AK 1/2

**Sample 12** (Turn 2)
- **Customer**: sem falar que tive um problema no site que me fez perder uma promoção relâmpago e eles deram um jeito nisso pra mim! &lt;3
- **AmazonHelp Reply**: Servimos bem, para servir sempre. 💕😄 ^VL
- **Prior Context**: `[Customer]: já declarei aqui meu amor pela mas os caras sempre se superam. fiz uma compra na black friday e entregaram hoje, NUM DOMINGO`

**Sample 13** (Turn 1)
- **Customer**: Can we expect homeland series soon ?
- **AmazonHelp Reply**: New content is constantly being added, Kinshuk. We might see this one coming our way soon too. Stay tuned :) ^JC

**Sample 14** (Turn 1)
- **Customer**: your delivery guy in Lincoln park NJ took my friends puppy. Need help now!!! Police next call... Thank you!
- **AmazonHelp Reply**: We'd like to have a specialist look into this. Please fill out detailed information here:https://t.co/ApFOFi4o5E ^SK

**Sample 15** (Turn 1)
- **Customer**: got a request I want these 3 and Thomas all session available in both English &amp; Hindi please it's 4 the inner kid in meplease https://t.co/JiiCVLcH2j
- **AmazonHelp Reply**: movies and TV shows we have available. Further, I'll surely forward this as feedback internally. (2/2)^SQ

**Sample 16** (Turn 2)
- **Customer**: ah yeah went through the process further and one of my items wasn't available for tomorrow! Thank you :)
- **AmazonHelp Reply**: Sure thing! 😊 ^LB
- **Prior Context**: `[Customer]: i want to get a delivery on prime tomorrow to a locker thats only open 11-5, is it guaranteed delivery still? | [AmazonHelp]: Are you able to get a delivery date for that locker for tomorrow? ^MC`

**Sample 17** (Turn 1)
- **Customer**: till now my money is not refunded it’s been a month. Unable to contact customer care. Order no-408-9697378-6505948
- **AmazonHelp Reply**: Please report this to our team here: https://t.co/TdDksLo6Mf and we'll look into this. Use the chat/email option. 2/2 ^HD

**Sample 18** (Turn 6)
- **Customer**: ur job is only to cash monies from ppl by putting up sales,u've no value of a customer and their experience that's it
- **AmazonHelp Reply**: We value your time. Our team is working on the pick up. It will be done at the earliest. ^CB
- **Prior Context**: `[AmazonHelp]: Also, please don’t provide your order/account details on public platform as we consider them to be personal information. ^HK | [Customer]: Again I have got a msg that my pickup has been cancelled coz I have told them to cancel this is ridiculous | [AmazonHelp]: Please share your details in the link given earlier and we'll investigate this issue. Appreciate your understanding. ^ZH | [Customer]: I have shared the details on the link check now | [AmazonHelp]: We've received your details and we'll reach out to you soon. ^GK | [Customer]: Courier guy came &amp; told me that have to check the mei no &amp;again I told him that customer care ppl told me to show invoice nw he is checking | [AmazonHelp]: We are still working on your pick up. Kindly wait and we will get back to you with an update at the earliest. ^HR | [Customer]: I want the pickup to be done today I can't wait for Tom u have the invoice with u with IMEI no so show them | [AmazonHelp]: I understand your concern, we're right on it! Request you to kindly wait. ^AG | [Customer]: What's the update? It's 7:35 and still pick up has not been made.. do I need to follow up with u guys? This time I am really pissed off`

**Sample 19** (Turn 1)
- **Customer**: do u i need to pay extra amount to watch movies are annual subscription is enough
- **AmazonHelp Reply**: Are you referring to watching Prime videos under Prime subscription ? ^MK

**Sample 20** (Turn 1)
- **Customer**: your performance over the Diwali sale had been great to worse! I waited for a week for my order and had to cancel it finally..
- **AmazonHelp Reply**: Write to us here: https://t.co/6sMH3R0veK &amp; we'll take necessary actions to make things right. 2/2 ^OS

**Sample 21** (Turn 1)
- **Customer**: Wat do u mean where did we go wrong. You provide such a pathetic service and can't u read my tweets? Read my tweets https://t.co/mJbVsQkx3A
- **AmazonHelp Reply**: I beg your pardon. We're following up on your concern. We'll update you soon. ^SG

**Sample 22** (Turn 1)
- **Customer**: hope u aware that shinchan is very famous in Tamilnadu for past few days. Most of us just started watching. Suddenly today, it says shinchan season 01 is not available for my area. Season 02 &amp; 03 are available. Season 01 was so fun &amp; suddenly u removed season 01. https://t.co/vhR00oo4Qw
- **AmazonHelp Reply**: I'm sorry for the trouble caused over this streaming. Please connect with our support team here: https://t.co/2t6DQoUmNZ &amp; they will assist you with an alternative over it. ^RW

**Sample 23** (Turn 1)
- **Customer**: Link???
- **AmazonHelp Reply**: Please understand we've forwarded a request to our concerned team internally for the winner's list. Appreciate your patience. ^KA

**Sample 24** (Turn 3)
- **Customer**: Raised the ticket yesterday,,almost more than 12 hours..no response
- **AmazonHelp Reply**: You could check for the same here https://t.co/8DAc119Io4. (2/2) ^VM
- **Prior Context**: `[Customer]: #AmazonIndia pathetic delivery services, ORDER # 408-8706809-0683542 No response , | [AmazonHelp]: Sorry for the inconvenience, May I know if you contacted our support team here: https://t.co/vlvfJr4nN9? (1/2)^BS | [Customer]: Already Contacted your support team,,but no pertinent solution, being a prime member, useless. | [AmazonHelp]: We would like to help you. Please share your details with us here: https://t.co/GIJyeYqKE0. We'll get in touch shortly. ^CB`

**Sample 25** (Turn 1)
- **Customer**: your customer service sucks!!!
- **AmazonHelp Reply**: I'm so sorry, Erin! Without sharing personal account information, could you please explain further what happened? ^LS

**Sample 26** (Turn 1)
- **Customer**: got query regarding Alexa Please reply to my DM Thank You !
- **AmazonHelp Reply**: Thanks! We'll get back to you via DM. ^YP

**Sample 27** (Turn 1)
- **Customer**: Where can I forward emails from sellers that ask me to review products and give 5 star reviews, plus a free product?
- **AmazonHelp Reply**: Thanks for your post, Trevor. We can take your information via the link and get this escalated. https://t.co/hApLpMlfHN ^EM

**Sample 28** (Turn 1)
- **Customer**: could you help me with a return ?
- **AmazonHelp Reply**: For sure! Was this an item shipped from and sold by Amazon or by a third-party seller: https://t.co/Y5jpI9gRhE? ^ST

**Sample 29** (Turn 1)
- **Customer**: Mario game was at Concord, NC yesterday. I live a mile from there. U say it will arrive by 8pm today. USPS says Mon. Not acceptable.
- **AmazonHelp Reply**: You might be able to contact your local USPS and arrange for a pick-up. ^AF

**Sample 30** (Turn 1)
- **Customer**: どうして電子書籍やデジタル音楽など国際クレジットカードでも買えないですか?（日本語がちょっとへた、すみません）
- **AmazonHelp Reply**: 恐れ入ります。https://t.co/st4oU5QbhPで販売されているKindle本の多くやAmazon Musicストアの楽曲は、販売権の地域の制約等の理由により、日本国内のお客様のみがご利用になれます。https://t.co/YGHehS9Onm / https://t.co/Pw56cCjUhN ご希望にそえず申し訳ありません。RI

**Sample 31** (Turn 2)
- **Customer**: DMきて
- **AmazonHelp Reply**: 恐れ入りますが、DMでのご案内をご希望の場合はお客様より本アカウント宛にDMをお送りください。なお、ご不明点はカスタマサービスでもご案内が可能です。 カスタマサービスへのお問い合わせ方法はこちら⇒https://t.co/Bh1BJr2z5A SM
- **Prior Context**: `[Customer]: だれかAmazonの使い方教えて | [AmazonHelp]: リプライ失礼いたします、Amazon公式です。何か当サイトのご利用のご不明な点などございましたでしょうか。注文方法については、ヘルプページでもご案内しておりましたので、ご参照ください。https://t.co/kNcoM8S2lW HM`

**Sample 32** (Turn 2)
- **Customer**: Thanks. Looks like someone's abusing your affiliate program. Worth looking into 'cos it hurts your brand image, not theirs.
- **AmazonHelp Reply**: I've noted your comments and have forwarded your feedback internally so that they are aware of this issue. ^SM
- **Prior Context**: `[Customer]: Hey , can you please confirm that this Whatsapp forward (link: https://t.co/R5MM5Omn5M) is spam? Too many people falling for this. https://t.co/AhPYrTFLl8 | [AmazonHelp]: This is a fake offer, Dhruv. Please refer to Amazon.in for genuine offers. ^MK`

**Sample 33** (Turn 2)
- **Customer**: Call me option is not working. PL call back
- **AmazonHelp Reply**: https://t.co/vlvfJr4nN9 &gt; Login to your Amazon.in account &gt; Under 'Tell us more about your issue', select 2/3 ^KA
- **Prior Context**: `[Customer]: problem with your Subscribe &amp; Save. Call back and resolve. | [AmazonHelp]: click here: https://t.co/HFwmNgMbDZ after selecting the appropriate issue, click on 'call me' option and a 2/3 ^SH`

**Sample 34** (Turn 5)
- **Customer**: came bck from wrk only bcos cust exec assured tht delivery wd cm before 5 PM.looks like delivery agents are boss here.such a pity!
- **AmazonHelp Reply**: We haven't received your details yet. Kindly drop in your details in the link provided earlier. We'll look into it for you.^RB
- **Prior Context**: `[Customer]: horrifying customer service this highly arrogant portal gives.On selecting call me,I am made to listen a an IVR | [AmazonHelp]: We're sorry about the stretch with your order. Were you not connected to one of our representative after the IVR on call? ^HK | [Customer]: no.IVR just parrots where my order is and runs pre-recorded options (no option to talk to exec) and recommends me to go back to the website | [AmazonHelp]: Apologies for the trouble. I'd like to help. Please share your details here: https://t.co/JzRdscmdH9 ^AK | [Customer]: reduce the amt of time a customer needs to seek 'amazon help (my foot)' and fill the form again and again. | [AmazonHelp]: This being a social platform we would not be able to access your details here. If you've already shared your details (1/2) ^GS | [Customer]: i hve shared update many times.amazon as a brand needs to store it.u hve no business wasting my time | [AmazonHelp]: I'll be sure to pass your comments as feedback to our concerned team for review. ^GS`

**Sample 35** (Turn 1)
- **Customer**: Hi I frustrated with Amazon Customer care services , Amazon has no Business Ethics &amp; values "I Placed order on 17th October which is expected to deliver by 21st October" cz of late delivery I lost offer who will bare the cost now ???
- **AmazonHelp Reply**: I'm sorry for the unpleasant experience you have had with this order. Call us here: https://t.co/vlvfJr4nN9 &amp; we'll help you with the available options. ^MJ

**Sample 36** (Turn 1)
- **Customer**: Please upload jaya janaki nayaka full movie bro
- **AmazonHelp Reply**: We will continue to add new content. Stay tuned for updates! ^EM

**Sample 37** (Turn 1)
- **Customer**: The real battery capacity of Sony xz1 is 2700mah,but amazon fooled me into buying this as i saw it has 3430mah battery. Order id:407-3621368-7052332 https://t.co/jya1lfpI88
- **AmazonHelp Reply**: I'm sorry about that raghavendra; please fill this form: https://t.co/beaaDm0muc and I’ll contact you soon. Please don't provide your order details, we consider it to be personal information. Our page is visible to the public. ^SP

**Sample 38** (Turn 2)
- **Customer**: Yeah, and sometimes the customer chooses 2-day shipping and your asses arbitrarily change that to FIVE DAY shipping
- **AmazonHelp Reply**: What was the estimated delivery date given to you for this order? Can you tell me the current status that shows here: https://t.co/Y5jpI9gRhE and who the carrier is? Keep us posted! ^FR
- **Prior Context**: `[Customer]: Still amazed that cust svc told me yesterday "see if item arrives tomorrow if not call back/cancel the order." She says go buy it elsewhere. My time is not valuable &amp; I should buy elsewhere? Yeah I didn't need that monior, and can wait forever. Corrrect ? | [AmazonHelp]: Unforeseen delays can occur. In most cases, your original order will arrive sooner than any other option we might have. We wouldn't want to burden you with the hassle of returning an extra item, if a replacement and the original arrives. ^AF`

**Sample 39** (Turn 1)
- **Customer**: Hey do you still monitor the Amazon Cares twitter account?
- **AmazonHelp Reply**: Hello Bruce - We're always happy to help. What can we do for you? ^MB

**Sample 40** (Turn 1)
- **Customer**: whats the benefit of online shopping when direct customer to service Center for product delivered within 48hrs.pathetic service
- **AmazonHelp Reply**: I'm sorry about the situation. Kindly share your details here: https://t.co/GIJyeYqKE0 and I'll look into this. ^SG

