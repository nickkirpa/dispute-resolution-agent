# Northwind Bank: account and card servicing (fictional). Not dispute rules; included in v2 for realism.

## POL-FEE-01: Monthly account fee
Applies to: fees
Standard accounts pay a monthly fee of 3 EUR, waived when at least 500 EUR is deposited in the calendar month.

## POL-FEE-02: Foreign transaction fee
Applies to: fees
Card payments in a currency other than EUR carry a 1.5% foreign transaction fee, shown as a separate line on the statement.

## POL-FEE-03: ATM withdrawal fees
Applies to: fees, atm
The first five ATM withdrawals per month are free. Further withdrawals cost 1 EUR each. Operator surcharges are shown on the ATM screen and are not refunded.

## POL-FEE-04: Fee refunds
Applies to: fees
A fee charged in error is refunded automatically once confirmed. Correctly applied fees are not refunded as goodwill more than once per year.

## POL-ATM-01: Cash not dispensed
Applies to: atm
If an ATM debits the account but does not dispense cash, the customer should report it within 30 days; the ATM operator's reconciliation is requested and the amount is credited if the operator confirms the error.

## POL-ATM-02: Card retained by ATM
Applies to: atm, card_management
If an ATM keeps the card, the card is blocked immediately and a replacement is sent free of charge.

## POL-FX-01: Exchange rates
Applies to: fx
Card payments in foreign currencies are converted at the card network rate on the day the transaction settles, which can differ from the rate on the day of purchase.

## POL-FX-02: Dynamic currency conversion
Applies to: fx
If the customer chose to pay in EUR at a foreign merchant (dynamic currency conversion), the merchant set the rate; differences are not disputable with the bank.

## POL-CARD-01: Lost or stolen card
Applies to: card_management
A lost or stolen card must be frozen in the app or by phone. Frozen cards can be unfrozen within 7 days; after that a replacement is issued.

## POL-CARD-02: Card replacement fee
Applies to: card_management, fees
The first replacement card per year is free. Further replacements cost 5 EUR, except replacements after fraud, which are always free.

## POL-CARD-03: Contactless limits
Applies to: card_management
Single contactless payments are limited to 50 EUR. After cumulative contactless spending of 150 EUR, the next payment requires the PIN.

## POL-CARD-04: Card delivery times
Applies to: card_management
New and replacement cards arrive within 5 business days. Express delivery within 2 business days costs 15 EUR.

## POL-ACC-01: Account closure
Applies to: account
Customers can close their account at any time. Pending card transactions must settle first; the remaining balance is transferred to an account of the customer's choice.

## POL-ACC-02: Dormant accounts
Applies to: account
Accounts with no activity for 24 months are marked dormant. Dormant accounts are not charged the monthly fee.

## POL-ACC-03: Statement corrections
Applies to: account
Customers who believe their statement is wrong for reasons other than a card dispute (for example a missing salary payment) are directed to the payments team, not the dispute team.

## POL-TRF-01: Transfer not received by recipient
Applies to: transfers
Bank transfers usually arrive within one business day. If a recipient has not received a transfer after 3 business days, the bank traces the payment with the receiving bank.

## POL-TRF-02: Transfer sent to the wrong account
Applies to: transfers
If the customer sent money to the wrong account, the bank asks the receiving bank to return it. A return cannot be guaranteed because the recipient must agree.

## POL-PEND-01: Pending transactions
Applies to: pending
Card payments can show as pending for up to 7 days. A pending payment that is cancelled by the merchant disappears without a refund line; it cannot be disputed while pending.

## POL-PEND-02: Hotel and car rental pre-authorisations
Applies to: pending
Hotels and car rental companies may block an amount on the card as a pre-authorisation. The block is released by the merchant, usually within 10 days, and is not a charge.

## POL-CB-01: Chargeback process timeline
Applies to: chargeback_process
After a provisional credit, the bank files a chargeback with the card network. The merchant has up to 45 days to respond. If the merchant proves the charge was valid, the provisional credit is reversed with 14 days' notice.

## POL-CB-02: Merchant evidence
Applies to: chargeback_process
If the merchant provides proof of delivery or proof that the customer authorised the payment, the dispute officer reviews it before the provisional credit is reversed.

## POL-SEC-01: Phishing and authorised push payments
Applies to: security
Payments the customer authorised themselves after being tricked (for example a fake delivery message) are handled by the fraud team under the scam reimbursement policy, not as card disputes.

## POL-SEC-02: Card details shared with a merchant
Applies to: security
If the customer shared card details with a merchant, later charges by that merchant are treated as authorised unless the customer cancelled the agreement in writing.

## POL-SUB-01: Cancelling subscriptions
Applies to: subscriptions
Subscriptions must be cancelled with the merchant. The bank can block future payments to a merchant on request, but cannot cancel the subscription itself.
