## POL-DUP-01: Duplicate charge refunds
Applies to: duplicate_charge
When two or more identical charges from the same merchant for the same amount are posted within 3 days, all but one charge are refunded to the customer as a provisional credit. The customer does not need to contact the merchant first.

## POL-DUP-02: Recurring payments are not duplicates
Applies to: duplicate_charge
Charges from the same merchant for the same amount that are more than 3 days apart are treated as separate (for example recurring subscription) payments, not duplicates. The agent should reject the duplicate claim and explain that subscription cancellations must be made with the merchant.
