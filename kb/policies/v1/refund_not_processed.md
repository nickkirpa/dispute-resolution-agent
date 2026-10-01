## POL-REF-01: Merchant refund not processed
Applies to: refund_not_processed
If a merchant promised a refund that has not appeared on the account, and no matching credit from that merchant exists in the ledger, the original transaction amount is refunded as a provisional credit.

## POL-REF-02: Refund already credited
Applies to: refund_not_processed
If the ledger already shows a credit from the merchant matching the refund, the dispute is rejected and the customer is shown the date the credit was posted.
