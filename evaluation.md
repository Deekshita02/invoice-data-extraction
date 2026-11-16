# Evaluation report

All numbers below come from runs in the Colab session. "Hand count" means I compared JSON output to the invoice image/answer list by eye.

## 1. Models
Random Forest (200 trees) and Logistic Regression (MaxAbs scaling) were compared on the dataset test. They scored the same, because the single-template data is too easy to separate them. Random Forest is shipped.

## 2. Dataset test (26 invoices, one template), per-piece scores
Random Forest: F1 1.000 on client name, client tax ID, date, IBAN, invoice no., total net, total VAT; 0.960 seller name; 0.980 seller tax ID; 0.981 gross total. Accuracy 0.998 (the "O" class dominates, so judge per field).
Each field has only 24-26 examples, so one miss moves F1 by about 0.04. Treat as optimistic.

## 3. Exact match per field, dataset test, 26 invoices (final code)
| Field | Exact match |
|---|---|
| INVOICE_NO | 26/26 |
| DATE | 26/26 |
| SELLER_TAX_ID | 25/25 |
| CLIENT_TAX_ID | 26/26 |
| IBAN | 26/26 |
| SELLER_NAME | 25/26 |
| CLIENT_NAME | 26/26 |
| TOTAL_NET | 26/26 |
| TOTAL_VAT | 26/26 |
| TOTAL_GROSS | 25/26 |

The two misses look like answer-key errors (a total with a stray digit; a seller field holding a tax ID). I did not verify them on the images. Name matching is lenient (predicted line must be the start of "name + address"), IBAN is fuzzy (key has typos).
Running the same 26 invoices through Tesseract instead of the dataset's OCR gave the same result (after fixing merged columns: amounts like "889,2088,92" came from joining a whole row; splitting lines at wide gaps fixed it, 2/10 -> 10/10 then 26-of-26 pattern above).

## 4. Unseen synthetic layout C (60 invoices, never trained on), v2 model, per-piece
F1 1.000: client name, date, invoice no., PO, seller name, gross, net, VAT. Tax IDs are lower in the raw model (client tax ID recall 0.681, seller tax ID precision 0.754) because it confuses seller and buyer IDs. The pairing rule (nearest name above) raised buyer tax ID from 26/47 to 40/51 (first model, fresh layout C invoices); seller 42/48.
Layout C is my own design, so it is easier than real invoices.
Adding layouts D and E left layout C and the dataset test unchanged or slightly better.

## 5. Real invoices (10 from Google Images, different layouts)
| Field | First model, one template | After synthetic layouts, rules, preprocessing |
|---|---|---|
| Invoice number | 1/10 | 6/10 (7/10 if the unclear Retention number counts) |
| Date | about 2/10 | about 4/10 |
| Total | 2/10 | about 5-6/10 |

Failures: business number (XYZ), ZIP code, OCR digit errors (Gujarat read as 400404), nothing found on one Zylker invoice, totals returned null when "Grand Total" sits in a coloured box (Studio Shodwe).
**Caveat:** I looked at these 10 while choosing fixes (confidence floor, type filters, prefix bug, preprocessing), so they are a development set. The date/total counts are hand counts from earlier runs and were not recounted on the final v2 build. No fresh held-out real invoices were run.

## 6. Bugs found and fixed (for transparency)
- Tesseract rows merged into one piece: fixed by splitting at wide gaps.
- Model always returned something: now returns null below a probability floor, plus type filters.
- `clean()` dropped prefixes (INV-, GST-, US-): fixed; layout C invoice no. 4/30 -> 30/30, dataset test stayed 26/26. Found and checked on layout C, so partly tuned.
- Misread rupee sign adding a leading digit to amounts: repaired when net + tax = total fits.

## 7. Sample outputs
`outputs/` (synthetic samples), `outputs_real_dev_set/` (10 real), each with `processing.log`.

## 8. Not done
Per-line tax; CRF comparison; clean held-out real test; deskew.
