# Invoice Data Extraction with Traditional ML (OCR + Random Forest)

Assessment project for Sarvatr TradSys LLP (AI/ML Intern). Author: Pilaka Deekshita.

**No LLMs, generative AI or pretrained AI models are used for extraction.** OCR is done with Tesseract (allowed by the brief).
Everything else is classical ML (scikit-learn Random Forest on hand-built features) plus plain regex/arithmetic rules.

## 1. Pipeline

image/PDF -> preprocessing -> Tesseract (word boxes) -> group words into text pieces (split at wide column gaps)
-> feature engineering -> Random Forest labels each piece (invoice no., date, seller, buyer, PO, tax IDs, 3 totals or "O")
-> cleaning + validation rules -> line-item parser -> JSON + log

**What is ML and what is a rule**

| Step | Type |
|---|---|
| Deciding which text piece is which field | ML (Random Forest, 200 trees) |
| Features (position, text shape, neighbouring words) | Feature engineering |
| Pairing the two tax IDs with the nearest seller/buyer name | Rule |
| Type filters (a date must look like a date, etc.), GSTIN repair/validation | Rules |
| Checking net + tax = total, repairing a misread leading digit | Validation rule |
| Currency detection, line-item column assignment (by x-position of header words) | Rules |

## 2. Features (per text piece)
- Position/size: normalised x1, y1, x2, y2, width, height.
- Text shape: length, digit count, letter count, has `$`, has `:`, date pattern, US-style tax ID pattern, "iban", GSTIN pattern, amount pattern, "PO" word, starts with capital.
- Context: keywords in the piece itself, the previous piece in reading order, the piece to its left on the same line, and the piece above it.
- Vectorised with scikit-learn `DictVectorizer`; keyword vocabulary keeps words seen at least 20 times.

## 3. Training data
1. **Public dataset** `mychen76/invoices-and-receipts_ocr_v1` (Hugging Face). License: **[FILL IN from the dataset page, or write "not stated"]**.
   Only about 425 train / 50 valid / 26 test rows are invoices (one template); the rest are receipts and are skipped. Dataset has no label column, so labels were created automatically by matching the answer key to OCR text (exact, prefix and fuzzy match; 192-200 of 200 invoices matched per field on a first check).
   The answer key contains typos (e.g. IBANs, one seller field holding a tax ID); these were not edited.
2. **Synthetic invoices** (`src/generate_synthetic.py`, `src/extra_layouts.py`): drawn with Pillow, so every value's position is known. Random currencies (USD, INR, EUR), Indian and Western number formats, GSTIN or US-style tax IDs (sometimes missing), optional PO numbers, varied label wording and date formats. Rendered, then OCR-ed with Tesseract so training pieces look like real OCR output.
   Layouts A, B, D, E are used for training; **layout C is never trained on** and is the unseen-layout test.

## 4. Setup
System tools (pip cannot install these):
- Ubuntu/Colab: `sudo apt-get install tesseract-ocr poppler-utils`
- macOS: `brew install tesseract poppler`; Windows: install Tesseract and Poppler and add them to PATH.

```
pip install -r requirements.txt
```

## 5. Run
Batch (images and PDFs, first page of each PDF):
```
python run_batch.py <input_folder> <output_folder>
```
Writes one JSON per file and `processing.log` (warnings per file; each file is wrapped in try/except so one failure does not stop the batch).
Samples: `data/samples/synthetic/` -> `outputs/`; real invoices -> `outputs_real_dev_set/`.

Retrain (about 15 min):
```
python src/generate_synthetic.py --out data/synth.pkl      # layouts A, B, C
python src/extra_layouts.py --out data/synth2.pkl          # layouts A, B, D, E train; C test
python src/train.py --synth data/synth2.pkl --out models/invoice_rf.joblib
```
`train.py` downloads the Hugging Face dataset (internet needed).

## 6. JSON output
`fields` (INVOICE_NO, DATE, SELLER_NAME, CLIENT_NAME (buyer), PO_NO, SELLER_TAX_ID, CLIENT_TAX_ID, IBAN, TOTAL_NET, TOTAL_VAT, TOTAL_GROSS, CURRENCY), `items` (description, qty, unit_price, line_total), `warnings`, `file`. Missing fields are `null`. Net/tax/total are numbers. There is no separate per-line tax field yet.

## 7. Libraries and why
Tesseract + pytesseract (OCR), Pillow (image handling, synthetic invoices), pdf2image + Poppler (PDF to image), scikit-learn (Random Forest, DictVectorizer, metrics), joblib (model files), Faker (fake names/companies for synthetic data), datasets (download the public dataset), numpy, matplotlib (bundled DejaVu font), tqdm. No pretrained models are used.

## 8. Results (details and caveats in `reports/evaluation.md`)
- Dataset test, 26 invoices, one template: exact match 26/26 on 8 fields, 25/26 on seller name and gross total. **Optimistic**: single template, tiny test set.
- Synthetic unseen layout C (60 invoices): F1 1.000 on invoice no., date, names, PO and all three totals.
- 10 real invoices from Google Images (different layouts): invoice number right on 6/10 (was 1/10 with the first model). Date and total are weaker (hand count about 4-5/10).
  These 10 were used while fixing bugs, so they are a **development set, not a clean test**.

## 9. Known limitations
- Only one real-looking template in the public data plus 5 synthetic layouts I wrote: weak on unseen real layouts (labels in boxes, "Grand Total" in coloured boxes, etc.).
- Tesseract misreads small images (rupee sign read as %, 2 or 4; digits swapped). Repair rules fix some totals but not all.
- Phone numbers, business numbers and ZIP codes are sometimes picked as the invoice number; due/PO dates are sometimes picked as the invoice date. Date text can still contain the label (e.g. "date of issue: 22-mar-2026").
- Line items only work when the header uses known words; missed on several real invoices. No per-line tax column.
- Seller/buyer tax IDs: pairing rule is imperfect (40/51 buyer IDs right on layout C with the first model).
- Preprocessing is basic (upscale, grayscale, contrast stretch, median filter); no deskew or binarisation.
- Only the first page of a PDF is read. No handwriting/stamp handling.
- Labels for the public dataset are auto-generated and noisy.

## 10. Possible improvements
More layouts and real annotated invoices; features describing the value type; CRF (sklearn-crfsuite) sequence labelling; deskew; per-line tax; a proper held-out set of real invoices.

## 11. Use of AI assistance
An AI coding assistant helped write and debug the code and documentation. The extraction itself uses no AI/LLM model.
