import re, io, ast, json, logging, difflib
from collections import Counter
import numpy as np
import pytesseract
from PIL import Image

GSTIN_RE = re.compile(r"\d{2}[a-z]{5}\d{4}[a-z]\d[a-z\d][a-z\d]")
AMT_RE   = re.compile(r"\d[\d,]*[.,]\d{2}\b")
GSTIN_FULL = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z\d]{2}$")
DIGIT_FIX  = {"o": "0", "l": "1", "i": "1", "s": "5", "b": "8", "z": "2"}
LETTER_FIX = {"0": "o", "1": "l", "5": "s", "8": "b", "2": "z"}
PATTERN = "DDLLLLLDDDDLDZA"
COLS = {'description': 'description', 'item': 'description', 'particulars': 'description', 'qty': 'qty', 'quantity': 'qty', 'rate': 'unit_price', 'price': 'unit_price', 'unit price': 'unit_price', 'tax': 'tax', 'amount': 'line_total', 'total': 'line_total', 'product/service': 'description', 'product': 'description', 'service': 'description', 'details': 'description', 'line total': 'line_total', 'unit cost': 'unit_price', 'cost': 'unit_price', 'hours': 'qty', 'amt': 'line_total'}
STOP = re.compile('(sub\\s*total|total|grand|gst|igst|cgst|sgst|vat|tax|balance|amount due|amount payable|discount)')
NUM = ['x1', 'y1', 'x2', 'y2', 'w', 'h', 'n_chars', 'n_digits', 'n_alpha', 'has_dollar', 'has_colon', 'has_date', 'has_taxid', 'has_iban', 'starts_upper']
NUM2 = NUM + ["has_gstin", "has_amount", "has_po"]
TRUE_CUR = {"€": "EUR", "$": "USD", "Rs.": "INR", "₹": "INR", "": None}


def norm(s):
    return re.sub(r"\s+", "", str(s))


def toks(s):
    return re.findall(r"[a-z]{2,}", s)


def unwrap(t):
    """Return a dict with 'header' and 'summary' no matter which format."""
    if list(t.keys()) == ["None"]:
        t = t["None"]
    if "header" in t:
        return t
    header = {k: t.get(k) for k in
              ("invoice_no", "invoice_date", "seller", "client",
               "seller_tax_id", "client_tax_id", "iban")}
    summary = {k: t.get(k) for k in
               ("total_net_worth", "total_vat", "total_gross_worth")}
    return {"header": header, "summary": summary, "items": []}


def truth_values(t):
    t = unwrap(t); h, s = t["header"], t["summary"]
    return {"INVOICE_NO": h.get("invoice_no"), "DATE": h.get("invoice_date"),
            "SELLER_TAX_ID": h.get("seller_tax_id"), "CLIENT_TAX_ID": h.get("client_tax_id"),
            "IBAN": h.get("iban"), "SELLER_NAME": h.get("seller"), "CLIENT_NAME": h.get("client"),
            "TOTAL_NET": s.get("total_net_worth"), "TOTAL_VAT": s.get("total_vat"),
            "TOTAL_GROSS": s.get("total_gross_worth")}


def label_pieces(truth, words):
    truth   = unwrap(truth)
    header  = truth.get("header", {})
    summary = truth.get("summary", {})

    exact = {
        "INVOICE_NO":    header.get("invoice_no"),
        "DATE":          header.get("invoice_date"),
        "SELLER_TAX_ID": header.get("seller_tax_id"),
        "CLIENT_TAX_ID": header.get("client_tax_id"),
        "TOTAL_NET":     summary.get("total_net_worth"),
        "TOTAL_VAT":     summary.get("total_vat"),
        "TOTAL_GROSS":   summary.get("total_gross_worth"),
    }
    names = {
        "SELLER_NAME": header.get("seller"),
        "CLIENT_NAME": header.get("client"),
    }
    iban = header.get("iban")

    labels = []
    for w in words:
        label = "O"
        for name, value in exact.items():
            if value and norm(value) in norm(w):
                label = name
        for name, value in names.items():
            piece = norm(w.strip(" ."))
            if value and len(piece) > 3 and norm(value).startswith(piece):
                label = name
        if iban and w.strip().upper().startswith("IBAN"):
            piece = w.split(":", 1)[-1]
            if close_match(iban, piece):
                label = "IBAN"
        labels.append(label)
    return labels


def invoice_to_rows(words, boxes, labels=None):
    # page size estimate from the boxes themselves
    all_b = [box_xyxy(b) for b in boxes]
    W = max(b[2] for b in all_b) or 1
    H = max(b[3] for b in all_b) or 1

    # reading order: top to bottom, then left to right
    order = sorted(range(len(words)), key=lambda i: (all_b[i][1], all_b[i][0]))

    rows, ys = [], []
    for pos, i in enumerate(order):
        w = words[i]
        x1, y1, x2, y2 = all_b[i]
        prev_text = words[order[pos - 1]] if pos > 0 else ""
        next_text = words[order[pos + 1]] if pos < len(order) - 1 else ""

        # the piece directly to the left on the same line (e.g. "Seller:" label)
        same_line = [j for j in range(len(words))
                     if j != i and abs(all_b[j][1] - y1) < 25 and all_b[j][2] <= x1]
        left_text = words[max(same_line, key=lambda j: all_b[j][2])] if same_line else ""

        # the piece directly above (closest one above that overlaps in x)
        above = [j for j in range(len(words))
                 if all_b[j][3] <= y1 + 5 and all_b[j][2] > x1 and all_b[j][0] < x2]
        above_text = words[max(above, key=lambda j: all_b[j][3])] if above else ""

        f = {
            "x1": x1 / W, "y1": y1 / H, "x2": x2 / W, "y2": y2 / H,
            "w": (x2 - x1) / W, "h": (y2 - y1) / H,
            "text": w.lower(),
            "n_chars": len(w),
            "n_digits": sum(c.isdigit() for c in w),
            "n_alpha": sum(c.isalpha() for c in w),
            "has_dollar": "$" in w,
            "has_colon": ":" in w,
            "has_date": bool(re.search(r"\d{2}/\d{2}/\d{4}", w)),
            "has_taxid": bool(re.search(r"\d{3}-\d{2}-\d{4}", w)),
            "has_iban": "iban" in w.lower(),
            "starts_upper": w.strip()[:1].isupper(),
            "prev_text": prev_text.lower(),
            "next_text": next_text.lower(),
            "left_text": left_text.lower(),
            "above_text": above_text.lower(),
        }
        rows.append(f)
        ys.append(labels[i] if labels else None)
    return rows, ys


def tesseract_pieces(img, gap_factor=1.0):
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    lines = {}
    for i, t in enumerate(d["text"]):
        if t.strip() and float(d["conf"][i]) >= 0:
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            lines.setdefault(key, []).append(i)

    words, boxes = [], []
    for idxs in lines.values():
        idxs = sorted(idxs, key=lambda i: d["left"][i])
        groups = [[idxs[0]]]
        for i in idxs[1:]:
            prev = groups[-1][-1]
            gap = d["left"][i] - (d["left"][prev] + d["width"][prev])
            if gap > gap_factor * d["height"][i]:   # big gap = new column
                groups.append([i])
            else:
                groups[-1].append(i)
        for g in groups:
            txt = " ".join(d["text"][i] for i in g)
            x1 = min(d["left"][i] for i in g)
            y1 = min(d["top"][i] for i in g)
            x2 = max(d["left"][i] + d["width"][i] for i in g)
            y2 = max(d["top"][i] + d["height"][i] for i in g)
            words.append(txt)
            boxes.append([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]], (txt, 1.0)])
    return words, boxes


def clean(field, text):
    t = text.strip()
    if field == "INVOICE_NO":
        return clean_invno(t)
    if field == "DATE":
        m = re.search(r"\d{2}/\d{2}/\d{4}", t);       return m.group(0) if m else t
    if field in ("SELLER_TAX_ID", "CLIENT_TAX_ID"):
        m = re.search(r"\d{3}-\d{2}-\d{4}", t);       return m.group(0) if m else t
    if field == "IBAN":
        return t.split(":", 1)[-1].replace(" ", "").upper()
    if field.startswith("TOTAL"):
        return re.sub(r"[\s$]", "", t)
    return t.strip(" .")          # names


def extra(f):
    t = f["text"]
    f["has_gstin"]  = bool(GSTIN_RE.search(t.replace(" ", "")))
    f["has_amount"] = bool(AMT_RE.search(t))
    f["has_po"]     = bool(re.search(r"\bp\.?o\b|purchase order", t))
    return f


def to_dict(f):                      # replaces the old to_dict, so extract() uses it
    d = {k: float(f[k]) for k in NUM2}
    for key in ("left_text", "above_text", "prev_text", "text"):
        for t in toks(f[key]):
            if t in vocab:
                d[f"{key}:{t}"] = 1.0
    return d


def extract(words, boxes, min_conf=0.3):
    out = extract_base(words, boxes, min_conf)
    for f in ("SELLER_TAX_ID", "CLIENT_TAX_ID"):
        v = out.get(f)
        if v:
            core = re.sub(r"[^a-z0-9]", "", v.lower().split(":")[-1])
            if len(core) == 15:          # looks like a GSTIN: repair it
                out[f] = fix_gstin(v)
    return out


def fix_gstin(text):
    t = text.lower().split(":")[-1]          # drop a "gstin:" prefix
    t = re.sub(r"[^a-z0-9]", "", t)          # keep only letters and digits
    if len(t) != 15:                         # wrong length: do not guess
        return t.upper()
    out = []
    for ch, kind in zip(t, PATTERN):
        if kind == "D":
            ch = DIGIT_FIX.get(ch, ch)
        elif kind == "L":
            ch = LETTER_FIX.get(ch, ch)
        elif kind == "Z":
            ch = "z"
        out.append(ch)
    return "".join(out).upper()


def validate_gstin(v):
    """returns (is_valid, reason)"""
    if not v:
        return False, "missing"
    core = re.sub(r"[^A-Za-z0-9]", "", v.split(":")[-1]).upper()
    if len(core) != 15:
        return False, f"length {len(core)} (should be 15)"
    if not GSTIN_FULL.match(core):
        return False, "does not match GSTIN pattern"
    if not (1 <= int(core[:2]) <= 38):
        return False, f"state code {core[:2]} not valid"
    return True, "ok"


def extract_checked(words, boxes, min_conf=0.3):
    out = extract(words, boxes, min_conf)
    warnings = []

    for f in ("SELLER_TAX_ID", "CLIENT_TAX_ID"):
        v = out.get(f)
        if v and re.sub(r"[^A-Za-z0-9]", "", v.split(":")[-1]).__len__() >= 12:
            ok, why = validate_gstin(v)      # only judge GSTIN-looking values
            if not ok:
                warnings.append(f"{f}: {why}")
        elif v is None:
            warnings.append(f"{f}: missing")

    return {"fields": out, "warnings": warnings}


def clean_po(v):
    if not v:
        return v
    v = v.strip()
    if re.search(r"[:#]", v):                      # "purchase order: po11201" -> "po11201"
        v = re.split(r"[:#]", v)[-1]
    else:                                          # "PO Number 2312/2019" -> "2312/2019"
        v = re.sub(r"^(purchase\s*order|p\.?o\.?)\s*(no\.?|number)\s+", "", v, flags=re.I)
    return v.strip()


def detect_currency(words, fields):
    t = " ".join(words).lower()
    if "€" in t or re.search(r"\beur\b", t):
        return "EUR"
    if "₹" in t or re.search(r"\binr\b", t) or re.search(r"\brs\.?\s*\d", t):
        return "INR"
    if "$" in t or re.search(r"\busd\b", t):
        return "USD"
    for f in ("SELLER_TAX_ID", "CLIENT_TAX_ID"):   # fallback: GSTIN => Indian invoice
        v = fields.get(f)
        if v and len(re.sub(r"[^a-z0-9]", "", v.lower().split(":")[-1])) == 15:
            return "INR"
    return None


def to_num(s):
    m = re.search(r"\d[\d,]*\.?\d*", s or "")
    return float(m.group().replace(",", "")) if m else None


def find_header(lines):
    for k, (y, ps) in enumerate(lines):
        c = {}
        for p in ps:
            name = COLS.get(p["text"].strip(" :."))
            if name:
                c[name] = p["cx"]
        if len(c) >= 3 and ("qty" in c or "unit_price" in c):
            if "description" not in c:             # leftmost piece is the description
                left = min(ps, key=lambda p: p["x1"])
                c["description"] = left["cx"]
            return k, c
    return None, {}


def extract_items(words, boxes):
    rows, _ = invoice_to_rows(words, boxes)
    for r in rows:
        r["cx"] = (r["x1"] + r["x2"]) / 2
    lines = []
    for r in sorted(rows, key=lambda r: r["y1"]):
        if lines and abs(r["y1"] - lines[-1][0]) < 0.02:
            lines[-1][1].append(r)
        else:
            lines.append([r["y1"], [r]])
    hdr, cols = find_header(lines)
    if hdr is None:
        return [], ["no item table header found"]
    items = []
    for y, ps in lines[hdr + 1:]:
        ps = sorted(ps, key=lambda p: p["x1"])
        if STOP.match(ps[0]["text"].strip()):
            break
        item = {}
        for p in ps:
            col = min(cols, key=lambda c: abs(cols[c] - p["cx"]))
            item[col] = (item.get(col, "") + " " + p["text"]).strip()
        it = {"description": item.get("description"),
              "qty": to_num(item.get("qty")),
              "unit_price": to_num(item.get("unit_price")),
              "line_total": to_num(item.get("line_total"))}
        if it["description"] and it["qty"] is not None:
            items.append(it)
    return items, []


def fix_items(items):
    warns = []
    for n, it in enumerate(items, 1):
        q, p, t = it["qty"], it["unit_price"], it["line_total"]
        if None in (q, p, t) or abs(q * p - t) < 0.02 * max(1, q):
            continue                                   # maths fits: leave it
        s = f"{p:.2f}"
        if len(s) > 4 and abs(q * float(s[1:]) - t) < 0.02 * max(1, q):
            it["unit_price"] = float(s[1:])            # extra leading digit removed
            warns.append(f"item {n}: unit_price repaired {p} -> {it['unit_price']}")
        else:
            warns.append(f"item {n}: qty x price != line total")
    return items, warns


def is_correct(field, pred, true):
    if not true: return None                      # no answer key: skip
    if field in ("SELLER_NAME", "CLIENT_NAME"):   # key = name + address
        return norm(true).lower().startswith(norm(pred).lower())
    if field == "IBAN":                           # key has typos, so fuzzy
        return close_match(true, pred)
    return norm(pred).lower() == norm(true).lower().replace("$", "")


def exact_match(split, ds):
    hit, tot, bad = Counter(), Counter(), []
    for r in ds[split]:
        t = ast.literal_eval(json.loads(r["parsed_data"])["json"])
        raw = json.loads(r["raw_data"])
        ws = ast.literal_eval(raw["ocr_words"]); bs = ast.literal_eval(raw["ocr_boxes"])
        if all(l == "O" for l in label_pieces(t, ws)):
            continue
        pred, tv = extract(ws, bs), truth_values(t)
        for f, v in tv.items():
            ok = is_correct(f, pred.get(f, ""), v)
            if ok is None: continue
            tot[f] += 1; hit[f] += ok
            if not ok: bad.append((f, pred.get(f), v))
    for f in tot:
        print(f"{f:14} {hit[f]}/{tot[f]} = {hit[f]/tot[f]:.3f}")
    return bad




def extract_base(words, boxes, min_conf=0.3):
    rows, _ = invoice_to_rows(words, boxes)
    for f in rows:                      # NEW: add has_gstin / has_amount / has_po
        extra(f)
    X = dv.transform([to_dict(f) for f in rows])
    proba = model.predict_proba(X)
    cls = list(model.classes_)
    P = lambda name: proba[:, cls.index(name)] if name in cls else proba[:, 0] * 0

    out = {}
    skip = ("O", "SELLER_TAX_ID", "CLIENT_TAX_ID")
    for field in cls:
        if field in skip:
            continue
        i = proba[:, cls.index(field)].argmax()
        ok = proba[i, cls.index(field)] >= min_conf
        out[field] = clean(field, rows[i]["text"]) if ok else None

    # ---- tax IDs: one combined score, then pair with the nearest name above ----
    tax_score = P("SELLER_TAX_ID") + P("CLIENT_TAX_ID")
    cand = [i for i in tax_score.argsort()[::-1][:2] if tax_score[i] >= min_conf]
    s_i, c_i = P("SELLER_NAME").argmax(), P("CLIENT_NAME").argmax()

    def dist(t, n):
        if rows[n]["y1"] > rows[t]["y1"]:
            return 1e9
        dy = rows[t]["y1"] - rows[n]["y1"]
        dx = abs(rows[t]["x1"] - rows[n]["x1"])
        return dy + 2 * dx

    out["SELLER_TAX_ID"] = out["CLIENT_TAX_ID"] = None
    owner = {}
    for t in cand:
        owner[t] = "SELLER_TAX_ID" if dist(t, s_i) <= dist(t, c_i) else "CLIENT_TAX_ID"
    if len(cand) == 2 and owner[cand[0]] == owner[cand[1]]:
        name = s_i if owner[cand[0]] == "SELLER_NAME" else c_i
        far = max(cand, key=lambda t: dist(t, name))
        owner[far] = ("CLIENT_TAX_ID" if owner[far] == "SELLER_TAX_ID" else "SELLER_TAX_ID")
    for t, f in owner.items():
        out[f] = clean(f, rows[t]["text"])
    return out



import joblib

def load_model(path):
    global model, dv, vocab
    d = joblib.load(path)
    model, dv, vocab = d["model"], d["dv"], d["vocab"]


def box_xyxy(item):
    corners = item[0]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys)

def close_match(value, piece, threshold=0.85):
    return difflib.SequenceMatcher(None, norm(value), norm(piece)).ratio() >= threshold


def parse_amount(v):
    """'%11,716.08' -> 11716.08 (strips junk, handles 1,23,456.00)"""
    if not v:
        return None
    m = re.search(r"\d[\d,]*\.?\d*", str(v))
    return float(m.group().replace(",", "")) if m else None

def finalize(words, boxes, min_conf=0.3):
    res = extract_checked(words, boxes, min_conf)
    f, warns = res["fields"], list(res["warnings"])
    f = {str(k): v for k, v in f.items()}
    f["PO_NO"] = clean_po(f.get("PO_NO"))
    net, vat, gross = (parse_amount(f.get(k)) for k in ("TOTAL_NET", "TOTAL_VAT", "TOTAL_GROSS"))
    if None not in (net, vat, gross):
        if abs(net + vat - gross) > 0.02:
            # try dropping one leading digit from the tax (a rupee sign misread as a digit)
            s = f"{vat:.2f}"
            fixed = float(s[1:]) if len(s) > 4 else None
            if fixed is not None and abs(net + fixed - gross) <= 0.02:
                warns.append(f"TOTAL_VAT repaired {vat} -> {fixed}")
                f["TOTAL_VAT"] = f"{fixed:.2f}"
            else:
                warns.append("net + tax != total")
    for k in ("TOTAL_NET", "TOTAL_VAT", "TOTAL_GROSS"):
        f[k] = parse_amount(f.get(k))
    f["CURRENCY"] = detect_currency(words, f)
    items, w2 = extract_items(words, boxes)
    items, w3 = fix_items(items)
    return {"fields": f, "items": items, "warnings": warns + w2 + w3}


def drop_lead(x):
    """9928.88 stays; 39928.88 -> 9928.88 (a rupee sign misread as a digit)"""
    s = f"{x:.2f}"
    return float(s[1:]) if len(s) >= 6 else None

def repair_amounts(net, vat, gross):
    """Try each amount as read, or with its first digit dropped.
    Return the combination that satisfies net + tax = total, fewest edits first."""
    def opts(x):
        d = drop_lead(x)
        return [(x, 0)] + ([(d, 1)] if d is not None else [])
    best = None
    for n, en in opts(net):
        for v, ev in opts(vat):
            for g, eg in opts(gross):
                if abs(n + v - g) <= 0.02:
                    edits = en + ev + eg
                    if best is None or edits < best[0]:
                        best = (edits, n, v, g)
    return best

finalize_base = finalize

def finalize(words, boxes, min_conf=0.3):
    res = finalize_base(words, boxes, min_conf)
    f = res["fields"]
    if "net + tax != total" in res["warnings"]:
        net, vat, gross = f["TOTAL_NET"], f["TOTAL_VAT"], f["TOTAL_GROSS"]
        if None not in (net, vat, gross):
            fixed = repair_amounts(net, vat, gross)
            if fixed:
                f["TOTAL_NET"], f["TOTAL_VAT"], f["TOTAL_GROSS"] = fixed[1:]
                res["warnings"].remove("net + tax != total")
                res["warnings"].append("amounts repaired: leading junk digit dropped")
    return res


def fill_line_totals(items):
    warns = []
    for n, it in enumerate(items, 1):
        if it.get("line_total") is None and it.get("qty") is not None and it.get("unit_price") is not None:
            it["line_total"] = round(it["qty"] * it["unit_price"], 2)
            warns.append(f"item {n}: line_total computed from qty x unit_price")
    return items, warns

finalize_base2 = finalize

def finalize(words, boxes, min_conf=0.3):
    res = finalize_base2(words, boxes, min_conf)
    res["items"], w = fill_line_totals(res["items"])
    res["warnings"] += w
    return res


DATE_RE = re.compile(r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}(?:st|nd|rd|th)?[ \-]?[a-z]{3,9}[ ,\-]*\d{4}|[a-z]{3,9}\.? \d{1,2},? \d{4})", re.I)
BAD_NO = re.compile(r"phone|tel\b|\bpan\b|gstin|busin|\bdue\b|\bpo\b|p\.o|retention|proforma|purchase|invoice$|date", re.I)

def clean_invno(t):
    t = t.strip()
    if re.search(r"[:#]", t):
        t = re.split(r"[:#]", t)[-1]
    else:
        toks_ = [w for w in t.split() if re.search(r"\d", w)]
        t = toks_[-1] if toks_ else t
    return t.strip(" .:")

def type_ok(field, text):
    t = text.lower()
    if field == "DATE":
        return bool(DATE_RE.search(t)) and not re.search(r"\b[do]ue\b", t)
    if field == "INVOICE_NO":
        core = clean_invno(text)
        return bool(re.search(r"\d", core)) and len(core) >= 3 and not BAD_NO.search(t) \
               and not DATE_RE.search(t)
    if field in ("TOTAL_NET", "TOTAL_VAT", "TOTAL_GROSS"):
        return bool(AMT_RE.search(t))
    return True

extract_prev = extract

def extract(words, boxes, min_conf=0.3, floor=0.10):
    out = extract_prev(words, boxes, min_conf)
    rows, _ = invoice_to_rows(words, boxes)
    proba = model.predict_proba(dv.transform([to_dict(extra(f)) for f in rows]))
    cls = list(model.classes_)
    for field in ("INVOICE_NO", "DATE", "TOTAL_NET", "TOTAL_VAT", "TOTAL_GROSS"):
        cur = out.get(field)
        if cur and type_ok(field, str(cur)):
            continue                                   # current value already looks right
        k = cls.index(field)
        for i in proba[:, k].argsort()[::-1][:5]:      # best candidates first
            if proba[i, k] < floor:
                break
            if type_ok(field, rows[i]["text"]):
                txt = rows[i]["text"]
                if field == "DATE":
                    out[field] = DATE_RE.search(txt).group(0)
                elif field == "INVOICE_NO":
                    out[field] = clean_invno(txt)
                else:
                    out[field] = clean(field, txt)
                break
        else:
            out[field] = None
    return out

def typed_fallback():   # marker so this cell is not applied twice
    pass


PHONE_RE = re.compile(r"\d{3}[-. ]\d{3}[-. ]\d{4}|\+\d{2}")
type_ok_prev = type_ok

def type_ok(field, text):
    if field == "INVOICE_NO":
        core = clean_invno(text)
        if PHONE_RE.search(core) or (core.isdigit() and len(core) >= 9 and core.startswith("0")):
            return False
    return type_ok_prev(field, text)


extract_prev2 = extract

def extract(words, boxes, min_conf=0.3, floor=0.10):
    out = extract_prev2(words, boxes, min_conf, floor)
    for field in ("INVOICE_NO", "DATE", "TOTAL_NET", "TOTAL_VAT", "TOTAL_GROSS"):
        v = out.get(field)
        if v and not type_ok(field, str(v)):
            out[field] = None          # a value that fails its type check is dropped
        elif v and field == "INVOICE_NO":
            out[field] = clean_invno(str(v))
    return out
