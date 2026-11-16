"""Generate synthetic invoices (3 layouts), OCR them with Tesseract, auto-label the pieces.
Usage: python src/generate_synthetic.py --n-train 300 --n-test 60 --out data/synth.pkl
Layouts A and B are for training. Layout C is held out as an unseen-layout test."""
import argparse, os, pickle, random, re, string
import matplotlib
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from faker import Faker
from tqdm.auto import tqdm
from collections import Counter
from pipeline_functions import tesseract_pieces, invoice_to_rows, box_xyxy

fake = Faker(["en_IN", "en_US", "en_GB"])
ITEMS = ['Website design', 'Consulting hours', 'Brochure printing', 'LED monitor', 'HDMI cable', 'Annual license', 'Repair labour', 'Office chairs', 'Steel brackets', 'Cloud hosting']

def _find_font(bold):
    dejavu = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    liberation = "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
    candidates = [
        os.path.join(matplotlib.get_data_path(), "fonts", "ttf", dejavu),
        "/usr/share/fonts/truetype/dejavu/" + dejavu,
        "/usr/share/fonts/truetype/liberation/" + liberation,
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No font found. Run: !find / -name '*.ttf' | head")


def font(size, bold=False):
    return ImageFont.truetype(_find_font(bold), size)


def money(x, cur, indian):
    whole, dec = f"{x:.2f}".split(".")
    if indian and len(whole) > 3:                      # 1,23,456.00 style
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", whole[:-3])
        whole = head + "," + whole[-3:]
    elif not indian:
        whole = f"{int(whole):,}"
    s = whole + "." + dec
    return f"{cur}{random.choice(['', ' ']) if cur == 'Rs.' else ''}{s}" if cur else s


def gstin():   # format only, the check character is random
    L = string.ascii_uppercase
    return (f"{random.randint(1, 37):02d}" + "".join(random.choices(L, k=5)) + f"{random.randint(0, 9999):04d}"
            + random.choice(L) + str(random.randint(1, 9)) + "Z" + random.choice(L + string.digits))


def short_company():
    while True:
        n = fake.company()
        if len(n) <= 28:
            return n


def make_data():
    cur = random.choice(["$", "₹", "€", "Rs.", ""])
    indian = cur in ("₹", "Rs.") or random.random() < 0.15
    items = [(random.choice(ITEMS), random.randint(1, 20), round(random.uniform(10, 5000), 2))
             for _ in range(random.randint(1, 4))]
    net = round(sum(q * r for _, q, r in items), 2)
    pct = random.choice([5, 10, 12, 18, 6.25, 8.5])
    vat = round(net * pct / 100, 2)

    def tax_id():
        if random.random() < 0.2:
            return None
        return gstin() if indian else f"{random.randint(900, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"

    po = (random.choice(["PO-", "PO", ""]) + str(random.randint(100, 99999))) if random.random() < 0.5 else None
    date = fake.date_between("-4y", "today").strftime(
        random.choice(["%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%b %d, %Y", "%d %B %Y", "%Y-%m-%d"]))
    return dict(
        cur=cur, indian=indian, items=items, pct=pct, net=net, vat=vat, gross=round(net + vat, 2),
        inv_no=random.choice(["INV-", "INV", "GST-", "US-", ""]) + str(random.randint(1000, 99999999)),
        inv_key=random.choice(["Invoice No:", "Invoice #", "Inv No:", "Bill No:", "Invoice Number:"]),
        date=date, date_key=random.choice(["Date:", "Invoice Date:", "Dated:", "Date of Issue:"]),
        po=po, po_key=random.choice(["PO No:", "P.O.#", "Purchase Order:", "PO Number:"]),
        seller=short_company(), client=short_company(), s_addr=fake.city(), c_addr=fake.city(),
        tax_key="GSTIN" if indian else random.choice(["Tax ID", "VAT No", "Tax Id"]),
        s_tax=tax_id(), c_tax=tax_id())


class Canvas:
    def __init__(self, W=1240, H=1754):
        self.img = Image.new("RGB", (W, H), "white")
        self.d = ImageDraw.Draw(self.img)
        self.fields = []                       # (label, box) of each value the model must find

    def text(self, x, y, s, size=26, bold=False, label="O", anchor="la", fill="black"):
        f = font(size, bold)
        self.d.text((x, y), s, font=f, fill=fill, anchor=anchor)
        box = self.d.textbbox((x, y), s, font=f, anchor=anchor)
        if label != "O":
            self.fields.append((label, box))
        return box

    def kv(self, x, y, key, val, label, size=26):
        b = self.text(x, y, key, size)
        return self.text(b[2] + 12, y, val, size, True, label)


def rkv(c, xr, y, key, val, label, size=26):               # right-aligned version
    w = font(size, True).getlength(val)
    c.text(xr, y, val, size, True, label, anchor="ra")
    c.text(xr - w - 12, y, key, size, anchor="ra")


def meta(c, d, x, y, step=45):
    rows = [(d["inv_key"], d["inv_no"], "INVOICE_NO"), (d["date_key"], d["date"], "DATE")]
    if d["po"]:
        rows.append((d["po_key"], d["po"], "PO_NO"))
    for key, val, lab in rows:
        c.kv(x, y, key, val, lab); y += step


def party(c, x, y, name, city, tax_key, tax, name_lab, tax_lab, head=None):
    if head:
        c.text(x, y, head, 26, True); y += 42
    c.text(x, y, name, 28, True, name_lab)
    c.text(x, y + 42, city, 24)
    if tax:
        c.kv(x, y + 82, tax_key + ":", tax, tax_lab, 24)


def table(c, d, y, x0=60, x1=1180):
    bg, fg = random.choice([("#e6e6e6", "black"), ("#1f4e9c", "white"), ("#444444", "white")])
    c.d.rectangle([x0, y, x1, y + 50], fill=bg)
    names = random.choice([("Description", "Qty", "Rate", "Amount"), ("Item", "Quantity", "Unit Price", "Total"),
                           ("Product/Service", "Qty", "Price", "Line Total")])
    cols = [x0 + 15, x1 - 560, x1 - 400]
    for x, n in zip(cols, names[:3]):
        c.text(x, y + 10, n, 24, True, fill=fg)
    c.text(x1 - 15, y + 10, names[3], 24, True, anchor="ra", fill=fg)
    y += 70
    for desc, qty, rate in d["items"]:
        c.text(cols[0], y, desc, 24)
        c.text(cols[1], y, str(qty), 24)
        c.text(cols[2], y, money(rate, d["cur"], d["indian"]), 24)
        c.text(x1 - 15, y, money(qty * rate, d["cur"], d["indian"]), 24, anchor="ra")
        y += 55
    c.d.line([x0, y, x1, y], fill="black", width=2)
    return y + 30


def totals(c, d, y, xl, xr):
    rows = [
        (random.choice(["Subtotal", "Sub Total", "Taxable Value", "Net Amount", "Amount"]), d["net"], "TOTAL_NET"),
        (random.choice([f"Tax ({d['pct']}%)", f"GST {d['pct']}%", "VAT", "Tax", "Sales Tax"]), d["vat"], "TOTAL_VAT"),
        (random.choice(["Total", "Grand Total", "Balance Due", "Total Due", "Amount Payable"]), d["gross"], "TOTAL_GROSS"),
    ]
    for key, val, lab in rows:
        big = lab == "TOTAL_GROSS"
        c.text(xl, y, key, 28 if big else 25, big)
        c.text(xr, y, money(val, d["cur"], d["indian"]), 28 if big else 25, big, lab, anchor="ra")
        y += 55


def layout_a(c, d):      # seller top-left, details top-right, totals bottom-right
    party(c, 60, 60, d["seller"], d["s_addr"], d["tax_key"], d["s_tax"], "SELLER_NAME", "SELLER_TAX_ID")
    c.text(1180, 60, "INVOICE", 46, True, anchor="ra")
    meta(c, d, 760, 140)
    party(c, 60, 330, d["client"], d["c_addr"], d["tax_key"], d["c_tax"], "CLIENT_NAME", "CLIENT_TAX_ID",
          head=random.choice(["Bill To", "Billed To", "Customer"]))
    totals(c, d, table(c, d, 560), 700, 1180)


def layout_b(c, d):      # centered title, two shaded "Billed By / Billed To" boxes
    c.text(620, 50, random.choice(["TAX INVOICE", "INVOICE", "Purchase Invoice"]), 44, True, anchor="ma")
    meta(c, d, 60, 140)
    for x, head, who in [(60, "Billed By", "s"), (640, "Billed To", "c")]:
        c.d.rectangle([x - 10, 330, x + 530, 560], fill="#ece8f8")
        party(c, x, 345, d["seller" if who == "s" else "client"], d["s_addr" if who == "s" else "c_addr"],
              d["tax_key"], d[who + "_tax"], "SELLER_NAME" if who == "s" else "CLIENT_NAME",
              "SELLER_TAX_ID" if who == "s" else "CLIENT_TAX_ID", head=head)
    totals(c, d, table(c, d, 600), 700, 1180)


def layout_c(c, d):      # big title top-right, right-aligned details, client block on the left
    party(c, 60, 60, d["seller"], d["s_addr"], d["tax_key"], d["s_tax"], "SELLER_NAME", "SELLER_TAX_ID")
    c.text(1180, 60, "INVOICE", 56, True, anchor="ra")
    rkv(c, 1180, 140, d["inv_key"], d["inv_no"], "INVOICE_NO")
    rkv(c, 1180, 185, d["date_key"], d["date"], "DATE")
    if d["po"]:
        rkv(c, 1180, 230, d["po_key"], d["po"], "PO_NO")
    party(c, 60, 330, d["client"], d["c_addr"], d["tax_key"], d["c_tax"], "CLIENT_NAME", "CLIENT_TAX_ID",
          head=random.choice(["Bill to:", "Invoice to:", "Sold To"]))
    totals(c, d, table(c, d, 560), 760, 1180)


def render(layout=None):
    d = make_data()
    c = Canvas()
    (layout or random.choice(LAYOUTS))(c, d)
    img = c.img
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.choice([0.5, 0.8])))
    return img, c.fields, d


def label_tess(boxes, fields):
    """Give each Tesseract piece the label of the drawn value it mostly covers."""
    labs = []
    for b in boxes:
        x1, y1, x2, y2 = box_xyxy(b)
        best, best_a = "O", 0
        for lab, (fx1, fy1, fx2, fy2) in fields:
            iw, ih = min(x2, fx2) - max(x1, fx1), min(y2, fy2) - max(y1, fy1)
            if iw > 0 and ih > 0:
                a = iw * ih
                if a / ((fx2 - fx1) * (fy2 - fy1)) >= 0.5 and a > best_a:
                    best, best_a = lab, a
        labs.append(best)
    return labs


def build_synth(n, layouts, seed):
    random.seed(seed)
    X, y = [], []
    for _ in tqdm(range(n)):
        img, fields, d = render(random.choice(layouts))
        ws, bs = tesseract_pieces(img)
        labs = label_tess(bs, fields)
        rows, ys = invoice_to_rows(ws, bs, labs)
        X += rows
        y += ys
    return X, y


LAYOUTS = [layout_a, layout_b, layout_c]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--out", default="data/synth.pkl")
    a = ap.parse_args()
    Faker.seed(10)
    Xs_train, ys_train = build_synth(a.n_train, [layout_a, layout_b], seed=10)
    Faker.seed(20)
    Xs_test, ys_test = build_synth(a.n_test, [layout_c], seed=20)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump((Xs_train, ys_train, Xs_test, ys_test), open(a.out, "wb"))
    print(len(Xs_train), len(Xs_test), Counter(ys_train).most_common())
