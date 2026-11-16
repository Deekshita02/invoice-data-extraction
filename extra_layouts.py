"""Two extra synthetic layouts (D, E) and a generator that trains on A, B, D, E and keeps C unseen.
Usage: python src/extra_layouts.py --out data/synth2.pkl"""
import argparse, os, pickle, random
from collections import Counter
from faker import Faker
from generate_synthetic import (Canvas, party, table, totals, render, build_synth,
                                layout_a, layout_b, layout_c)

def stack(c, x, y, key, val, lab):          # small grey label, value underneath
    c.text(x, y, key.rstrip(":#").strip(), 20, fill="#777777")
    c.text(x, y + 26, val, 26, True, lab)

def layout_d(c, d):
    c.text(60, 50, random.choice(["Purchase Invoice", "Tax Invoice", "Invoice"]), 44, True, fill="#4b2fb0")
    rows = [(d["inv_key"], d["inv_no"], "INVOICE_NO"), (d["date_key"], d["date"], "DATE")]
    if d["po"]:
        rows.append((d["po_key"], d["po"], "PO_NO"))
    for k, (key, val, lab) in enumerate(rows):
        stack(c, 60, 140 + k * 80, key, val, lab)
    for x, head, who in [(60, "Billed By", "s"), (640, "Billed To", "c")]:
        c.d.rectangle([x - 10, 400, x + 530, 640], fill="#ece8f8")
        party(c, x, 415, d["seller" if who == "s" else "client"], d["s_addr" if who == "s" else "c_addr"],
              d["tax_key"], d[who + "_tax"], "SELLER_NAME" if who == "s" else "CLIENT_NAME",
              "SELLER_TAX_ID" if who == "s" else "CLIENT_TAX_ID", head=head)
    totals(c, d, table(c, d, 690), 700, 1180)

def layout_e(c, d):
    party(c, 60, 60, d["seller"], d["s_addr"], d["tax_key"], d["s_tax"], "SELLER_NAME", "SELLER_TAX_ID")
    c.text(1180, 60, "INVOICE", 60, True, anchor="ra", fill="#1f4e9c")
    c.text(1180, 140, "# " + d["inv_no"], 28, False, "INVOICE_NO", anchor="ra")
    c.text(1180, 190, d["date_key"], 22, False, anchor="ra", fill="#777777")
    c.text(1180, 220, d["date"], 26, True, "DATE", anchor="ra")
    if d["po"]:
        c.text(1180, 270, d["po_key"] + " " + d["po"], 22, False, "PO_NO", anchor="ra")
    party(c, 60, 340, d["client"], d["c_addr"], d["tax_key"], d["c_tax"], "CLIENT_NAME", "CLIENT_TAX_ID",
          head=random.choice(["Invoice to :", "Bill To", "Customer"]))
    y = table(c, d, 580)
    c.d.rectangle([680, y + 105, 1190, y + 165], fill="#c9dcf5")      # box behind the grand total
    totals(c, d, y, 700, 1180)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=450)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--out", default="data/synth2.pkl")
    a = ap.parse_args()
    Faker.seed(10)
    Xs_train, ys_train = build_synth(a.n_train, [layout_a, layout_b, layout_d, layout_e], seed=10)
    Faker.seed(20)
    Xs_test, ys_test = build_synth(a.n_test, [layout_c], seed=20)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump((Xs_train, ys_train, Xs_test, ys_test), open(a.out, "wb"))
    print(len(Xs_train), len(Xs_test), Counter(ys_train).most_common())
