"""Train the Random Forest on the Hugging Face dataset + synthetic invoices.
Usage: python src/train.py --synth data/synth.pkl --out models/invoice_rf.joblib"""
import argparse, ast, json, pickle
from collections import Counter
import joblib
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
import pipeline_functions as pf

def load_split(ds, split):
    X, y, kept = [], [], 0
    for r in ds[split]:
        t = ast.literal_eval(json.loads(r["parsed_data"])["json"])
        raw = json.loads(r["raw_data"])
        ws = ast.literal_eval(raw["ocr_words"])
        bs = ast.literal_eval(raw["ocr_boxes"])
        labs = pf.label_pieces(t, ws)
        if all(l == "O" for l in labs):      # receipts / no usable answer key
            continue
        rows, ys = pf.invoice_to_rows(ws, bs, labs)
        X += [pf.extra(f) for f in rows]
        y += ys
        kept += 1
    print(split, "invoices kept:", kept)
    return X, y

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="data/synth.pkl")
    ap.add_argument("--out", default="models/invoice_rf.joblib")
    a = ap.parse_args()

    ds = load_dataset("mychen76/invoices-and-receipts_ocr_v1")
    X_train, y_train = load_split(ds, "train")
    X_test, y_test = load_split(ds, "test")
    Xs_train, ys_train, Xs_test, ys_test = pickle.load(open(a.synth, "rb"))
    for X in (Xs_train, Xs_test):
        for f in X:
            pf.extra(f)

    X_all, y_all = X_train + Xs_train, y_train + ys_train
    cnt = Counter()
    for f in X_all:
        for key in ("left_text", "above_text", "prev_text", "text"):
            cnt.update(set(pf.toks(f[key])))
    pf.vocab = {w for w, c in cnt.items() if c >= 20}

    dv = DictVectorizer()
    Xa = dv.fit_transform([pf.to_dict(f) for f in X_all])
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    model.fit(Xa, y_all)

    print("=== UNSEEN LAYOUT C (synthetic) ===")
    print(classification_report(ys_test, model.predict(dv.transform([pf.to_dict(f) for f in Xs_test])), digits=3))
    print("=== DATASET TEST ===")
    print(classification_report(y_test, model.predict(dv.transform([pf.to_dict(f) for f in X_test])), digits=3))

    joblib.dump({"model": model, "dv": dv, "vocab": pf.vocab}, a.out)
    print("saved", a.out)
