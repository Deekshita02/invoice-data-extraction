import argparse, json, logging, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import pipeline_functions as pf
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def load_pages(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pdf2image import convert_from_path
        return [p.convert("RGB") for p in convert_from_path(path, dpi=200, first_page=1, last_page=1)]
    return [Image.open(path).convert("RGB")]

def preprocess(img, min_width=1600):
    from PIL import ImageOps, ImageFilter
    if img.width < min_width:
        s = min_width / img.width
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    img = ImageOps.autocontrast(img.convert("L"), cutoff=1)   # grayscale + contrast stretch
    img = img.filter(ImageFilter.MedianFilter(3))             # light noise removal
    return img.convert("RGB")

def main():
    ap = argparse.ArgumentParser(description="Invoice extraction (OCR + classical ML)")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "invoice_rf.joblib"))
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(a.output_dir, "processing.log")),
                                  logging.StreamHandler()])
    pf.load_model(a.model)
    files = sorted(f for f in os.listdir(a.input_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXT | {".pdf"})
    logging.info("found %d files", len(files))
    ok = 0
    for f in files:
        t0 = time.time()
        try:
            img = preprocess(load_pages(os.path.join(a.input_dir, f))[0])
            ws, bs = pf.tesseract_pieces(img)
            res = pf.finalize(ws, bs)
            res["file"] = f
            with open(os.path.join(a.output_dir, os.path.splitext(f)[0] + ".json"), "w") as fh:
                json.dump(res, fh, indent=2, default=str)
            for w in res["warnings"]:
                logging.warning("%s: %s", f, w)
            logging.info("OK %s (%.1fs)", f, time.time() - t0)
            ok += 1
        except Exception as e:
            logging.error("FAILED %s: %s", f, e)
    logging.info("done: %d/%d succeeded", ok, len(files))

if __name__ == "__main__":
    main()
