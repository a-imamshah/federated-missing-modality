"""Run on Kaggle (GPU on, Internet on). MM-IMDb, MULTI-LABEL: every movie with at least one of the 23 genres and a plot.
Extracts frozen CLIP ViT-B/32 features (image + first plot) and a 23-column multi-hot label matrix.
Output: /kaggle/working/feats_imdb23/{train,test}_{img,txt,y}.npy + classes.json (y is float32, shape [N, 23])
and the same folder zipped as /kaggle/working/feats_imdb23.zip.

KEEPING THE RESULT: files in /kaggle/working vanish when the session ends. Either download the zip from the
notebook's Output panel (or the link printed at the end) BEFORE closing, or use "Save Version" -> "Save & Run All"
so Kaggle keeps the output. If the script is re-run and a split is already extracted, that split is skipped.

MM-IMDb images: if you added an MM-IMDb dataset through "Add Input" (a folder with <id>.jpeg and <id>.json files),
the script uses it. Otherwise it downloads the original archive from archive.org, which needs Internet on."""
import os, json, glob, collections, subprocess, shutil, urllib.request
import numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

Image.MAX_IMAGE_PIXELS = None   # a few MM-IMDb posters are huge; PIL warns about them, they are safe to open
OUT = "/kaggle/working/feats_imdb23"; WORK = "/kaggle/temp/mmimdb"; BS = 128
os.makedirs(OUT, exist_ok=True); os.makedirs(WORK, exist_ok=True)
GENRES = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure', 'Horror', 'Documentary', 'Mystery',
          'Sci-Fi', 'Fantasy', 'Family', 'Biography', 'War', 'History', 'Music', 'Animation', 'Musical', 'Western',
          'Sport', 'Short', 'Film-Noir']
FILES = [f"{sp}_{k}.npy" for sp in ("train", "test") for k in ("img", "txt", "y")] + ["classes.json"]


def zip_output():
    shutil.make_archive("/kaggle/working/feats_imdb23", "zip", "/kaggle/working", "feats_imdb23")
    print("zipped:", "/kaggle/working/feats_imdb23.zip", round(os.path.getsize("/kaggle/working/feats_imdb23.zip") / 1e6), "MB")
    print("Download it NOW from the Output panel (right side, /kaggle/working), or use the link below.")
    try:
        from IPython.display import FileLink, display
        display(FileLink("feats_imdb23.zip"))
    except Exception: pass


if all(os.path.exists(f"{OUT}/{f}") for f in FILES):
    print("all feature files already exist, nothing to extract")
    zip_output(); raise SystemExit

def find(pattern):
    return glob.glob(f"/kaggle/input/**/{pattern}", recursive=True) + glob.glob(f"{WORK}/**/{pattern}", recursive=True)
if not find("*.jpeg"):                      # not added as a Kaggle input: download the original archive
    print("no MM-IMDb images under /kaggle/input; downloading from archive.org (large, needs Internet on)...")
    try:
        subprocess.run(f"cd {WORK} && wget -c --tries=5 --timeout=60 -O mmimdb.tar.gz https://archive.org/download/mmimdb/mmimdb.tar.gz "
                       f"&& tar -xzf mmimdb.tar.gz && rm mmimdb.tar.gz", shell=True, check=True)
    except subprocess.CalledProcessError:
        raise SystemExit("Download failed (wget exit 4 = network). Turn Internet ON in the notebook Settings, or use 'Add Input' "
                         "to add an MM-IMDb dataset (folder with .jpeg and .json files) and run again.")
imgs = {os.path.basename(p)[:-5]: p for p in find("*.jpeg")}
jsons = {os.path.basename(p)[:-5]: p for p in find("*.json") if os.path.basename(p)[:-5] in imgs}
splits = find("split.json")
if splits: split = json.load(open(splits[0]))
else:
    urllib.request.urlretrieve("https://raw.githubusercontent.com/YiLunLee/missing_aware_prompts/refs/heads/main/datasets/mmimdb/split.json", f"{WORK}/split.json")
    split = json.load(open(f"{WORK}/split.json"))
print(len(imgs), "images,", len(jsons), "label files; split sizes", {k: len(v) for k, v in split.items()})
split["train"] = split["train"] + split.get("dev", [])

def load(ids):
    out = []
    for i in ids:
        if i not in imgs or i not in jsons: continue
        j = json.load(open(jsons[i]))
        genres = [g for g in j.get("genres", []) if g in GENRES]
        plots = [p for p in j.get("plot", []) if p.strip()]
        if genres and plots: out.append((i, genres, plots[0]))
    return out
rows = {sp: load(split[sp]) for sp in ("train", "test")}

# order the genre columns by how often they occur in train (most frequent first)
count = collections.Counter(g for _, gs, _ in rows["train"] for g in gs)
classes = [g for g, _ in count.most_common()] + [g for g in GENRES if g not in count]
col = {g: k for k, g in enumerate(classes)}
json.dump(classes, open(f"{OUT}/classes.json", "w"))
for sp, r in rows.items():
    print(sp, "movies:", len(r), "| labels per movie:", round(sum(len(gs) for _, gs, _ in r) / len(r), 2))
print("train genre counts:", dict(count.most_common()))

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).half().eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
for sp, rs in rows.items():
    if all(os.path.exists(f"{OUT}/{sp}_{k}.npy") for k in ("img", "txt", "y")):
        print(sp, "already extracted, skipping"); continue
    fi, ft, ys = [], [], []
    with torch.no_grad():
        for s in range(0, len(rs), BS):
            ims, tx, yy = [], [], []
            for i, gs, t in rs[s:s + BS]:
                try: im = Image.open(imgs[i]).convert("RGB")
                except Exception as e: print("skip", i, e); continue
                y = np.zeros(len(classes), np.float32)
                for g in gs: y[col[g]] = 1
                ims.append(im); tx.append(t); yy.append(y)
            pi = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev).half()
            pt = proc(text=tx, return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
            vo = model.vision_model(pixel_values=pi); ei = model.visual_projection(vo.pooler_output)
            to = model.text_model(input_ids=pt["input_ids"], attention_mask=pt["attention_mask"]); et = model.text_projection(to.pooler_output)
            fi.append(torch.nn.functional.normalize(ei.float(), dim=-1).cpu().numpy())
            ft.append(torch.nn.functional.normalize(et.float(), dim=-1).cpu().numpy()); ys += yy
            if (s // BS) % 20 == 0: print(sp, s, "/", len(rs), flush=True)
    np.save(f"{OUT}/{sp}_img.npy", np.concatenate(fi).astype(np.float16))
    np.save(f"{OUT}/{sp}_txt.npy", np.concatenate(ft).astype(np.float16))
    np.save(f"{OUT}/{sp}_y.npy", np.stack(ys).astype(np.float32)); print(sp, "done", len(ys), flush=True)
zip_output()
