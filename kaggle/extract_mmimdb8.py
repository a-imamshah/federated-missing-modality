"""Kaggle (GPU + Internet on). Rebuilds Fed-PRIME's MM-IMDb subset (movies with exactly one of the 23 genres,
8 most frequent genres) and extracts frozen CLIP ViT-B/32 features.
Output: /kaggle/working/feats_imdb8/{train,test}_{img,txt,y}.npy + classes.json
Set ALL_PLOTS=True to make one sample per (movie, plot) pair like the repo's dataset class does;
the default uses the first plot of each movie. The script prints both counts; the paper reports 5,778 samples."""
import os, json, glob, collections, subprocess, urllib.request
import numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

Image.MAX_IMAGE_PIXELS = None   # a few MM-IMDb posters are huge; PIL warns about them, they are safe to open

ALL_PLOTS = False
OUT = "/kaggle/working/feats_imdb8"; WORK = "/kaggle/temp/mmimdb"; BS = 128
os.makedirs(OUT, exist_ok=True); os.makedirs(WORK, exist_ok=True)
GENRES = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure', 'Horror', 'Documentary', 'Mystery',
          'Sci-Fi', 'Fantasy', 'Family', 'Biography', 'War', 'History', 'Music', 'Animation', 'Musical', 'Western',
          'Sport', 'Short', 'Film-Noir']

def find(pattern):
    return glob.glob(f"/kaggle/input/**/{pattern}", recursive=True) + glob.glob(f"{WORK}/**/{pattern}", recursive=True)
if not find("*.jpeg"):                      # not added as a Kaggle input: download the original archive (the repo's method)
    print("downloading MM-IMDb from archive.org (large)...")
    subprocess.run(f"cd {WORK} && wget -q -O mmimdb.tar.gz https://archive.org/download/mmimdb/mmimdb.tar.gz && tar -xzf mmimdb.tar.gz && rm mmimdb.tar.gz", shell=True, check=True)
imgs = {os.path.basename(p)[:-5]: p for p in find("*.jpeg")}
jsons = {os.path.basename(p)[:-5]: p for p in find("*.json") if os.path.basename(p)[:-5] in imgs}
splits = find("split.json")
if splits: split = json.load(open(splits[0]))
else:
    urllib.request.urlretrieve("https://raw.githubusercontent.com/YiLunLee/missing_aware_prompts/refs/heads/main/datasets/mmimdb/split.json", f"{WORK}/split.json")
    split = json.load(open(f"{WORK}/split.json"))
print(len(imgs), "images,", len(jsons), "label files; split sizes", {k: len(v) for k, v in split.items()})
split["train"] = split["train"] + split.get("dev", [])       # the repo merges dev into train

def load(ids):
    out = []
    for i in ids:
        if i not in imgs or i not in jsons: continue
        j = json.load(open(jsons[i])); g = [x for x in j.get("genres", []) if x in GENRES]
        if len(g) == 1 and j.get("plot"): out.append((i, GENRES.index(g[0]), [p for p in j["plot"] if p.strip()] or [""]))
    return out
data = {sp: load(split[sp]) for sp in ("train", "test")}
# repo quirk: it ranks the top-8 inside each split separately; here the top-8 comes from train and is used for both
top8 = [g for g, _ in collections.Counter(g for _, g, _ in data["train"]).most_common(8)]
for sp in data:
    rank = [g for g, _ in collections.Counter(g for _, g, _ in data[sp]).most_common(8)]
    print(sp, "single-genre movies:", len(data[sp]), "| per-split top-8:", [GENRES[g] for g in rank])
print("train top-8 used:", [GENRES[g] for g in top8])
cmap = {g: k for k, g in enumerate(top8)}
rows = {}
for sp in data:
    r = []
    for i, g, plots in data[sp]:
        if g not in cmap: continue
        for t in (plots if ALL_PLOTS else plots[:1]): r.append((i, cmap[g], t))
    rows[sp] = r
    print(sp, "samples:", len(r), "(movies:", len({x[0] for x in r}), ")")
print("total samples:", sum(len(v) for v in rows.values()), "- paper reports 5,778")

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).half().eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
for sp, rs in rows.items():
    fi, ft, ys = [], [], []
    with torch.no_grad():
        for s in range(0, len(rs), BS):
            ims, tx, yy = [], [], []
            for i, c, t in rs[s:s + BS]:
                try: im = Image.open(imgs[i]).convert("RGB")
                except Exception as e: print("skip", i, e); continue
                ims.append(im); tx.append(t); yy.append(c)
            pi = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev).half()
            pt = proc(text=tx, return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
            vo = model.vision_model(pixel_values=pi); ei = model.visual_projection(vo.pooler_output)
            to = model.text_model(input_ids=pt["input_ids"], attention_mask=pt["attention_mask"]); et = model.text_projection(to.pooler_output)
            fi.append(torch.nn.functional.normalize(ei.float(), dim=-1).cpu().numpy())
            ft.append(torch.nn.functional.normalize(et.float(), dim=-1).cpu().numpy()); ys += yy
            print(sp, s, "/", len(rs), flush=True)
    np.save(f"{OUT}/{sp}_img.npy", np.concatenate(fi).astype(np.float16))
    np.save(f"{OUT}/{sp}_txt.npy", np.concatenate(ft).astype(np.float16))
    np.save(f"{OUT}/{sp}_y.npy", np.array(ys, dtype=np.int64)); print(sp, "done", len(ys))
json.dump([GENRES[g] for g in top8], open(f"{OUT}/classes.json", "w"))
