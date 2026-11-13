"""Kaggle (GPU + Internet on, UPMC Food-101 added as input). Rebuilds the 8-class Food-101 subset used by
Fed-PRIME (same split.json / text.json / class_idx.json and the same row cut as the repo) and extracts
frozen CLIP ViT-B/32 features. Output: /kaggle/working/feats8/{train,test}_{img,txt,y}.npy + classes.json.
Then click Save Version (Save & Run All) so the files are kept, and download them or add them as a Dataset."""
import os, json, glob, urllib.request, collections
import numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

OUT = "/kaggle/working/feats8"; JS = "/kaggle/working/food_json"; BS = 128
os.makedirs(OUT, exist_ok=True); os.makedirs(JS, exist_ok=True)
base = "https://raw.githubusercontent.com/YiLunLee/missing_aware_prompts/refs/heads/main/datasets/Food101/"
for f in ("class_idx.json", "split.json", "text.json"):
    if not os.path.exists(f"{JS}/{f}"): urllib.request.urlretrieve(base + f, f"{JS}/{f}")
cls_idx = json.load(open(f"{JS}/class_idx.json")); text = json.load(open(f"{JS}/text.json")); split = json.load(open(f"{JS}/split.json"))
split["train"] = split["train"] + split.get("val", [])          # the repo merges val into train

cands = [p for p in glob.glob("/kaggle/input/**/images", recursive=True) if os.path.isdir(p + "/train")]
assert cands, "No images/train folder under /kaggle/input. Run: !find /kaggle/input -maxdepth 4 | head -50"
IMG = cands[0]; print("images:", IMG)

# same rule as the repo: keep samples with text, then only the first rows (train: n // 13.86, test: n // 12.458)
rows = {}
for sp, div in (("train", 13.86), ("test", 12.458)):
    ok = [s for s in split[sp] if s in text]
    keep = ok[: int(len(ok) // div)]
    rows[sp] = [(s, s[: s.rindex("_")]) for s in keep]
    print(sp, "all rows", len(ok), "kept", len(keep), collections.Counter(c for _, c in rows[sp]))
classes = sorted({c for sp in rows for _, c in rows[sp]}, key=lambda c: cls_idx[c])
print(len(classes), "classes:", classes); assert len(classes) == 8, "expected 8 classes"
cmap = {c: i for i, c in enumerate(classes)}

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).half().eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
for sp, rs in rows.items():
    fi, ft, ys = [], [], []
    with torch.no_grad():
        for s in range(0, len(rs), BS):
            ims, tx, yy = [], [], []
            for name, c in rs[s:s + BS]:
                try: im = Image.open(f"{IMG}/{'train' if sp == 'train' else 'test'}/{c}/{name}").convert("RGB")
                except Exception as e: print("skip", name, e); continue
                ims.append(im); tx.append(text[name] if isinstance(text[name], str) else " ".join(text[name])); yy.append(cmap[c])
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
json.dump(classes, open(f"{OUT}/classes.json", "w"))
