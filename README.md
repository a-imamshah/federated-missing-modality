# Missing-modality federated learning on frozen CLIP features

Federated image-text classification when many training samples are missing the image or the text.
CLIP ViT-B/32 encodes every image and text once (on Kaggle); a small model of about 1M parameters is then trained
with federated averaging on the cached 512-d vectors, so a full experiment runs on a laptop CPU in minutes.

The data protocol follows [Fed-PRIME](https://arxiv.org/html/2602.07081) (ICCV 2025): 8-class Food-101 and 8-genre
MM-IMDb subsets, 20 clients, 70% of training samples missing one modality. The backbone, model and optimizer are
different (see [Differences from Fed-PRIME](#differences-from-fed-prime)), so the results are comparable in setup, not a
like-for-like replication.

## Method

- **Features.** Frozen CLIP image and text encoders. Features are L2-normalized and stored as float16 `.npy` files.
- **Model** (`model.py`). One adapter per modality (Linear + GELU), a fusion layer, a classifier. A missing input is
  **zero-masked**: its adapter output is multiplied by 0. Optional **modality dropout** hides one input at random
  during training.
- **Federation** (`federated.py`). Each round, every client (or a random fraction of them) trains a copy of the global
  model for one local epoch with Adam; the server averages the weights, weighted by sample count (FedAvg).
- **Missing data** (`data.py`). A fraction `eta` of training samples lose exactly one input. Pattern `image` drops images,
  `text` drops texts, `both` drops the image of `eta/2` of the samples and the text of another `eta/2`.
- **Evaluation** (`utils.py`). The test set is scored with both inputs (`full`), one input (`image_only`, `text_only`),
  the training pattern (`missing`, the paper's "~Train") and the `both` pattern (`miss_both`). Metrics: F1-micro
  (equals accuracy for single-label data), F1-macro, and ECE before and after temperature scaling (fitted per scenario
  on a 10% validation split of train). Bytes per round are computed as parameters x 4 bytes x 2 directions x clients.

Single-label data (class indices, shape `[N]`) uses softmax and cross-entropy. Multi-label data (multi-hot matrix, shape
`[N, C]`) uses sigmoid and binary cross-entropy with a 0.5 threshold. The mode is read from the labels.

## Repository layout

```
main.py                    run one experiment
data.py  model.py  federated.py  utils.py
summarize.py               average results over seeds into a table
check_simulator.py         sanity checks: clients share no data or weights, FedAvg is a weighted mean
run_paper_experiments.sh   experiments from the Fed-PRIME paper
run_client_experiments.sh  federated learning vs number of clients
kaggle/                    scripts that extract the CLIP features (run on Kaggle, GPU + Internet)
results/                   finished runs: paper.jsonl, clients.jsonl (one JSON line per run)
```

## Installation

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The features are not stored in the repository. Create them once with the scripts in `kaggle/`, in a Kaggle notebook with
GPU and Internet on, then zip the output folder, download it and unzip it next to the code.

| Folder | Script | Content |
|---|---|---|
| `feats8/` | `kaggle/extract_food8.py` | 8-class UPMC Food-101 (4,904 train, 1,823 test). Needs the UPMC Food-101 images added as a Kaggle input |
| `feats_imdb8/` | `kaggle/extract_mmimdb8.py` | 8 single-label MM-IMDb genres (4,030 train, 1,748 test) |
| `feats_imdb23/` | `kaggle/extract_mmimdb23.py` | all 23 MM-IMDb genres, multi-label (every movie with a plot and a genre) |

Each folder holds `{train,test}_{img,txt,y}.npy` and `classes.json`; row i of the three files is the same sample. The
Food-101 and MM-IMDb splits come from the Fed-PRIME / [missing_aware_prompts](https://github.com/YiLunLee/missing_aware_prompts) data files, which the scripts download. The
MM-IMDb script downloads the movie archive from archive.org unless you add an MM-IMDb dataset as a Kaggle input.

## Usage

```
python main.py --feats feats8 --rounds 250
python main.py --rounds 2 --eval_every 1        # quick check that everything runs
```

| Option | Default | Meaning |
|---|---|---|
| `--feats` | `feats8` | Folder with the features |
| `--clients` | 20 | Number of clients (1 = centralized training) |
| `--rounds` | 50 | Communication rounds |
| `--local_epochs` | 1 | Passes over its own data per client and round |
| `--participation` | 1.0 | Fraction of clients that train in each round |
| `--alpha` | none | Dirichlet alpha for a non-IID split (single-label only); default is an IID split |
| `--eta` | 0.7 | Fraction of training samples with a missing input |
| `--missing` | `both` | `image`, `text` or `both` |
| `--modality_dropout` | 0 | 0 = plain FedAvg with zero-masking |
| `--lr`, `--batch_size` | 1e-3, 64 | Adam learning rate and batch size |
| `--seed` | 0 | Random seed |
| `--eval_every` | 5 | Evaluate every this many rounds |
| `--out` | `results.jsonl` | File that each run appends one JSON line to |

Each run prints a table per scenario and appends its settings, results and evaluation history to the output file.

## Experiments

```
bash run_paper_experiments.sh table1|eta|noniid|scale   # then: python summarize.py results/paper.jsonl
bash run_client_experiments.sh                          # then: python summarize.py results/clients.jsonl
```

| Experiment | Settings | Status |
|---|---|---|
| `table1` | 2 datasets x 3 missing patterns, eta 0.7, 3 seeds, 250 rounds | done |
| `eta` | Food-101, missing `both`, eta 0 to 1 | done |
| client sweep | 1 to 100 clients, both datasets, eta 0 and 0.7 | done |
| `noniid` | Food-101, Dirichlet alpha 0.1 | not run |
| `scale` | 100 clients, 10% to 50% participation | not run |
| modality dropout | `--modality_dropout 0.3` against plain zero-masking | not run |
| 23-genre multi-label | `--feats feats_imdb23` | implemented, not run on real features |

`results/paper.jsonl` holds 33 runs and `results/clients.jsonl` 84 runs. `summarize.py` reports F1-macro for MM-IMDb and
F1-micro for Food-101, as the paper does.

## Results

Mean of 3 seeds (see `results/` for the standard deviations), 20 clients, 250 rounds, eta 0.7. Rows are the training
pattern (which input was dropped); columns are the test scenario. Fed-PRIME values are from Table 1 of the paper
(verify before quoting). Food-101 is accuracy (%), MM-IMDb is F1-macro (%).

**Food-101**

| Training pattern | Method | ~Train | Miss Both | Full | Text only | Image only |
|---|---|---|---|---|---|---|
| Miss Text | ours | 91.9 | 92.1 | 98.2 | 90.9 | 89.3 |
| Miss Text | Fed-PRIME | 78.9 | 80.4 | 92.1 | 73.0 | 76.8 |
| Miss Image | ours | 93.5 | 90.8 | 97.9 | 92.0 | 85.5 |
| Miss Image | Fed-PRIME | 90.6 | 79.1 | 92.9 | 90.2 | 54.1 |
| Miss Both | ours | 92.8 | same as ~Train | 98.2 | 92.0 | 88.5 |
| Miss Both | Fed-PRIME | 84.4 | same as ~Train | 93.6 | 88.0 | 72.4 |

**MM-IMDb**

| Training pattern | Method | ~Train | Miss Both | Full | Text only | Image only |
|---|---|---|---|---|---|---|
| Miss Text | ours | 55.1 | 53.8 | 64.7 | 50.8 | 49.8 |
| Miss Text | Fed-PRIME | 31.9 | 31.5 | 37.7 | 30.6 | 30.7 |
| Miss Image | ours | 59.9 | 54.7 | 61.4 | 60.6 | 42.5 |
| Miss Image | Fed-PRIME | 36.1 | 31.4 | 38.5 | 36.9 | 18.2 |
| Miss Both | ours | 59.0 | same as ~Train | 68.4 | 59.7 | 47.5 |
| Miss Both | Fed-PRIME | 32.0 | same as ~Train | 38.7 | 31.0 | 26.0 |

**Number of clients** (Miss Both, eta 0.7, ~Train score; rounds to 95% = first evaluated round where the score reaches
95% of its final value; 1 client is centralized training, so its traffic is not real communication)

| Clients | MB per round | Food-101 accuracy | rounds to 95% | MM-IMDb F1-macro | rounds to 95% |
|---|---|---|---|---|---|
| 1 | 8.4 | 91.4 | 10 | 57.9 | 10 |
| 5 | 42.2 | 92.1 | 10 | 56.6 | 27 |
| 10 | 84.3 | 92.4 | 10 | 58.3 | 73 |
| 20 | 168.7 | 92.8 | 10 | 59.0 | 137 |
| 30 | 253.0 | 92.9 | 17 | 56.8 | 190 |
| 50 | 421.7 | 93.4 | 20 | 54.4 | 197 |
| 100 | 843.4 | 93.2 | 40 | 47.0 | 170 |

On Food-101 the number of clients changes only how fast training converges. On MM-IMDb, 50 and 100 clients are still
improving at round 250, because each client takes few gradient steps per round.

## Differences from Fed-PRIME

| | Fed-PRIME | This project |
|---|---|---|
| Backbone | Frozen ViLT with prompt tuning | Frozen CLIP ViT-B/32, adapters on cached features |
| Missing input | Blank image or empty text through the backbone | Adapter output multiplied by 0 |
| Optimizer, batch | SGD (lr 0.05 / 0.01), batch 512 / 256 | Adam (lr 1e-3), batch 64 |
| MM-IMDb text | 40 tokens | first plot, up to 77 tokens |
| Training data | all of train | 90% of train (10% for temperature scaling) |

The same: 20 IID clients, 250 rounds, 1 local epoch, eta 0.7, the same data splits and test scenarios. I did not run
ablations, so the reasons for the higher scores (most likely the CLIP features) are not measured.

## Known limitations

- **Food-101 overlap.** 38% of test texts have a near-identical text in train (cosine above 0.99), which comes from the
  split files the paper also uses. It inflates text-based scores: a nearest-neighbour check gives 99.1% on those samples
  and 86.8% on the rest.
- **MM-IMDb noise.** The rarest genres have 12 to 56 test samples, so F1-macro varies by 1 to 4 points between seeds.
- **Simulated federation.** Clients run one after another in one process, as in the paper's easyFL-based code. Traffic is
  computed, not measured.
- **Spread.** `summarize.py` reports the population standard deviation over seeds (`ddof=0`), which understates the sample
  standard deviation for 3 seeds.

## Acknowledgements

- Phung et al., [Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](https://arxiv.org/html/2602.07081), ICCV 2025 (data protocol and reference results).
- Lee et al., Multimodal Prompting with Missing Modalities for Visual Recognition, CVPR 2023 ([missing_aware_prompts](https://github.com/YiLunLee/missing_aware_prompts), data splits).
- Radford et al., Learning Transferable Visual Models From Natural Language Supervision (CLIP), 2021.
- UPMC Food-101 (Wang et al., 2015) and MM-IMDb (Arevalo et al., 2017).

## License

MIT, see `LICENSE`.
