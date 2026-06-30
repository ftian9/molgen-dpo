"""
Phase 4: Evaluate and compare base, RLHF, and RLVR models.

Metrics per model:
  - Validity       : fraction of generated SMILES parseable by RDKit
  - Uniqueness     : fraction of valid SMILES that are unique
  - MolSkill mean  : mean chemist-preference score (LOWER = better)
  - QED mean       : mean drug-likeness by formula (HIGHER = better)
  - SA mean        : mean synthetic accessibility (LOWER = easier to synthesize)

Outputs:
  results/metrics.csv          -- numeric summary table
  results/molskill_dist.png    -- MolSkill score distributions (core result figure)
  results/qed_dist.png         -- QED distributions
  results/umap.png             -- chemical space UMAP (optional, requires umap-learn)

Usage:
  python do_4_eval.py --rlhf molgpt_rlhf --rlvr molgpt_rlvr --n 5000
"""

import os
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

warnings.filterwarnings("ignore")

MOLGPT_PATH = os.path.join("/drive3/ftian/Resources/HuggingFace/models", "jonghyunlee/MolGPT_pretrained-by-ZINC15")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--base",  default=MOLGPT_PATH,  help="Path to base MolGPT")
parser.add_argument("--rlhf",  required=True,  help="Path to RLHF model")
parser.add_argument("--rlvr",  required=True,  help="Path to RLVR model")
parser.add_argument("--n",     type=int, default=5000, help="Molecules per model")
parser.add_argument("--no_sa", action="store_true",
                    help="Skip SA score (requires sascorer in PYTHONPATH)")
args = parser.parse_args()

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# MolSkill setup
# ---------------------------------------------------------------------------

from molskill.scorer import MolSkillScorer
scorer = MolSkillScorer()

# ---------------------------------------------------------------------------
# Molecule utilities
# ---------------------------------------------------------------------------

ORGANIC = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"}

def clean_smiles(smi):
    if not smi or not isinstance(smi, str):
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    if "." in smi:
        return None
    if any(a.GetSymbol() not in ORGANIC for a in m.GetAtoms()):
        return None
    return Chem.MolToSmiles(m)

def sa_score(smi):
    """Return SA score if sascorer is available, else NaN."""
    try:
        from rdkit.Contrib.SA_Score import sascorer
        m = Chem.MolFromSmiles(smi)
        return sascorer.calculateScore(m) if m else np.nan
    except Exception:
        return np.nan

# ---------------------------------------------------------------------------
# Model loading + sampling
# ---------------------------------------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path: str):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        print(f"Warning: pad_token is not set for {path}")
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    return model, tok

def sample(model, tok, n: int, batch_size: int = 100, max_len: int = 128):
    out = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bsz = min(batch_size, n - i)
            ids = torch.full((bsz, 1), tok.bos_token_id, dtype=torch.long, device=device)
            gen = model.generate(
                input_ids=ids, do_sample=True, temperature=1.0,
                max_length=max_len,
                pad_token_id=tok.pad_token_id,
                bos_token_id=tok.bos_token_id,
                eos_token_id=tok.eos_token_id,
            )
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out

# ---------------------------------------------------------------------------
# Evaluate one model
# ---------------------------------------------------------------------------

def evaluate(name: str, path: str, n: int) -> dict:
    print(f"\n[eval] {name}  ({path})")
    model, tok = load_model(path)
    model.eval()
    raw = sample(model, tok, n)

    valid = [s for s in (clean_smiles(r) for r in raw) if s]
    unique = list(set(valid))

    validity   = len(valid)  / len(raw)
    uniqueness = len(unique) / len(valid) if valid else 0.0

    ms_scores = scorer.score(valid) if valid else np.array([])
    qed_scores = [QED.qed(Chem.MolFromSmiles(s)) for s in valid]

    row = {
        "model":        name,
        "n_sampled":    len(raw),
        "validity":     round(validity,   4),
        "uniqueness":   round(uniqueness, 4),
        "molskill_mean": round(float(np.mean(ms_scores)),  3) if len(ms_scores) else np.nan,
        "molskill_std":  round(float(np.std(ms_scores)),   3) if len(ms_scores) else np.nan,
        "qed_mean":      round(float(np.mean(qed_scores)), 3),
        "qed_std":       round(float(np.std(qed_scores)),  3),
    }

    if not args.no_sa:
        sa = [sa_score(s) for s in valid[:500]]  # cap at 500, SA is slow
        sa = [x for x in sa if not np.isnan(x)]
        row["sa_mean"] = round(float(np.mean(sa)), 3) if sa else np.nan

    print(f"  validity={row['validity']:.3f}  uniqueness={row['uniqueness']:.3f}  "
          f"molskill_mean={row['molskill_mean']}  qed_mean={row['qed_mean']}")

    # Store raw scores for plotting
    row["_ms_scores"]  = ms_scores
    row["_qed_scores"] = np.array(qed_scores)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return row

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

results = []
for name, path in [("base", args.base), ("rlhf", args.rlhf), ("rlvr", args.rlvr)]:
    results.append(evaluate(name, path, args.n))

# ---------------------------------------------------------------------------
# Save numeric table
# ---------------------------------------------------------------------------

metrics_cols = ["model", "n_sampled", "validity", "uniqueness",
                "molskill_mean", "molskill_std", "qed_mean", "qed_std"]
if not args.no_sa:
    metrics_cols.append("sa_mean")

metrics_df = pd.DataFrame([{k: r[k] for k in metrics_cols if k in r} for r in results])
metrics_df.to_csv("results/metrics.csv", index=False)
print(f"\n[results]\n{metrics_df.to_string(index=False)}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

colors = {"base": "#6c757d", "rlhf": "#0077b6", "rlvr": "#f77f00"}
labels = {"base": "Base MolGPT",
          "rlhf": "RLHF (MolSkill)",
          "rlvr": "RLVR (QED)"}

# -- MolSkill distribution --
fig, ax = plt.subplots(figsize=(8, 4))
for r in results:
    scores = r["_ms_scores"]
    if len(scores):
        ax.hist(scores, bins=40, alpha=0.55, density=True,
                color=colors[r["model"]], label=labels[r["model"]])
ax.set_xlabel("MolSkill score  (lower = more preferred by chemists)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("MolSkill Score Distribution: Base vs RLHF vs RLVR", fontsize=12)
ax.legend()
ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("results/molskill_dist.png", dpi=150)
plt.close()
print("[plot] results/molskill_dist.png")

# -- QED distribution --
fig, ax = plt.subplots(figsize=(8, 4))
for r in results:
    qeds = r["_qed_scores"]
    ax.hist(qeds, bins=40, alpha=0.55, density=True,
            color=colors[r["model"]], label=labels[r["model"]])
ax.set_xlabel("QED  (higher = more drug-like by formula)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("QED Distribution: Base vs RLHF vs RLVR", fontsize=12)
ax.legend()
fig.tight_layout()
fig.savefig("results/qed_dist.png", dpi=150)
plt.close()
print("[plot] results/qed_dist.png")

# -- Optional UMAP --
try:
    from umap import UMAP
    from rdkit.Chem import AllChem

    def get_fp(smiles_list, radius=2, nbits=2048):
        fps = []
        for s in smiles_list:
            m = Chem.MolFromSmiles(s)
            if m:
                fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
                fps.append(list(fp))
        return np.array(fps, dtype=np.uint8)

    cap = 1000
    all_fps, all_labels = [], []
    for r in results:
        smis = list({s for s in (clean_smiles(x) for x in []) if s})
        # re-sample a small set for fingerprints (re-use valid from scorer call)
        # use valid SMILES from the scored pool instead
        smis = pd.read_parquet("scored_pool.parquet")["smiles"].sample(
            min(cap, len(pd.read_parquet("scored_pool.parquet"))), random_state=0
        ).tolist() if r["model"] == "base" else []
        if not smis:
            continue
        fps = get_fp(smis[:cap])
        all_fps.append(fps)
        all_labels.extend([r["model"]] * len(fps))

    if all_fps:
        X = np.vstack(all_fps)
        embedding = UMAP(n_components=2, random_state=42).fit_transform(X)
        fig, ax = plt.subplots(figsize=(7, 6))
        for name in ["base", "rlhf", "rlvr"]:
            idx = [i for i, l in enumerate(all_labels) if l == name]
            if idx:
                ax.scatter(embedding[idx, 0], embedding[idx, 1],
                           s=4, alpha=0.4, color=colors[name], label=labels[name])
        ax.legend(markerscale=3)
        ax.set_title("Chemical Space (UMAP of Morgan fingerprints)", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig("results/umap.png", dpi=150)
        plt.close()
        print("[plot] results/umap.png")

except ImportError:
    print("[skip] UMAP not installed; skipping chemical space plot.")

print("\n[done] All evaluation outputs in results/")
