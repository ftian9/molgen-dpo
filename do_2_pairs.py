"""
Phase 2: Build RLHF and RLVR preference pairs from the scored molecule pool.

Two signal sources:
  A. On-policy pairs from MolSkill scores  (dense, synthetic)
  B. Real human chemist annotations from molskill/data/production_public.csv (sparse, real)

Score convention (LOWER = more preferred):
  - chosen  = lower MolSkill score  (more drug-like per chemists)
  - rejected = higher MolSkill score (less preferred)

For the RLVR baseline (Phase 6), QED is used instead:
  - chosen  = HIGHER QED  (QED is 0-1, higher = more drug-like by formula)
  - rejected = lower QED

Output files:
  pairs_rlhf.parquet  -- MolSkill-based pairs  (RLHF arm)
  pairs_rlvr.parquet  -- QED-based pairs       (RLVR arm)

Both have columns: prompt | chosen | rejected
"""

import os
import pandas as pd
import numpy as np

SCORED_POOL   = os.environ.get("SCORED_POOL",   "scores.parquet")
HUMAN_ANNOT   = os.environ.get("HUMAN_ANNOT",   "")   # path to production_public.csv, optional
N_PAIRS       = int(os.environ.get("N_PAIRS",   "8000"))
SCORE_GAP     = float(os.environ.get("SCORE_GAP", "2.0"))  # min MolSkill gap between pair members
QED_GAP       = float(os.environ.get("QED_GAP",   "0.15")) # min QED gap

# ---------------------------------------------------------------------------
# Load scores
# ---------------------------------------------------------------------------

df = pd.read_parquet(SCORED_POOL)
print(f"[pairs] Loaded {len(df)} molecules from {SCORED_POOL}")

# ---------------------------------------------------------------------------
# Pair function: build pairs from a scored DataFrame
# ---------------------------------------------------------------------------

def make_pairs(df: pd.DataFrame,
               score_col: str,
               lower_is_better: bool,
               n_pairs: int,
               min_gap: float) -> pd.DataFrame:
    """
    Sample (chosen, rejected) pairs where |score_chosen - score_rejected| >= min_gap.

    lower_is_better=True  -> chosen has the lower score  (MolSkill arm)
    lower_is_better=False -> chosen has the higher score (QED arm)
    """
    sorted_asc = df.sort_values(score_col, ascending=True).reset_index(drop=True)
    n = len(sorted_asc)
    third = n // 3

    if lower_is_better:
        good = sorted_asc.iloc[:third]      # low scores = preferred
        bad  = sorted_asc.iloc[2*third:]    # high scores = not preferred
    else:
        good = sorted_asc.iloc[2*third:]    # high scores = preferred (QED)
        bad  = sorted_asc.iloc[:third]

    pairs = []
    rng = np.random.default_rng(42)

    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        c = good.iloc[rng.integers(len(good))]
        r = bad.iloc[rng.integers(len(bad))]
        gap = abs(c[score_col] - r[score_col])
        if gap >= min_gap:
            pairs.append({
                "prompt":   "",
                "chosen":   c["smiles"],
                "rejected": r["smiles"],
            })

    if len(pairs) < n_pairs:
        print(f"  [warn] Only built {len(pairs)}/{n_pairs} pairs "
              f"(gap threshold {min_gap} may be too strict).")
    return pd.DataFrame(pairs)

# ---------------------------------------------------------------------------
# A. MolSkill-based pairs (RLHF arm)
# ---------------------------------------------------------------------------

print(f"[pairs] Building MolSkill pairs (lower score = chosen) ...")
pairs_rlhf = make_pairs(df, "molskill", lower_is_better=True,
                       n_pairs=N_PAIRS, min_gap=SCORE_GAP)

# Optionally mix in real human annotations from molskill repo
if HUMAN_ANNOT and os.path.exists(HUMAN_ANNOT):
    print(f"[pairs] Mixing in real human annotations from {HUMAN_ANNOT} ...")
    human = pd.read_csv(HUMAN_ANNOT)
    # label=1 means smiles_j was chosen by the chemist
    human["chosen"]   = human.apply(
        lambda r: r["smiles_j"] if r["label"] == 1 else r["smiles_i"], axis=1)
    human["rejected"] = human.apply(
        lambda r: r["smiles_i"] if r["label"] == 1 else r["smiles_j"], axis=1)
    human["prompt"] = ""
    human_pairs = human[["prompt", "chosen", "rejected"]].dropna()
    pairs_rlhf = pd.concat([pairs_rlhf, human_pairs], ignore_index=True)
    print(f"  total pairs after mixing: {len(pairs_rlhf)}")

pairs_rlhf.to_parquet("pairs_rlhf.parquet", index=False)
pairs_rlhf.to_csv("pairs_rlhf.txt", index=False, sep="\t")
print(f"[pairs] Saved {len(pairs_rlhf)} RLHF pairs")

# ---------------------------------------------------------------------------
# B. QED-based pairs (RLVR baseline arm)
# ---------------------------------------------------------------------------

print(f"[pairs] Building QED pairs (higher QED = chosen) ...")
pairs_rlvr = make_pairs(df, "qed", lower_is_better=False,
                        n_pairs=N_PAIRS, min_gap=QED_GAP)
pairs_rlvr.to_parquet("pairs_rlvr.parquet", index=False)
pairs_rlvr.to_csv("pairs_rlvr.txt", index=False, sep="\t")
print(f"[pairs] Saved {len(pairs_rlvr)} RLVR pairs")

# ---------------------------------------------------------------------------
# Quick checks
# ---------------------------------------------------------------------------

def verify_pairs(pairs: pd.DataFrame, label: str):
    assert "chosen" in pairs.columns and "rejected" in pairs.columns
    assert not pairs["chosen"].isna().any(),   f"{label}: NaN in chosen"
    assert not pairs["rejected"].isna().any(), f"{label}: NaN in rejected"
    assert (pairs["chosen"] != pairs["rejected"]).all(), \
        f"{label}: chosen == rejected in some rows"
    print(f"[verify] {label}: {len(pairs)} pairs, no NaN, no trivial pairs. OK")

verify_pairs(pairs_rlhf, "RLHF (MolSkill)")
verify_pairs(pairs_rlvr, "RLVR (QED)")
