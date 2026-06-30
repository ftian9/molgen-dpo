"""
Phase 1: Sample molecules from MolGPT, score with MolSkill and QED.

Run in the MolSkill environment (NOT the TRL environment).
MolGPT is loaded from a local directory (offline, no HF network access needed).

Confirmed facts from source inspection:
  - MolSkill score: LOWER = more preferred by chemists (validated empirically)
  - Score direction: label=1 means smiles_j was chosen; RankNet trains s_i > s_j
    when j is preferred, so chosen molecules get LOWER scores.
  - MolSkill is NOT an LLM; it is a RankNet MLP on molecular fingerprints.
"""

import os
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

# ---------------------------------------------------------------------------
# 0. Import MolGPT
# ---------------------------------------------------------------------------

MOLGPT_PATH = os.path.join("/drive3/ftian/Resources/HuggingFace/models", "jonghyunlee/MolGPT_pretrained-by-ZINC15")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"[setup] Loading MolGPT from {MOLGPT_PATH}")
tok = AutoTokenizer.from_pretrained(MOLGPT_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[setup] Using device: {device}")

base_model = AutoModelForCausalLM.from_pretrained(MOLGPT_PATH).to(device)
base_model.eval()

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# 1. Sampling function
# ---------------------------------------------------------------------------

def sample_smiles(model, tokenizer, n: int = 50000,
                  batch_size: int = 100, max_len: int = 128,
                  temperature: float = 1.0) -> list[str]:
    """Sample raw SMILES strings from MolGPT (unconditional generation)."""
    results = []
    n_batches = (n + batch_size - 1) // batch_size

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    print(f"[sample] pad_id={pad_id} bos_id={bos_id} eos_id={eos_id}")

    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            bsz = min(batch_size, n - i * batch_size)
            input_ids = torch.full(
                (bsz, 1), bos_id, dtype=torch.long, device=device
            )
            out = model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=temperature,
                max_length=max_len,
                pad_token_id=pad_id,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
            )
            for seq in out:
                smi = tokenizer.decode(seq, skip_special_tokens=True).strip()
                results.append(smi)

            if (i + 1) % 50 == 0:
                print(f"  sampled {(i+1)*batch_size}/{n} ...")
    return results

# ---------------------------------------------------------------------------
# 2. Molecule cleaning function
#    - MolSkill is out-of-domain for: multi-fragment SMILES, non-organic atoms
#    - Always canonicalize before scoring
# ---------------------------------------------------------------------------

ORGANIC = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"}

def clean_smiles(smi: str):
    """Return canonical SMILES or None if invalid / out of MolSkill domain."""
    if not smi or not isinstance(smi, str):
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    if "." in smi:                                      # multi-fragment
        return None
    if any(a.GetSymbol() not in ORGANIC for a in m.GetAtoms()):
        return None
    return Chem.MolToSmiles(m)                          # canonical form

# ---------------------------------------------------------------------------
# 3. Generate molecules from MolGPT
# ---------------------------------------------------------------------------

N_SAMPLE = 50000

print(f"Sampling {N_SAMPLE} molecules from MolGPT ...")
raw = sample_smiles(base_model, tok, n=N_SAMPLE)

print("Cleaning and deduplicating ...")
valid = [s for s in (clean_smiles(r) for r in raw) if s]
cleaned = list(dict.fromkeys(valid))

valid_rate = len(valid) / len(raw)
unique_rate = len(cleaned) / len(valid) if valid else 0.0
print(f"  valid : {len(valid)} / {len(raw)} ({100*valid_rate:.1f}%)")
print(f"  unique: {len(cleaned)} / {len(valid)} ({100*unique_rate:.1f}%)")

assert valid_rate > 0.5, (
    f"Validity too low ({valid_rate:.2%}). "
    "Check MolGPT tokenizer or max_length setting."
)

# ---------------------------------------------------------------------------
# 4. Import molskill
# history:
# fixed /drive3/ftian/Tools/miniconda3/envs/rlhf/lib/python3.10/site-packages/molskill/data/standardization.py
# downloaded moments and save in /drive3/ftian/Tools/miniconda3/envs/rlhf/lib/python3.10/site-packages/data/assets/
# downloaded models and save in /drive3/ftian/Tools/miniconda3/envs/rlhf/lib/python3.10/site-packages/
# ---------------------------------------------------------------------------

from molskill.scorer import MolSkillScorer
scorer = MolSkillScorer()

# ---------------------------------------------------------------------------
# 5. Sanity check: confirm score direction BEFORE building pairs
#    Lower MolSkill score = more preferred (confirmed empirically)
# ---------------------------------------------------------------------------

def check_score_direction(scorer: MolSkillScorer):
    drugs = [
        "CC(=O)Oc1ccccc1C(=O)O",       # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # caffeine
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",   # ibuprofen
        "CC(=O)Nc1ccc(O)cc1",            # paracetamol
    ]
    junk = ["C", "CC", "CCC", "CCO", "CCCCCCCC", "C=C"]
    drug_scores = scorer.score(drugs)
    junk_scores = scorer.score(junk)
    print(f"[sanity] drug mean score : {np.mean(drug_scores):.3f}")
    print(f"[sanity] junk mean score : {np.mean(junk_scores):.3f}")
    assert np.mean(drug_scores) < np.mean(junk_scores), (
        "Score direction WRONG: expected drugs < junk (lower = more preferred). "
        "Check MolSkill version or label convention before building pairs."
    )
    print("[sanity] PASSED: lower score = more preferred by chemists.")

check_score_direction(scorer)

# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

print("Scoring with MolSkill ...")
scores = scorer.score(cleaned)

print("Scoring with QED ...")
qed_scores = [QED.qed(Chem.MolFromSmiles(s)) for s in cleaned]

df = pd.DataFrame({"smiles": cleaned, "molskill": scores, "qed": qed_scores})
df.to_parquet("scores.parquet", index=False)
df.to_csv("scores.txt", index=False, sep="\t")

print(f"Saved {len(df)} rows")
print(f"  MolSkill: mean={df.molskill.mean():.3f}  "
      f"min={df.molskill.min():.3f}  max={df.molskill.max():.3f}")
print(f"  QED     : mean={df.qed.mean():.3f}  "
      f"min={df.qed.min():.3f}  max={df.qed.max():.3f}")
print("Done")
