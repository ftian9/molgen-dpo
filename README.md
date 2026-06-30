# molgen-dpo

Aligning MolGPT to medicinal-chemistry preferences via DPO, using MolSkill (RLHF) or QED (RLVR).

## Overview

This project fine-tunes a SMILES molecular generator ([MolGPT](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00600)) using Direct Preference Optimization (DPO) and compares two alignment signals:

| Arm | Signal | Type | Source |
|---|---|---|---|
| RLHF | [MolSkill](https://www.nature.com/articles/s41467-023-42242-1) score | Human preference | 5,000+ pairwise annotations from 35 Novartis medicinal chemists |
| RLVR | RDKit QED | Verifiable reward | Quantitative Estimate of Drug-likeness (computable formula) |

The comparison isolates what human feedback captures that a computable reward misses — and vice versa.

## Results

Evaluated on 5,000 molecules sampled from each model after 1 epoch of DPO training:

| Model | Validity | Uniqueness | MolSkill mean ↓ | QED mean ↑ |
|---|---|---|---|---|
| Base MolGPT | 99.5% | 100.0% | +0.321 | 0.647 |
| DPO (MolSkill / RLHF) | 99.6% | 100.0% | **-0.659** | 0.660 |
| DPO (QED / RLVR) | 99.6% | 99.98% | -0.543 | **0.694** |

- MolSkill: lower = more preferred by chemists
- QED: higher = more drug-like by formula
- Both aligned models maintain validity and uniqueness relative to base

Key finding: RLHF (MolSkill) more efficiently optimizes chemist preference score (-0.980 shift vs -0.864), while RLVR (QED) more efficiently optimizes the computable drug-likeness formula (+0.047 vs +0.013). The divergence between the two reflects what human intuition encodes beyond what a formula captures.

## Methods

**Base model**: [MolGPT](https://huggingface.co/jonghyunlee/MolGPT_pretrained-by-ZINC15) — a GPT-2 architecture (6.9M parameters) pretrained on ZINC15 to generate drug-like SMILES.

**Preference signal (RLHF arm)**: [MolSkill](https://github.com/microsoft/molskill) — a RankNet MLP trained on pairwise chemist annotations via learning-to-rank (Bradley-Terry model). Score is lower for more preferred molecules. The raw human annotation data (`production_public.csv`) is included in the MolSkill repository under MIT license.

**Verifiable reward (RLVR arm)**: RDKit QED — a weighted geometric mean of 8 physicochemical properties (MW, logP, HBA, HBD, PSA, rotatable bonds, aromaticity, structural alerts), calibrated against marketed drugs. Score is higher for more drug-like molecules.

**Alignment algorithm**: [DPO](https://arxiv.org/abs/2305.18290) (Rafailov et al., NeurIPS 2023) via HuggingFace TRL. Pairs constructed on-policy: sample molecules from base model, score with MolSkill or QED, pair top vs bottom tercile with minimum score gap.

**RLHF vs RLVR framing**: In genomics and molecular biology, ground truth is often verifiable (variant calls, bioassay readouts, QED), making RLVR the natural choice. Human feedback (RLHF) adds value specifically where the target cannot be fully expressed as a computable function — such as medicinal-chemistry intuition about synthetic accessibility, scaffold novelty, and route familiarity. This project makes that distinction concrete and quantifiable.

## Setup

```bash
conda create -n dpo python=3.10
conda activate dpo
conda install molskill=*=py310* -c msr-ai4science -c conda-forge
conda install requests
conda install conda-forge::transformers
conda install -c huggingface -c conda-forge datasets
```

## Usage

```bash
conda activate dpo

# Phase 1: sample and score
python do_1_scores.py
# Phase 2: make pairs
python do_2_pairs.py

# Phase 3: DPO training
python do_3_train.py --pairs pairs_rlhf.parquet --out molgpt-rlhf
python do_3_train.py --pairs pairs_rlvr.parquet --out molgpt-rlvr

# Phase 4: evaluation
python do_4_eval.py --rlhf molgpt_rlhf --rlvr molgpt_rlvr --n 5000
```

## References

- Bagal et al. MolGPT: Molecular Generation Using a Transformer-Decoder Model. *JCIM* 2022.
- Choung et al. Extracting medicinal chemistry intuition via preference machine learning. *Nature Communications* 2023.
- Rafailov et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS* 2023.
- Bickerton et al. Quantifying the chemical beauty of drugs. *Nature Chemistry* 2012.
