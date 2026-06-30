"""
Phase 3: RLHF and RLVR fine-tuning of MolGPT.

Run in the TRL environment (separate from the MolSkill environment).
Reads pair parquets produced by 02_build_pairs.py.

Usage:
  # RLHF arm (MolSkill preferences)
  python do_3_train.py --pairs pairs_rlhf.parquet --out molgpt_rlhf

  # RLVR arm (QED verifiable reward)
  python do_3_train.py --pairs pairs_rlvr.parquet --out molgpt_rlvr

TRL API note:
  Targets TRL >= 1.0 with DPOConfig + processing_class.
  TRL >= 1.7 removed max_prompt_length; use max_length for truncation.
  MolGPT unconditional pairs use empty prompts; we pre-tokenize with BOS-only prompt_ids
  to match do_1_scores.py generation and avoid TRL's empty-string tokenizer mismatch.
  Always check: python -c "import trl; print(trl.__version__)"
"""

import os
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
import argparse
import pandas as pd
import torch

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

MOLGPT_PATH = os.path.join("/drive3/ftian/Resources/HuggingFace/models", "jonghyunlee/MolGPT_pretrained-by-ZINC15")

def tokenize_molgpt_pairs(example: dict, tokenizer: PreTrainedTokenizerBase) -> dict:
    """
    Build DPO token columns for MolGPT.

    Unconditional generation (empty prompt) uses BOS-only prompt_ids, matching
    do_1_scores.py which seeds generation with a single BOS token.
    """
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    prompt_ids = [bos_id]
    chosen_ids = tokenizer.encode(example["chosen"], add_special_tokens=False) + [eos_id]
    rejected_ids = tokenizer.encode(example["rejected"], add_special_tokens=False) + [eos_id]
    return {"prompt_ids": prompt_ids, "chosen_ids": chosen_ids, "rejected_ids": rejected_ids}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--pairs",    required=True,
                    help="Input parquet with prompt/chosen/rejected columns")
parser.add_argument("--out",      required=True,
                    help="Output directory for fine-tuned model")
parser.add_argument("--base",     default=MOLGPT_PATH,
                    help="Path to base MolGPT (local directory)")
parser.add_argument("--beta",     type=float, default=0.1,
                    help="DPO beta: higher = stay closer to reference policy")
parser.add_argument("--lr",       type=float, default=1e-5)
parser.add_argument("--epochs",   type=int,   default=1)
parser.add_argument("--batch",    type=int,   default=8)
parser.add_argument("--grad_acc", type=int,   default=4)
parser.add_argument("--lora",     action="store_true",
                    help="Use LoRA (recommended if GPU VRAM < 16 GB)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load tokenizer + model (offline)
# ---------------------------------------------------------------------------

print(f"[setup] Loading base model from {args.base}")
tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForCausalLM.from_pretrained(args.base)

if args.lora:
    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],   # GPT-2 attention projection names
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

df = pd.read_parquet(args.pairs)
print(f"[data] Loaded {len(df)} pairs from {args.pairs}")

# DPO dataset must have: prompt, chosen, rejected (all strings)
df["prompt"]   = df["prompt"].fillna("").astype(str)
df["chosen"]   = df["chosen"].astype(str)
df["rejected"] = df["rejected"].astype(str)

dat = Dataset.from_pandas(df[["prompt", "chosen", "rejected"]], preserve_index=False)
print(f"[data] Dataset: {dat}")

print("[data] Pre-tokenizing for MolGPT (BOS prompt, SMILES + EOS) ...")
dam = dat.map(
    tokenize_molgpt_pairs,
    fn_kwargs={"tokenizer": tok},
    remove_columns=dat.column_names,
    desc="Tokenizing pairs",
)
ex = dam[0]
print(f"[data] Example: prompt_ids={ex['prompt_ids']} "
      f"chosen_len={len(ex['chosen_ids'])} rejected_len={len(ex['rejected_ids'])}")

# ---------------------------------------------------------------------------
# TRL DPO training
# ---------------------------------------------------------------------------

import trl
print(f"[setup] TRL version: {trl.__version__}")

from trl import DPOConfig, DPOTrainer


class MolGPTDPOTrainer(DPOTrainer):
    """Skip TRL's default tokenization when prompt_ids are already provided."""

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        first = next(iter(dataset))
        if {"prompt_ids", "chosen_ids", "rejected_ids"} <= first.keys():
            return dataset
        return super()._prepare_dataset(dataset, processing_class, args, dataset_name)

cfg_kwargs = dict(
    output_dir=args.out,
    beta=args.beta,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    gradient_accumulation_steps=args.grad_acc,
    max_length=160,                # SMILES are short; prompts are empty
    save_steps=200,
    logging_steps=20,
    bf16=torch.cuda.is_available(),
    remove_unused_columns=False,
)
# TRL < 1.7 still accepts max_prompt_length; removed in 1.7+
if "max_prompt_length" in DPOConfig.__dataclass_fields__:
    cfg_kwargs["max_prompt_length"] = 8

cfg = DPOConfig(**cfg_kwargs)
trainer = MolGPTDPOTrainer(
    model=model,
    ref_model=None,                # auto-copies base as frozen reference
    args=cfg,
    train_dataset=dam,
    processing_class=tok,
)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

print(f"[train] Starting training (beta={args.beta}, lr={args.lr}, "
      f"epochs={args.epochs}, lora={args.lora}) ...")
trainer.train()
trainer.save_model(args.out)
tok.save_pretrained(args.out)
print(f"[Done] Model saved to {args.out} directory")
