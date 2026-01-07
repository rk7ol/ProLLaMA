#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProteinGym-style zero-shot inference with ProLLaMA (causal LM).

This script mirrors the ESM/SaProt zero-shot runners in this workspace:
- reads one or more ProteinGym DMS substitution CSV files
- computes a per-variant score
- writes a per-CSV output with an added score column
- writes an aggregated `summary.csv`

Score definition (aligned to ProLLaMA's provided `scripts/mutation.py`):
  Δ = log P(mutant_sequence) - log P(wildtype_sequence)
    = -NLL(mutant_sequence) + NLL(wildtype_sequence)

Notes:
- Input CSV rows contain `mutated_sequence` (the full mutant sequence) and a
  `mutant` string like "A123G"; we reconstruct the WT sequence by restoring the
  WT residue at that position.
- ProLLaMA expects sequences in the "Seq=<...>" format; this wrapper is applied
  by default via `--seq_prefix/--seq_suffix`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from importlib.metadata import distributions
except Exception:  # pragma: no cover

    def distributions():  # type: ignore[override]
        return []


REQUIRED_COLS = {"mutant", "mutated_sequence", "DMS_score"}


def compute_spearman(pred_scores, true_scores) -> tuple[float | None, float | None]:
    rho, pval = spearmanr(pred_scores, true_scores, nan_policy="omit")
    rho_val = None if rho is None or (isinstance(rho, float) and math.isnan(rho)) else float(rho)
    pval_val = None if pval is None or (isinstance(pval, float) and math.isnan(pval)) else float(pval)
    return rho_val, pval_val


def _fmt_float(x: float | None, *, fmt: str) -> str:
    return "nan" if x is None else format(x, fmt)


def collect_installed_packages() -> list[str]:
    items: list[str] = []
    for dist in distributions():
        name = None
        try:
            name = dist.metadata.get("Name")
        except Exception:
            name = None
        if not name:
            continue
        items.append(f"{name}=={dist.version}")
    return sorted(set(items), key=str.lower)


def print_runtime_environment() -> None:
    print("========== Runtime ==========")
    print(f"Python:        {sys.version.replace(os.linesep, ' ')}")
    print(f"Executable:    {sys.executable}")
    print(f"Platform:      {sys.platform}")
    print("Packages:")
    for item in collect_installed_packages():
        print(f"  - {item}")
    print("=============================\n")


def parse_mutant(mut_str: str) -> tuple[str, int, str]:
    wt_aa = mut_str[0]
    mut_aa = mut_str[-1]
    pos1 = int(mut_str[1:-1])
    return wt_aa, pos1, mut_aa


def recover_wt_sequence(mut_seq: str, wt_aa: str, pos1: int) -> str:
    return mut_seq[: pos1 - 1] + wt_aa + mut_seq[pos1:]


def resolve_csv_paths(*, data_dir: Path, csv: str | None) -> list[Path]:
    if csv:
        p = Path(csv)
        if p.is_absolute() and p.exists():
            return [p]
        candidate = data_dir / csv
        if candidate.exists():
            return [candidate]
        raise FileNotFoundError(f"CSV not found: {csv} (looked in {data_dir} and absolute path)")

    return sorted([p for p in data_dir.glob("*.csv") if p.is_file()], key=lambda x: x.name.lower())


def _pick_device(device_arg: str | None) -> torch.device:
    if device_arg and device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pick_dtype(dtype: str, device: torch.device) -> torch.dtype:
    s = (dtype or "auto").lower()
    if s == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        return torch.float32
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("--dtype must be one of: auto, bf16, fp16, fp32")


def load_model(*, model_name_or_path: str, device: torch.device, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token/eos_token; cannot batch with padding.")
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return model, tokenizer


@torch.no_grad()
def sequence_total_nll(
    *,
    model,
    tokenizer,
    device: torch.device,
    sequences: list[str],
    batch_size: int,
    max_length: int | None,
) -> list[float]:
    if not sequences:
        return []

    all_nll: list[float] = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits

        shift_logits = logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].float()

        token_loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        token_loss = token_loss.reshape(shift_labels.size(0), shift_labels.size(1))
        token_loss = token_loss * shift_mask
        seq_nll = token_loss.sum(dim=1)
        all_nll.extend([float(x) for x in seq_nll.detach().cpu().tolist()])

    return all_nll


def run_one_csv(
    *,
    csv_path: Path,
    output_dir: Path,
    output_suffix: str,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int | None,
    seq_prefix: str,
    seq_suffix: str,
    progress_every: int,
    debug_alignment: bool,
    debug_rows: int,
) -> dict:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name}: missing required columns: {sorted(missing)}")

    mutants = df["mutant"].astype(str).tolist()
    mutated_seqs = df["mutated_sequence"].astype(str).tolist()
    true_scores = df["DMS_score"].tolist()

    wt_seqs: list[str] = []
    for i, (m, mut_seq) in enumerate(zip(mutants, mutated_seqs)):
        wt_aa, pos1, mut_aa = parse_mutant(m)
        if pos1 < 1 or pos1 > len(mut_seq):
            raise ValueError(f"{csv_path.name}: row {i}: pos1 out of range: {m} (len={len(mut_seq)})")
        if mut_seq[pos1 - 1] != mut_aa and debug_alignment and i < debug_rows:
            print(
                f"[debug_alignment] row {i}: mutated_sequence[pos1-1]={mut_seq[pos1-1]!r} != mut_aa={mut_aa!r} for mutant={m!r}"
            )
        wt_seqs.append(recover_wt_sequence(mut_seq, wt_aa, pos1))

    wt_inputs = [f"{seq_prefix}{s}{seq_suffix}" for s in wt_seqs]
    mut_inputs = [f"{seq_prefix}{s}{seq_suffix}" for s in mutated_seqs]

    if debug_alignment:
        for i in range(min(debug_rows, len(df))):
            print("\n========== Alignment Debug (ProLLaMA) ==========")
            wt_aa, pos1, mut_aa = parse_mutant(mutants[i])
            print(f"mutant: {mutants[i]!r} -> (wt_aa={wt_aa!r}, pos1={pos1}, mut_aa={mut_aa!r})")
            print(f"mutated_sequence[pos1-1]: {mutated_seqs[i][pos1-1]!r}")
            print(f"wt_sequence[pos1-1]:      {wt_seqs[i][pos1-1]!r}")
            print(f"WT input preview:  {wt_inputs[i][:120]!r}")
            print(f"MUT input preview: {mut_inputs[i][:120]!r}")
            print("===============================================")

    print(f"Scoring {csv_path.name} ...")
    nll_wt = sequence_total_nll(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sequences=wt_inputs,
        batch_size=batch_size,
        max_length=max_length,
    )
    if progress_every:
        print(f"  WT done ({len(nll_wt)} sequences)")
    nll_mut = sequence_total_nll(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sequences=mut_inputs,
        batch_size=batch_size,
        max_length=max_length,
    )
    if progress_every:
        print(f"  MUT done ({len(nll_mut)} sequences)")

    pred_scores = [(-nm + nw) for nw, nm in zip(nll_wt, nll_mut)]
    rho, pval = compute_spearman(pred_scores, true_scores)

    score_col = "prollama_delta_logp"
    df[score_col] = pred_scores

    out_name = f"{csv_path.stem}{output_suffix}"
    out_path = output_dir / out_name
    df.to_csv(out_path, index=False)

    print("\n========== ProteinGym zero-shot ==========")
    print("Model:        prollama")
    print(f"CSV:          {csv_path.name}")
    print(f"Variants:     {len(df)}")
    print(f"Spearman ρ:   {_fmt_float(rho, fmt='.4f')}")
    print(f"P-value:      {_fmt_float(pval, fmt='.2e')}")
    print(f"Saved to:     {out_path}")
    print("==========================================\n")

    return {
        "model": "prollama",
        "csv": csv_path.name,
        "variants": len(df),
        "spearman_rho": rho,
        "p_value": pval,
        "output_csv": out_path.name,
        "score_column": score_col,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="GreatCaptainNemo/ProLLaMA", help="HF repo id or local path.")
    p.add_argument("--hf_token", default=None, help="Optional HF token (or set env HF_TOKEN/HUGGINGFACE_HUB_TOKEN).")
    p.add_argument("--input_csv", default=None, help="Only process this CSV (basename under data_dir, or an absolute path).")
    p.add_argument("--data_dir", default="/opt/ml/processing/input/data")
    p.add_argument("--output_dir", default="/opt/ml/processing/output")
    p.add_argument("--output_suffix", default="_prollama_zeroshot.csv")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=None, help="Optional truncation length for the tokenizer/model.")
    p.add_argument("--device", default="auto", help="auto/cuda/cpu/cuda:0 ...")
    p.add_argument("--dtype", default="auto", help="auto/bf16/fp16/fp32")
    p.add_argument("--seq_prefix", default="Seq=<", help="Prefix to wrap raw AA sequences for ProLLaMA.")
    p.add_argument("--seq_suffix", default=">", help="Suffix to wrap raw AA sequences for ProLLaMA.")
    p.add_argument("--progress_every", type=int, default=1, help="Print lightweight progress (0 disables).")
    p.add_argument("--debug_alignment", action="store_true")
    p.add_argument("--debug_rows", type=int, default=1)
    args = p.parse_args()

    if args.hf_token:
        os.environ.setdefault("HF_TOKEN", str(args.hf_token))
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", str(args.hf_token))

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_runtime_environment()

    device = _pick_device(args.device)
    dtype = _pick_dtype(args.dtype, device)
    print(f"Device: {device}  dtype: {dtype}\n")

    model, tokenizer = load_model(model_name_or_path=args.model, device=device, dtype=dtype)
    if not getattr(model, "hf_device_map", None):
        model.to(device)

    csv_paths = resolve_csv_paths(data_dir=data_dir, csv=args.input_csv)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    summaries: list[dict] = []
    for csv_path in csv_paths:
        summaries.append(
            run_one_csv(
                csv_path=csv_path,
                output_dir=output_dir,
                output_suffix=args.output_suffix,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=max(1, int(args.batch_size)),
                max_length=args.max_length,
                seq_prefix=str(args.seq_prefix),
                seq_suffix=str(args.seq_suffix),
                progress_every=max(0, int(args.progress_every)),
                debug_alignment=bool(args.debug_alignment),
                debug_rows=max(1, int(args.debug_rows)),
            )
        )

    summary_path = output_dir / "summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
