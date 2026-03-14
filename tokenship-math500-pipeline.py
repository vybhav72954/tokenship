#!/usr/bin/env python3
"""
TokenSkip End-to-End Pipeline  -  Qwen2.5-7B-Instruct on MATH-500
==================================================================
Stage 1  . Load MATH train split (7,500 problems)
Stage 2  . Qwen inference -> raw CoT traces  (mathtraincot.jsonl)
Stage 3  . LLMLingua-2 compression @ ratio 0.5 / 0.6 / 0.7
Stage 4  . LoRA fine-tune one adapter per ratio  (3 x 3 epochs)
Stage 5  . MATH-500 evaluation (all prompt + truncation + LoRA methods)
Stage 6  . Generate all 8 figures + 2 CSVs
Stage 7  . Zip everything into a single downloadable archive

Usage:
  python tokenskip_pipeline.py                        # full run (ngrok from Kaggle Secret)
  python tokenskip_pipeline.py --eval-only            # stages 5+6+7 only
  python tokenskip_pipeline.py --plots-only           # stage 6 only
  python tokenskip_pipeline.py --no-kgout             # skip kgout/ngrok entirely
  python tokenskip_pipeline.py --dry-run              # print config, do nothing
  python tokenskip_pipeline.py --stages 3 4           # run only stages 3 and 4
  python tokenskip_pipeline.py --resume               # resume from checkpoint

Ngrok token priority:
  1. Kaggle Secret named  NGROK_AUTH_TOKEN  (recommended)
  2. Environment variable NGROK_AUTH_TOKEN
  If neither is found, kgout is silently skipped (pipeline still runs fine).
"""

# ======================================================================
#  0 . INSTALL DEPENDENCIES
# ======================================================================
import subprocess, sys, os

def pip(*pkgs, optional=False):
    """Install packages; if optional=True, swallow install errors."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *pkgs],
            stderr=subprocess.DEVNULL if optional else None,
        )
        print(f"  [pip] installed: {' '.join(pkgs)}")
    except Exception as exc:
        if optional:
            print(f"  [pip] OPTIONAL install failed (skipping): {pkgs} - {exc}")
        else:
            raise

print("\n[0] Installing dependencies ...")
pip("transformers==4.37.0", "accelerate", "flash-attn", "--no-build-isolation")
pip("peft", "llmlingua", "sentencepiece")
pip("datasets", "protobuf")
pip("seaborn", "matplotlib", "pandas", "tqdm")
pip("kgout[local]", optional=True)   # optional - pipeline works without it

# ======================================================================
#  1 . IMPORTS
# ======================================================================
print("\n[1] Importing libraries ...")
import re, time, json, shutil, zipfile, argparse, pprint, traceback
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from tqdm.auto import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# kgout is fully optional
try:
    from kgout import KgOut
    _KGOUT_AVAILABLE = True
    print("  [kgout] available")
except ImportError:
    _KGOUT_AVAILABLE = False
    print("  [kgout] not available - will use IPython FileLinks fallback")

# llmlingua
try:
    from llmlingua import PromptCompressor
    _LLMLINGUA_AVAILABLE = True
    print("  [llmlingua] available")
except ImportError:
    _LLMLINGUA_AVAILABLE = False
    print("  [llmlingua] NOT available - Stage 3 will be skipped")

# lazy globals (populated in run_pipeline when needed)
tokenizer  = None
base_model = None

# ======================================================================
#  2 . ARGPARSE
# ======================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        prog="tokenskip_pipeline",
        description=(
            "TokenSkip End-to-End Pipeline - Qwen2.5 on MATH-500\n\n"
            "ngrok token is read automatically from Kaggle Secret\n"
            "'NGROK_AUTH_TOKEN' or the same env-var. No CLI token needed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- kgout / serving ---------------------------------------------------
    parser.add_argument(
        "--no-kgout", action="store_true",
        help=(
            "Disable kgout/ngrok live serving entirely. "
            "Files are still zipped and available in the Kaggle Output tab."
        ),
    )

    # -- paths -------------------------------------------------------------
    parser.add_argument(
        "--output-dir", type=str, default="/kaggle/working", metavar="DIR",
        help="Root directory for all output files. (default: /kaggle/working)",
    )
    parser.add_argument(
        "--math500-path", type=str,
        default="/kaggle/input/math500/math500.jsonl", metavar="PATH",
        help=(
            "Path to MATH-500 canonical .jsonl. "
            "Falls back to full HuggingFace test split if file is missing."
        ),
    )
    parser.add_argument(
        "--adapter-dir", type=str, default=None, metavar="DIR",
        help=(
            "Path to a PRE-TRAINED LoRA adapter root directory. "
            "If given, Stage 4 is skipped and adapters are loaded from here. "
            "Expected sub-dirs: <adapter-dir>/qwenloraratio0.5, ...0.6, ...0.7"
        ),
    )

    # -- model -------------------------------------------------------------
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        metavar="HF_MODEL_ID",
        help="HuggingFace model ID for the base LLM.",
    )

    # -- stage selection ---------------------------------------------------
    parser.add_argument(
        "--stages", type=int, nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7],
        choices=[1, 2, 3, 4, 5, 6, 7], metavar="N",
        help=(
            "Explicit list of stages to run (space-separated). "
            "1=DataLoad 2=CoT 3=Compress 4=LoRA 5=Eval 6=Plots 7=Zip"
        ),
    )
    parser.add_argument(
        "--skip-stages", type=int, nargs="+", default=[],
        choices=[1, 2, 3, 4, 5, 6, 7], metavar="N",
        help="Stages to skip (subtracted from --stages).",
    )

    # -- shortcuts ---------------------------------------------------------
    parser.add_argument(
        "--eval-only", action="store_true",
        help=(
            "Shortcut: run only Stages 5 + 6 + 7 "
            "(assumes CoT traces + adapters already exist)."
        ),
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help=(
            "Shortcut: run only Stage 6 "
            "(assumes tokenskipresultsfinal.csv already exists)."
        ),
    )

    # -- compression -------------------------------------------------------
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=[0.5, 0.6, 0.7], metavar="R",
        help="Compression ratios for LLMLingua-2 and LoRA training.",
    )

    # -- inference / eval --------------------------------------------------
    parser.add_argument(
        "--eval-batch", type=int, default=64, metavar="N",
        help="Batch size for model inference during evaluation.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024, metavar="N",
        help="Max new tokens the model generates per sample.",
    )
    parser.add_argument(
        "--orig-acc", type=float, default=68.4, metavar="PCT",
        help="Baseline accuracy %% of unmodified Original CoT.",
    )
    parser.add_argument(
        "--orig-tokens", type=float, default=557.91, metavar="N",
        help="Baseline avg tokens of unmodified Original CoT.",
    )

    # -- training ----------------------------------------------------------
    parser.add_argument(
        "--train-batch", type=int, default=4, metavar="N",
        help="Per-device train batch size for LoRA fine-tuning.",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=8, metavar="N",
        help="Gradient accumulation steps (effective batch = train-batch x grad-accum).",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, metavar="N",
        help="Training epochs per LoRA adapter.",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, metavar="LR",
        help="Learning rate for LoRA fine-tuning.",
    )
    parser.add_argument(
        "--lora-r", type=int, default=16, metavar="R",
        help="LoRA rank r.",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32, metavar="A",
        help="LoRA alpha scaling factor.",
    )

    # -- flags -------------------------------------------------------------
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume from tokenskipcheckpoint.json if it exists; "
            "skip already-completed methods/ratios."
        ),
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip Stage 6 figure generation.",
    )
    parser.add_argument(
        "--no-zip", action="store_true",
        help="Skip Stage 7 final zip.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved configuration and exit without running anything.",
    )

    args = parser.parse_args()

    # -- resolve shortcuts -------------------------------------------------
    if args.eval_only:
        args.stages = [5, 6, 7]
    if args.plots_only:
        args.stages = [6]
    if args.no_plots and 6 in args.stages:
        args.stages.remove(6)
    if args.no_zip and 7 in args.stages:
        args.stages.remove(7)

    args.stages = sorted(set(args.stages) - set(args.skip_stages))

    return args


# ======================================================================
#  3 . RESOLVE ARGS + GLOBALS
# ======================================================================
args = parse_args()

OUTPUT_DIR     = args.output_dir
MODEL_NAME     = args.model
MAX_NEW_TOKENS = args.max_new_tokens
EVAL_BATCH     = args.eval_batch
TRAIN_BATCH    = args.train_batch
GRAD_ACCUM     = args.grad_accum
TRAIN_EPOCHS   = args.epochs
TARGET_RATIOS  = args.ratios
ORIG_ACC       = args.orig_acc
ORIG_TOKENS    = args.orig_tokens
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]

PROMPTS = {
    "Original":
        "Solve the following math problem step by step. Think carefully and "
        "show your work. Put your final answer within \\boxed{{}}.\n\n"
        "Problem: {question}",
    "Concise":
        "Solve concisely. Show key steps only. "
        "Final answer in \\boxed{{}}.\n\nProblem: {question}",
    "Structured":
        "Solve using: UNDERSTAND -> PLAN -> EXECUTE -> VERIFY.\n"
        "Final answer in \\boxed{{}}.\n\nProblem: {question}",
    "StepByStep":
        "Solve step-by-step. Number each step. "
        "Final answer in \\boxed{{}}.\n\nProblem: {question}",
    "DirectAnswer":
        "Solve directly with minimal explanation. "
        "Final answer in \\boxed{{}}.\n\nProblem: {question}",
}

COLORS = dict(
    trunc="tomato", prompt="mediumpurple",
    lora_orig="#90CAF9", lora_guided="darkorange", lora_soft="steelblue",
    orig="gray",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- dry run -------------------------------------------------------------------
if args.dry_run:
    print("\n[dry-run] Resolved configuration:")
    pprint.pprint(vars(args))
    print(f"\n  Stages that will run : {args.stages}")
    print(f"  Device               : {DEVICE}")
    print(f"  GPU                  : "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"  kgout available      : {_KGOUT_AVAILABLE}")
    sys.exit(0)

print(f"\n  Device  : {DEVICE}")
print(f"  GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"  Stages  : {args.stages}")
print(f"  Ratios  : {TARGET_RATIOS}")
print(f"  Model   : {MODEL_NAME}")
print(f"  OutDir  : {OUTPUT_DIR}")


# ======================================================================
#  4 . NGROK TOKEN RESOLUTION  (Kaggle Secret -> env-var fallback)
# ======================================================================
def resolve_ngrok_token():   # returns str token or None
    """
    Tries to obtain the ngrok auth token in order:
      1. Kaggle Secret  'NGROK_AUTH_TOKEN'  (safest - never in code/logs)
      2. Environment variable  NGROK_AUTH_TOKEN
    Returns the token string, or None if unavailable.
    """
    # -- 1. Kaggle Secrets -------------------------------------------------
    try:
        from kaggle_secrets import UserSecretsClient
        secret_val = UserSecretsClient().get_secret("NGROK_AUTH_TOKEN")
        if secret_val:
            print("  [ngrok] token loaded from Kaggle Secret")
            return secret_val
    except Exception as exc:
        print(f"  [ngrok] Kaggle Secret lookup failed ({type(exc).__name__}: {exc})")

    # -- 2. Environment variable -------------------------------------------
    env_val = os.environ.get("NGROK_AUTH_TOKEN", "").strip()
    if env_val:
        print("  [ngrok] token loaded from environment variable")
        return env_val

    print("  [ngrok] no token found - kgout will be disabled")
    return None


# ======================================================================
#  5 . SHARED UTILITIES
# ======================================================================
def extract_boxed(text):
    idx = text.find(r"\boxed{")
    if idx != -1:
        depth, start, end = 1, idx + 7, idx + 7
        while end < len(text) and depth:
            if   text[end] == "{": depth += 1
            elif text[end] == "}": depth -= 1
            end += 1
        if depth == 0:
            return text[start:end - 1]
    m = re.search(r"final answer is[:\s]*(.*)", text, re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def normalize(ans):
    ans = str(ans).strip()
    ans = re.sub(r"\s+",       " ",  ans)
    ans = re.sub(r",(\d{3})",  r"\1", ans)
    return re.sub(r"[,\-]",    "",   ans).lower()


def is_correct(pred, gt):
    p, g = normalize(extract_boxed(pred)), normalize(extract_boxed(gt))
    if p == g:
        return True
    try:
        return abs(float(p.replace(",", "")) - float(g.replace(",", ""))) < 1e-6
    except Exception:
        return False


def make_prompt(question, method="Original"):
    """Build a chat-formatted prompt string for a single question."""
    template = PROMPTS.get(method, PROMPTS["Original"])
    msg = [{"role": "user", "content": template.format(question=question)}]
    return tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    )


def save_checkpoint(results):
    path = os.path.join(OUTPUT_DIR, "tokenskipcheckpoint.json")
    with open(path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"    -> checkpoint saved ({len(results)} methods)")


def header(title):
    bar = "=" * 65
    print(f"\n{bar}\n  {title}\n{bar}")


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}")


# ======================================================================
#  6 . BATCHED EVALUATION HELPER
#     BUG FIX: reordering used sp->sp (no-op) instead of sp->oi
# ======================================================================
def evaluate_batched(df, method="Original", max_new_tokens=None,
                     original_avg_tokens=None, model=None,
                     custom_prompts=None):
    """
    Run batched greedy-decode inference on df.

    Prompts are sorted by length (bucketing) to minimise padding waste,
    then re-ordered back to the original df row order before accuracy
    is computed.

    Args:
        df               : DataFrame with at least an 'Answer' column.
        method           : Key into PROMPTS (ignored when custom_prompts given).
        max_new_tokens   : Override MAX_NEW_TOKENS for this call.
        original_avg_tokens: Denominator for Act Ratio metric.
        model            : Override base_model (e.g. a loaded LoRA model).
        custom_prompts   : Pre-built list of prompt strings, one per df row.

    Returns:
        (metrics_dict, responses_list, token_counts_list)
        responses_list and token_counts_list are aligned to df row order.
    """
    global base_model
    mdl            = model if model is not None else base_model
    max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
    start          = time.time()

    log(f"evaluate_batched: method={method}  n={len(df)}  "
        f"batch={EVAL_BATCH}  max_new={max_new_tokens}")

    # -- build (original_idx, prompt) list ---------------------------------
    if custom_prompts is not None:
        if len(custom_prompts) != len(df):
            raise ValueError(
                f"custom_prompts length {len(custom_prompts)} != df length {len(df)}"
            )
        prompts_indexed = list(enumerate(custom_prompts))
    else:
        prompts_indexed = [
            (seq_i, make_prompt(row["Question"], method))
            for seq_i, (_, row) in enumerate(df.iterrows())
        ]

    # -- sort by prompt length (ascending) for batching --------------------
    prompts_indexed.sort(key=lambda x: len(x[1]))
    sorted_orig_indices = [oi for oi, _ in prompts_indexed]
    sorted_prompts      = [p  for _, p  in prompts_indexed]

    all_responses    = []
    all_token_counts = []
    total_batches    = (len(sorted_prompts) + EVAL_BATCH - 1) // EVAL_BATCH

    for batch_num, bs in enumerate(
        tqdm(range(0, len(sorted_prompts), EVAL_BATCH),
             desc=f"{method}", total=total_batches, unit="batch")
    ):
        batch  = sorted_prompts[bs: bs + EVAL_BATCH]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(DEVICE)
        input_len = inputs["input_ids"].shape[1]

        log(f"  batch {batch_num+1}/{total_batches}  "
            f"size={len(batch)}  input_len={input_len}")

        with torch.no_grad():
            outputs = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated    = outputs[:, input_len:]
        token_counts = (generated != tokenizer.pad_token_id).sum(dim=1).tolist()
        responses    = tokenizer.batch_decode(generated, skip_special_tokens=True)

        all_token_counts.extend(token_counts)
        all_responses.extend(responses)

        del outputs, inputs, generated
        torch.cuda.empty_cache()

    # -- FIXED reordering: restore original df row order -------------------
    # sorted_orig_indices[sp] = the original row index that lives at sorted
    # position sp.  We must write all_responses[sp] -> reordered[oi].
    n = len(df)
    reordered_resp   = [None] * n
    reordered_tokens = [None] * n
    for sp, oi in enumerate(sorted_orig_indices):
        reordered_resp[oi]   = all_responses[sp]    # was [sp] in original (bug)
        reordered_tokens[oi] = all_token_counts[sp] # was [sp] in original (bug)

    if any(r is None for r in reordered_resp):
        log("WARNING: Some reordered responses are None - check index alignment!")

    elapsed = time.time() - start
    answers = df["Answer"].tolist()
    correct = sum(is_correct(r, g) for r, g in zip(reordered_resp, answers))
    avg_tok = sum(reordered_tokens) / n

    metrics = {
        "Method":     method,
        "Accuracy":   round(100 * correct / n, 2),
        "Avg Tokens": round(avg_tok, 2),
        "Latency(s)": round(elapsed / n, 3),
        "Act Ratio":  round(avg_tok / original_avg_tokens, 3)
                      if original_avg_tokens else 1.0,
        "Correct": correct,
        "Total":   n,
    }

    log(f"evaluate_batched DONE -> Acc={metrics['Accuracy']}%  "
        f"AvgTok={metrics['Avg Tokens']}  elapsed={elapsed:.1f}s")

    return metrics, reordered_resp, reordered_tokens


# ======================================================================
#  7 . PLOTTER  - all 8 figures
# ======================================================================
class Plotter:
    def __init__(self, df, out=None):
        self.df  = df.copy()
        self.out = out or OUTPUT_DIR

    def _save(self, name):
        p = os.path.join(self.out, name)
        plt.tight_layout()
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        sz = os.path.getsize(p) / 1e3
        log(f"[fig] saved -> {p}  ({sz:.0f} KB)")

    def truncation_analysis(self):
        df    = self.df
        trunc = df[df.Method.str.startswith("Truncation")].sort_values("Token Savings")
        tw    = pd.concat([trunc, df[df.Method == "Original"]]).sort_values("Avg Tokens")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(tw["Avg Tokens"], tw["Accuracy"], "o-",
                     color="#1565C0", lw=2, ms=8)
        for _, row in tw.iterrows():
            lbl = str(row.get("Ratio", "")) if pd.notna(row.get("Ratio", float("nan"))) else "Orig"
            axes[0].annotate(lbl, (row["Avg Tokens"], row["Accuracy"]),
                             textcoords="offset points", xytext=(5, 5), fontsize=8)
        axes[0].set_xlabel("Avg Tokens"); axes[0].set_ylabel("Accuracy %")
        axes[0].set_title("Accuracy vs Token Budget", fontsize=13, fontweight="bold")
        axes[0].grid(alpha=0.3)

        ax1 = axes[1]; ax2 = ax1.twinx()
        ax1.plot(trunc["Ratio"], trunc["Avg Tokens"], "o-", color="tab:blue", lw=2)
        ax2.plot(trunc["Ratio"], trunc["Latency(s)"], "s-", color="tab:red",  lw=2)
        ax1.set_xlabel("Truncation Ratio")
        ax1.set_ylabel("Avg Tokens",       color="tab:blue")
        ax2.set_ylabel("Latency s/sample", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax1.set_title("Tokens & Latency vs Ratio", fontsize=13, fontweight="bold")

        final = df[~df.Method.str.startswith("LoRA")]
        axes[2].scatter(final["Token Savings"], final["Accuracy"],
                        s=120, c="#1565C0", zorder=5, edgecolors="black", lw=0.5)
        for _, row in final.iterrows():
            axes[2].annotate(row["Method"],
                             (row["Token Savings"], row["Accuracy"]),
                             textcoords="offset points", xytext=(5, 3), fontsize=7)
        axes[2].set_xlabel("Token Savings %"); axes[2].set_ylabel("Accuracy %")
        axes[2].set_title("Pareto Frontier: Accuracy vs Savings",
                          fontsize=13, fontweight="bold")
        axes[2].axvline(0, color="gray", linestyle="--", lw=0.8)
        axes[2].grid(alpha=0.3)
        self._save("fig1_truncation_analysis.png")

    def method_heatmap(self):
        cols  = ["Accuracy", "Avg Tokens", "Token Savings",
                 "Latency(s)", "Efficiency Score"]
        cols  = [c for c in cols if c in self.df.columns]
        pivot = self.df.set_index("Method")[cols]
        norm  = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-9)
        fig, ax = plt.subplots(figsize=(10, max(6, len(self.df) * 0.38)))
        sns.heatmap(norm, annot=pivot.round(2), fmt="g", cmap="YlOrRd",
                    linewidths=0.5, ax=ax,
                    cbar_kws={"label": "Normalized Score"})
        ax.set_title(
            "TokenSkip Methods - Metric Heatmap\n(annotations = actual values)",
            fontsize=13, fontweight="bold",
        )
        self._save("fig2_method_heatmap.png")

    def token_distribution(self, all_token_counts):
        if not all_token_counts:
            log("[skip] token_distribution - no token-count data")
            return
        rows = [{"Method": m, "Tokens": c}
                for m, counts in all_token_counts.items() for c in counts]
        dist_df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.boxplot(data=dist_df, x="Method", y="Tokens", palette="Blues", ax=ax)
        ax.set_title("Token Length Distribution per Method",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("Generated Tokens")
        ax.tick_params(axis="x", rotation=25)
        self._save("fig3_token_distribution.png")

    def accuracy_drop_vs_savings(self):
        df     = self.df
        trunc  = df[df.Method.str.startswith("Truncation")].sort_values("Token Savings")
        soft   = df[df.Method.str.startswith("LoRASoft")].sort_values("Token Savings")
        guided = df[df.Method.str.startswith("LoRAGuided")].sort_values("Token Savings")
        fig, ax = plt.subplots(figsize=(9, 5))
        if len(trunc):
            ax.plot(trunc["Token Savings"],  trunc["Accuracy Drop"],  "o--",
                    color=COLORS["trunc"],       lw=2, ms=7, label="Truncation")
        if len(soft):
            ax.plot(soft["Token Savings"],   soft["Accuracy Drop"],   "s-",
                    color=COLORS["lora_soft"],   lw=2, ms=8, label="LoRA Soft")
        if len(guided):
            ax.plot(guided["Token Savings"], guided["Accuracy Drop"], "^-",
                    color=COLORS["lora_guided"], lw=2, ms=7, label="LoRA Guided")
        ax.axhline(0, linestyle=":", color="gray", lw=1.5, label="No-drop baseline")
        max_sav = df["Token Savings"].max() if len(df) else 1
        ax.fill_between([0, max_sav], 0, 3,
                        alpha=0.06, color="green", label="Accuracy gain zone")
        ax.set_xlabel("Token Savings %", fontsize=12)
        ax.set_ylabel("Accuracy Change (pp)", fontsize=12)
        ax.set_title("Accuracy Cost of Compression  (up = improved)", fontsize=13)
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        self._save("fig4_accuracy_drop_vs_savings.png")

    def grouped_by_ratio(self):
        df     = self.df
        ratios = TARGET_RATIOS

        def _val(col, method_name):
            r = df[df.Method == method_name]
            return float(r[col].values[0]) if len(r) else 0.0

        t_acc = [_val("Accuracy",      f"Truncation{r}") for r in ratios]
        s_acc = [_val("Accuracy",      f"LoRASoft{r}")   for r in ratios]
        t_sav = [_val("Token Savings", f"Truncation{r}") for r in ratios]
        s_sav = [_val("Token Savings", f"LoRASoft{r}")   for r in ratios]

        x = np.arange(len(ratios)); w = 0.35
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, ya, yb, ylabel, title in [
            (axes[0], t_acc, s_acc, "Accuracy %",      "Accuracy by Compression Ratio"),
            (axes[1], t_sav, s_sav, "Token Savings %", "Token Savings by Ratio"),
        ]:
            ax.bar(x - w/2, ya, w, label="Truncation",
                   color=COLORS["trunc"],     edgecolor="white")
            ax.bar(x + w/2, yb, w, label="LoRA Soft",
                   color=COLORS["lora_soft"], edgecolor="white")
            ax.axhline(ORIG_ACC if "Accuracy" in ylabel else 0,
                       linestyle="--", color="gray", lw=1.2)
            ax.set_xticks(x); ax.set_xticklabels([f"r={r}" for r in ratios])
            ax.set_ylabel(ylabel); ax.set_title(title)
            ax.legend(); ax.grid(axis="y", alpha=0.3)
            for i, (a, b) in enumerate(zip(ya, yb)):
                ax.text(i - w/2, a + 0.3, f"{a:.1f}",
                        ha="center", fontsize=9, fontweight="bold")
                ax.text(i + w/2, b + 0.3, f"{b:.1f}",
                        ha="center", fontsize=9, fontweight="bold")
        fig.suptitle(
            "Truncation vs TokenSkip LoRA Soft - By Compression Ratio",
            fontsize=13, y=1.01,
        )
        self._save("fig5_grouped_by_ratio.png")

    def lora_triplet(self):
        df = self.df

        def _acc(m):
            r = df[df.Method == m]
            return float(r["Accuracy"].values[0]) if len(r) else 0.0

        ratios = TARGET_RATIOS
        orig   = [_acc(f"LoRA{r}")       for r in ratios]
        guided = [_acc(f"LoRAGuided{r}") for r in ratios]
        soft   = [_acc(f"LoRASoft{r}")   for r in ratios]
        x = np.arange(len(ratios)); w = 0.25
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w,  orig,   w, label="LoRA Original",
               color=COLORS["lora_orig"],   edgecolor="white")
        ax.bar(x,      guided, w, label="LoRA Guided",
               color=COLORS["lora_guided"], edgecolor="white")
        ax.bar(x + w,  soft,   w, label="LoRA Soft",
               color=COLORS["lora_soft"],   edgecolor="white")
        ax.axhline(ORIG_ACC, linestyle="--", color="black",
                   lw=1.3, label=f"Baseline {ORIG_ACC}%")
        ax.set_xticks(x); ax.set_xticklabels([f"ratio={r}" for r in ratios])
        ax.set_ylabel("Accuracy %"); ax.set_ylim(35, 80)
        ax.set_title("LoRA Variants - Accuracy by Compression Ratio",
                     fontsize=13, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        for i, (a, b, c) in enumerate(zip(orig, guided, soft)):
            ax.text(i - w, a + 0.3, f"{a:.1f}", ha="center", fontsize=9)
            ax.text(i,     b + 0.3, f"{b:.1f}", ha="center", fontsize=9)
            ax.text(i + w, c + 0.3, f"{c:.1f}", ha="center", fontsize=9)
        self._save("fig6_lora_triplet.png")

    def all_methods_bar(self):
        dfp = self.df.sort_values("Accuracy", ascending=True)
        colors = []
        for m in dfp.Method:
            if   m.startswith("LoRASoft"):   colors.append(COLORS["lora_soft"])
            elif m.startswith("LoRAGuided"): colors.append(COLORS["lora_guided"])
            elif m.startswith("LoRA"):       colors.append(COLORS["lora_orig"])
            elif m.startswith("Truncation"): colors.append(COLORS["trunc"])
            elif m == "Original":            colors.append(COLORS["orig"])
            else:                            colors.append(COLORS["prompt"])
        fig, ax = plt.subplots(figsize=(9, max(6, len(dfp) * 0.4)))
        bars = ax.barh(dfp.Method, dfp.Accuracy, color=colors, edgecolor="white")
        ax.axvline(ORIG_ACC, linestyle="--", color="black", lw=1.2)
        ax.set_xlabel("Accuracy %"); ax.set_xlim(35, 82)
        ax.set_title("All Methods Ranked by Accuracy",
                     fontsize=13, fontweight="bold")
        for bar, val in zip(bars, dfp.Accuracy):
            ax.text(bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9)
        patches = [
            mpatches.Patch(color=COLORS["orig"],        label="Original"),
            mpatches.Patch(color=COLORS["prompt"],      label="Prompt Variant"),
            mpatches.Patch(color=COLORS["trunc"],       label="Truncation"),
            mpatches.Patch(color=COLORS["lora_orig"],   label="LoRA"),
            mpatches.Patch(color=COLORS["lora_guided"], label="LoRA Guided"),
            mpatches.Patch(color=COLORS["lora_soft"],   label="LoRA Soft"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=9)
        self._save("fig7_all_methods_bar.png")

    def subject_accuracy_heatmap(self, results_by_subject):
        if not results_by_subject:
            log("[skip] subject_accuracy_heatmap - no subject data")
            return
        subj_df = pd.DataFrame(results_by_subject).T
        fig, ax = plt.subplots(
            figsize=(max(10, len(subj_df.columns)), max(5, len(subj_df) * 0.4))
        )
        sns.heatmap(subj_df, annot=True, fmt=".1f", cmap="RdYlGn",
                    linewidths=0.4, ax=ax,
                    cbar_kws={"label": "Accuracy %"})
        ax.set_title("Per-Subject Accuracy Heatmap",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Subject"); ax.set_ylabel("Method")
        self._save("fig8_subject_accuracy.png")

    def run_all(self, all_token_counts=None, results_by_subject=None):
        header("STAGE 6 . Generating all 8 figures")
        self.truncation_analysis()
        self.method_heatmap()
        self.token_distribution(all_token_counts or {})
        self.accuracy_drop_vs_savings()
        self.grouped_by_ratio()
        self.lora_triplet()
        self.all_methods_bar()
        self.subject_accuracy_heatmap(results_by_subject or {})
        log("All 8 figures complete.")


# ======================================================================
#  8 . MAIN PIPELINE
# ======================================================================
def run_pipeline():
    global tokenizer, base_model

    # -- kgout setup -------------------------------------------------------
    ngrok_token = None
    use_kgout   = False

    if not args.no_kgout and _KGOUT_AVAILABLE:
        ngrok_token = resolve_ngrok_token()
        if ngrok_token:
            os.environ["NGROK_AUTH_TOKEN"] = ngrok_token
            use_kgout = True
        else:
            log("kgout disabled (no ngrok token). "
                "Files will be available in the Kaggle Output tab.")
    elif args.no_kgout:
        log("kgout disabled by --no-kgout flag.")
    else:
        log("kgout disabled (package not installed).")

    # -- build context manager ---------------------------------------------
    if use_kgout:
        try:
            ctx = KgOut("local")
            log("kgout context created - ngrok tunnel will open shortly.")
        except Exception as exc:
            log(f"WARNING: KgOut() failed ({exc}). Falling back to no-kgout mode.")
            use_kgout = False
            from contextlib import nullcontext
            ctx = nullcontext()
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:

        # -- Load model + tokenizer (shared across stages 2/4/5) -----------
        if any(s in args.stages for s in [2, 4, 5]):
            header("LOADING MODEL & TOKENIZER")
            log(f"Loading tokenizer from {MODEL_NAME} ...")
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, trust_remote_code=True, padding_side="left"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token    = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            log(f"Tokenizer ready  (vocab={tokenizer.vocab_size})")

            log(f"Loading base model from {MODEL_NAME} ...")
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            base_model.eval()
            log(f"Model on: {next(base_model.parameters()).device}")
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1e9
                log(f"GPU memory used after model load: {mem:.2f} GB")

        # ==================================================================
        #  STAGE 1 . LOAD MATH TRAIN SPLIT
        # ==================================================================
        train_df = None
        if 1 in args.stages:
            header("STAGE 1 . Loading MATH train split")
            all_splits = []
            for subj in SUBJECTS:
                log(f"Loading subject: {subj} ...")
                ds = load_dataset("EleutherAI/hendrycks_math", subj, split="train")
                log(f"  {subj:<30s}  {len(ds)} problems")
                all_splits.append(ds)
            math_ds  = concatenate_datasets(all_splits)
            train_df = pd.DataFrame({
                "Question": [ex["problem"]  for ex in math_ds],
                "Answer":   [ex["solution"] for ex in math_ds],
                "Level":    [ex["level"]    for ex in math_ds],
                "Type":     [ex["type"]     for ex in math_ds],
            }).reset_index(drop=True)
            log(f"Train set shape: {train_df.shape}")
            log(f"Level distribution:\n{train_df.Level.value_counts().to_string()}")

        # ==================================================================
        #  STAGE 2 . GENERATE RAW CoT TRACES
        # ==================================================================
        if 2 in args.stages:
            header("STAGE 2 . Generating raw CoT traces on MATH train split")
            if train_df is None:
                raise RuntimeError(
                    "Stage 2 requires Stage 1 (train_df). "
                    "Add stage 1 or use --stages 1 2 ..."
                )

            COT_PATH = os.path.join(OUTPUT_DIR, "mathtraincot.jsonl")
            done_ids = set()

            if os.path.exists(COT_PATH) and args.resume:
                with open(COT_PATH) as f:
                    done_ids = {json.loads(l)["id"] for l in f}
                log(f"Resuming - {len(done_ids)}/{len(train_df)} already done")
            else:
                log(f"Starting fresh - {len(train_df)} problems to process")

            remaining_mask = ~train_df.index.isin(done_ids)
            remaining_df   = train_df[remaining_mask].reset_index(drop=True)
            remaining_orig = train_df[remaining_mask].index.tolist()

            if len(remaining_df) == 0:
                log("All CoT traces already exist - skipping inference.")
            else:
                log(f"Running inference on {len(remaining_df)} problems ...")
                _, responses, token_counts = evaluate_batched(
                    remaining_df, method="Original"
                )
                with open(COT_PATH, "a") as f:
                    for li, (resp, tc) in enumerate(zip(responses, token_counts)):
                        orig_idx = remaining_orig[li]
                        row      = train_df.iloc[orig_idx]
                        f.write(json.dumps({
                            "id":         int(orig_idx),
                            "problem":    row["Question"],
                            "answer":     row["Answer"],
                            "fullcot":    resp,
                            "tokencount": tc,
                            "level":      row["Level"],
                            "subject":    row["Type"],
                        }) + "\n")
                log(f"Saved -> {COT_PATH}")

            with open(COT_PATH) as f:
                cot_records = [json.loads(l) for l in f]
            avg_tok = sum(r["tokencount"] for r in cot_records) / len(cot_records)
            log(f"CoT file: {len(cot_records)} records  |  avg tokens: {avg_tok:.1f}")

        # ==================================================================
        #  STAGE 3 . LLMLingua-2 COMPRESSION
        # ==================================================================
        if 3 in args.stages:
            header("STAGE 3 . LLMLingua-2 compression")
            if not _LLMLINGUA_AVAILABLE:
                log("ERROR: llmlingua not installed - skipping Stage 3.")
            else:
                COT_PATH = os.path.join(OUTPUT_DIR, "mathtraincot.jsonl")
                if not os.path.exists(COT_PATH):
                    raise FileNotFoundError(
                        f"CoT file not found: {COT_PATH}\n"
                        "Run Stage 2 first (or add --stages 2 3 ...)."
                    )
                with open(COT_PATH) as f:
                    cot_records = [json.loads(l) for l in f]
                log(f"Loaded {len(cot_records)} CoT records for compression")

                log("Loading LLMLingua-2 on CPU ...")
                compressor = PromptCompressor(
                    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                    use_llmlingua2=True,
                    device_map="cpu",
                )
                log("LLMLingua-2 ready!")

                for ratio in TARGET_RATIOS:
                    out_path = os.path.join(
                        OUTPUT_DIR, f"mathtraincompressedratio{ratio}.jsonl"
                    )
                    if os.path.exists(out_path) and args.resume:
                        with open(out_path) as f:
                            n_exist = sum(1 for _ in f)
                        log(f"ratio={ratio} already exists ({n_exist} records) - skipping.")
                        continue

                    log(f"Compressing {len(cot_records)} samples at ratio={ratio} ...")
                    t0 = time.time()
                    n_errors = 0
                    with open(out_path, "w") as fout:
                        for rec in tqdm(cot_records, desc=f"LLMLingua {ratio}"):
                            try:
                                result     = compressor.compress_prompt(
                                    rec["fullcot"], rate=ratio,
                                    force_tokens=[".", "?", "\n"],
                                )
                                compressed = result["compressed_prompt"]
                            except Exception as exc:
                                n_errors  += 1
                                compressed = rec["fullcot"]
                            fout.write(json.dumps({
                                "id":             rec["id"],
                                "problem":        rec["problem"],
                                "answer":         rec["answer"],
                                "compressedcot":  compressed,
                                "originaltokens": rec["tokencount"],
                                "ratio":          ratio,
                                "level":          rec.get("level", ""),
                                "subject":        rec.get("subject", ""),
                            }) + "\n")
                    elapsed = (time.time() - t0) / 60
                    log(f"ratio={ratio} done in {elapsed:.1f} min "
                        f"(fallbacks={n_errors}) -> {out_path}")

                log("All compression ratios complete!")
                del compressor
                torch.cuda.empty_cache()

        # ==================================================================
        #  STAGE 4 . LoRA TRAINING
        # ==================================================================
        if 4 in args.stages:
            header("STAGE 4 . LoRA fine-tuning")

            class CoTDataset(Dataset):
                def __init__(self, records, tkz, max_length=1024):
                    self.samples = []
                    log(f"  Tokenising {len(records)} training samples ...")
                    for rec in tqdm(records, desc="Tokenising", leave=False):
                        prompt   = make_prompt(rec["problem"], "Original")
                        fulltext = prompt + rec["compressedcot"] + tkz.eos_token
                        enc      = tkz(fulltext, truncation=True,
                                       max_length=max_length, return_tensors="pt")
                        self.samples.append(
                            {k: v.squeeze(0) for k, v in enc.items()}
                        )
                def __len__(self):        return len(self.samples)
                def __getitem__(self, i): return self.samples[i]

            for ratio in TARGET_RATIOS:
                if args.adapter_dir:
                    adapter_dir = os.path.join(
                        args.adapter_dir, f"qwenloraratio{ratio:.1f}"
                    )
                else:
                    adapter_dir = os.path.join(
                        OUTPUT_DIR, f"qwenloraratio{ratio:.1f}"
                    )
                zip_path = adapter_dir + ".zip"

                if os.path.exists(zip_path) and args.resume:
                    log(f"Adapter ratio={ratio} ZIP exists - skipping."); continue
                if os.path.isdir(adapter_dir) and args.resume:
                    log(f"Adapter ratio={ratio} dir exists - skipping."); continue

                print(f"\n{'--'*32}")
                print(f"  LoRA Training  ratio={ratio}  epochs={TRAIN_EPOCHS}")
                print(f"  lr={args.lr}  r={args.lora_r}  alpha={args.lora_alpha}")
                print(f"  output -> {adapter_dir}")
                print(f"{'--'*32}")

                compressed_path = os.path.join(
                    OUTPUT_DIR, f"mathtraincompressedratio{ratio}.jsonl"
                )
                if not os.path.exists(compressed_path):
                    log(f"ERROR: compressed file not found: {compressed_path} - "
                        f"Run Stage 3 for ratio={ratio}. Skipping.")
                    continue

                with open(compressed_path) as f:
                    records = [json.loads(l) for l in f]
                log(f"Loaded {len(records)} compressed training records")

                lora_cfg = LoraConfig(
                    r=args.lora_r, lora_alpha=args.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.05, bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                lora_model = get_peft_model(base_model, lora_cfg)
                lora_model.print_trainable_parameters()

                dataset  = CoTDataset(records, tokenizer, max_length=1024)
                collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=False
                )
                train_args = TrainingArguments(
                    output_dir=os.path.join(OUTPUT_DIR, f"lorackpt{ratio}"),
                    num_train_epochs=TRAIN_EPOCHS,
                    per_device_train_batch_size=TRAIN_BATCH,
                    gradient_accumulation_steps=GRAD_ACCUM,
                    learning_rate=args.lr,
                    warmup_ratio=0.1,
                    lr_scheduler_type="cosine",
                    bf16=True,
                    logging_steps=10,
                    save_strategy="no",
                    report_to="none",
                    dataloader_num_workers=2,
                )
                trainer = Trainer(
                    model=lora_model, args=train_args,
                    train_dataset=dataset, data_collator=collator,
                )
                log(f"Starting trainer for ratio={ratio} ...")
                trainer.train()
                log(f"Training complete for ratio={ratio}")

                os.makedirs(adapter_dir, exist_ok=True)
                lora_model.save_pretrained(adapter_dir)
                tokenizer.save_pretrained(adapter_dir)
                log(f"Adapter saved -> {adapter_dir}")

                shutil.make_archive(adapter_dir, "zip", adapter_dir)
                sz = os.path.getsize(zip_path) / 1e6
                log(f"Adapter ZIP -> {zip_path}  ({sz:.1f} MB)")

                log("Unloading LoRA adapter - restoring base_model ...")
                try:
                    base_model = lora_model.unload()
                    if base_model is None:
                        raise AttributeError("unload() returned None")
                except Exception:
                    base_model = lora_model.base_model.model
                    if hasattr(base_model, "peft_config"):
                        del base_model.peft_config

                del lora_model, trainer, dataset
                torch.cuda.empty_cache()
                base_model.eval()
                log("base_model restored and set to eval mode.")

            log("ALL 3 ADAPTERS TRAINED AND SAVED")

        # ==================================================================
        #  STAGE 5 . MATH-500 EVALUATION
        # ==================================================================
        results_df   = None
        all_tok_dict = {}
        subj_results = {}

        if 5 in args.stages:
            header("STAGE 5 . MATH-500 Evaluation")

            if os.path.exists(args.math500_path):
                log(f"Loading MATH-500 canonical: {args.math500_path}")
                with open(args.math500_path) as f:
                    m500 = [json.loads(l) for l in f]
                test_df = pd.DataFrame({
                    "Question": [x["problem"]       for x in m500],
                    "Answer":   [x["solution"]      for x in m500],
                    "Level":    [x.get("level", "") for x in m500],
                    "Type":     [x.get("type",  "") for x in m500],
                })
                log(f"MATH-500 canonical loaded: {len(test_df)} problems")
            else:
                log(f"Canonical file not found ({args.math500_path}). "
                    "Loading from HuggingFace test split ...")
                test_splits = []
                for subj in SUBJECTS:
                    ds = load_dataset("EleutherAI/hendrycks_math", subj, split="test")
                    test_splits.append(ds)
                test_ds = concatenate_datasets(test_splits)
                test_df = pd.DataFrame({
                    "Question": [ex["problem"]  for ex in test_ds],
                    "Answer":   [ex["solution"] for ex in test_ds],
                    "Level":    [ex["level"]    for ex in test_ds],
                    "Type":     [ex["type"]     for ex in test_ds],
                }).reset_index(drop=True)
                log(f"HuggingFace test split loaded: {len(test_df)} problems")

            results   = []
            done_meth = set()
            CHECKPOINT = os.path.join(OUTPUT_DIR, "tokenskipcheckpoint.json")
            if os.path.exists(CHECKPOINT) and args.resume:
                with open(CHECKPOINT) as f:
                    results = json.load(f).get("results", [])
                done_meth = {r["Method"] for r in results}
                log(f"Checkpoint loaded - {len(done_meth)} methods already done: "
                    f"{sorted(done_meth)}")
            else:
                log("Starting evaluation from scratch.")

            def record_subj_acc(method, responses):
                cb, tb = {}, {}
                for resp, row in zip(responses, test_df.itertuples()):
                    s      = row.Type
                    cb[s]  = cb.get(s, 0) + int(is_correct(resp, row.Answer))
                    tb[s]  = tb.get(s, 0) + 1
                subj_results[method] = {
                    s: round(100 * cb[s] / tb[s], 1) for s in tb
                }
                log(f"  Subject accuracies recorded for {method}")

            def run_method(name, model=None, prompt_override=None):
                if name in done_meth:
                    log(f"[{name}] checkpoint hit - skipping."); return
                log(f"Starting evaluation: {name} ...")
                row, resp, tok = evaluate_batched(
                    test_df,
                    method=prompt_override or name,
                    original_avg_tokens=ORIG_TOKENS,
                    model=model,
                )
                row["Method"] = name
                results.append(row)
                all_tok_dict[name] = tok
                record_subj_acc(name, resp)
                done_meth.add(name)
                save_checkpoint(results)
                log(f"[{name}]  Acc={row['Accuracy']}%  "
                    f"AvgTok={row['Avg Tokens']}  Latency={row['Latency(s)']}s")

            # -- prompt methods --------------------------------------------
            header("  5a . Prompt-engineering methods")
            run_method("Original")
            for pm in ["Concise", "Structured", "StepByStep", "DirectAnswer"]:
                run_method(pm)

            # -- truncation ------------------------------------------------
            header("  5b . Truncation methods")
            for ratio in TARGET_RATIOS:
                mname = f"Truncation{ratio}"
                if mname in done_meth:
                    log(f"[{mname}] checkpoint hit - skipping."); continue

                log(f"Building truncated prompts at ratio={ratio} ...")

                def _trunc_prompt(q, r=ratio):
                    words = PROMPTS["Original"].format(question=q).split()
                    trunc = " ".join(words[:int(len(words) * r)])
                    msg   = [{"role": "user", "content": trunc}]
                    return tokenizer.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True
                    )

                trunc_prompts = [_trunc_prompt(row["Question"])
                                 for _, row in test_df.iterrows()]
                row, resp, tok = evaluate_batched(
                    test_df, method=mname,
                    original_avg_tokens=ORIG_TOKENS,
                    custom_prompts=trunc_prompts,
                )
                row["Ratio"] = ratio
                results.append(row)
                all_tok_dict[mname] = tok
                record_subj_acc(mname, resp)
                done_meth.add(mname)
                save_checkpoint(results)
                log(f"[{mname}]  Acc={row['Accuracy']}%  AvgTok={row['Avg Tokens']}")

            # -- LLMLingua compressed eval ---------------------------------
            header("  5c . LLMLingua compressed evaluation")

            def eval_llmlingua(path, label):
                mname = f"LLMLingua{label}"
                if mname in done_meth:
                    log(f"[{mname}] checkpoint hit - skipping."); return
                log(f"Loading compressed records: {path} ...")
                with open(path) as f:
                    recs = [json.loads(l) for l in f]
                compressed_prompts = []
                for rec in recs:
                    msg = [{"role": "user", "content": rec["compressedcot"]}]
                    compressed_prompts.append(
                        tokenizer.apply_chat_template(
                            msg, tokenize=False, add_generation_prompt=True
                        )
                    )
                tmp_df = pd.DataFrame({
                    "Question": [r["problem"] for r in recs],
                    "Answer":   [r["answer"]  for r in recs],
                })
                row, resp, tok = evaluate_batched(
                    tmp_df, method=mname,
                    original_avg_tokens=ORIG_TOKENS,
                    custom_prompts=compressed_prompts,
                )
                row["Ratio"] = float(label)
                results.append(row)
                all_tok_dict[mname] = tok
                record_subj_acc(mname, resp)   # was missing in original
                done_meth.add(mname)
                save_checkpoint(results)
                log(f"[{mname}]  Acc={row['Accuracy']}%  AvgTok={row['Avg Tokens']}")

            for ratio in TARGET_RATIOS:
                p = os.path.join(
                    OUTPUT_DIR, f"mathtraincompressedratio{ratio}.jsonl"
                )
                if os.path.exists(p):
                    eval_llmlingua(p, str(ratio))
                else:
                    log(f"Compressed file not found for ratio={ratio} - skipping "
                        "LLMLingua eval. (Run Stage 3 to generate it.)")

            # -- LoRA adapter evaluations ----------------------------------
            header("  5d . LoRA adapter evaluations")
            for ratio in TARGET_RATIOS:
                if args.adapter_dir:
                    adapter_dir = os.path.join(
                        args.adapter_dir, f"qwenloraratio{ratio:.1f}"
                    )
                else:
                    adapter_dir = os.path.join(
                        OUTPUT_DIR, f"qwenloraratio{ratio:.1f}"
                    )
                zip_path = adapter_dir + ".zip"

                if not os.path.isdir(adapter_dir) and os.path.exists(zip_path):
                    log(f"Unpacking adapter ZIP: {zip_path} ...")
                    shutil.unpack_archive(zip_path, adapter_dir)

                if not os.path.isdir(adapter_dir):
                    log(f"[SKIP] adapter not found for ratio={ratio}: {adapter_dir}")
                    log("  (Run Stage 4, or supply --adapter-dir with pre-trained adapters)")
                    continue

                log(f"Loading LoRA adapter from {adapter_dir} ...")
                lora_model = PeftModel.from_pretrained(base_model, adapter_dir)
                lora_model.eval()
                log(f"LoRA adapter ratio={ratio} loaded")

                run_method(f"LoRA{ratio}", model=lora_model)

                for suffix, prompt in [("Guided", "Structured"),
                                       ("Soft",   "Concise")]:
                    mname = f"LoRA{suffix}{ratio}"
                    if mname in done_meth:
                        log(f"[{mname}] checkpoint hit - skipping."); continue
                    log(f"Evaluating {mname} ...")
                    row, resp, tok = evaluate_batched(
                        test_df, method=prompt,
                        original_avg_tokens=ORIG_TOKENS,
                        model=lora_model,
                    )
                    row["Method"] = mname
                    row["Ratio"]  = ratio
                    results.append(row)
                    all_tok_dict[mname] = tok
                    record_subj_acc(mname, resp)
                    done_meth.add(mname)
                    save_checkpoint(results)
                    log(f"[{mname}]  Acc={row['Accuracy']}%  AvgTok={row['Avg Tokens']}")

                del lora_model
                torch.cuda.empty_cache()
                log(f"LoRA ratio={ratio} done - GPU memory freed.")

            # -- build results DataFrame -----------------------------------
            results_df = pd.DataFrame(results)
            results_df["Token Savings"]    = (
                (ORIG_TOKENS - results_df["Avg Tokens"]) / ORIG_TOKENS * 100
            ).round(2)
            results_df["Accuracy Drop"]    = (
                results_df["Accuracy"] - ORIG_ACC
            ).round(2)
            results_df["Efficiency Score"] = (
                results_df["Accuracy"] / results_df["Avg Tokens"] * 100
            ).round(4)

            base_csv  = os.path.join(OUTPUT_DIR, "tokenskipresults.csv")
            final_csv = os.path.join(OUTPUT_DIR, "tokenskipresultsfinal.csv")
            results_df[~results_df.Method.str.startswith("LoRA")].to_csv(
                base_csv, index=False
            )
            results_df.to_csv(final_csv, index=False)
            log(f"CSVs saved:\n    {base_csv}\n    {final_csv}")

            summary_cols = ["Method", "Accuracy", "Avg Tokens",
                            "Token Savings", "Latency(s)"]
            print("\n" + results_df[summary_cols].to_string(index=False))

        # ==================================================================
        #  STAGE 6 . GENERATE ALL 8 FIGURES
        # ==================================================================
        if 6 in args.stages:
            if results_df is None:
                final_csv = os.path.join(OUTPUT_DIR, "tokenskipresultsfinal.csv")
                if not os.path.exists(final_csv):
                    raise FileNotFoundError(
                        f"Results CSV not found: {final_csv}\n"
                        "Run Stage 5 first (or use --stages 5 6 ...)."
                    )
                results_df = pd.read_csv(final_csv)
                log(f"Loaded results from {final_csv}  ({len(results_df)} rows)")
                if not all_tok_dict:
                    log("Note: per-method token distributions unavailable "
                        "(Fig 3 will be skipped in plots-only mode).")

            Plotter(results_df).run_all(
                all_token_counts=all_tok_dict if all_tok_dict else None,
                results_by_subject=subj_results if subj_results else None,
            )

        # ==================================================================
        #  STAGE 7 . ZIP EVERYTHING
        # ==================================================================
        if 7 in args.stages:
            header("STAGE 7 . Zipping all outputs")
            ZIP_FILE = os.path.join(OUTPUT_DIR, "tokenskip_full_outputs.zip")
            n_files  = 0
            with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(OUTPUT_DIR):
                    dirs[:] = [d for d in dirs
                                if not d.startswith("lorackpt")]
                    for fname in sorted(files):
                        if fname == "tokenskip_full_outputs.zip":
                            continue
                        fpath   = os.path.join(root, fname)
                        arcname = os.path.relpath(fpath, OUTPUT_DIR)
                        zf.write(fpath, arcname)
                        n_files += 1
                        log(f"  added to ZIP: {arcname}")

            sz = os.path.getsize(ZIP_FILE) / 1e6
            log(f"Master ZIP -> {ZIP_FILE}  ({sz:.1f} MB, {n_files} files)")

        # ==================================================================
        #  FINAL MANIFEST
        # ==================================================================
        print("\n" + "="*65 + "\n  OUTPUT MANIFEST\n" + "="*65)
        total_size = 0
        for root, _, files in os.walk(OUTPUT_DIR):
            for fname in sorted(files):
                fpath  = os.path.join(root, fname)
                sz_mb  = os.path.getsize(fpath) / 1e6
                total_size += sz_mb
                relpath = os.path.relpath(fpath, OUTPUT_DIR)
                print(f"  {relpath:<55s}  {sz_mb:7.2f} MB")
        print(f"\n  {'TOTAL':<55s}  {total_size:7.2f} MB")
        print("="*65)
        print("\n  ALL STAGES COMPLETE")

        if use_kgout:
            print("  Every file above is available at your ngrok URL.")
        else:
            print("  Files are available in the /kaggle/working Output tab.")
            print("  Download tokenskip_full_outputs.zip for the full bundle.")
            try:
                from IPython.display import FileLinks, display
                display(FileLinks(OUTPUT_DIR))
            except Exception:
                pass


# ======================================================================
#  ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    run_pipeline()
