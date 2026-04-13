"""
TokenSkip — Scaling Law & Cross-Benchmark Analysis
====================================================
All 5 model sizes (0.5B, 1.5B, 3B, 7B, 14B) × 2 benchmarks (GSM8K, MATH-500)

Generates:
  Fig 1:  Scaling Law — Baseline accuracy vs model size (both benchmarks)
  Fig 2:  Scaling Law — LoRA best accuracy vs model size by γ
  Fig 3:  Compression Collapse Threshold — accuracy drop vs model size per γ
  Fig 4:  Pareto Frontiers — Accuracy vs Token Savings (per benchmark, overlay sizes)
  Fig 5:  LoRA Triplet Comparison across model sizes
  Fig 6:  Method Family Heatmap — accuracy across sizes (GSM8K + MATH-500)
  Fig 7:  Token Savings Efficiency — savings per accuracy-point-lost
  Fig 8:  Cross-Benchmark Correlation — GSM8K acc vs MATH-500 acc per method
  Fig 9:  Adaptive vs Best-Fixed γ comparison across sizes
  Fig 10: Prompt-based methods scaling comparison

  Table 1: Master results CSV (all methods × all sizes × both benchmarks)
  Table 2: Scaling law summary CSV
  Table 3: Best Pareto-optimal methods per size
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ======================================================================
#  CONFIG
# ======================================================================
OUTPUT_DIR = "./final_analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})

COLORS = {
    "orig":        "#2196F3",
    "prompt":      "#9C27B0",
    "trunc":       "#FF9800",
    "lora_orig":   "#4CAF50",
    "lora_guided": "#E91E63",
    "lora_soft":   "#00BCD4",
    "llmlingua":   "#795548",
    "adaptive":    "#F44336",
}

MODEL_SIZES = [0.5, 1.5, 3, 7, 14]
MODEL_LABELS = ["0.5B", "1.5B", "3B", "7B", "14B"]
MODEL_PARAMS = {0.5: 0.5e9, 1.5: 1.5e9, 3: 3e9, 7: 7e9, 14: 14e9}

GAMMAS = [0.5, 0.6, 0.7, 0.8, 0.9]

def log(msg):
    print(f"  → {msg}")

def save_fig(name):
    p = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"Saved {name} ({os.path.getsize(p)/1e3:.0f} KB)")

# ======================================================================
#  RAW DATA — All results pasted in
# ======================================================================

def build_df(rows):
    """Parse list of (method, acc, avg_tokens, token_savings, latency) tuples."""
    return pd.DataFrame(rows, columns=["Method", "Accuracy", "Avg Tokens", "Token Savings", "Latency(s)"])

# ---------- MATH-500 ----------

MATH500_14B = build_df([
    ("Original",       65.0, 581.68,  0.00, 1.499),
    ("BeConcise",      66.8, 505.55, 13.09, 1.483),
    ("OnlyNumbers",    26.4,  29.45, 94.94, 0.827),
    ("AbbreWords",     66.2, 407.31, 29.98, 1.474),
    ("LC-Prompt",      65.4, 338.03, 41.89, 1.392),
    ("Truncation0.5",  39.6, 442.75, 23.88, 0.554),
    ("Truncation0.6",  48.6, 488.82, 15.96, 0.710),
    ("Truncation0.7",  55.4, 523.22, 10.05, 0.883),
    ("Truncation0.8",  59.6, 549.30,  5.57, 1.074),
    ("Truncation0.9",  63.6, 568.39,  2.28, 1.277),
    ("LLMLingua0.5",   33.2, 547.51,  5.87, 1.379),
    ("LLMLingua0.6",   48.8, 572.15,  1.64, 1.406),
    ("LLMLingua0.7",   56.8, 570.05,  2.00, 1.431),
    ("LLMLingua0.8",   61.2, 581.85, -0.03, 1.451),
    ("LLMLingua0.9",   64.8, 581.47,  0.04, 1.471),
    ("LoRA0.5",        49.4, 358.83, 38.31, 0.567),
    ("LoRAGuided0.5",  51.8, 349.95, 39.84, 0.570),
    ("LoRASoft0.5",    53.8, 350.51, 39.74, 0.582),
    ("LoRA0.6",        53.6, 411.16, 29.32, 0.726),
    ("LoRAGuided0.6",  56.0, 400.62, 31.13, 0.730),
    ("LoRASoft0.6",    58.4, 390.60, 32.85, 0.744),
    ("LoRA0.7",        57.6, 466.11, 19.87, 0.902),
    ("LoRAGuided0.7",  60.6, 434.63, 25.28, 0.902),
    ("LoRASoft0.7",    62.2, 421.87, 27.47, 0.921),
    ("LoRA0.8",        61.2, 499.63, 14.11, 1.094),
    ("LoRAGuided0.8",  62.8, 471.00, 19.03, 1.099),
    ("LoRASoft0.8",    64.6, 444.45, 23.59, 1.118),
    ("LoRA0.9",        61.8, 536.29,  7.80, 1.302),
    ("LoRAGuided0.9",  63.6, 497.00, 14.56, 1.306),
    ("LoRASoft0.9",    64.4, 462.06, 20.56, 1.326),
    ("LoRAAdaptive",   61.6, 495.75, 14.77, 1.277),
])

MATH500_7B = build_df([
    ("Original",       61.0, 571.68,  0.00, 0.657),
    ("BeConcise",      61.2, 524.54,  8.25, 0.654),
    ("OnlyNumbers",    61.4, 446.35, 21.92, 0.660),
    ("AbbreWords",     60.8, 452.25, 20.89, 0.655),
    ("LC-Prompt",      61.0, 422.73, 26.05, 0.619),
    ("Truncation0.5",  40.6, 438.67, 23.27, 0.251),
    ("Truncation0.6",  47.6, 483.33, 15.45, 0.320),
    ("Truncation0.7",  52.2, 516.96,  9.57, 0.395),
    ("Truncation0.8",  55.8, 542.31,  5.14, 0.475),
    ("Truncation0.9",  59.4, 559.90,  2.06, 0.563),
    ("LLMLingua0.5",   31.2, 517.79,  9.43, 0.598),
    ("LLMLingua0.6",   44.8, 543.67,  4.90, 0.621),
    ("LLMLingua0.7",   53.0, 552.68,  3.32, 0.632),
    ("LLMLingua0.8",   59.6, 564.02,  1.34, 0.639),
    ("LLMLingua0.9",   61.8, 561.74,  1.74, 0.647),
    ("LoRA0.5",        43.8, 337.21, 41.01, 0.257),
    ("LoRAGuided0.5",  44.8, 321.72, 43.72, 0.258),
    ("LoRASoft0.5",    48.4, 318.16, 44.35, 0.262),
    ("LoRA0.6",        51.4, 391.12, 31.58, 0.326),
    ("LoRAGuided0.6",  50.8, 375.75, 34.27, 0.328),
    ("LoRASoft0.6",    50.4, 361.82, 36.71, 0.333),
    ("LoRA0.7",        56.4, 445.87, 22.01, 0.401),
    ("LoRAGuided0.7",  56.6, 431.25, 24.56, 0.402),
    ("LoRASoft0.7",    55.4, 411.99, 27.93, 0.410),
    ("LoRA0.8",        57.8, 494.67, 13.47, 0.485),
    ("LoRAGuided0.8",  60.4, 476.82, 16.59, 0.485),
    ("LoRASoft0.8",    59.2, 462.89, 19.03, 0.492),
    ("LoRA0.9",        62.2, 527.47,  7.73, 0.573),
    ("LoRAGuided0.9",  61.0, 509.24, 10.92, 0.573),
    ("LoRASoft0.9",    60.8, 489.86, 14.31, 0.576),
    ("LoRAAdaptive",   60.2, 483.09, 15.50, 0.562),
])

MATH500_3B = build_df([
    ("Original",       55.6, 580.37,  0.00, 0.507),
    ("BeConcise",      54.0, 516.73, 10.97, 0.512),
    ("OnlyNumbers",    55.8, 518.87, 10.60, 0.514),
    ("AbbreWords",     54.6, 566.21,  2.44, 0.515),
    ("LC-Prompt",      53.2, 507.39, 12.57, 0.520),
    ("Truncation0.5",  34.8, 439.54, 24.27, 0.209),
    ("Truncation0.6",  43.4, 486.55, 16.17, 0.262),
    ("Truncation0.7",  47.8, 521.54, 10.14, 0.318),
    ("Truncation0.8",  51.8, 548.22,  5.54, 0.379),
    ("Truncation0.9",  54.0, 566.92,  2.32, 0.444),
    ("LLMLingua0.5",   22.2, 508.51, 12.38, 0.471),
    ("LLMLingua0.6",   36.8, 536.23,  7.61, 0.491),
    ("LLMLingua0.7",   43.2, 553.48,  4.63, 0.495),
    ("LLMLingua0.8",   51.0, 574.49,  1.01, 0.499),
    ("LLMLingua0.9",   53.8, 569.23,  1.92, 0.503),
    ("LoRA0.5",        33.4, 348.65, 39.93, 0.211),
    ("LoRAGuided0.5",  34.6, 335.75, 42.15, 0.210),
    ("LoRASoft0.5",    34.8, 330.99, 42.97, 0.210),
    ("LoRA0.6",        39.0, 406.66, 29.93, 0.263),
    ("LoRAGuided0.6",  40.6, 391.14, 32.61, 0.259),
    ("LoRASoft0.6",    40.6, 380.02, 34.52, 0.263),
    ("LoRA0.7",        45.8, 462.66, 20.28, 0.318),
    ("LoRAGuided0.7",  48.2, 435.91, 24.89, 0.315),
    ("LoRASoft0.7",    43.4, 424.93, 26.78, 0.321),
    ("LoRA0.8",        49.2, 502.43, 13.43, 0.379),
    ("LoRAGuided0.8",  52.4, 473.01, 18.50, 0.379),
    ("LoRASoft0.8",    50.6, 453.88, 21.79, 0.383),
    ("LoRA0.9",        54.2, 538.74,  7.17, 0.446),
    ("LoRAGuided0.9",  54.6, 499.20, 13.99, 0.445),
    ("LoRASoft0.9",    56.2, 479.93, 17.31, 0.450),
    ("LoRAAdaptive",   47.2, 506.69, 12.70, 0.446),
])

MATH500_1_5B = build_df([
    ("Original",       43.6, 572.14,  0.00, 0.339),
    ("BeConcise",      44.4, 554.56,  3.07, 0.339),
    ("OnlyNumbers",    41.8, 567.14,  0.87, 0.341),
    ("AbbreWords",     45.0, 577.18, -0.88, 0.341),
    ("LC-Prompt",      39.6, 541.19,  5.41, 0.343),
    ("Truncation0.5",  31.6, 434.52, 24.05, 0.147),
    ("Truncation0.6",  37.2, 479.22, 16.24, 0.180),
    ("Truncation0.7",  40.0, 512.07, 10.50, 0.216),
    ("Truncation0.8",  42.0, 537.41,  6.07, 0.254),
    ("Truncation0.9",  43.2, 557.12,  2.63, 0.297),
    ("LLMLingua0.5",   13.2, 524.11,  8.39, 0.319),
    ("LLMLingua0.6",   24.6, 550.79,  3.73, 0.323),
    ("LLMLingua0.7",   33.0, 568.65,  0.61, 0.330),
    ("LLMLingua0.8",   38.6, 575.24, -0.54, 0.333),
    ("LLMLingua0.9",   42.8, 565.60,  1.14, 0.334),
    ("LoRA0.5",        24.0, 362.02, 36.73, 0.144),
    ("LoRAGuided0.5",  23.8, 356.64, 37.67, 0.145),
    ("LoRASoft0.5",    19.4, 352.96, 38.31, 0.147),
    ("LoRA0.6",        29.8, 405.47, 29.13, 0.177),
    ("LoRAGuided0.6",  29.2, 392.27, 31.44, 0.177),
    ("LoRASoft0.6",    27.6, 393.17, 31.28, 0.178),
    ("LoRA0.7",        32.6, 447.82, 21.73, 0.214),
    ("LoRAGuided0.7",  31.0, 444.77, 22.26, 0.213),
    ("LoRASoft0.7",    29.8, 438.45, 23.37, 0.215),
    ("LoRA0.8",        36.8, 495.43, 13.41, 0.252),
    ("LoRAGuided0.8",  34.6, 478.85, 16.31, 0.256),
    ("LoRASoft0.8",    37.0, 479.02, 16.28, 0.259),
    ("LoRA0.9",        37.6, 521.08,  8.92, 0.295),
    ("LoRAGuided0.9",  36.4, 508.11, 11.19, 0.298),
    ("LoRASoft0.9",    35.0, 507.00, 11.39, 0.301),
    ("LoRAAdaptive",   35.4, 492.89, 13.85, 0.299),
])

MATH500_0_5B = build_df([
    ("Original",       25.8, 553.48,  0.00, 0.252),
    ("BeConcise",      25.2, 550.22,  0.59, 0.250),
    ("OnlyNumbers",    23.0, 535.54,  3.24, 0.251),
    ("AbbreWords",     24.8, 550.61,  0.52, 0.250),
    ("LC-Prompt",      20.8, 582.87, -5.31, 0.256),
    ("Truncation0.5",  20.8, 424.42, 23.32, 0.127),
    ("Truncation0.6",  23.0, 467.57, 15.52, 0.151),
    ("Truncation0.7",  24.6, 500.44,  9.58, 0.174),
    ("Truncation0.8",  25.4, 523.95,  5.34, 0.195),
    ("Truncation0.9",  25.8, 540.66,  2.32, 0.223),
    ("LLMLingua0.5",    9.0, 516.59,  6.67, 0.245),
    ("LLMLingua0.6",   14.0, 538.92,  2.63, 0.256),
    ("LLMLingua0.7",   17.8, 555.25, -0.32, 0.252),
    ("LLMLingua0.8",   21.6, 578.08, -4.44, 0.251),
    ("LLMLingua0.9",   24.4, 585.14, -5.72, 0.252),
    ("LoRA0.5",        13.8, 397.37, 28.21, 0.122),
    ("LoRAGuided0.5",  15.2, 378.21, 31.67, 0.122),
    ("LoRASoft0.5",    12.6, 401.02, 27.55, 0.123),
    ("LoRA0.6",        16.8, 429.02, 22.49, 0.146),
    ("LoRAGuided0.6",  15.2, 414.21, 25.16, 0.150),
    ("LoRASoft0.6",    14.4, 440.02, 20.50, 0.151),
    ("LoRA0.7",        16.2, 460.78, 16.75, 0.175),
    ("LoRAGuided0.7",  14.8, 450.16, 18.67, 0.176),
    ("LoRASoft0.7",    12.8, 477.95, 13.65, 0.175),
    ("LoRA0.8",        17.4, 472.11, 14.70, 0.197),
    ("LoRAGuided0.8",  15.8, 479.53, 13.36, 0.196),
    ("LoRASoft0.8",    14.8, 493.64, 10.81, 0.201),
    ("LoRA0.9",        18.2, 495.20, 10.53, 0.225),
    ("LoRAGuided0.9",  16.8, 495.00, 10.57, 0.225),
    ("LoRASoft0.9",    14.8, 514.27,  7.08, 0.229),
    ("LoRAAdaptive",   17.8, 498.24,  9.98, 0.227),
])

# ---------- GSM8K ----------

GSM8K_14B = build_df([
    ("Original",       92.42, 314.27,  0.00, 0.475),
    ("BeConcise",      94.39, 227.71, 27.54, 0.393),
    ("OnlyNumbers",    37.45,   8.97, 97.15, 0.057),
    ("AbbreWords",     91.66, 143.07, 54.48, 0.263),
    ("LC-Prompt",      92.34, 107.90, 65.67, 0.200),
    ("Truncation0.5",  33.36, 245.34, 21.93, 0.193),
    ("Truncation0.6",  54.97, 276.53, 12.01, 0.242),
    ("Truncation0.7",  73.16, 295.92,  5.84, 0.296),
    ("Truncation0.8",  84.15, 306.61,  2.44, 0.353),
    ("Truncation0.9",  90.45, 311.83,  0.78, 0.415),
    ("LLMLingua0.5",   49.51, 300.69,  4.32, 0.443),
    ("LLMLingua0.6",   64.67, 304.90,  2.98, 0.453),
    ("LLMLingua0.7",   77.48, 312.28,  0.63, 0.453),
    ("LLMLingua0.8",   87.41, 313.31,  0.31, 0.465),
    ("LLMLingua0.9",   91.21, 314.24,  0.01, 0.465),
    ("LoRA0.5",        89.76, 174.83, 44.37, 0.321),
    ("LoRAGuided0.5",  90.83, 168.09, 46.51, 0.280),
    ("LoRASoft0.5",    92.49, 165.33, 47.39, 0.290),
    ("LoRA0.6",        92.42, 201.90, 35.76, 0.376),
    ("LoRAGuided0.6",  92.27, 189.07, 39.84, 0.325),
    ("LoRASoft0.6",    92.12, 182.41, 41.96, 0.330),
    ("LoRA0.7",        93.18, 237.17, 24.53, 0.412),
    ("LoRAGuided0.7",  92.57, 219.22, 30.24, 0.377),
    ("LoRASoft0.7",    92.57, 196.64, 37.43, 0.326),
    ("LoRA0.8",        93.93, 268.13, 14.68, 0.441),
    ("LoRAGuided0.8",  93.03, 245.39, 21.92, 0.405),
    ("LoRASoft0.8",    93.40, 205.91, 34.48, 0.359),
    ("LoRA0.9",        93.71, 286.81,  8.74, 0.468),
    ("LoRAGuided0.9",  93.33, 250.93, 20.15, 0.402),
    ("LoRASoft0.9",    93.63, 211.34, 32.75, 0.373),
])

GSM8K_7B = build_df([
    ("Original",       90.45, 299.80,  0.00, 0.215),
    ("BeConcise",      90.22, 231.62, 22.74, 0.186),
    ("OnlyNumbers",    89.46, 180.23, 39.88, 0.164),
    ("AbbreWords",     88.86, 181.51, 39.46, 0.179),
    ("LC-Prompt",      89.54, 178.45, 40.48, 0.148),
    ("Truncation0.5",  39.12, 241.47, 19.46, 0.094),
    ("Truncation0.6",  61.49, 268.74, 10.36, 0.116),
    ("Truncation0.7",  75.51, 285.11,  4.90, 0.141),
    ("Truncation0.8",  84.69, 293.88,  1.97, 0.166),
    ("Truncation0.9",  88.63, 297.97,  0.61, 0.192),
    ("LLMLingua0.5",   41.39, 287.21,  4.20, 0.208),
    ("LLMLingua0.6",   54.28, 301.36, -0.52, 0.209),
    ("LLMLingua0.7",   68.61, 299.05,  0.25, 0.209),
    ("LLMLingua0.8",   81.80, 303.56, -1.25, 0.209),
    ("LLMLingua0.9",   87.79, 308.00, -2.74, 0.216),
    ("LoRA0.5",        83.78, 164.37, 45.17, 0.142),
    ("LoRAGuided0.5",  83.47, 154.84, 48.35, 0.155),
    ("LoRASoft0.5",    82.87, 153.34, 48.85, 0.138),
    ("LoRA0.6",        86.50, 186.78, 37.70, 0.162),
    ("LoRAGuided0.6",  87.19, 174.05, 41.94, 0.156),
    ("LoRASoft0.6",    85.52, 164.80, 45.03, 0.143),
    ("LoRA0.7",        88.55, 225.19, 24.89, 0.187),
    ("LoRAGuided0.7",  88.86, 209.69, 30.06, 0.167),
    ("LoRASoft0.7",    87.41, 192.70, 35.72, 0.163),
    ("LoRA0.8",        89.92, 254.88, 14.98, 0.202),
    ("LoRAGuided0.8",  89.69, 238.92, 20.31, 0.183),
    ("LoRASoft0.8",    88.93, 225.88, 24.66, 0.183),
    ("LoRA0.9",        89.99, 274.06,  8.59, 0.209),
    ("LoRAGuided0.9",  90.45, 254.03, 15.27, 0.195),
    ("LoRASoft0.9",    89.54, 237.92, 20.64, 0.197),
])

GSM8K_3B = build_df([
    ("Original",       82.79, 318.50,  0.00, 0.187),
    ("BeConcise",      81.88, 242.61, 23.83, 0.168),
    ("OnlyNumbers",    84.00, 200.53, 37.04, 0.153),
    ("AbbreWords",     83.09, 294.70,  7.47, 0.182),
    ("LC-Prompt",      82.18, 225.34, 29.25, 0.155),
    ("Truncation0.5",  31.69, 244.63, 23.19, 0.089),
    ("Truncation0.6",  49.66, 276.76, 13.11, 0.107),
    ("Truncation0.7",  65.88, 297.74,  6.52, 0.125),
    ("Truncation0.8",  75.89, 309.51,  2.82, 0.144),
    ("Truncation0.9",  79.83, 315.41,  0.97, 0.166),
    ("LLMLingua0.5",   31.69, 287.65,  9.69, 0.181),
    ("LLMLingua0.6",   45.03, 308.84,  3.03, 0.185),
    ("LLMLingua0.7",   57.47, 316.95,  0.49, 0.184),
    ("LLMLingua0.8",   72.93, 318.06,  0.14, 0.183),
    ("LLMLingua0.9",   78.77, 319.71, -0.38, 0.185),
    ("LoRA0.5",        73.69, 173.49, 45.53, 0.143),
    ("LoRAGuided0.5",  73.69, 164.25, 48.43, 0.139),
    ("LoRASoft0.5",    72.18, 157.89, 50.43, 0.145),
    ("LoRA0.6",        78.54, 216.95, 31.88, 0.158),
    ("LoRAGuided0.6",  78.24, 200.21, 37.14, 0.144),
    ("LoRASoft0.6",    74.53, 175.10, 45.02, 0.127),
    ("LoRA0.7",        82.41, 251.50, 21.04, 0.164),
    ("LoRAGuided0.7",  80.44, 235.98, 25.91, 0.168),
    ("LoRASoft0.7",    79.68, 208.71, 34.47, 0.149),
    ("LoRA0.8",        81.96, 274.64, 13.77, 0.174),
    ("LoRAGuided0.8",  82.64, 258.78, 18.75, 0.167),
    ("LoRASoft0.8",    81.27, 238.09, 25.25, 0.163),
    ("LoRA0.9",        84.08, 290.14,  8.90, 0.177),
    ("LoRAGuided0.9",  83.17, 272.34, 14.49, 0.174),
    ("LoRASoft0.9",    82.79, 252.75, 20.64, 0.172),
])

GSM8K_1_5B = build_df([
    ("Original",       71.04, 316.28,  0.00, 0.141),
    ("BeConcise",      69.29, 309.22,  2.23, 0.137),
    ("OnlyNumbers",    71.65, 312.51,  1.19, 0.135),
    ("AbbreWords",     69.52, 315.27,  0.32, 0.139),
    ("LC-Prompt",      63.00, 254.39, 19.57, 0.138),
    ("Truncation0.5",  29.72, 244.85, 22.58, 0.069),
    ("Truncation0.6",  47.01, 275.82, 12.79, 0.082),
    ("Truncation0.7",  59.06, 295.90,  6.44, 0.099),
    ("Truncation0.8",  65.43, 307.11,  2.90, 0.109),
    ("Truncation0.9",  70.05, 313.22,  0.97, 0.126),
    ("LLMLingua0.5",   22.06, 290.37,  8.19, 0.138),
    ("LLMLingua0.6",   29.49, 302.35,  4.40, 0.139),
    ("LLMLingua0.7",   43.14, 315.91,  0.12, 0.139),
    ("LLMLingua0.8",   58.76, 319.87, -1.14, 0.140),
    ("LLMLingua0.9",   67.93, 313.25,  0.96, 0.135),
    ("LoRA0.5",        50.87, 180.71, 42.86, 0.128),
    ("LoRAGuided0.5",  50.80, 176.68, 44.14, 0.114),
    ("LoRASoft0.5",    53.90, 172.17, 45.56, 0.107),
    ("LoRA0.6",        56.56, 210.00, 33.60, 0.123),
    ("LoRAGuided0.6",  57.39, 198.94, 37.10, 0.119),
    ("LoRASoft0.6",    55.04, 199.57, 36.90, 0.118),
    ("LoRA0.7",        61.94, 249.18, 21.22, 0.135),
    ("LoRAGuided0.7",  61.11, 232.96, 26.34, 0.126),
    ("LoRASoft0.7",    60.27, 226.84, 28.28, 0.126),
    ("LoRA0.8",        66.49, 274.62, 13.17, 0.135),
    ("LoRAGuided0.8",  66.03, 261.15, 17.43, 0.134),
    ("LoRASoft0.8",    66.19, 246.68, 22.01, 0.128),
    ("LoRA0.9",        66.49, 298.40,  5.65, 0.136),
    ("LoRAGuided0.9",  67.70, 282.23, 10.77, 0.136),
    ("LoRASoft0.9",    64.22, 266.38, 15.78, 0.135),
])

GSM8K_0_5B = build_df([
    ("Original",       43.14, 302.48,  0.00, 0.122),
    ("BeConcise",      43.29, 296.53,  1.97, 0.124),
    ("OnlyNumbers",    43.75, 292.85,  3.18, 0.121),
    ("AbbreWords",     44.35, 296.44,  2.00, 0.123),
    ("LC-Prompt",      40.41, 320.29, -5.89, 0.123),
    ("Truncation0.5",  24.87, 240.82, 20.38, 0.062),
    ("Truncation0.6",  34.42, 268.08, 11.37, 0.073),
    ("Truncation0.7",  39.42, 284.81,  5.84, 0.085),
    ("Truncation0.8",  42.08, 294.19,  2.74, 0.098),
    ("Truncation0.9",  42.53, 299.43,  1.01, 0.110),
    ("LLMLingua0.5",   10.16, 273.18,  9.69, 0.122),
    ("LLMLingua0.6",   14.56, 289.34,  4.34, 0.123),
    ("LLMLingua0.7",   20.85, 295.23,  2.40, 0.121),
    ("LLMLingua0.8",   32.90, 305.48, -0.99, 0.118),
    ("LLMLingua0.9",   39.50, 307.01, -1.50, 0.120),
    ("LoRA0.5",        32.98, 229.06, 24.27, 0.114),
    ("LoRAGuided0.5",  26.91, 193.90, 35.90, 0.114),
    ("LoRASoft0.5",    28.35, 216.82, 28.32, 0.116),
    ("LoRA0.6",        34.72, 233.51, 22.80, 0.114),
    ("LoRAGuided0.6",  30.63, 198.93, 34.23, 0.114),
    ("LoRASoft0.6",    28.51, 220.62, 27.06, 0.120),
    ("LoRA0.7",        35.94, 234.60, 22.44, 0.120),
    ("LoRAGuided0.7",  30.33, 199.16, 34.16, 0.113),
    ("LoRASoft0.7",    27.07, 225.42, 25.48, 0.120),
    ("LoRA0.8",        37.45, 240.41, 20.52, 0.117),
    ("LoRAGuided0.8",  34.50, 209.96, 30.59, 0.114),
    ("LoRASoft0.8",    28.13, 222.03, 26.60, 0.112),
    ("LoRA0.9",        39.35, 273.57,  9.56, 0.114),
    ("LoRAGuided0.9",  36.77, 232.97, 22.98, 0.111),
    ("LoRASoft0.9",    30.10, 216.30, 28.49, 0.111),
])

# ======================================================================
#  Organize into dicts for easy iteration
# ======================================================================

MATH500 = {0.5: MATH500_0_5B, 1.5: MATH500_1_5B, 3: MATH500_3B, 7: MATH500_7B, 14: MATH500_14B}
GSM8K   = {0.5: GSM8K_0_5B,   1.5: GSM8K_1_5B,   3: GSM8K_3B,   7: GSM8K_7B,   14: GSM8K_14B}

BENCHMARKS = {"GSM8K": GSM8K, "MATH-500": MATH500}

# ======================================================================
#  HELPER: extract method info
# ======================================================================

def get_acc(data, method):
    """Get accuracy for a specific method from a DataFrame."""
    row = data[data.Method == method]
    return row.Accuracy.values[0] if len(row) > 0 else np.nan

def get_best_lora(data, gamma, variant="best"):
    """Get best LoRA accuracy at given gamma across all 3 variants."""
    methods = [f"LoRA{gamma}", f"LoRAGuided{gamma}", f"LoRASoft{gamma}"]
    accs = [get_acc(data, m) for m in methods]
    accs = [a for a in accs if not np.isnan(a)]
    return max(accs) if accs else np.nan

def get_lora_variant_acc(data, gamma, variant):
    """Get specific LoRA variant accuracy."""
    prefix = {"lora": "LoRA", "guided": "LoRAGuided", "soft": "LoRASoft"}[variant]
    return get_acc(data, f"{prefix}{gamma}")


# ======================================================================
#  FIGURE 1: Scaling Law — Baseline + Best LoRA per benchmark
# ======================================================================

def fig1_scaling_law_baselines():
    print("\n=== Fig 1: Scaling Law — Baselines ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for idx, (bname, bdata) in enumerate(BENCHMARKS.items()):
        ax = axes[idx]
        # Baseline
        orig = [get_acc(bdata[s], "Original") for s in MODEL_SIZES]
        ax.plot(MODEL_LABELS, orig, "o-", color=COLORS["orig"], lw=2.5, ms=9,
                label="Original (no compression)", zorder=5)

        # Best LoRA per gamma
        for g in [0.5, 0.7, 0.9]:
            best = [get_best_lora(bdata[s], g) for s in MODEL_SIZES]
            style = {0.5: ("s--", COLORS["lora_orig"]),
                     0.7: ("D-.", COLORS["lora_guided"]),
                     0.9: ("^:", COLORS["lora_soft"])}[g]
            ax.plot(MODEL_LABELS, best, style[0], color=style[1], lw=2, ms=8,
                    label=f"Best LoRA γ={g}")

        # Truncation at same gammas for reference
        for g in [0.5, 0.7]:
            trunc = [get_acc(bdata[s], f"Truncation{g}") for s in MODEL_SIZES]
            ax.plot(MODEL_LABELS, trunc, "x--", color=COLORS["trunc"], lw=1.2, ms=7,
                    alpha=0.5, label=f"Truncation γ={g}" if idx == 0 else "")

        ax.set_xlabel("Model Size")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{bname}", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Scaling Law: Accuracy vs Model Size", fontsize=15, fontweight="bold", y=1.02)
    save_fig("fig1_scaling_law_baselines.png")


# ======================================================================
#  FIGURE 2: Accuracy Drop from Baseline per γ (Compression Collapse)
# ======================================================================

def fig2_compression_collapse():
    print("\n=== Fig 2: Compression Collapse Threshold ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, (bname, bdata) in enumerate(BENCHMARKS.items()):
        ax = axes[idx]
        for g in GAMMAS:
            drops = []
            for s in MODEL_SIZES:
                orig = get_acc(bdata[s], "Original")
                best_lora = get_best_lora(bdata[s], g)
                drops.append(orig - best_lora)
            ax.plot(MODEL_LABELS, drops, "o-", lw=2, ms=8, label=f"γ={g}")

        ax.axhline(0, color="black", linestyle="--", lw=1, alpha=0.5)
        ax.axhline(5, color="red", linestyle=":", lw=1, alpha=0.4, label="5pp threshold")
        ax.set_xlabel("Model Size")
        ax.set_ylabel("Accuracy Drop (pp) — lower is better")
        ax.set_title(f"{bname}", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.invert_yaxis()  # so "better" is up

    fig.suptitle("Compression Collapse: Accuracy Drop from Baseline (Best LoRA variant)",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig("fig2_compression_collapse.png")


# ======================================================================
#  FIGURE 3: Pareto Frontiers — Accuracy vs Token Savings
# ======================================================================

def fig3_pareto_frontiers():
    print("\n=== Fig 3: Pareto Frontiers ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    size_colors = {0.5: "#E91E63", 1.5: "#FF9800", 3: "#4CAF50", 7: "#2196F3", 14: "#9C27B0"}

    for idx, (bname, bdata) in enumerate(BENCHMARKS.items()):
        ax = axes[idx]
        for s, label in zip(MODEL_SIZES, MODEL_LABELS):
            df = bdata[s]
            # Only LoRA methods (the interesting Pareto frontier)
            lora = df[df.Method.str.startswith("LoRA") & ~df.Method.str.contains("Adaptive")]
            ax.scatter(lora["Token Savings"], lora["Accuracy"],
                       s=50, alpha=0.6, color=size_colors[s], edgecolors="none")
            # Connect Pareto-optimal points (sorted by savings)
            pareto = lora.sort_values("Token Savings")
            ax.plot(pareto["Token Savings"], pareto["Accuracy"],
                    "-", alpha=0.3, color=size_colors[s])
            # Mark Original baseline
            orig_acc = get_acc(df, "Original")
            ax.plot(0, orig_acc, "*", color=size_colors[s], ms=14, zorder=10,
                    markeredgecolor="black", markeredgewidth=0.5)
            ax.annotate(label, (0, orig_acc), textcoords="offset points",
                        xytext=(-15, 5), fontsize=8, fontweight="bold", color=size_colors[s])

        ax.set_xlabel("Token Savings (%)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{bname} — LoRA Methods", fontweight="bold")
        ax.grid(alpha=0.3)
        # Legend
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=size_colors[s],
                              ms=8, label=label) for s, label in zip(MODEL_SIZES, MODEL_LABELS)]
        ax.legend(handles=handles, fontsize=9, title="Model Size")

    fig.suptitle("Pareto Frontiers: Accuracy vs Token Savings (★ = Baseline)",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig("fig3_pareto_frontiers.png")


# ======================================================================
#  FIGURE 4: LoRA Triplet Comparison Across Sizes
# ======================================================================

def fig4_lora_triplet_scaling():
    print("\n=== Fig 4: LoRA Triplet Scaling ===")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, (bname, bdata) in enumerate(BENCHMARKS.items()):
        for gi, g in enumerate([0.5, 0.7, 0.9]):
            ax = axes[idx][gi]
            x = np.arange(len(MODEL_SIZES))
            w = 0.25

            lora_acc    = [get_lora_variant_acc(bdata[s], g, "lora") for s in MODEL_SIZES]
            guided_acc  = [get_lora_variant_acc(bdata[s], g, "guided") for s in MODEL_SIZES]
            soft_acc    = [get_lora_variant_acc(bdata[s], g, "soft") for s in MODEL_SIZES]

            ax.bar(x - w, lora_acc,   w, label="LoRA", color=COLORS["lora_orig"], edgecolor="white")
            ax.bar(x,     guided_acc, w, label="LoRAGuided", color=COLORS["lora_guided"], edgecolor="white")
            ax.bar(x + w, soft_acc,   w, label="LoRASoft", color=COLORS["lora_soft"], edgecolor="white")

            # Baseline line
            baselines = [get_acc(bdata[s], "Original") for s in MODEL_SIZES]
            ax.plot(x, baselines, "k--", lw=1.3, alpha=0.6, label="Baseline")

            ax.set_xticks(x)
            ax.set_xticklabels(MODEL_LABELS)
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"{bname} — γ={g}", fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            if gi == 0 and idx == 0:
                ax.legend(fontsize=8)

    fig.suptitle("LoRA Variant Comparison Across Model Sizes", fontsize=15, fontweight="bold", y=1.01)
    save_fig("fig4_lora_triplet_scaling.png")


# ======================================================================
#  FIGURE 5: Method Family Heatmap
# ======================================================================

def fig5_method_family_heatmap():
    print("\n=== Fig 5: Method Family Heatmap ===")

    # For each benchmark, build a matrix: rows = method families, cols = model sizes
    families = {
        "Original":       lambda df: get_acc(df, "Original"),
        "BeConcise":      lambda df: get_acc(df, "BeConcise"),
        "LC-Prompt":      lambda df: get_acc(df, "LC-Prompt"),
        "Truncation γ=0.7": lambda df: get_acc(df, "Truncation0.7"),
        "LLMLingua γ=0.7":  lambda df: get_acc(df, "LLMLingua0.7"),
        "LoRA γ=0.7":       lambda df: get_acc(df, "LoRA0.7"),
        "LoRAGuided γ=0.7": lambda df: get_acc(df, "LoRAGuided0.7"),
        "LoRASoft γ=0.7":   lambda df: get_acc(df, "LoRASoft0.7"),
        "Best LoRA γ=0.5":  lambda df: get_best_lora(df, 0.5),
        "Best LoRA γ=0.7":  lambda df: get_best_lora(df, 0.7),
        "Best LoRA γ=0.9":  lambda df: get_best_lora(df, 0.9),
        "LoRAAdaptive":     lambda df: get_acc(df, "LoRAAdaptive") if "LoRAAdaptive" in df.Method.values else np.nan,
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, (bname, bdata) in enumerate(BENCHMARKS.items()):
        matrix = []
        for fname, fn in families.items():
            row = [fn(bdata[s]) for s in MODEL_SIZES]
            matrix.append(row)

        heatmap_df = pd.DataFrame(matrix, index=list(families.keys()), columns=MODEL_LABELS)
        ax = axes[idx]
        sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="RdYlGn",
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Accuracy %"},
                    vmin=0, vmax=100)
        ax.set_title(f"{bname}", fontweight="bold")
        ax.set_xlabel("Model Size")

    fig.suptitle("Method Accuracy Across Model Sizes", fontsize=15, fontweight="bold", y=1.02)
    save_fig("fig5_method_family_heatmap.png")


# ======================================================================
#  FIGURE 6: Cross-Benchmark Scatter (GSM8K vs MATH-500)
# ======================================================================

def fig6_cross_benchmark():
    print("\n=== Fig 6: Cross-Benchmark Correlation ===")
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)
    size_colors = {0.5: "#E91E63", 1.5: "#FF9800", 3: "#4CAF50", 7: "#2196F3", 14: "#9C27B0"}

    for si, (s, label) in enumerate(zip(MODEL_SIZES, MODEL_LABELS)):
        ax = axes[si]
        gsm = GSM8K[s]
        math = MATH500[s]

        # Merge on method name
        merged = gsm.merge(math, on="Method", suffixes=("_gsm", "_math"))

        # Color by method type
        for _, row in merged.iterrows():
            m = row.Method
            if m.startswith("LoRASoft"):    c = COLORS["lora_soft"]
            elif m.startswith("LoRAGuided"): c = COLORS["lora_guided"]
            elif m.startswith("LoRA"):       c = COLORS["lora_orig"]
            elif m.startswith("Truncation"): c = COLORS["trunc"]
            elif m.startswith("LLMLingua"):  c = COLORS["llmlingua"]
            elif m == "Original":            c = COLORS["orig"]
            else:                            c = COLORS["prompt"]
            ax.scatter(row.Accuracy_gsm, row.Accuracy_math, s=40, color=c,
                       alpha=0.7, edgecolors="black", linewidth=0.3)

        # Diagonal reference
        lo = min(merged.Accuracy_gsm.min(), merged.Accuracy_math.min()) - 5
        hi = max(merged.Accuracy_gsm.max(), merged.Accuracy_math.max()) + 5
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, lw=1)

        # Correlation
        corr = merged[["Accuracy_gsm", "Accuracy_math"]].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f"r={corr:.3f}", transform=ax.transAxes,
                fontsize=10, va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("GSM8K Accuracy (%)")
        if si == 0:
            ax.set_ylabel("MATH-500 Accuracy (%)")
        ax.set_title(label, fontweight="bold")
        ax.grid(alpha=0.3)

    fig.suptitle("Cross-Benchmark Correlation: GSM8K vs MATH-500",
                 fontsize=14, fontweight="bold", y=1.05)
    # Shared legend
    patches = [
        mpatches.Patch(color=COLORS["orig"],        label="Original"),
        mpatches.Patch(color=COLORS["prompt"],      label="Prompt"),
        mpatches.Patch(color=COLORS["trunc"],       label="Truncation"),
        mpatches.Patch(color=COLORS["llmlingua"],   label="LLMLingua"),
        mpatches.Patch(color=COLORS["lora_orig"],   label="LoRA"),
        mpatches.Patch(color=COLORS["lora_guided"], label="LoRAGuided"),
        mpatches.Patch(color=COLORS["lora_soft"],   label="LoRASoft"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))
    save_fig("fig6_cross_benchmark.png")


# ======================================================================
#  FIGURE 7: Adaptive vs Best Fixed γ
# ======================================================================

def fig7_adaptive_vs_fixed():
    print("\n=== Fig 7: Adaptive vs Best Fixed γ ===")
    # Only MATH-500 has LoRAAdaptive
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, (bname, bdata) in enumerate(BENCHMARKS.items()):
        ax = axes[idx]
        adaptive_accs = []
        best_fixed_accs = []
        best_fixed_gammas = []

        for s in MODEL_SIZES:
            df = bdata[s]
            adap = get_acc(df, "LoRAAdaptive")
            adaptive_accs.append(adap if not np.isnan(adap) else 0)

            # Best across all gammas and variants
            best_g = 0
            best_a = 0
            for g in GAMMAS:
                for v in ["lora", "guided", "soft"]:
                    a = get_lora_variant_acc(df, g, v)
                    if not np.isnan(a) and a > best_a:
                        best_a = a
                        best_g = g
            best_fixed_accs.append(best_a)
            best_fixed_gammas.append(best_g)

        x = np.arange(len(MODEL_SIZES))
        w = 0.35
        bars1 = ax.bar(x - w/2, adaptive_accs, w, label="LoRAAdaptive",
                        color=COLORS["adaptive"], edgecolor="white")
        bars2 = ax.bar(x + w/2, best_fixed_accs, w, label="Best Fixed γ",
                        color=COLORS["lora_guided"], edgecolor="white")

        # Annotate with gamma values
        for i, (a, bf, bg) in enumerate(zip(adaptive_accs, best_fixed_accs, best_fixed_gammas)):
            ax.text(i - w/2, a + 0.3, f"{a:.1f}", ha="center", fontsize=8)
            ax.text(i + w/2, bf + 0.3, f"{bf:.1f}\n(γ={bg})", ha="center", fontsize=7)

        # Baselines
        baselines = [get_acc(bdata[s], "Original") for s in MODEL_SIZES]
        ax.plot(x, baselines, "k--", lw=1.5, alpha=0.6, label="Baseline")

        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{bname}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Adaptive Compression vs Best Fixed γ", fontsize=15, fontweight="bold", y=1.02)
    save_fig("fig7_adaptive_vs_fixed.png")


# ======================================================================
#  FIGURE 8: Token Savings Efficiency (savings per accuracy point lost)
# ======================================================================

def fig8_efficiency():
    print("\n=== Fig 8: Token Savings Efficiency ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, (bname, bdata) in enumerate(BENCHMARKS.items()):
        ax = axes[idx]
        for g in [0.5, 0.7, 0.9]:
            efficiencies = []
            for s in MODEL_SIZES:
                orig = get_acc(bdata[s], "Original")
                best = get_best_lora(bdata[s], g)
                drop = orig - best
                # Best savings among the 3 variants
                savings_list = []
                for prefix in ["LoRA", "LoRAGuided", "LoRASoft"]:
                    row = bdata[s][bdata[s].Method == f"{prefix}{g}"]
                    if len(row) > 0:
                        savings_list.append(row["Token Savings"].values[0])
                savings = max(savings_list) if savings_list else 0
                eff = savings / max(drop, 0.1)  # avoid div by 0
                efficiencies.append(eff)

            ax.plot(MODEL_LABELS, efficiencies, "o-", lw=2, ms=8, label=f"γ={g}")

        ax.set_xlabel("Model Size")
        ax.set_ylabel("Efficiency (Token Savings % / Accuracy Drop pp)")
        ax.set_title(f"{bname}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Compression Efficiency: Token Savings per Accuracy Point Lost (Best LoRA)",
                 fontsize=13, fontweight="bold", y=1.02)
    save_fig("fig8_efficiency.png")


# ======================================================================
#  FIGURE 9: LoRASoft vs LoRA — "Does soft prompting help at scale?"
# ======================================================================

def fig9_soft_vs_standard():
    print("\n=== Fig 9: LoRASoft Advantage Across Sizes ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, (bname, bdata) in enumerate(BENCHMARKS.items()):
        ax = axes[idx]
        for g in GAMMAS:
            deltas = []
            for s in MODEL_SIZES:
                soft = get_lora_variant_acc(bdata[s], g, "soft")
                lora = get_lora_variant_acc(bdata[s], g, "lora")
                deltas.append(soft - lora)
            ax.plot(MODEL_LABELS, deltas, "o-", lw=1.8, ms=7, label=f"γ={g}")

        ax.axhline(0, color="black", linestyle="--", lw=1, alpha=0.5)
        ax.set_xlabel("Model Size")
        ax.set_ylabel("LoRASoft − LoRA (pp)")
        ax.set_title(f"{bname}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("LoRASoft Advantage Over Standard LoRA (positive = Soft better)",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig("fig9_soft_vs_standard.png")


# ======================================================================
#  FIGURE 10: Prompt Methods Scaling
# ======================================================================

def fig10_prompt_methods():
    print("\n=== Fig 10: Prompt Methods Scaling ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    prompt_methods = ["Original", "BeConcise", "OnlyNumbers", "AbbreWords", "LC-Prompt"]
    pm_colors = ["#2196F3", "#9C27B0", "#FF5722", "#009688", "#FF9800"]

    for idx, (bname, bdata) in enumerate(BENCHMARKS.items()):
        ax = axes[idx]
        for mi, m in enumerate(prompt_methods):
            accs = [get_acc(bdata[s], m) for s in MODEL_SIZES]
            ax.plot(MODEL_LABELS, accs, "o-", color=pm_colors[mi], lw=2, ms=8, label=m)

        ax.set_xlabel("Model Size")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{bname}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Prompt-Based Methods Across Model Sizes",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig("fig10_prompt_methods.png")


# ======================================================================
#  TABLE 1: Master Results CSV
# ======================================================================

def table1_master_csv():
    print("\n=== Table 1: Master Results CSV ===")
    all_rows = []
    for bname, bdata in BENCHMARKS.items():
        for s, label in zip(MODEL_SIZES, MODEL_LABELS):
            df = bdata[s].copy()
            df.insert(0, "Benchmark", bname)
            df.insert(1, "Model Size", label)
            df.insert(2, "Params (B)", s)
            all_rows.append(df)

    master = pd.concat(all_rows, ignore_index=True)
    p = os.path.join(OUTPUT_DIR, "table1_master_results.csv")
    master.to_csv(p, index=False)
    log(f"Saved {p} ({len(master)} rows)")
    return master


# ======================================================================
#  TABLE 2: Scaling Law Summary
# ======================================================================

def table2_scaling_summary():
    print("\n=== Table 2: Scaling Law Summary ===")
    rows = []
    for bname, bdata in BENCHMARKS.items():
        for s, label in zip(MODEL_SIZES, MODEL_LABELS):
            df = bdata[s]
            orig = get_acc(df, "Original")

            for g in GAMMAS:
                best = get_best_lora(df, g)
                drop = orig - best
                # Find which variant is best
                best_var = "LoRA"
                for v, vn in [("lora", "LoRA"), ("guided", "LoRAGuided"), ("soft", "LoRASoft")]:
                    if get_lora_variant_acc(df, g, v) == best:
                        best_var = vn

                # Token savings of best variant
                row_data = df[df.Method == f"{best_var}{g}"]
                savings = row_data["Token Savings"].values[0] if len(row_data) > 0 else np.nan
                tokens  = row_data["Avg Tokens"].values[0] if len(row_data) > 0 else np.nan

                rows.append({
                    "Benchmark": bname,
                    "Model": label,
                    "Params_B": s,
                    "Gamma": g,
                    "Baseline_Acc": orig,
                    "Best_LoRA_Acc": best,
                    "Acc_Drop_pp": round(drop, 2),
                    "Best_Variant": best_var,
                    "Token_Savings_pct": round(savings, 2),
                    "Avg_Tokens": round(tokens, 1),
                })

            # Adaptive
            adap = get_acc(df, "LoRAAdaptive")
            if not np.isnan(adap):
                adap_row = df[df.Method == "LoRAAdaptive"]
                rows.append({
                    "Benchmark": bname,
                    "Model": label,
                    "Params_B": s,
                    "Gamma": "adaptive",
                    "Baseline_Acc": orig,
                    "Best_LoRA_Acc": adap,
                    "Acc_Drop_pp": round(orig - adap, 2),
                    "Best_Variant": "LoRAAdaptive",
                    "Token_Savings_pct": round(adap_row["Token Savings"].values[0], 2),
                    "Avg_Tokens": round(adap_row["Avg Tokens"].values[0], 1),
                })

    summary = pd.DataFrame(rows)
    p = os.path.join(OUTPUT_DIR, "table2_scaling_summary.csv")
    summary.to_csv(p, index=False)
    log(f"Saved {p} ({len(summary)} rows)")

    # Print key findings to console
    print("\n  KEY FINDINGS:")
    for bname in ["GSM8K", "MATH-500"]:
        sub = summary[(summary.Benchmark == bname) & (summary.Gamma != "adaptive")]
        # Where does compression collapse (<5pp drop at γ=0.5)?
        g05 = sub[sub.Gamma == 0.5]
        no_collapse = g05[g05.Acc_Drop_pp <= 5]
        print(f"\n  {bname}:")
        print(f"    γ=0.5 with ≤5pp drop: {', '.join(no_collapse.Model.tolist()) if len(no_collapse) > 0 else 'None'}")

        # Best efficiency point
        sub["eff"] = sub["Token_Savings_pct"] / sub["Acc_Drop_pp"].clip(lower=0.1)
        best_eff = sub.loc[sub.eff.idxmax()]
        print(f"    Best efficiency: {best_eff.Model} γ={best_eff.Gamma} "
              f"({best_eff.Token_Savings_pct:.1f}% savings, {best_eff.Acc_Drop_pp:.1f}pp drop)")

    return summary


# ======================================================================
#  TABLE 3: Paper-style comparison table (like Table 1 in paper)
# ======================================================================

def table3_paper_style():
    print("\n=== Table 3: Paper-Style Comparison ===")
    # Replicate the paper's Table 1 format for 7B Qwen
    rows = []
    for bname in ["GSM8K", "MATH-500"]:
        bdata = BENCHMARKS[bname]
        df = bdata[7]
        orig = get_acc(df, "Original")
        orig_tok = df[df.Method == "Original"]["Avg Tokens"].values[0]

        for method in ["Original", "BeConcise", "OnlyNumbers", "AbbreWords"]:
            r = df[df.Method == method]
            if len(r) == 0: continue
            rows.append({
                "Method": method, "Ratio": "-", "Benchmark": bname,
                "Accuracy": r.Accuracy.values[0],
                "Tokens": r["Avg Tokens"].values[0],
                "Speedup": f"{r['Latency(s)'].values[0] / df[df.Method=='Original']['Latency(s)'].values[0]:.1f}x" if bname == "GSM8K" else "",
            })

        for g in [0.9, 0.7, 0.5]:
            for prefix in ["LC-Prompt", "Truncation", "LoRA"]:
                m = f"{prefix}{g}" if prefix != "LC-Prompt" else "LC-Prompt"
                if prefix == "LC-Prompt" and g != 0.5: continue  # LC-Prompt only once
                r = df[df.Method == m]
                if len(r) == 0: continue
                rows.append({
                    "Method": prefix if prefix == "LC-Prompt" else prefix,
                    "Ratio": g, "Benchmark": bname,
                    "Accuracy": r.Accuracy.values[0],
                    "Tokens": r["Avg Tokens"].values[0],
                    "Speedup": "",
                })

    paper_df = pd.DataFrame(rows)
    p = os.path.join(OUTPUT_DIR, "table3_paper_style.csv")
    paper_df.to_csv(p, index=False)
    log(f"Saved {p}")


# ======================================================================
#  BONUS: Print key statistics to console
# ======================================================================

def print_key_stats():
    print("\n" + "="*70)
    print("  KEY STATISTICS SUMMARY")
    print("="*70)

    for bname, bdata in BENCHMARKS.items():
        print(f"\n  [{bname}]")
        for s, label in zip(MODEL_SIZES, MODEL_LABELS):
            orig = get_acc(bdata[s], "Original")
            best_09 = get_best_lora(bdata[s], 0.9)
            best_07 = get_best_lora(bdata[s], 0.7)
            best_05 = get_best_lora(bdata[s], 0.5)
            print(f"    {label:>5s}: Baseline={orig:5.1f}  "
                  f"LoRA@0.9={best_09:5.1f} (Δ{orig-best_09:+.1f})  "
                  f"LoRA@0.7={best_07:5.1f} (Δ{orig-best_07:+.1f})  "
                  f"LoRA@0.5={best_05:5.1f} (Δ{orig-best_05:+.1f})")

    # Highlight 14B GSM8K result (paper's headline claim)
    gsm14 = GSM8K[14]
    orig_14 = get_acc(gsm14, "Original")
    best_lora_09 = get_best_lora(gsm14, 0.9)
    print(f"\n  ★ Paper headline: Qwen2.5-14B on GSM8K at γ=0.9")
    print(f"    Baseline: {orig_14}%  Best LoRA: {best_lora_09}%  Drop: {orig_14-best_lora_09:.2f}pp")
    print(f"    (Paper claims <0.4% drop for 14B — we get {orig_14-best_lora_09:.2f}pp)")


# ======================================================================
#  MAIN
# ======================================================================

if __name__ == "__main__":
    print("TokenSkip Scaling Law Analysis")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Figures
    fig1_scaling_law_baselines()
    fig2_compression_collapse()
    fig3_pareto_frontiers()
    fig4_lora_triplet_scaling()
    fig5_method_family_heatmap()
    fig6_cross_benchmark()
    fig7_adaptive_vs_fixed()
    fig8_efficiency()
    fig9_soft_vs_standard()
    fig10_prompt_methods()

    # Tables
    master = table1_master_csv()
    summary = table2_scaling_summary()
    table3_paper_style()

    # Stats
    print_key_stats()

    print(f"\n{'='*70}")
    print(f"  DONE — All outputs in {OUTPUT_DIR}")
    print(f"{'='*70}")
