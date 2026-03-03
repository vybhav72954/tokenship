# ---- Cell 1 ----
import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
COLORS = {
    'llama_gsm8k':  '#1565C0', 'llama_math500':'#42A5F5',
    'qwen_gsm8k':   '#B71C1C', 'qwen_math500': '#EF9A9A',
    'llama': '#1565C0', 'qwen': '#B71C1C',
}
OUT = os.path.join("Outputs", "comparison")
os.makedirs(OUT, exist_ok=True)

BASE = "Outputs"
llama_gsm  = pd.read_csv(f"{BASE}/llama3.1/gsm8k/results.csv")
llama_math = pd.read_csv(f"{BASE}/llama3.1/math500/results.csv")
qwen_gsm   = pd.read_csv(f"{BASE}/qwen2.5/gsm8k/results.csv")
qwen_math  = pd.read_csv(f"{BASE}/qwen2.5/math500/tokenskip_results.csv")

for df, model, dataset in [
    (llama_gsm,  'LLaMA-3.1-8B', 'GSM8K'),
    (llama_math, 'LLaMA-3.1-8B', 'MATH-500'),
    (qwen_gsm,   'Qwen2.5-7B',   'GSM8K'),
    (qwen_math,  'Qwen2.5-7B',   'MATH-500'),
]:
    df['Model'] = model; df['Dataset'] = dataset

all_df = pd.concat([llama_gsm, llama_math, qwen_gsm, qwen_math], ignore_index=True)
# Fix Ratio column
all_df['Ratio'] = all_df['Ratio'].fillna(
    all_df['Method'].str.extract(r'(0\.[0-9])')[0].astype(float).where(
        all_df['Method'].str.startswith('Truncation'), other=np.nan))

print(f"Loaded {len(all_df)} rows across 4 configs")
print(all_df[['Model','Dataset','Method','Accuracy','Avg Tokens','Latency(s)']].to_string(index=False))


# ---- Cell 2 ----
import openpyxl

# Table 1: Baseline
orig = all_df[all_df['Method']=='Original'][['Model','Dataset','Accuracy','Avg Tokens','Latency(s)','Correct','Total']].copy()
orig.to_csv(f"{OUT}/table1_baseline.csv", index=False)

# Table 2a: Accuracy by prompt method
prompt_methods = ['Original','BeConcise','OnlyNumbers','AbbreWords']
prompt_df = all_df[all_df['Method'].isin(prompt_methods)]
t2a = prompt_df.pivot_table(index='Method', columns=['Model','Dataset'], values='Accuracy').round(2).reset_index()
t2a.columns = ['_'.join(c).strip('_') for c in t2a.columns]
t2a.to_csv(f"{OUT}/table2a_accuracy_by_prompt.csv", index=False)

# Table 2b: Tokens by prompt method
t2b = prompt_df.pivot_table(index='Method', columns=['Model','Dataset'], values='Avg Tokens').round(1).reset_index()
t2b.columns = ['_'.join(c).strip('_') for c in t2b.columns]
t2b.to_csv(f"{OUT}/table2b_tokens_by_prompt.csv", index=False)

# Table 3: Truncation accuracy
trunc_df = all_df[all_df['Method'].str.startswith('Truncation')]
t3 = trunc_df.pivot_table(index='Ratio', columns=['Model','Dataset'], values='Accuracy').sort_index(ascending=False).round(2).reset_index()
t3.columns = ['_'.join([str(c) for c in col]).strip('_') for col in t3.columns]
t3.to_csv(f"{OUT}/table3_truncation_accuracy.csv", index=False)

# Table 4: Full metrics
full = all_df[['Model','Dataset','Method','Accuracy','Avg Tokens','Latency(s)',
               'Act Ratio','Token Savings (%)', 'Accuracy Drop','Efficiency Score']].copy()
full = full.sort_values(['Dataset','Model','Accuracy'], ascending=[True,True,False])
full.to_csv(f"{OUT}/table4_full_metrics.csv", index=False)

# All tables in one Excel workbook
with pd.ExcelWriter(f"{OUT}/all_tables.xlsx", engine='openpyxl') as writer:
    orig.to_excel(writer, sheet_name='1_Baseline',        index=False)
    t2a.to_excel(writer,  sheet_name='2a_Accuracy_Prompt',index=False)
    t2b.to_excel(writer,  sheet_name='2b_Tokens_Prompt',  index=False)
    t3.to_excel(writer,   sheet_name='3_Truncation',      index=False)
    full.to_excel(writer, sheet_name='4_Full_Metrics',    index=False)
print("Saved: 4 CSVs + all_tables.xlsx")


# ---- Cell 3 ----
# Key findings — print + save to text file
findings = []
def pf(s=""): print(s); findings.append(s)
pf("=" * 70)
pf("KEY FINDINGS SUMMARY")
pf("=" * 70)
pf("\nBASELINE:")
for _, row in orig.iterrows():
    pf(f"  {row['Model']:20s} | {row['Dataset']:8s} | Acc={row['Accuracy']:.1f}%  Tokens={row['Avg Tokens']:.0f}  Lat={row['Latency(s)']:.3f}s")
pf("\nBEST PROMPT METHOD (by Efficiency Score):")
for dataset in ["GSM8K","MATH-500"]:
    for model in ["LLaMA-3.1-8B","Qwen2.5-7B"]:
        sub = all_df[(all_df["Dataset"]==dataset)&(all_df["Model"]==model)&
                     (all_df["Method"].isin(["BeConcise","OnlyNumbers","AbbreWords"]))]
        best = sub.loc[sub["Efficiency Score"].idxmax()]
        pf(f"  {model:20s} | {dataset:8s} | {best['Method']:12s}  Acc={best['Accuracy']:.1f}%  Savings={best['Token Savings (%)']:.1f}%  Eff={best['Efficiency Score']:.2f}")
pf("\nTRUNCATION at ratio=0.7:")
for _, row in all_df[all_df["Method"]=="Truncation_0.7"].iterrows():
    pf(f"  {row['Model']:20s} | {row['Dataset']:8s} | Acc={row['Accuracy']:.1f}%  Drop={row['Accuracy Drop']:.1f}%")
pf("\nACCURACY DELTA (Qwen - LLaMA) at baseline:")
for dataset in ["GSM8K","MATH-500"]:
    qw = orig[(orig["Model"]=="Qwen2.5-7B")&(orig["Dataset"]==dataset)]["Accuracy"].values[0]
    ll = orig[(orig["Model"]=="LLaMA-3.1-8B")&(orig["Dataset"]==dataset)]["Accuracy"].values[0]
    pf(f"  {dataset:8s}: Qwen={qw:.1f}%  LLaMA={ll:.1f}%  Delta={qw-ll:+.1f}%")
with open(f"{OUT}/key_findings.txt", "w") as f:
    f.write("\n".join(findings))
print("Saved: key_findings.txt")

prompt_methods = ['Original','BeConcise','OnlyNumbers','AbbreWords']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, dataset in zip(axes, ['GSM8K','MATH-500']):
    sub = all_df[(all_df['Method'].isin(prompt_methods)) & (all_df['Dataset']==dataset)]
    x = np.arange(len(prompt_methods)); w = 0.35
    for i, (model, color) in enumerate([('LLaMA-3.1-8B',COLORS['llama']),('Qwen2.5-7B',COLORS['qwen'])]):
        vals = [sub[(sub['Model']==model)&(sub['Method']==m)]['Accuracy'].values[0] for m in prompt_methods]
        bars = ax.bar(x+i*w-w/2, vals, w, label=model, color=color, edgecolor='black', lw=0.6, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4, f'{v:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(prompt_methods, rotation=15)
    ax.set_title(f'Accuracy by Prompt Method — {dataset}', fontweight='bold')
    ax.set_ylabel('Accuracy (%)'); ax.legend(); ax.set_ylim(0, max(sub['Accuracy'].max()+15, 60))
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_accuracy_prompt_methods.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 7 ----
# Fig 2: Baseline accuracy side-by-side
orig = all_df[all_df['Method']=='Original']
fig, ax = plt.subplots(figsize=(8,5))
datasets = ['GSM8K','MATH-500']; x = np.arange(2); w = 0.35
for i, (model, color) in enumerate([('LLaMA-3.1-8B',COLORS['llama']),('Qwen2.5-7B',COLORS['qwen'])]):
    vals = [orig[(orig['Model']==model)&(orig['Dataset']==d)]['Accuracy'].values[0] for d in datasets]
    bars = ax.bar(x+i*w-w/2, vals, w, label=model, color=color, edgecolor='black', lw=0.6, alpha=0.88)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=12)
ax.set_ylabel('Accuracy (%)'); ax.set_title('Baseline Accuracy: LLaMA-3.1-8B vs Qwen2.5-7B', fontweight='bold', fontsize=13)
ax.legend(fontsize=11); ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_baseline_accuracy.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 8 ----
# Fig 3: Accuracy vs truncation ratio
trunc = all_df[all_df['Method'].str.startswith('Truncation')].copy()
orig_acc = all_df[all_df['Method']=='Original'][['Model','Dataset','Accuracy']].rename(columns={'Accuracy':'BaseAcc'})
trunc = trunc.merge(orig_acc, on=['Model','Dataset'])
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, dataset in zip(axes, ['GSM8K','MATH-500']):
    sub = trunc[trunc['Dataset']==dataset].sort_values('Ratio')
    for model, color, ls in [('LLaMA-3.1-8B',COLORS['llama'],'-o'),('Qwen2.5-7B',COLORS['qwen'],'-s')]:
        m = sub[sub['Model']==model]
        ax.plot(m['Ratio'], m['Accuracy'], ls, color=color, lw=2.2, ms=8, label=model)
        ax.axhline(m['BaseAcc'].iloc[0], color=color, lw=1.2, ls='--', alpha=0.5)
    ax.set_xlabel('Truncation Ratio'); ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Accuracy vs Truncation Ratio — {dataset}', fontweight='bold')
    ax.legend(); ax.invert_xaxis()
plt.suptitle('Dashed lines = no-truncation baseline', fontsize=9, y=0)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_truncation_accuracy.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 9 ----
# Fig 4: Avg tokens per method
prompt_methods = ['Original','BeConcise','OnlyNumbers','AbbreWords']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, dataset in zip(axes, ['GSM8K','MATH-500']):
    sub = all_df[(all_df['Method'].isin(prompt_methods))&(all_df['Dataset']==dataset)]
    x = np.arange(len(prompt_methods)); w = 0.35
    for i, (model, color) in enumerate([('LLaMA-3.1-8B',COLORS['llama']),('Qwen2.5-7B',COLORS['qwen'])]):
        vals = [sub[(sub['Model']==model)&(sub['Method']==m)]['Avg Tokens'].values[0] for m in prompt_methods]
        bars = ax.bar(x+i*w-w/2, vals, w, label=model, color=color, edgecolor='black', lw=0.6, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{v:.0f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(prompt_methods, rotation=15)
    ax.set_title(f'Avg CoT Tokens — {dataset}', fontweight='bold')
    ax.set_ylabel('Avg Tokens'); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_avg_tokens.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 10 ----
# Fig 5: Token savings
prompt_methods = ['BeConcise','OnlyNumbers','AbbreWords']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, dataset in zip(axes, ['GSM8K','MATH-500']):
    sub = all_df[(all_df['Method'].isin(prompt_methods))&(all_df['Dataset']==dataset)]
    x = np.arange(len(prompt_methods)); w = 0.35
    for i, (model, color) in enumerate([('LLaMA-3.1-8B',COLORS['llama']),('Qwen2.5-7B',COLORS['qwen'])]):
        vals = [sub[(sub['Model']==model)&(sub['Method']==m)]['Token Savings (%)'].values[0] for m in prompt_methods]
        bars = ax.bar(x+i*w-w/2, vals, w, label=model, color=color, edgecolor='black', lw=0.6, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(prompt_methods)
    ax.set_title(f'Token Savings vs Original — {dataset}', fontweight='bold')
    ax.set_ylabel('Token Savings (%)'); ax.legend(); ax.set_ylim(0, 70)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_token_savings.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 11 ----
# Fig 6: Pareto — accuracy drop vs token savings
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
markers = {'LLaMA-3.1-8B':'o','Qwen2.5-7B':'s'}
for ax, dataset in zip(axes, ['GSM8K','MATH-500']):
    sub = all_df[all_df['Dataset']==dataset]
    for model, color in [('LLaMA-3.1-8B',COLORS['llama']),('Qwen2.5-7B',COLORS['qwen'])]:
        m = sub[sub['Model']==model]
        ax.scatter(m['Token Savings (%)'], m['Accuracy Drop'], color=color, marker=markers[model],
                   s=90, label=model, edgecolors='black', lw=0.5, zorder=5)
        for _, row in m.iterrows():
            ax.annotate(row['Method'].replace('Truncation_','T='),
                        (row['Token Savings (%)'], row['Accuracy Drop']),
                        textcoords="offset points", xytext=(4,4), fontsize=7.5)
    ax.axhline(0, color='gray', lw=1, ls='--'); ax.axvline(0, color='gray', lw=1, ls='--')
    ax.set_xlabel('Token Savings (%)'); ax.set_ylabel('Accuracy Drop (%) [neg = improvement]')
    ax.set_title(f'Pareto: Accuracy Drop vs Token Savings — {dataset}', fontweight='bold')
    ax.legend(); ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_pareto.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 12 ----
# Fig 7: Latency per sample
prompt_methods = ['Original','BeConcise','OnlyNumbers','AbbreWords']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, dataset in zip(axes, ['GSM8K','MATH-500']):
    sub = all_df[(all_df['Method'].isin(prompt_methods))&(all_df['Dataset']==dataset)]
    x = np.arange(len(prompt_methods)); w = 0.35
    for i, (model, color) in enumerate([('LLaMA-3.1-8B',COLORS['llama']),('Qwen2.5-7B',COLORS['qwen'])]):
        vals = [sub[(sub['Model']==model)&(sub['Method']==m)]['Latency(s)'].values[0] for m in prompt_methods]
        bars = ax.bar(x+i*w-w/2, vals, w, label=model, color=color, edgecolor='black', lw=0.6, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{v:.3f}s', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(prompt_methods, rotation=15)
    ax.set_title(f'Latency per Sample — {dataset}', fontweight='bold')
    ax.set_ylabel('Latency (s/sample)'); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_latency.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 13 ----
# Fig 8: Latency vs token count scatter
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, dataset in zip(axes, ['GSM8K','MATH-500']):
    sub = all_df[all_df['Dataset']==dataset]
    for model, color in [('LLaMA-3.1-8B',COLORS['llama']),('Qwen2.5-7B',COLORS['qwen'])]:
        m = sub[sub['Model']==model]
        ax.scatter(m['Avg Tokens'], m['Latency(s)'], color=color, s=80, label=model, edgecolors='black', lw=0.5, zorder=5)
        z = np.polyfit(m['Avg Tokens'], m['Latency(s)'], 1)
        xs = np.linspace(m['Avg Tokens'].min(), m['Avg Tokens'].max(), 50)
        ax.plot(xs, np.poly1d(z)(xs), color=color, lw=1.5, ls='--', alpha=0.6)
    ax.set_xlabel('Avg Tokens'); ax.set_ylabel('Latency (s/sample)')
    ax.set_title(f'Latency vs Token Count — {dataset}', fontweight='bold'); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig8_latency_vs_tokens.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 14 ----
# Fig 9: Efficiency heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, dataset in zip(axes, ['GSM8K','MATH-500']):
    sub = all_df[all_df['Dataset']==dataset]
    pivot = sub.pivot_table(index='Method', columns='Model', values='Efficiency Score')
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGn', ax=ax,
                linewidths=0.5, cbar_kws={'label':'Efficiency Score'})
    ax.set_title(f'Efficiency Score — {dataset}', fontweight='bold'); ax.set_xlabel(''); ax.set_ylabel('')
plt.tight_layout()
plt.savefig(f"{OUT}/fig9_efficiency_heatmap.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 15 ----
# Fig 10: Efficiency ranking
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, dataset in zip(axes, ['GSM8K','MATH-500']):
    sub = all_df[all_df['Dataset']==dataset].sort_values('Efficiency Score', ascending=True)
    bar_colors = [COLORS['llama'] if 'LLaMA' in m else COLORS['qwen'] for m in sub['Model']]
    labels = [f"{row['Method']} ({row['Model'].split('-')[0]})" for _,row in sub.iterrows()]
    bars = ax.barh(labels, sub['Efficiency Score'], color=bar_colors, edgecolor='black', lw=0.5, alpha=0.88)
    for bar, v in zip(bars, sub['Efficiency Score']):
        ax.text(v+0.1, bar.get_y()+bar.get_height()/2, f'{v:.2f}', va='center', fontsize=8)
    ax.set_title(f'Efficiency Ranking — {dataset}', fontweight='bold'); ax.set_xlabel('Efficiency Score')
    ax.legend(handles=[mpatches.Patch(color=COLORS['llama'],label='LLaMA-3.1-8B'),
                       mpatches.Patch(color=COLORS['qwen'], label='Qwen2.5-7B')])
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_efficiency_ranking.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 16 ----
# Fig 11: Accuracy delta heatmap (Qwen - LLaMA)
fig, ax = plt.subplots(figsize=(10, 7))
pivot = all_df.pivot_table(index='Method', columns=['Model','Dataset'], values='Accuracy')
delta = pd.concat([
    (pivot[('Qwen2.5-7B','GSM8K')]   - pivot[('LLaMA-3.1-8B','GSM8K')]).rename('GSM8K'),
    (pivot[('Qwen2.5-7B','MATH-500')] - pivot[('LLaMA-3.1-8B','MATH-500')]).rename('MATH-500')
], axis=1)
delta = delta.loc[delta.abs().max(axis=1).sort_values(ascending=False).index]
sns.heatmap(delta, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'label':'Qwen Acc − LLaMA Acc (%)'})
ax.set_title('Accuracy Delta: Qwen2.5 minus LLaMA (green=Qwen wins, red=LLaMA wins)',
             fontweight='bold', fontsize=12); ax.set_ylabel('')
plt.tight_layout()
plt.savefig(f"{OUT}/fig11_delta_heatmap.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 17 ----
# Fig 12: Radar chart
N = 4
labels_radar = ['Accuracy','Token\nSavings','Speed\n(inv. lat)','Efficiency']
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
norm_cols = ['Acc_norm','Sav_norm','Lat_norm','Eff_norm']

pm = all_df[all_df['Method'].isin(['Original','BeConcise','OnlyNumbers','AbbreWords'])]
agg = pm.groupby(['Model','Dataset'])[['Accuracy','Token Savings (%)','Latency(s)','Efficiency Score']].mean().reset_index()
for col, inv in [('Accuracy',False),('Token Savings (%)',False),('Latency(s)',True),('Efficiency Score',False)]:
    mn, mx = agg[col].min(), agg[col].max()
    n = (agg[col]-mn)/(mx-mn+1e-9)
    agg[['Acc_norm','Sav_norm','Lat_norm','Eff_norm'][['Accuracy','Token Savings (%)','Latency(s)','Efficiency Score'].index(col)]] = 1-n if inv else n

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))
for ax, dataset in zip(axes, ['GSM8K','MATH-500']):
    for model, color in [('LLaMA-3.1-8B',COLORS['llama']),('Qwen2.5-7B',COLORS['qwen'])]:
        row = agg[(agg['Model']==model)&(agg['Dataset']==dataset)][norm_cols].values[0].tolist()
        row += row[:1]
        ax.plot(angles, row, color=color, lw=2, label=model)
        ax.fill(angles, row, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_radar, fontsize=10)
    ax.set_ylim(0,1); ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_title(f'Radar — {dataset}', fontweight='bold', pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35,1.1), fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig12_radar.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 18 ----
# Fig 13: Accuracy drop all methods
non_orig = all_df[all_df['Method']!='Original'].copy()
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, dataset in zip(axes, ['GSM8K','MATH-500']):
    sub = non_orig[non_orig['Dataset']==dataset]
    piv = sub.pivot_table(index='Method', columns='Model', values='Accuracy Drop').sort_values('LLaMA-3.1-8B')
    x = np.arange(len(piv)); w = 0.35
    for i, (model, color) in enumerate([('LLaMA-3.1-8B',COLORS['llama']),('Qwen2.5-7B',COLORS['qwen'])]):
        bars = ax.bar(x+i*w-w/2, piv[model], w, label=model, color=color, edgecolor='black', lw=0.5, alpha=0.88)
        for bar, v in zip(bars, piv[model]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(0.3 if v>=0 else -1.5),
                    f'{v:.1f}', ha='center', fontsize=7.5)
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=35, ha='right')
    ax.set_title(f'Accuracy Drop vs Baseline — {dataset}', fontweight='bold')
    ax.set_ylabel('Accuracy Drop (%) [neg = improvement]'); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig13_accuracy_drop_all.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 19 ----
# Fig 14: GSM8K vs MATH-500 scatter
gsm_acc  = all_df[all_df['Dataset']=='GSM8K' ][['Model','Method','Accuracy']].rename(columns={'Accuracy':'GSM8K'})
math_acc = all_df[all_df['Dataset']=='MATH-500'][['Model','Method','Accuracy']].rename(columns={'Accuracy':'MATH500'})
merged   = gsm_acc.merge(math_acc, on=['Model','Method'])
fig, ax = plt.subplots(figsize=(9,7))
for model, color, mk in [('LLaMA-3.1-8B',COLORS['llama'],'o'),('Qwen2.5-7B',COLORS['qwen'],'s')]:
    m = merged[merged['Model']==model]
    ax.scatter(m['GSM8K'], m['MATH500'], color=color, s=90, label=model,
               marker=mk, edgecolors='black', lw=0.5, zorder=5)
    for _, row in m.iterrows():
        ax.annotate(row['Method'].replace('Truncation_','T='),
                    (row['GSM8K'], row['MATH500']), textcoords="offset points", xytext=(4,4), fontsize=7.5)
ax.set_xlabel('GSM8K Accuracy (%)'); ax.set_ylabel('MATH-500 Accuracy (%)')
ax.set_title('GSM8K vs MATH-500 Accuracy per Method & Model', fontweight='bold'); ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/fig14_gsm8k_vs_math500.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 20 ----
# Fig 15: Truncation accuracy drop
trunc = all_df[all_df['Method'].str.startswith('Truncation')].copy()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, dataset in zip(axes, ['GSM8K','MATH-500']):
    sub = trunc[trunc['Dataset']==dataset].sort_values('Ratio')
    for model, color in [('LLaMA-3.1-8B',COLORS['llama']),('Qwen2.5-7B',COLORS['qwen'])]:
        m = sub[sub['Model']==model]
        ax.plot(m['Ratio'], m['Accuracy Drop'], '-o', color=color, lw=2.2, ms=8, label=model)
        for _, r in m.iterrows():
            ax.annotate(f"{r['Accuracy Drop']:.1f}", (r['Ratio'], r['Accuracy Drop']),
                        textcoords="offset points", xytext=(4,4), fontsize=8)
    ax.axhline(0, color='gray', lw=1, ls='--')
    ax.set_xlabel('Truncation Ratio'); ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title(f'Accuracy Drop under Truncation — {dataset}', fontweight='bold'); ax.legend(); ax.invert_xaxis()
plt.tight_layout()
plt.savefig(f"{OUT}/fig15_truncation_drop.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 21 ----
# Fig 16: Actual vs target ratio
trunc = all_df[all_df['Method'].str.startswith('Truncation')].copy()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, dataset in zip(axes, ['GSM8K','MATH-500']):
    sub = trunc[trunc['Dataset']==dataset].sort_values('Ratio')
    ax.plot([0.45,1.0],[0.45,1.0],'k--', lw=1.2, label='Perfect adherence', alpha=0.5)
    for model, color, mk in [('LLaMA-3.1-8B',COLORS['llama'],'-o'),('Qwen2.5-7B',COLORS['qwen'],'-s')]:
        m = sub[sub['Model']==model]
        ax.plot(m['Ratio'], m['Act Ratio'], mk, color=color, lw=2, ms=8, label=model)
    ax.set_xlabel('Target Ratio'); ax.set_ylabel('Actual Ratio')
    ax.set_title(f'Target vs Actual Compression Ratio — {dataset}', fontweight='bold'); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig16_ratio_adherence.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 22 ----
# Fig 17: 6-panel dashboard
fig = plt.figure(figsize=(20, 15))
gs_ = GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

# Panel A — baseline accuracy
ax_a = fig.add_subplot(gs_[0,0])
orig = all_df[all_df['Method']=='Original']; datasets=['GSM8K','MATH-500']
x = np.arange(2); w = 0.35
for i, (model, color) in enumerate([('LLaMA-3.1-8B',COLORS['llama']),('Qwen2.5-7B',COLORS['qwen'])]):
    vals=[orig[(orig['Model']==model)&(orig['Dataset']==d)]['Accuracy'].values[0] for d in datasets]
    bars=ax_a.bar(x+i*w-w/2,vals,w,label=model,color=color,edgecolor='black',lw=0.6,alpha=0.88)
    for bar,v in zip(bars,vals): ax_a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax_a.set_xticks(x); ax_a.set_xticklabels(datasets); ax_a.set_ylim(0,100)
ax_a.set_title('A. Baseline Accuracy', fontweight='bold'); ax_a.legend(fontsize=9); ax_a.set_ylabel('Accuracy (%)')

# Panel B — baseline tokens
ax_b = fig.add_subplot(gs_[0,1])
for i, (model, color) in enumerate([('LLaMA-3.1-8B',COLORS['llama']),('Qwen2.5-7B',COLORS['qwen'])]):
    vals=[orig[(orig['Model']==model)&(orig['Dataset']==d)]['Avg Tokens'].values[0] for d in datasets]
    bars=ax_b.bar(x+i*w-w/2,vals,w,label=model,color=color,edgecolor='black',lw=0.6,alpha=0.88)
    for bar,v in zip(bars,vals): ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')
ax_b.set_xticks(x); ax_b.set_xticklabels(datasets)
ax_b.set_title('B. Baseline Avg Tokens', fontweight='bold'); ax_b.legend(fontsize=9); ax_b.set_ylabel('Avg Tokens')

# Panel C — truncation lines all combos
ax_c = fig.add_subplot(gs_[1,:])
trunc = all_df[all_df['Method'].str.startswith('Truncation')].copy()
combos=[('LLaMA-3.1-8B','GSM8K',COLORS['llama_gsm8k'],'-o'),
        ('LLaMA-3.1-8B','MATH-500',COLORS['llama_math500'],'-s'),
        ('Qwen2.5-7B','GSM8K',COLORS['qwen_gsm8k'],'--o'),
        ('Qwen2.5-7B','MATH-500',COLORS['qwen_math500'],'--s')]
for model,dataset,color,ls in combos:
    sub=trunc[(trunc['Model']==model)&(trunc['Dataset']==dataset)].sort_values('Ratio')
    ax_c.plot(sub['Ratio'],sub['Accuracy'],ls,color=color,lw=2,ms=7,
              label=f"{model.split('-')[0]}/{dataset}")
ax_c.set_xlabel('Truncation Ratio'); ax_c.set_ylabel('Accuracy (%)')
ax_c.set_title('C. Accuracy under Truncation — All Combinations', fontweight='bold')
ax_c.legend(ncol=2,fontsize=9); ax_c.invert_xaxis()

# Panel D — best prompt token savings
ax_d = fig.add_subplot(gs_[2,0])
prompt_methods=['BeConcise','OnlyNumbers','AbbreWords']
best_sav=all_df[all_df['Method'].isin(prompt_methods)].groupby(['Model','Dataset'])['Token Savings (%)'].max().reset_index()
for i,(model,color) in enumerate([('LLaMA-3.1-8B',COLORS['llama']),('Qwen2.5-7B',COLORS['qwen'])]):
    vals=[best_sav[(best_sav['Model']==model)&(best_sav['Dataset']==d)]['Token Savings (%)'].values[0] for d in datasets]
    bars=ax_d.bar(x+i*w-w/2,vals,w,label=model,color=color,edgecolor='black',lw=0.6,alpha=0.88)
    for bar,v in zip(bars,vals): ax_d.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,f'{v:.1f}%',ha='center',fontsize=9)
ax_d.set_xticks(x); ax_d.set_xticklabels(datasets)
ax_d.set_title('D. Max Token Savings (Prompt Methods)', fontweight='bold'); ax_d.legend(fontsize=9); ax_d.set_ylabel('Token Savings (%)')

# Panel E — efficiency score bar
ax_e = fig.add_subplot(gs_[2,1])
eff=all_df[all_df['Method'].isin(['Original','BeConcise','OnlyNumbers','AbbreWords'])]
eff_piv=eff.pivot_table(index=['Model','Dataset'],values='Efficiency Score',aggfunc='max').reset_index()
eff_piv['label']=eff_piv['Model'].str.split('-').str[0]+'/'+eff_piv['Dataset']
bar_c=[COLORS['llama_gsm8k'],COLORS['llama_math500'],COLORS['qwen_gsm8k'],COLORS['qwen_math500']]
bars=ax_e.bar(eff_piv['label'],eff_piv['Efficiency Score'],color=bar_c,edgecolor='black',lw=0.6,alpha=0.88)
for bar,v in zip(bars,eff_piv['Efficiency Score']):
    ax_e.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,f'{v:.2f}',ha='center',fontsize=9)
ax_e.set_title('E. Peak Efficiency Score', fontweight='bold'); ax_e.set_ylabel('Efficiency Score')
ax_e.tick_params(axis='x',rotation=15)

plt.suptitle('TokenSkip — LLaMA-3.1-8B vs Qwen2.5-7B Summary Dashboard',
             fontsize=16, fontweight='bold', y=1.01)
plt.savefig(f"{OUT}/fig17_dashboard.png", dpi=150, bbox_inches='tight')
# plt.show()


# ---- Cell 23 ----
print("=" * 80)
print("KEY FINDINGS SUMMARY")
print("=" * 80)
orig = all_df[all_df['Method']=='Original']
print("\nBASELINE:")
for _, row in orig.iterrows():
    print(f"  {row['Model']:20s} | {row['Dataset']:8s} | Acc={row['Accuracy']:.1f}%  Tokens={row['Avg Tokens']:.0f}  Lat={row['Latency(s)']:.3f}s")

print("\nBEST PROMPT METHOD (by Efficiency Score):")
for dataset in ['GSM8K','MATH-500']:
    for model in ['LLaMA-3.1-8B','Qwen2.5-7B']:
        sub=all_df[(all_df['Dataset']==dataset)&(all_df['Model']==model)&
                   (all_df['Method'].isin(['BeConcise','OnlyNumbers','AbbreWords']))]
        best=sub.loc[sub['Efficiency Score'].idxmax()]
        print(f"  {model:20s} | {dataset:8s} | {best['Method']:12s}  Acc={best['Accuracy']:.1f}%  Savings={best['Token Savings (%)']:.1f}%  Eff={best['Efficiency Score']:.2f}")

print("\nTRUNCATION ROBUSTNESS at ratio=0.7:")
t07=all_df[all_df['Method']=='Truncation_0.7']
for _,row in t07.iterrows():
    print(f"  {row['Model']:20s} | {row['Dataset']:8s} | Acc={row['Accuracy']:.1f}%  Drop={row['Accuracy Drop']:.1f}%")

print("\nACCURACY DELTA (Qwen - LLaMA) at baseline:")
for dataset in ['GSM8K','MATH-500']:
    qw=orig[(orig['Model']=='Qwen2.5-7B')&(orig['Dataset']==dataset)]['Accuracy'].values[0]
    ll=orig[(orig['Model']=='LLaMA-3.1-8B')&(orig['Dataset']==dataset)]['Accuracy'].values[0]
    print(f"  {dataset:8s}: Qwen={qw:.1f}%  LLaMA={ll:.1f}%  Delta={qw-ll:+.1f}%")

print(f"\nAll {len([f for f in os.listdir(OUT) if f.endswith('.png')])} plots saved to: {OUT}/")