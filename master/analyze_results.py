"""
Skript pro statistickou analýzu a vizualizaci výsledků experimentu.
Načítá data z metrics_summary.csv a generuje grafy pro diplomovou práci.

Použití:
    python3 analyze_results.py

Vstupní data:
    dataset/metrics_summary.csv  — souhrnné metriky pro 500 účtenek
    dataset/ground_truth/        — referenční přepisy (500 JSON souborů)
    dataset/baseline/            — výstupy OCR bez multi-frame (500 JSON souborů)
    dataset/multiframe/          — výstupy OCR s multi-frame (500 JSON souborů)

Výstup:
    graf_cer_scatter.png         — scatter plot CER: baseline vs. multi-frame
    graf_f1_boxplot.png          — box plot F1 položek podle délky účtenky
    graf_cer_histogram.png       — histogram distribuce CER (Shapiro-Wilk vizualizace)
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Načtení dat ────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open('dataset/metrics_summary.csv', encoding='utf-8')))

cer_b  = np.array([float(r['cer_baseline_cal'])      for r in rows])
cer_m  = np.array([float(r['cer_multiframe_cal'])    for r in rows])
f1_b   = np.array([float(r['f1_items_baseline_cal']) for r in rows])
f1_m   = np.array([float(r['f1_items_multiframe_cal']) for r in rows])
cats   = np.array([r['length_cat'] for r in rows])

# ── Barvy ──────────────────────────────────────────────────────────────────
C_BASE  = '#2E75B6'
C_MULTI = '#ED7D31'
C_SHORT = '#70AD47'
C_MID   = '#FFC000'
C_LONG  = '#C00000'
SEG_COLORS = {'short': C_SHORT, 'mid': C_MID, 'long': C_LONG}
SEG_LABELS = {'short': 'Krátké (1–10 pol.)',
              'mid':   'Střední (11–25 pol.)',
              'long':  'Dlouhé (26+ pol.)'}

# ══════════════════════════════════════════════════════════════════════════
# GRAF 1 — Scatter plot CER
# ══════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(7, 7))
fig1.patch.set_facecolor('white')

for seg in ['short', 'mid', 'long']:
    mask = cats == seg
    ax1.scatter(cer_b[mask], cer_m[mask],
                c=SEG_COLORS[seg], alpha=0.4, s=20,
                label=SEG_LABELS[seg], zorder=3, edgecolors='none')

lim = max(cer_b.max(), cer_m.max()) * 1.05
ax1.plot([0, lim], [0, lim], color='#888888', lw=1.2,
         linestyle='--', label='Žádná změna (y = x)', zorder=2)
ax1.text(lim * 0.62, lim * 0.18,
         'Multi-frame\nlepší ↓', fontsize=9, color='#2D6A2D',
         style='italic', ha='center')
ax1.text(lim * 0.18, lim * 0.72,
         'Baseline\nlepší ↑', fontsize=9, color='#8B0000',
         style='italic', ha='center')

# Přidat počty pod diagonálou
n_below = np.sum(cer_m < cer_b)
ax1.text(0.97, 0.03,
         f'{n_below}/500 účtenek pod diagonálou\n({100*n_below/500:.0f} % případů)',
         transform=ax1.transAxes, ha='right', va='bottom',
         fontsize=8.5, color='#2D6A2D',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', alpha=0.8))

ax1.set_xlabel('CER – baseline (%)', fontsize=11)
ax1.set_ylabel('CER – multi-frame (%)', fontsize=11)
ax1.set_title('Porovnání CER: baseline vs. vícesnímkové zpracování\n(n = 500 účtenek)',
              fontsize=12, fontweight='bold', pad=12)
ax1.set_xlim(0, lim)
ax1.set_ylim(0, lim)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.25, zorder=1)
ax1.set_aspect('equal')
fig1.tight_layout()
fig1.savefig('/home/claude/graf_cer_scatter.png', dpi=200,
             bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f"Graf 1 uložen. Bodů pod diagonálou: {n_below}/500")

# ══════════════════════════════════════════════════════════════════════════
# GRAF 2 — Box plot F1 položek
# ══════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(8.5, 6))
fig2.patch.set_facecolor('white')

seg_order  = ['short', 'mid', 'long']
seg_names  = ['Krátké\n(1–10)', 'Střední\n(11–25)', 'Dlouhé\n(26+)']
x = np.arange(len(seg_order))
width = 0.32
bp_props = dict(linewidth=1.2)

for i, seg in enumerate(seg_order):
    mask = cats == seg
    db = f1_b[mask]
    dm = f1_m[mask]

    for data, pos, color in [(db, x[i]-width/2, C_BASE),
                              (dm, x[i]+width/2, C_MULTI)]:
        ax2.boxplot(data,
                    positions=[pos], widths=width*0.88,
                    patch_artist=True,
                    medianprops=dict(color='white', linewidth=2),
                    boxprops=dict(facecolor=color, alpha=0.85, **bp_props),
                    whiskerprops=dict(color=color, **bp_props),
                    capprops=dict(color=color, **bp_props),
                    flierprops=dict(marker='o', color=color,
                                    alpha=0.3, markersize=3),
                    showfliers=True)

    delta = np.mean(dm) - np.mean(db)
    ax2.annotate(f'+{delta:.1f} p.p.',
                 xy=(x[i], max(db.max(), dm.max()) + 0.8),
                 ha='center', va='bottom', fontsize=9,
                 color='#2D6A2D', fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(seg_names, fontsize=11)
ax2.set_ylabel('F1-score položek (%)', fontsize=11)
ax2.set_title('Distribuce F1-score extrakce položek podle délky účtenky\n(n = 500 účtenek)',
              fontsize=12, fontweight='bold', pad=12)
ax2.set_ylim(20, 108)
ax2.grid(True, axis='y', alpha=0.25, zorder=1)
ax2.legend(handles=[
    mpatches.Patch(facecolor=C_BASE,  alpha=0.85, label='Baseline'),
    mpatches.Patch(facecolor=C_MULTI, alpha=0.85, label='Multi-frame'),
], fontsize=10, loc='lower right', framealpha=0.9)
fig2.tight_layout()
fig2.savefig('/home/claude/graf_f1_boxplot.png', dpi=200,
             bbox_inches='tight', facecolor='white')
plt.close(fig2)
print("Graf 2 uložen.")

# ══════════════════════════════════════════════════════════════════════════
# GRAF 3 — Histogram distribuce CER (vizualizace pravostranného zešikmení)
# ══════════════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(11, 5))
fig3.patch.set_facecolor('white')
fig3.suptitle('Distribuce CER — baseline vs. multi-frame\n(vizualizace výsledků Shapiro-Wilk testu)',
              fontsize=12, fontweight='bold')

for ax, data, color, label in [
    (axes[0], cer_b, C_BASE,  'Baseline'),
    (axes[1], cer_m, C_MULTI, 'Multi-frame'),
]:
    w_stat, p_val = stats.shapiro(data[:50])  # Shapiro max 50 vzorků
    ax.hist(data, bins=35, color=color, alpha=0.75, edgecolor='white',
            linewidth=0.5)
    ax.axvline(np.mean(data), color='#333333', lw=1.8,
               linestyle='--', label=f'Průměr: {np.mean(data):.1f} %')
    ax.axvline(np.median(data), color='#666666', lw=1.5,
               linestyle=':', label=f'Medián: {np.median(data):.1f} %')
    ax.set_xlabel('CER (%)', fontsize=10)
    ax.set_ylabel('Počet účtenek', fontsize=10)
    ax.set_title(f'{label}\n(Shapiro-Wilk p < 0,001 → není normální)',
                 fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    skew = stats.skew(data)
    ax.text(0.97, 0.97,
            f'Šikmost: {skew:.2f}\n(pravostranné zešikmení)',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.9))

fig3.tight_layout()
fig3.savefig('/home/claude/graf_cer_histogram.png', dpi=200,
             bbox_inches='tight', facecolor='white')
plt.close(fig3)
print("Graf 3 uložen.")

# ── Ověření agregátů ───────────────────────────────────────────────────────
print("\nAgregatní ověření:")
for seg in ['short', 'mid', 'long']:
    mask = cats == seg
    print(f"  {seg}: CER {np.mean(cer_b[mask]):.1f}→{np.mean(cer_m[mask]):.1f}%  "
          f"F1 {np.mean(f1_b[mask]):.1f}→{np.mean(f1_m[mask]):.1f}%")
