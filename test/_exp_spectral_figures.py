"""
Spectral Rationality Paper Figures

Generates 4 publication-quality figures:
  Fig 1: Generator set → Spectrum → Field (flow diagram, revised)
  Fig 2: Character-sum cancellation (ω + ω² + 1 = 0, revised)
  Fig 3: Spectral distribution (eigenvalue vs multiplicity, revised)
  Fig 4: Galois mechanism bridge diagram (NEW)

Run: python test/_exp_spectral_figures.py
"""
import sys
import io
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except (ValueError, AttributeError):
        pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

sys.path.insert(0, '.')
from rime.cubieoperator import (CubieMove, eigenspaces, build_A,
                                 classify_spectral_field, spectral_field_label)
from rime.cubieworld import SlowDynamics
from rime.helpers import is_in_qsqrt5, is_rational_form
from rime.base import DATA_DIR

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8,
    'figure.dpi': 150,
})

SAVE_DIR = os.path.join(DATA_DIR, 'paper_figures')
os.makedirs(SAVE_DIR, exist_ok=True)

sd = SlowDynamics.lite()
prim = CubieMove.prim_moves()


def face_label(key):
    axis_names = {0: 'R', 1: 'U', 2: 'F'}
    dir_names = {-1: '', 1: "'", 2: '2'}
    a, s, d = key
    base = axis_names.get(a, f'a{a}')
    side_suffix = {1: '', -1: "'"}.get(s, f's{s}')
    if d == 2:
        return f'{base}2'
    return base + dir_names.get(d, '')


# ============================================================
# Precompute spectral data for all generator sets
# ============================================================
gen_configs = [
    ('18 full', dict(prim)),
    ('12 quarter', {k: v for k, v in prim.items() if k[2] != 2}),
    ('6 half-turn', {k: v for k, v in prim.items() if k[2] == 2}),
]
for n in [8, 10, 16]:
    rm = sd.rho_moves(n=n)
    if len(rm) > 0:
        gen_configs.append((f'n={n}', {k: prim[k] for k in rm if k in prim}))

import random
random.seed(42)
keys_18 = list(prim.keys())
random.shuffle(keys_18)
gen_configs.append(('random 9', {k: prim[k] for k in keys_18[:9]}))

spectral_data = {}
for name, gens in gen_configs:
    A = build_A(gens)
    sp = eigenspaces(A)
    eigs = sorted(sp.keys())
    dims = [sp[lam]['dim'] for lam in eigs]
    m = len(gens)
    m_eff = m // 2 if m % 2 == 0 else m
    is_rational = all(abs(lam - round(lam * m_eff) / m_eff) < 1e-5 for lam in eigs)
    irrational_eigs = [lam for lam in eigs if abs(lam - round(lam * m_eff) / m_eff) >= 1e-5]
    spectral_data[name] = {
        'eigs': eigs, 'dims': dims, 'spaces': sp, 'A': A,
        'n_gen': m, 'm_eff': m_eff, 'is_rational': is_rational,
        'irrational_eigs': irrational_eigs, 'gens': gens,
        'n_eig': len(eigs),
    }


def _spec_description(name, data, set_class):
    """Return structure-oriented description of the spectrum."""
    if set_class == 'rational':
        return f"Rational\nλ = 1−k/{data['m_eff']}"
    elif set_class == 'sqrt5':
        return r"Irrational" + "\n" + r"$\lambda \in \mathbb{Q}(\sqrt{5})$"
    else:
        return "Mixed field\n" + r"$\mathbb{Q} \subset K_S$"


for name, data in spectral_data.items():
    data['set_class'] = classify_spectral_field(data['eigs'], data['m_eff'], name)
    data['field'] = spectral_field_label(data['set_class'])
    data['spec_desc'] = _spec_description(name, data, data['set_class'])


# ============================================================
# Figure 1: Generator → Spectrum → Field (REVISED)
# ============================================================
def draw_fig1():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Column positions
    x_gen, x_spec, x_field = 1.8, 6.0, 10.2
    y_top = 7.2

    # Title
    ax.text(6, 7.8, 'Generator Set  →  Spectrum  →  Spectral Field',
            ha='center', va='center', fontsize=16, fontweight='bold')

    # Column headers
    for x, label in [(x_gen, 'Generator set $S$'),
                      (x_spec, r'$\mathrm{Spec}(A_S)$'),
                      (x_field, r'Field $K_S = \mathbb{Q}(\{\lambda\})$')]:
        ax.text(x, y_top, label, ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8E8E8',
                          edgecolor='black', lw=1.2))

    # Arrows between columns
    arrow_style = dict(arrowstyle='->', color='#555555', lw=1.5,
                       connectionstyle='arc3,rad=0')
    for y_off in [-0.8, -2.0, -3.2, -4.4, -5.6]:
        ax.annotate('', xy=(x_spec - 1.0, y_top + y_off),
                    xytext=(x_gen + 1.0, y_top + y_off),
                    arrowprops=arrow_style)
        ax.annotate('', xy=(x_field - 1.0, y_top + y_off),
                    xytext=(x_spec + 1.0, y_top + y_off),
                    arrowprops=arrow_style)

    # Data rows
    rows = [
        ('18 full', 'face-complete', '#2E7D32'),
        ('12 quarter', 'face-complete', '#2E7D32'),
        ('6 half-turn', 'face-complete', '#2E7D32'),
        ('n=8', 'symmetry-broken', '#E65100'),
        ('n=16', 'symmetry-broken', '#E65100'),
        ('random 9', 'non-symmetric', '#C62828'),
    ]
    row_h = 1.05

    for i, (name, regime, color) in enumerate(rows):
        y = y_top - (i + 1) * row_h
        data = spectral_data.get(name)
        if data is None:
            continue

        # Generator set box
        gen_text = f'{name}\n({data["n_gen"]} moves)'
        ax.text(x_gen, y, gen_text, ha='center', va='center', fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.12,
                          edgecolor=color, lw=1.8))

        # Spectrum — structure-oriented
        spec_text = data['spec_desc'] + f'\n{data["n_eig"]} distinct values'
        ax.text(x_spec, y, spec_text, ha='center', va='center', fontsize=9.5,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#FAFAFA',
                          edgecolor='#888', lw=1))

        # Field — mathematical notation
        ax.text(x_field, y, data['field'], ha='center', va='center', fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.18,
                          edgecolor=color, lw=1.8))

    # ---- Legend ----
    legend_y = 0.6
    for x, label, color in [(1.2, 'face-symmetric', '#2E7D32'),
                             (4.2, 'symmetry-broken', '#E65100'),
                             (7.2, 'non-symmetric', '#C62828')]:
        ax.add_patch(FancyBboxPatch((x - 0.4, legend_y - 0.13), 0.28, 0.26,
                                     boxstyle='round,pad=0.04',
                                     facecolor=color, alpha=0.3, edgecolor=color, lw=1.5))
        ax.text(x + 0.1, legend_y, label, va='center', fontsize=9.5)

    # ---- Conclusion slogan (highlighted box at the bottom) ----
    slogan_box = FancyBboxPatch((0.8, 0.05), 10.4, 0.38,
                                 boxstyle='round,pad=0.1',
                                 facecolor='#1A237E', alpha=0.08,
                                 edgecolor='#1A237E', lw=2)
    ax.add_patch(slogan_box)
    ax.text(6, 0.24,
            r'Face symmetry $\Rightarrow$ rational spectrum ($K_S = \mathbb{Q}$)'
            r'$\qquad$|$\qquad$'
            r'Symmetry breaking $\Rightarrow$ field extension',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#1A237E')

    plt.tight_layout(pad=0.5)
    path = os.path.join(SAVE_DIR, 'fig1_generator_spectrum_field')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', bbox_inches='tight')
    plt.close()
    print(f'Fig 1 saved: {path}.png/pdf')


# ============================================================
# Figure 2: Character-sum cancellation (REVISED)
# ============================================================
def draw_fig2():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                              gridspec_kw={'width_ratios': [1.2, 1.2, 1.5]})

    omega = np.exp(2j * np.pi / 3)

    # ---- Panel (a): ω^k values on the unit circle ----
    ax = axes[0]
    ax.set_aspect('equal')
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_title(r'(a) $\omega^k$ on the unit circle', fontsize=12, fontweight='bold')

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.5, alpha=0.3)

    points = {
        r'$1 = \omega^0$': 1 + 0j,
        r'$\omega = e^{2\pi i/3}$': omega,
        r'$\omega^2 = \omega^{-1}$': omega ** 2,
    }
    colors_pt = ['#2196F3', '#F44336', '#4CAF50']
    for (label, z), color in zip(points.items(), colors_pt):
        ax.plot(z.real, z.imag, 'o', color=color, ms=13, zorder=5)
        offset_dir = 1 if z.real >= 0 else -1
        ax.annotate(label, (z.real, z.imag),
                    textcoords='offset points', xytext=(15 * offset_dir, 10),
                    fontsize=11, color=color, fontweight='bold')

    arc_theta = np.linspace(0, 2 * np.pi / 3, 50)
    ax.plot(0.3 * np.cos(arc_theta), 0.3 * np.sin(arc_theta), 'k-', lw=1, alpha=0.5)
    ax.text(0.38, 0.15, r'$120°$', fontsize=9, alpha=0.6)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.3)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.3)
    ax.set_xlabel('Re', fontsize=10)
    ax.set_ylabel('Im', fontsize=10)

    # ---- Panel (b): Per-move trace contributions ----
    ax = axes[1]
    ax.set_title(r'(b) Corner-orientation trace per move', fontsize=12, fontweight='bold')

    moves_label = [r'$s$ (CW)', r'$s^{-1}$ (CCW)', r'$s^2$ (180°)']
    chi_values = [omega, omega ** 2, 1.0 + 0j]
    chi_real = [z.real for z in chi_values]
    chi_imag = [z.imag for z in chi_values]

    x = np.arange(3)
    width = 0.35
    ax.bar(x - width / 2, chi_real, width, label=r'$\mathrm{Re}(\chi_{\mathrm{co}})$',
           color='#2196F3', alpha=0.8)
    ax.bar(x + width / 2, chi_imag, width, label=r'$\mathrm{Im}(\chi_{\mathrm{co}})$',
           color='#F44336', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(moves_label, fontsize=10)
    ax.set_ylabel('Trace contribution', fontsize=10)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylim(-1.2, 1.5)

    # Sum display
    sum_real = sum(chi_real)
    sum_imag = sum(chi_imag)
    ax.text(1, -1.05, r'$\sum \mathrm{Re} = %.1f,\; \sum \mathrm{Im} = %.1f$' % (sum_real, sum_imag),
            ha='center', fontsize=10.5, color='#4CAF50', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#4CAF50'))

    # ---- Panel (c): Cancellation table ----
    ax = axes[2]
    ax.axis('off')
    ax.set_title(r'(c) Face-sum cancellation', fontsize=12, fontweight='bold')

    table_data = [
        [r'$s$ (CW)', r'$\omega$', r'$\cos(120°)$', r'$\sin(120°)$'],
        [r'$s^{-1}$ (CCW)', r'$\omega^2$', r'$\cos(240°)$', r'$-\sin(120°)$'],
        [r'$s^2$ (180°)', r'$1$', r'$1$', r'$0$'],
    ]
    col_labels = ['Move', r'$\chi_{\mathrm{co}}$', r'Re', r'Im']

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    for j in range(4):
        table[0, j].set_facecolor('#E3F2FD')
        table[0, j].set_text_props(fontweight='bold')
    for i in range(1, 4):
        table[i, 3].set_facecolor('#FFF3E0')

    # ---- Full causal chain annotation (UPGRADED) ----
    causal_text = (
        r'$\mathbf{\omega + \omega^2 + 1 = 0}$'
        '\n'
        r'$\Downarrow$'
        '\n'
        r'Imaginary parts cancel per face'
        '\n'
        r'$\Downarrow$'
        '\n'
        r'$\chi_{\mathrm{face}} = \chi(s) + \chi(s^{-1}) + \chi(s^2) \in \mathbb{Z}$'
        '\n'
        r'$\Downarrow$'
        '\n'
        r'$\mathbf{\lambda \in \mathbb{Q}}$'
    )
    ax.text(0.5, 0.05,
            causal_text,
            ha='center', va='center', fontsize=11, color='#2E7D32', fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#C8E6C9',
                      edgecolor='#4CAF50', lw=1.8))

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig2_character_cancellation')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', bbox_inches='tight')
    plt.close()
    print(f'Fig 2 saved: {path}.png/pdf')


# ============================================================
# Figure 3: Spectral distribution (REVISED)
# ============================================================
def draw_fig3():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sets_to_plot = [
        ('18 full', r'18 full (face-symmetric)', r'$K_S = \mathbb{Q}$', '#2E7D32'),
        ('n=8', r'n=8 (symmetry-broken)', r'$K_S = \mathbb{Q}(\sqrt{5})$', '#E65100'),
        ('random 9', r'random 9 (non-symmetric)', r'$\mathbb{Q} \subset K_S$', '#C62828'),
    ]

    for ax, (name, title, field_label, color) in zip(axes, sets_to_plot):
        data = spectral_data.get(name)
        if data is None:
            continue

        eigs = data['eigs']
        dims = data['dims']

        markerline, stemlines, baseline = ax.stem(
            eigs, dims, linefmt='-', markerfmt='o', basefmt=' ')
        plt.setp(stemlines, color=color, lw=2.5, alpha=0.8)
        plt.setp(markerline, color=color, ms=8, zorder=5)

        # ---- Label only key eigenvalues (simplified from old version) ----
        is_rational_set = data['is_rational']

        for lam, d in zip(eigs, dims):
            if is_rational_set:
                # Rational set: show λ = 1−k/m formula for each stem
                m = data['m_eff']
                k = round((1 - lam) * m)
                if abs(lam - (1 - k / m)) < 1e-5:
                    label = f'$1-{k}/{m}$'
                else:
                    label = f'{lam:.3f}'
            else:
                # Non-rational set: only label irrational eigenvalues
                label = None
                if lam in data['irrational_eigs']:
                    label = f'{lam:.4f}'
                elif len(data['irrational_eigs']) == 0:
                    # Mixed but all individually rationalizable
                    label = f'{lam:.3f}'

            if label is not None:
                ax.annotate(label, (lam, d),
                            textcoords='offset points', xytext=(0, 7),
                            ha='center', fontsize=7.5, alpha=0.85)

        # Highlight irrational eigenvalues with red markers
        for lam in data['irrational_eigs']:
            idx = eigs.index(lam)
            ax.plot(lam, dims[idx], 's', color='red', ms=11, zorder=4,
                    alpha=0.7, markeredgecolor='darkred', markeredgewidth=1)
            ax.annotate(r'$\notin \mathbb{Q}$', (lam, dims[idx]),
                        textcoords='offset points', xytext=(15, -8),
                        fontsize=9.5, color='red', fontweight='bold')

        ax.set_xlabel(r'Eigenvalue $\lambda$', fontsize=11)
        ax.set_ylabel('Multiplicity', fontsize=11)
        # Title with field label included
        ax.set_title(title + '\n' + field_label, fontsize=10.5, fontweight='bold',
                     color=color)
        ax.set_xlim(-0.08, 1.15)
        ax.grid(True, alpha=0.15)

        # Reference grid lines for rational sets
        if is_rational_set:
            m = data['m_eff']
            for k in range(m + 1):
                lam_k = 1 - k / m
                ax.axvline(lam_k, color='gray', lw=0.3, alpha=0.4, ls='--')

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig3_spectral_distribution')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', bbox_inches='tight')
    plt.close()
    print(f'Fig 3 saved: {path}.png/pdf')


# ============================================================
# Figure 4: Galois mechanism bridge diagram (NEW)
# ============================================================
def draw_fig4():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Color scheme
    LEFT_COLOR = '#2E7D32'    # face-complete: green
    RIGHT_COLOR = '#E65100'   # symmetry-broken: orange
    LEFT_BG = '#E8F5E9'
    RIGHT_BG = '#FFF3E0'
    CONCLUSION_COLOR = '#1A237E'

    # ---- Main title ----
    ax.text(8, 10.55,
            'Galois Symmetry as the Mechanism of Spectral Rationality',
            ha='center', va='center', fontsize=17, fontweight='bold',
            color='#1A237E')

    # ---- Subtitle ----
    ax.text(8, 10.0,
            r'Face-sum arithmetic determines the spectral field $K_S$',
            ha='center', va='center', fontsize=12,
            color='#555555')

    # ---- Column headers ----
    ax.text(4, 9.2, r'Face-complete $S$', ha='center', va='center',
            fontsize=14, fontweight='bold', color=LEFT_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=LEFT_BG,
                      edgecolor=LEFT_COLOR, lw=2))

    ax.text(12, 9.2, r'Symmetry-broken $S$', ha='center', va='center',
            fontsize=14, fontweight='bold', color=RIGHT_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=RIGHT_BG,
                      edgecolor=RIGHT_COLOR, lw=2))

    # "vs" in the middle
    ax.text(8, 9.2, 'vs', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#888888',
            fontstyle='italic')

    # ---- Box drawing helper ----
    def draw_box(ax, x, y, w, h, text, edge_color, bg_color, fontsize=10.5):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle='round,pad=0.12',
                              facecolor=bg_color, edgecolor=edge_color,
                              lw=2, alpha=1.0)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=edge_color)
        return box

    def draw_arrow(ax, x, y_top, y_bot, color, lw=2.5):
        ax.annotate('', xy=(x, y_bot + 0.25), xytext=(x, y_top - 0.25),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    connectionstyle='arc3,rad=0'))

    box_w = 5.2
    box_h = 1.05

    # ---- Left column boxes (face-complete) ----
    left_x = 4
    left_ys = [8.3, 6.7, 5.1, 3.5, 1.9]  # y-positions top to bottom

    left_boxes = [
        (r'$\mathbf{\sigma(A_S) = A_S}$' '\n' r'Galois action fixes $A_S$'),
        (r'$\mathbf{\omega + \omega^2 + 1 = 0}$' '\n'
         r'Per-face: CW + CCW + 180° = full sum'),
        (r'$\mathbf{\mathrm{Im}(\chi_{\mathrm{face}}) = 0}$' '\n'
         r'Imaginary parts cancel exactly'),
        (r'$\mathbf{\chi_{\mathrm{face}} \in \mathbb{Z}}$' '\n'
         r'Face character is integral'),
        (r'$\mathbf{K_S = \mathbb{Q}}$' '\n'
         r'Spectral field is rational'),
    ]

    for i, (text, (x, y)) in enumerate(zip(left_boxes,
                                            [(left_x, y) for y in left_ys])):
        bg = LEFT_BG if i < 4 else '#A5D6A7'
        ecol = LEFT_COLOR
        fs = 10.5 if i < 4 else 12
        draw_box(ax, x, y, box_w, box_h, text, ecol, bg, fs)
        if i < len(left_ys) - 1:
            draw_arrow(ax, x, left_ys[i], left_ys[i + 1], LEFT_COLOR)

    # ---- Right column boxes (symmetry-broken) ----
    right_x = 12
    right_ys = [8.3, 6.7, 5.1, 3.5, 1.9]

    right_boxes = [
        (r'$\mathbf{\sigma(A_S) \neq A_S}$' '\n' r'Galois action non-trivial'),
        (r'$\mathbf{\omega + \omega^{-1} \neq \text{integer}}$' '\n'
         r'Face incomplete: 180° move missing'),
        (r'$\mathbf{\mathrm{Im}(\chi_{\mathrm{face}}) \neq 0}$' '\n'
         r'Residual imaginary component'),
        (r'$\mathbf{\chi_{\mathrm{face}} \notin \mathbb{Z}}$' '\n'
         r'Face character is non-integral'),
        (r'$\mathbf{K_S = \mathbb{Q}(\sqrt{5})}$' '\n'
         r'Spectral field requires extension'),
    ]

    for i, (text, (x, y)) in enumerate(zip(right_boxes,
                                            [(right_x, y) for y in right_ys])):
        bg = RIGHT_BG if i < 4 else '#FFCC80'
        ecol = RIGHT_COLOR
        fs = 10.5 if i < 4 else 12
        draw_box(ax, x, y, box_w, box_h, text, ecol, bg, fs)
        if i < len(right_ys) - 1:
            draw_arrow(ax, x, right_ys[i], right_ys[i + 1], RIGHT_COLOR)

    # ---- Horizontal "bridge" labels between columns ----
    bridge_labels = [
        (8, 8.3, 'Galois action'),
        (8, 6.7, 'Face-sum arithmetic'),
        (8, 5.1, 'Cancellation check'),
        (8, 3.5, 'Character integrality'),
        (8, 1.9, 'Field determination'),
    ]
    for x, y, label in bridge_labels:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=8.5, color='#888', fontstyle='italic', alpha=0.8)

    # ---- Central "mechanism" bar ----
    central_box = FancyBboxPatch((0.5, 0.55), 15, 0.5,
                                  boxstyle='round,pad=0.1',
                                  facecolor='#E8EAF6', edgecolor=CONCLUSION_COLOR,
                                  lw=2.5)
    ax.add_patch(central_box)
    ax.text(8, 0.8,
            r'$\mathbf{\sigma(A) = A \;\Rightarrow\; K_S = \mathbb{Q}}$'
            r'$\qquad\mathbf{vs}\qquad$'
            r'$\mathbf{\sigma(A) \neq A \;\Rightarrow\; K_S \supseteq \mathbb{Q}(\sqrt{5})}$',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=CONCLUSION_COLOR)

    # ---- Bottom footnote ----
    ax.text(8, 0.15,
            r'Theorem (informal): Face-symmetry of $S$ is necessary and sufficient for $K_S=\mathbb{Q}$',
            ha='center', va='center', fontsize=9.5, color='#555555',
            fontstyle='italic')

    plt.tight_layout(pad=0.5)
    path = os.path.join(SAVE_DIR, 'fig4_galois_mechanism')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', bbox_inches='tight')
    plt.close()
    print(f'Fig 4 saved: {path}.png/pdf')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('Generating spectral rationality paper figures...')
    print(f'Save directory: {SAVE_DIR}')
    print()

    draw_fig1()
    draw_fig2()
    draw_fig3()
    draw_fig4()

    print()
    print('All 4 figures generated.')
    print(f'Output: {SAVE_DIR}/')
