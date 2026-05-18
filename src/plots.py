# plots.py — Generación de figuras para el manuscrito
# Responsable: Personas 6 y 7
# Cada figura es una función independiente.
# Guardar siempre en outputs/ con 300 dpi.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gates import n_inf, hk_inf
from params import V_half_n, V_half_hk

# ── Estilo global coherente para el manuscrito ─────────────
COLORS = {
    'wt'  : '#2196F3',   # azul  — silvestre
    'mut' : '#F44336',   # rojo  — V408A
    'ref' : '#9E9E9E',   # gris  — referencia / fuga
}
LW        = 1.8    # linewidth
FONTLABEL = 11     # tamaño etiquetas
FONTTITLE = 12     # tamaño títulos
DPI       = 300
OUT       = 'outputs/'

def _style(ax, xlabel, ylabel, title, legend=True):
    """Aplica estilo común a todos los ejes."""
    ax.set_xlabel(xlabel, fontsize=FONTLABEL)
    ax.set_ylabel(ylabel, fontsize=FONTLABEL)
    ax.set_title(title, fontsize=FONTTITLE, fontweight='bold')
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legend:
        ax.legend(fontsize=9, framealpha=0.7)


def fig1_action_potential(result_wt, result_mut):
    """
    FIGURA 1 — Potencial de acción único: WT vs V408A superpuestos.
    Anotar: voltaje pico, umbral aproximado, hiperpolarización post-PA.

    INPUT:
        result_wt, result_mut : dict — salida de run_simulation()
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(result_wt['t'],  result_wt['V'],
            color=COLORS['wt'],  lw=LW, label='Kv1.1 WT')
    ax.plot(result_mut['t'], result_mut['V'],
            color=COLORS['mut'], lw=LW, label='Kv1.1 V408A', linestyle='--')

    # TODO: anotar Vpico con ax.annotate()
    # TODO: marcar umbral con línea horizontal punteada

    ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)

    _style(ax,
           xlabel='Tiempo (ms)',
           ylabel='Voltaje de membrana (mV)',
           title='Figura 1 — Potencial de acción: Kv1.1 WT vs V408A')

    plt.tight_layout()
    plt.savefig(f'{OUT}figura_1_PA.png', dpi=DPI, bbox_inches='tight')
    print(f"✓ figura_1_PA.png guardada")
    plt.close()


def fig2_ionic_currents(result_wt, result_mut):
    """
    FIGURA 2 — Corrientes iónicas: INa, IK, IL para WT y V408A.
    3 subplots verticales.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    pairs = [
        ('INa', 'Corriente Na⁺ (µA/cm²)', 'I$_{Na}$'),
        ('IK',  'Corriente K⁺  (µA/cm²)', 'I$_{K}$ Kv1.1'),
        ('IL',  'Corriente fuga (µA/cm²)', 'I$_{L}$'),
    ]

    for ax, (key, ylabel, label) in zip(axes, pairs):
        ax.plot(result_wt['t'],  result_wt[key],
                color=COLORS['wt'],  lw=LW, label=f'{label} WT')
        ax.plot(result_mut['t'], result_mut[key],
                color=COLORS['mut'], lw=LW, label=f'{label} V408A', linestyle='--')
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.4)
        _style(ax, xlabel='Tiempo (ms)', ylabel=ylabel, title='')
        ax.set_title('')

    axes[0].set_title('Figura 2 — Corrientes iónicas: WT vs V408A',
                      fontsize=FONTTITLE, fontweight='bold')
    axes[2].set_xlabel('Tiempo (ms)', fontsize=FONTLABEL)

    plt.tight_layout()
    plt.savefig(f'{OUT}figura_2_corrientes.png', dpi=DPI, bbox_inches='tight')
    print(f"✓ figura_2_corrientes.png guardada")
    plt.close()


def fig3_train(result_wt_train, result_mut_train, spikes_wt=None, spikes_mut=None):
    """
    FIGURA 3 — Tren de estímulos: hiperexcitabilidad V408A.
    2 subplots: WT arriba, V408A abajo.
    Anotar número de spikes en cada panel.

    INPUT:
        spikes_wt, spikes_mut : int — de count_spikes() en analysis.py
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(result_wt_train['t'], result_wt_train['V'],
             color=COLORS['wt'], lw=1.2)
    if spikes_wt is not None:
        ax1.text(0.02, 0.88, f'Spikes: {spikes_wt}',
                 transform=ax1.transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax2.plot(result_mut_train['t'], result_mut_train['V'],
             color=COLORS['mut'], lw=1.2)
    if spikes_mut is not None:
        ax2.text(0.02, 0.88, f'Spikes: {spikes_mut}',
                 transform=ax2.transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    _style(ax1, '', 'Vm (mV)', 'WT — Kv1.1 silvestre', legend=False)
    _style(ax2, 'Tiempo (ms)', 'Vm (mV)', 'V408A — Hiperexcitabilidad', legend=False)

    fig.suptitle('Figura 3 — Tren de estímulos: WT vs V408A',
                 fontsize=FONTTITLE, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}figura_3_tren.png', dpi=DPI, bbox_inches='tight')
    print(f"✓ figura_3_tren.png guardada")
    plt.close()


def fig4_boltzmann():
    """
    FIGURA 4 — Curvas de estado estacionario (Boltzmann) de las compuertas Kv1.1.
    Muestra: n∞⁴, hk∞ WT, hk∞ V408A.
    Marcar V½ con línea punteada vertical.
    """
    V_range = np.linspace(-100, 60, 400)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(V_range, n_inf(V_range)**4,
            color=COLORS['wt'], lw=LW, label='n$^4_\\infty$ — Activación Kv1.1')
    ax.plot(V_range, hk_inf(V_range, mutant=False),
            color=COLORS['wt'], lw=LW, linestyle='--',
            label='hk$_\\infty$ WT — Inactivación')
    ax.plot(V_range, hk_inf(V_range, mutant=True),
            color=COLORS['mut'], lw=LW, linestyle='--',
            label='hk$_\\infty$ V408A — Inactivación')

    # Marcar V½ de activación
    ax.axvline(x=V_half_n, color=COLORS['wt'], linestyle=':', alpha=0.6,
               label=f'V½ act. = {V_half_n} mV')
    # Marcar V½ de inactivación
    ax.axvline(x=V_half_hk, color='gray', linestyle=':', alpha=0.6,
               label=f'V½ inact. = {V_half_hk} mV')

    ax.axhline(y=0.5, color='black', linestyle=':', linewidth=0.6, alpha=0.4)

    _style(ax,
           xlabel='Voltaje de membrana (mV)',
           ylabel='Probabilidad de apertura',
           title='Figura 4 — Curvas de Boltzmann: compuertas Kv1.1')
    ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig(f'{OUT}figura_4_boltzmann.png', dpi=DPI, bbox_inches='tight')
    print(f"✓ figura_4_boltzmann.png guardada")
    plt.close()


def fig5_threshold_bar(thresh_wt, thresh_mut):
    """
    FIGURA 5 — Comparación de umbral de disparo (gráfica de barras).
    Anotar el valor encima de cada barra.

    INPUT:
        thresh_wt, thresh_mut : float — de find_threshold() en analysis.py
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    labels = ['WT', 'V408A']
    values = [thresh_wt, thresh_mut]
    bar_colors = [COLORS['wt'], COLORS['mut']]

    bars = ax.bar(labels, values, color=bar_colors, width=0.45,
                  edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'{val:.2f} µA/cm²',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_ylabel('Corriente umbral (µA/cm²)', fontsize=FONTLABEL)
    ax.set_title('Figura 5 — Umbral de disparo: WT vs V408A',
                 fontsize=FONTTITLE, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Anotar reducción porcentual
    delta = ((thresh_wt - thresh_mut) / thresh_wt) * 100
    ax.text(0.5, 0.92, f'Reducción: {delta:.1f}%',
            transform=ax.transAxes, ha='center', fontsize=10,
            color=COLORS['mut'], fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}figura_5_umbral.png', dpi=DPI, bbox_inches='tight')
    print(f"✓ figura_5_umbral.png guardada")
    plt.close()


# ══════════════════════════════════════════════════════════
# TEST RÁPIDO — correr con: python plots.py
# (requiere model.py, gates.py, stimulus.py, params.py funcionando)
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)

    print("Generando figura de prueba (Boltzmann)...\n")
    fig4_boltzmann()
    print("\n✅ fig4_boltzmann generada sin errores.")
    print("Para generar todas las figuras, correr main.py")
