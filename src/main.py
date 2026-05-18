# main.py — Orquestador del pipeline completo
# Responsable: Persona 3 (integración final)
# Correr con: python main.py
# Genera todos los outputs: 5 figuras + 1 CSV + resumen en terminal

import os
import numpy as np

# Crear carpeta de outputs si no existe
os.makedirs('outputs', exist_ok=True)

from params import gKv11_WT, gKv11_MUT
from stimulus import I_single, I_train
from model import run_simulation
from analysis import (find_threshold, get_spike_metrics,
                      count_spikes, build_summary_table)
from plots import (fig1_action_potential, fig2_ionic_currents,
                   fig3_train, fig4_boltzmann, fig5_threshold_bar)


def main():
    print("=" * 60)
    print("  Pipeline Kv1.1 WT vs V408A — Hodgkin-Huxley")
    print("=" * 60)

    # ── 1. Simulaciones ─────────────────────────────────────
    print("\n[1/4] Corriendo simulaciones...\n")

    print("  · WT  — pulso único")
    result_wt      = run_simulation(gKv11_WT,  mutant=False, stim_func=I_single)

    print("  · V408A — pulso único")
    result_mut     = run_simulation(gKv11_MUT, mutant=True,  stim_func=I_single)

    print("  · WT  — tren de estímulos")
    result_wt_tr   = run_simulation(gKv11_WT,  mutant=False, stim_func=I_train)

    print("  · V408A — tren de estímulos")
    result_mut_tr  = run_simulation(gKv11_MUT, mutant=True,  stim_func=I_train)

    # ── 2. Análisis ─────────────────────────────────────────
    print("\n[2/4] Calculando métricas...\n")

    print("  · Umbral WT  (búsqueda binaria)...")
    thresh_wt  = find_threshold(gKv11_WT,  mutant=False)

    print("  · Umbral V408A (búsqueda binaria)...")
    thresh_mut = find_threshold(gKv11_MUT, mutant=True)

    wt_metrics  = get_spike_metrics(result_wt)
    mut_metrics = get_spike_metrics(result_mut)

    spikes_wt   = count_spikes(result_wt_tr)
    spikes_mut  = count_spikes(result_mut_tr)

    # ── 3. Tabla CSV ────────────────────────────────────────
    print("\n[3/4] Generando tabla de resultados...\n")
    df = build_summary_table(wt_metrics, mut_metrics,
                             thresh_wt, thresh_mut,
                             spikes_wt, spikes_mut)
    print(df.to_string(index=False))

    # ── 4. Figuras ──────────────────────────────────────────
    print("\n[4/4] Generando figuras...\n")
    fig1_action_potential(result_wt, result_mut)
    fig2_ionic_currents(result_wt, result_mut)
    fig3_train(result_wt_tr, result_mut_tr, spikes_wt, spikes_mut)
    fig4_boltzmann()
    fig5_threshold_bar(thresh_wt, thresh_mut)

    # ── Resumen final ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESUMEN DE RESULTADOS")
    print("=" * 60)
    delta_thresh = ((thresh_wt - thresh_mut) / thresh_wt) * 100
    print(f"  Umbral WT        : {thresh_wt:.3f} µA/cm²")
    print(f"  Umbral V408A     : {thresh_mut:.3f} µA/cm²")
    print(f"  Reducción umbral : {delta_thresh:.1f}% → mayor excitabilidad")
    print(f"  Spikes WT (tren) : {spikes_wt}")
    print(f"  Spikes V408A     : {spikes_mut}")
    print(f"\n  Archivos generados en outputs/:")
    for f in ['figura_1_PA.png', 'figura_2_corrientes.png',
              'figura_3_tren.png', 'figura_4_boltzmann.png',
              'figura_5_umbral.png', 'tabla_parametros.csv']:
        print(f"    ✓ {f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
