# analysis.py — Análisis cuantitativo de resultados
# Responsable: Persona 5
# Extrae métricas del potencial de acción y produce la tabla del manuscrito.
# Depende de: model.py, stimulus.py, params.py

import numpy as np
import pandas as pd
from params import gKv11_WT, gKv11_MUT, t_stim_start


def find_threshold(gKv11, mutant, I_min=1.0, I_max=20.0):
    """
    Determina la corriente mínima (umbral) para generar un potencial de acción.
    Método: búsqueda binaria (bisección) — 20 iteraciones → precisión ~0.01 µA/cm²

    INPUT:
        gKv11  : float — conductancia Kv1.1 (WT o MUT)
        mutant : bool  — True = V408A
        I_min  : float — límite inferior de búsqueda (µA/cm²)
        I_max  : float — límite superior de búsqueda (µA/cm²)

    OUTPUT:
        float — corriente umbral en µA/cm²
    """
    # Import aquí para evitar importación circular
    from model import run_simulation
    from stimulus import make_pulse

    lo, hi = I_min, I_max

    for _ in range(20):
        mid = (lo + hi) / 2
        stim = make_pulse(mid, t_stim_start, 1.0)
        result = run_simulation(gKv11, mutant, stim, t_end_=200.0)

        # Criterio de disparo: el voltaje supera 0 mV
        fired = np.max(result['V']) > 0.0

        if fired:
            hi = mid   # umbral está por debajo
        else:
            lo = mid   # umbral está por encima

    return (lo + hi) / 2


def get_spike_metrics(result):
    """
    Calcula métricas cuantitativas del potencial de acción.

    INPUT:
        result : dict — salida de run_simulation()

    OUTPUT:
        dict con:
            'V_peak'      : float (mV)  — voltaje pico
            'V_trough'    : float (mV)  — hiperpolarización máxima
            'AP_width'    : float (ms)  — ancho a mitad de amplitud (FWHM)
            'latency'     : float (ms)  — tiempo del estímulo al pico
            'IK_peak'     : float (µA/cm²) — corriente K+ máxima
            'IK_integral' : float       — carga total de K+ (integral de IK)
    """
    t  = result['t']
    V  = result['V']
    IK = result['IK']

    metrics = {}

    # Pico
    idx_peak       = np.argmax(V)
    metrics['V_peak']   = V[idx_peak]
    metrics['latency']  = t[idx_peak] - t_stim_start

    # Hiperpolarización post-PA
    idx_after = idx_peak + int(1.0 / (t[1]-t[0]))   # buscar 1 ms después del pico
    if idx_after < len(V):
        metrics['V_trough'] = np.min(V[idx_after:idx_after + int(20.0/(t[1]-t[0]))])
    else:
        metrics['V_trough'] = np.min(V)

    # Ancho a mitad de amplitud (FWHM)
    # TODO: calcular AP_width
    # Pista: encontrar los índices donde V cruza (V_peak + V_rest) / 2
    # AP_width = t[idx_down] - t[idx_up]
    metrics['AP_width'] = None   # reemplazar con cálculo real

    # Corriente de K+
    metrics['IK_peak']     = np.max(IK)
    metrics['IK_integral'] = np.trapz(np.abs(IK), t)   # carga total µA·ms/cm²

    return metrics


def count_spikes(result, threshold_V=0.0):
    """
    Cuenta potenciales de acción en una simulación (útil para tren de estímulos).

    INPUT:
        result      : dict — salida de run_simulation()
        threshold_V : float — voltaje de cruce para detección (mV)

    OUTPUT:
        int — número de spikes detectados
    """
    V = result['V']
    # Detección por cruce ascendente del umbral
    crossings = np.where((V[:-1] < threshold_V) & (V[1:] >= threshold_V))[0]
    return len(crossings)


def build_summary_table(wt_metrics, mut_metrics, thresh_wt, thresh_mut,
                        spikes_wt, spikes_mut):
    """
    Construye la Tabla 1 del manuscrito — comparativa WT vs V408A.

    OUTPUT:
        pd.DataFrame con columnas: Parámetro | WT | V408A | Δ%
        También exporta tabla_parametros.csv
    """
    def delta(a, b):
        if a and b and a != 0:
            return f"{((b-a)/abs(a))*100:.1f}%"
        return "—"

    rows = [
        ("I umbral (µA/cm²)",   thresh_wt,                    thresh_mut),
        ("V pico (mV)",         wt_metrics.get('V_peak'),     mut_metrics.get('V_peak')),
        ("V mínimo AHP (mV)",   wt_metrics.get('V_trough'),   mut_metrics.get('V_trough')),
        ("Latencia al pico (ms)",wt_metrics.get('latency'),   mut_metrics.get('latency')),
        ("IK pico (µA/cm²)",    wt_metrics.get('IK_peak'),    mut_metrics.get('IK_peak')),
        ("Carga IK (µA·ms/cm²)",wt_metrics.get('IK_integral'),mut_metrics.get('IK_integral')),
        ("Spikes en tren",      spikes_wt,                    spikes_mut),
    ]

    data = []
    for param, wt_val, mut_val in rows:
        wt_str  = f"{wt_val:.3f}"  if isinstance(wt_val, float)  else str(wt_val)
        mut_str = f"{mut_val:.3f}" if isinstance(mut_val, float) else str(mut_val)
        d       = delta(wt_val, mut_val)
        data.append({'Parámetro': param, 'WT': wt_str, 'V408A': mut_str, 'Δ%': d})

    df = pd.DataFrame(data)
    df.to_csv('outputs/tabla_parametros.csv', index=False)
    print("✓ Tabla guardada en outputs/tabla_parametros.csv")
    return df


# ══════════════════════════════════════════════════════════
# TESTS — correr con: python analysis.py
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    from model import run_simulation
    from stimulus import I_single, I_train

    print("Corriendo tests de analysis.py...\n")

    # Test 1: umbral WT > umbral V408A (V408A más excitable)
    print("[1/3] Calculando umbral WT (búsqueda binaria)...")
    thresh_wt  = find_threshold(gKv11_WT,  False)
    print("[2/3] Calculando umbral V408A...")
    thresh_mut = find_threshold(gKv11_MUT, True)
    assert thresh_mut < thresh_wt, \
        f"FALLO: umbral V408A ({thresh_mut:.3f}) debe ser menor que WT ({thresh_wt:.3f})"
    print(f"✓ Umbral WT = {thresh_wt:.3f} µA/cm²")
    print(f"✓ Umbral V408A = {thresh_mut:.3f} µA/cm²")
    print(f"✓ Reducción = {((thresh_wt-thresh_mut)/thresh_wt)*100:.1f}% → mayor excitabilidad")

    # Test 2: métricas del PA
    print("\n[3/3] Calculando métricas del PA...")
    result_wt  = run_simulation(gKv11_WT,  False, I_single)
    result_mut = run_simulation(gKv11_MUT, True,  I_single)
    wt_m  = get_spike_metrics(result_wt)
    mut_m = get_spike_metrics(result_mut)
    assert wt_m['V_peak'] > 0,  "FALLO: V_peak WT debe ser > 0 mV"
    print(f"✓ V_peak WT = {wt_m['V_peak']:.1f} mV | V408A = {mut_m['V_peak']:.1f} mV")
    print(f"✓ IK_peak WT = {wt_m['IK_peak']:.2f} | V408A = {mut_m['IK_peak']:.2f} µA/cm²")

    # Test 3: contar spikes en tren
    r_wt_tr  = run_simulation(gKv11_WT,  False, I_train)
    r_mut_tr = run_simulation(gKv11_MUT, True,  I_train)
    spikes_wt  = count_spikes(r_wt_tr)
    spikes_mut = count_spikes(r_mut_tr)
    print(f"✓ Spikes WT = {spikes_wt} | V408A = {spikes_mut}")

    print("\n✅ Todos los tests pasaron. analysis.py listo.")
