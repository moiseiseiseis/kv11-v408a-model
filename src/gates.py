# gates.py — Funciones de compuerta Nav y Kv1.1
# Responsable: Persona 2
# Solo calcula valores puntuales — sin ODEs, sin simulaciones.
# Todas las funciones aceptan float o np.ndarray.

import numpy as np
from params import (gNa, ENa, gL, EL, EK,
                    V_half_n, k_n, V_half_hk, k_hk,
                    tau_scale_WT, tau_scale_MUT)

# ══════════════════════════════════════════════════════════
# CANAL Nav — Hodgkin-Huxley estándar
# ══════════════════════════════════════════════════════════

def alpha_m(V):
    """Tasa de apertura compuerta m (ms⁻¹). CUIDADO: singularidad en V = -40 mV."""
    # Desvío seguro para evitar advertencias de división por cero en el denominador
    V_safe = np.where(np.abs(V + 40.0) < 1e-6, V + 1e-6, V)
    alpha = 0.1 * (V_safe + 40.0) / (1.0 - np.exp(-(V_safe + 40.0) / 10.0))
    # Aplicar el límite de L'Hôpital en la singularidad
    return np.where(np.abs(V + 40.0) < 1e-6, 1.0, alpha)

def beta_m(V):
    """Tasa de cierre compuerta m (ms⁻¹)."""
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    """Tasa de apertura compuerta h (ms⁻¹)."""
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    """Tasa de cierre compuerta h (ms⁻¹)."""
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
def m_inf(V):
    """Estado estacionario de activación Nav."""
    return alpha_m(V) / (alpha_m(V) + beta_m(V))

def h_inf(V):
    """Estado estacionario de inactivación Nav."""
    return alpha_h(V) / (alpha_h(V) + beta_h(V))

def tau_m(V):
    """Constante de tiempo de activación Nav (ms)."""
    return 1.0 / (alpha_m(V) + beta_m(V))

def tau_h(V):
    """Constante de tiempo de inactivación Nav (ms)."""
    return 1.0 / (alpha_h(V) + beta_h(V))


# ══════════════════════════════════════════════════════════
# CANAL Kv1.1 — Retificador tardío
# ══════════════════════════════════════════════════════════

def n_inf(V):
    """
    Estado estacionario de activación Kv1.1 — curva de Boltzmann.
    n_inf = 1 / (1 + exp(-(V - V_half_n) / k_n))
    """
    return 1.0 / (1.0 + np.exp(-(V - V_half_n) / k_n))

def tau_n(V):
    """
    Constante de tiempo de activación Kv1.1 (ms).
    Usa las tasas alpha/beta del HH clásico como aproximación.
    """
    # Singularidad en V = -55 mV -> Límite es 0.1
    V_safe = np.where(np.abs(V + 55.0) < 1e-6, V + 1e-6, V)
    alpha_n = 0.01 * (V_safe + 55.0) / (1.0 - np.exp(-(V_safe + 55.0) / 10.0))
    alpha_n = np.where(np.abs(V + 55.0) < 1e-6, 0.1, alpha_n)
    
    beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
    return 1.0 / (alpha_n + beta_n) + 0.5

def hk_inf(V, mutant=False):
    """
    Estado estacionario de inactivación lenta Kv1.1.
    Misma curva WT y V408A — la mutación NO desplaza V½ (Miceli 2023).
    hk_inf = 1 / (1 + exp((V - V_half_hk) / k_hk))
    """
    # El signo positivo indica que se inactiva con la depolarización
    return 1.0 / (1.0 + np.exp((V - V_half_hk) / k_hk))

def tau_hk(V, mutant=False):
    """
    Constante de tiempo de inactivación lenta Kv1.1 (ms).
    Forma de campana con dependencia de voltaje.
    V408A: multiply por tau_scale_MUT (0.6) → 40% más rápido.
    """
    scale = tau_scale_MUT if mutant else tau_scale_WT
    tau_base = 200.0 / (np.exp((V + 40.0) / 20.0) + np.exp(-(V + 40.0) / 20.0)) + 50.0
    return tau_base * scale

# ══════════════════════════════════════════════════════════
# TESTS — correr con: python gates.py
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    V_test = np.linspace(-100, 60, 200)

    print("Corriendo tests de gates.py...\n")

    # Test 1: rangos [0,1]
    n = n_inf(V_test)
    assert np.all(n >= 0) and np.all(n <= 1), "FALLO: n_inf fuera de [0,1]"
    print("✓ n_inf está en [0,1]")

    hk_wt  = hk_inf(V_test, mutant=False)
    hk_mut = hk_inf(V_test, mutant=True)
    assert np.all(hk_wt  >= 0) and np.all(hk_wt  <= 1), "FALLO: hk_inf WT fuera de [0,1]"
    assert np.all(hk_mut >= 0) and np.all(hk_mut <= 1), "FALLO: hk_inf MUT fuera de [0,1]"
    print("✓ hk_inf WT y MUT están en [0,1]")

    # Test 2: V408A inactiva MÁS RÁPIDO
    tau_wt  = tau_hk(-20.0, mutant=False)
    tau_mut = tau_hk(-20.0, mutant=True)
    assert tau_mut < tau_wt, f"FALLO: tau_hk MUT ({tau_mut:.1f}) debe ser < WT ({tau_wt:.1f})"
    print(f"✓ tau_hk V408A ({tau_mut:.1f} ms) < WT ({tau_wt:.1f} ms) — inactivación acelerada")

    # Test 3: m_inf y h_inf en reposo
    m0 = m_inf(-65.0)
    h0 = h_inf(-65.0)
    assert 0 < m0 < 0.1, f"FALLO: m_inf en reposo debe ser ~0.05, got {m0:.4f}"
    assert 0.5 < h0 < 1.0, f"FALLO: h_inf en reposo debe ser ~0.6, got {h0:.4f}"
    print(f"✓ En reposo: m_inf = {m0:.4f}, h_inf = {h0:.4f}")

    # Test 4: n_inf sube con depolarización
    assert n_inf(0.0) > n_inf(-65.0), "FALLO: n_inf debe subir con depolarización"
    print("✓ n_inf(0 mV) > n_inf(-65 mV) — activación correcta")

    print("\n Todos los tests pasaron. gates.py listo.")
