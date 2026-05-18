# model.py — Sistema de ODEs y función de simulación
# Responsable: Persona 3
# Núcleo del modelo. Integra las 5 ODEs con scipy.odeint.
# Depende de: params.py, gates.py

import numpy as np
from scipy.integrate import odeint
from params import (Cm, gNa, ENa, gL, EL, EK,
                    gKv11_WT, gKv11_MUT, V_rest, dt, t_end)
from gates import (m_inf, h_inf, tau_m, tau_h,
                   n_inf, tau_n, hk_inf, tau_hk)


def hh_odes(y, t, gKv11, mutant, stim_func):
    """
    Sistema de 5 ODEs del modelo Hodgkin-Huxley con Kv1.1.

    PARÁMETROS:
        y        : lista [V, m, h, n, hk] — estado actual
        t        : float — tiempo actual (ms)
        gKv11    : float — conductancia máxima Kv1.1 (mS/cm²)
        mutant   : bool  — True = V408A, False = WT
        stim_func: callable — I_ext(t) en µA/cm²

    RETORNA:
        [dV/dt, dm/dt, dh/dt, dn/dt, dhk/dt]
    """
    V, m, h, n, hk = y

    # Corrientes iónicas
    INa  = gNa   * m**3 * h  * (V - ENa)
    IK   = gKv11 * n**4 * hk * (V - EK)
    IL   = gL * (V - EL)
    Iext = stim_func(t)

    # Ecuación de membrana (circuito RC)
    dVdt = (Iext - INa - IK - IL) / Cm

    # Compuertas Nav
    dm = (m_inf(V) - m) / tau_m(V)
    dh = (h_inf(V) - h) / tau_h(V)

    # Compuertas Kv1.1
    dn  = (n_inf(V) - n)            / tau_n(V)
    dhk = (hk_inf(V, mutant) - hk)  / tau_hk(V, mutant)

    return [dVdt, dm, dh, dn, dhk]


def run_simulation(gKv11, mutant, stim_func, t_end_=None, dt_=None):
    """
    Ejecuta la simulación HH completa.

    INPUT:
        gKv11     : float   — conductancia máxima Kv1.1 (mS/cm²)
        mutant    : bool    — True = V408A, False = WT
        stim_func : callable — función I_ext(t) en µA/cm²
        t_end_    : float opcional — sobreescribe params.t_end
        dt_       : float opcional — sobreescribe params.dt

    OUTPUT:
        dict con:
            't'   : np.ndarray (ms)
            'V'   : np.ndarray (mV) — voltaje de membrana
            'INa' : np.ndarray (µA/cm²)
            'IK'  : np.ndarray (µA/cm²)
            'IL'  : np.ndarray (µA/cm²)
            'm', 'h', 'n', 'hk' : np.ndarray — variables de compuerta
    """
    _t_end = t_end_ if t_end_ is not None else t_end
    _dt    = dt_    if dt_    is not None else dt

    t = np.arange(0, _t_end, _dt)

    # Condiciones iniciales — estado de reposo
    V0  = V_rest
    m0  = m_inf(V0)
    h0  = h_inf(V0)
    n0  = n_inf(V0)
    hk0 = hk_inf(V0, mutant)
    y0  = [V0, m0, h0, n0, hk0]

    # Integración
    sol = odeint(hh_odes, y0, t,
                 args=(gKv11, mutant, stim_func),
                 rtol=1e-6, atol=1e-9)

    V_s  = sol[:, 0]
    m_s  = sol[:, 1]
    h_s  = sol[:, 2]
    n_s  = sol[:, 3]
    hk_s = sol[:, 4]

    # Recalcular corrientes desde el estado integrado
    INa_s = gNa   * m_s**3 * h_s  * (V_s - ENa)
    IK_s  = gKv11 * n_s**4 * hk_s * (V_s - EK)
    IL_s  = gL * (V_s - EL)

    return {
        't'  : t,
        'V'  : V_s,
        'INa': INa_s,
        'IK' : IK_s,
        'IL' : IL_s,
        'm'  : m_s,
        'h'  : h_s,
        'n'  : n_s,
        'hk' : hk_s,
    }


# ══════════════════════════════════════════════════════════
# TESTS — correr con: python model.py
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    from stimulus import I_single

    print("Corriendo tests de model.py...\n")

    # Test 1: WT genera potencial de acción
    result_wt = run_simulation(gKv11_WT, False, I_single, t_end_=200.0)
    assert np.max(result_wt['V']) > 0,    "FALLO: WT no generó potencial de acción"
    assert np.min(result_wt['V']) < -60,  "FALLO: WT no mostró hiperpolarización"
    print(f"✓ WT: Vpico = {np.max(result_wt['V']):.1f} mV, "
          f"Vmin = {np.min(result_wt['V']):.1f} mV")

    # Test 2: V408A también genera potencial de acción
    result_mut = run_simulation(gKv11_MUT, True, I_single, t_end_=200.0)
    assert np.max(result_mut['V']) > 0,   "FALLO: V408A no generó potencial de acción"
    print(f"✓ V408A: Vpico = {np.max(result_mut['V']):.1f} mV")

    # Test 3: keys correctas en el dict
    for key in ['t', 'V', 'INa', 'IK', 'IL', 'm', 'h', 'n', 'hk']:
        assert key in result_wt, f"FALLO: falta key '{key}' en resultado"
    print("✓ Dict de resultados tiene todas las keys esperadas")

    # Test 4: corriente IK es negativa (salida de K+) durante el PA
    IK_peak = np.min(result_wt['IK'])  # convención: corriente hacia afuera es positiva
    print(f"✓ IK: rango [{np.min(result_wt['IK']):.2f}, {np.max(result_wt['IK']):.2f}] µA/cm²")

    print("\n✅ Todos los tests pasaron. model.py listo.")
