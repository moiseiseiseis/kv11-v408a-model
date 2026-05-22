# model.py — Sistema de ODEs y función de simulación
# Núcleo del modelo. Integra las 5 ODEs con scipy.solve_ivp.
# Depende de: params.py, gates.py
#
# MEJORAS SOBRE LA VERSIÓN ORIGINAL:
#   1. Soporte para parámetros de channelopatía (γα, γβ, γK, ΔV)
#      → Hafez & Gottschalk (2020), Tabla 1
#   2. Soporte para NFA (niflumic acid) como modificador de compuertas
#      → Servettini et al. (2023), PNAS
#   3. Modo heterocigoto WT + MUT
#      → Browne et al. (1994): EA1 es dominante y heterocigótica
#   4. Migración a scipy.solve_ivp (API moderna, más robusto que odeint)
#   5. Iext guardado en el diccionario de salida
#   6. Validación de entradas y mensajes de error claros
#   7. Tabla de channelopatías predefinida (Hafez 2020, Tabla 1)

import numpy as np
from scipy.integrate import solve_ivp
from params import (Cm, gNa, ENa, gL, EL, EK,
                    gKv11_WT, gKv11_MUT, V_rest, dt, t_end)
from gates import (m_inf, h_inf, tau_m, tau_h,
                   n_inf, tau_n, hk_inf, tau_hk)


# ══════════════════════════════════════════════════════════════════════
# TABLA DE CHANNELOPATÍAS Kv1.1 CONOCIDAS
# Fuente: Hafez & Gottschalk (2020), J Comput Neurosci, Tabla 1
#
# Columnas: (gamma_alpha, gamma_beta, gamma_K, delta_V_mV)
#   gamma_alpha : escala la tasa de activación α_n
#   gamma_beta  : escala la tasa de deactivación β_n
#   gamma_K     : escala la conductancia máxima gKv11 (expresión relativa)
#   delta_V     : desplazamiento de la sensibilidad de voltaje (mV)
#                 positivo → desplazamiento depolarizante
#
# Browne et al. (1994) identificó V408A como una de las primeras mutaciones
# causales de EA1. Servettini et al. (2023) caracterizó su efecto funcional
# con precisión en el modelo knock-in murino Kv1.1^(V408A/+).
# ══════════════════════════════════════════════════════════════════════

CHANNELOPATHY_TABLE = {
    # Mutacion : (γα,   γβ,    γK,    ΔV mV)
    'WT'    : (1.0,  1.0,   1.00,   0.0),
    'V174F' : (1.0,  1.0,   0.08,  +36.0),   # Adelman et al. 1995
    'F184C' : (0.5,  1.0,   0.15,  +27.0),   # Adelman et al. 1995
    'T226A' : (0.5,  0.5,   0.05,  +15.0),   # Zerr et al. 1998a
    'T226M' : (0.5,  0.5,   0.05,  +16.0),   # Zerr et al. 1998a
    'T226R' : (1/6,  0.5,   0.03,  +25.0),   # Zuberi et al. 1999
    'A242P' : (1.0,  1.0,   0.10,  -10.0),   # Eunson et al. 2000
    'P244H' : (1.0,  1.0,   1.00,   -5.0),   # Eunson et al. 2000
    'F249I' : (1.0,  1.0,   0.01,   0.0),    # Zerr et al. 1998b
    'G311S' : (1.0,  1.0,   0.23,  +27.0),   # Zerr et al. 1998a
    'E325D' : (6.0,  6.0,   0.08,  +64.0),   # Adelman et al. 1995
    'V404I' : (1.0,  1.0,   1.00,  +10.0),   # Eunson et al. 2000
    # V408A: γK = 0.70 (Servettini 2023: ~-30% corriente macroscópica)
    # τ_hk escalada por 0.6 (Peters et al. 2011: -40% en τ inactivacion)
    'V408A' : (1.0,  20.0,  0.68,   0.0),    # Adelman et al. 1995 + Servettini 2023
}


# ══════════════════════════════════════════════════════════════════════
# PARAMETROS NFA — Servettini et al. (2023), PNAS
#
# NFA (niflumic acid) actúa como activador de Kv1.1:
#   - Desplaza V½ de activación ~10 mV hacia potenciales hiperpolarizantes
#   - Enlentece la deactivación (γβ efectivo aumenta el τ deactivacion)
#   - No aumenta la conductancia máxima una vez alcanzada la prob. máx.
# ══════════════════════════════════════════════════════════════════════

NFA_PARAMS = {
    'delta_V'     : -10.0,   # mV — desplazamiento hiperpolarizante de V½
    'scale_deact' :   0.5,   # reduce β_n → deactivacion más lenta
}


# ══════════════════════════════════════════════════════════════════════
# FUNCION ODE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def hh_odes(t, y, gKv11, mutant, stim_func,
            gamma_alpha=1.0, gamma_beta=1.0, delta_V=0.0,
            nfa=False):
    """
    Sistema de 5 ODEs del modelo Hodgkin-Huxley con Kv1.1.

    PARAMETROS FUNDAMENTALES:
        t        : float — tiempo actual (ms)  [solve_ivp pasa t primero]
        y        : lista [V, m, h, n, hk] — estado actual
        gKv11    : float — conductancia máxima Kv1.1 (mS/cm²)
        mutant   : bool  — True = V408A, False = WT
        stim_func: callable — I_ext(t) en µA/cm²

    PARAMETROS DE CHANNELOPATIA (Hafez & Gottschalk 2020, Tabla 1):
        gamma_alpha : float — escala tasa de activacion α_n. Default=1.0 (WT)
        gamma_beta  : float — escala tasa de deactivacion β_n. Default=1.0 (WT)
        delta_V     : float — desplazamiento de voltaje del rectificador (mV).
                              Positivo = depolarizante. Default=0.0

    PARAMETROS DE TRATAMIENTO:
        nfa      : bool — True = aplicar efecto NFA (Servettini et al. 2023)

    RETORNA:
        [dV/dt, dm/dt, dh/dt, dn/dt, dhk/dt]
    """
    V, m, h, n, hk = y

    # Desplazamiento de voltaje efectivo: channelopatia + NFA si aplica
    dV_total = delta_V + (NFA_PARAMS['delta_V'] if nfa else 0.0)

    # Corrientes ionicas
    INa  = gNa   * m**3 * h  * (V - ENa)
    IK   = gKv11 * n**4 * hk * (V - EK)
    IL   = gL * (V - EL)
    Iext = stim_func(t)

    # Ecuacion de membrana (circuito RC)
    dVdt = (Iext - INa - IK - IL) / Cm

    # Compuertas Nav (Hodgkin-Huxley estandar)
    dmdt = (m_inf(V) - m) / tau_m(V)
    dhdt = (h_inf(V) - h) / tau_h(V)

    # Compuertas Kv1.1 con parametros de channelopatia
    # Hafez (2020): τ_n efectivo escala con γα y γβ
    tau_n_base = tau_n(V + dV_total)
    scale_tau_n = 2.0 / (gamma_alpha + gamma_beta)
    if nfa:
        scale_tau_n *= (1.0 / NFA_PARAMS['scale_deact'])
    tau_n_eff = tau_n_base * scale_tau_n
    n_inf_eff = n_inf(V + dV_total)
    dndt = (n_inf_eff - n) / tau_n_eff

    # tau_hk: V408A inactiva 40% más rapido (Peters et al. 2011)
    # NFA enlentece la inactivacion lenta (Servettini 2023, Fig 1I)
    tau_hk_eff = tau_hk(V + dV_total, mutant)
    if nfa:
        tau_hk_eff *= 1.65   # promedio τ_fast (+50%) y τ_slow (+80%)
    hk_inf_eff = hk_inf(V + dV_total, mutant)
    dhkdt = (hk_inf_eff - hk) / tau_hk_eff

    return [dVdt, dmdt, dhdt, dndt, dhkdt]


# ══════════════════════════════════════════════════════════════════════
# FUNCION DE SIMULACION PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def run_simulation(gKv11, mutant, stim_func,
                   t_end_=None, dt_=None,
                   gamma_alpha=1.0, gamma_beta=1.0, delta_V=0.0,
                   nfa=False,
                   heterozygous=False):
    """
    Ejecuta la simulacion HH completa con Kv1.1 WT, mutante, o heterocigótica.

    INPUT FUNDAMENTAL:
        gKv11     : float   — conductancia máxima Kv1.1 (mS/cm²)
        mutant    : bool    — True = V408A, False = WT
        stim_func : callable — funcion I_ext(t) en µA/cm²
        t_end_    : float opcional — sobreescribe params.t_end (ms)
        dt_       : float opcional — sobreescribe params.dt (ms)

    INPUT DE CHANNELOPATIA (Hafez & Gottschalk 2020):
        gamma_alpha : float — escala α_n (default=1.0 para WT)
        gamma_beta  : float — escala β_n (default=1.0 para WT)
        delta_V     : float — desplazamiento de voltaje del rectificador (mV)

    INPUT DE TRATAMIENTO (Servettini et al. 2023):
        nfa         : bool — True = simular efecto del NFA sobre Kv1.1

    INPUT CLINICO (Browne et al. 1994):
        heterozygous : bool — True = neuronas heterocigóticas EA1
                              Combina conductancias WT y MUT (50% de cada canal).
                              Fundamento: EA1 es autosómico dominante; pacientes
                              portan una copia WT y una copia mutante del KCNA1.

    OUTPUT:
        dict con:
            't'     : np.ndarray (ms)
            'V'     : np.ndarray (mV)
            'INa'   : np.ndarray (µA/cm²)
            'IK'    : np.ndarray (µA/cm²)
            'IL'    : np.ndarray (µA/cm²)
            'Iext'  : np.ndarray (µA/cm²) — estimulo aplicado [NUEVO]
            'm', 'h', 'n', 'hk' : np.ndarray
            'params': dict — parametros usados en esta simulacion [NUEVO]
    """
    # Validacion de entradas
    if gKv11 < 0:
        raise ValueError(f"gKv11 debe ser >= 0, recibido: {gKv11}")
    if not callable(stim_func):
        raise TypeError("stim_func debe ser una funcion callable f(t)")
    if gamma_alpha <= 0 or gamma_beta <= 0:
        raise ValueError("gamma_alpha y gamma_beta deben ser > 0 "
                         "(Hafez 2020: espacio fisiologico es gamma > 0)")

    _t_end = t_end_ if t_end_ is not None else t_end
    _dt    = dt_    if dt_    is not None else dt

    if _t_end <= 0:
        raise ValueError(f"t_end debe ser > 0, recibido: {_t_end}")
    if _dt <= 0 or _dt >= _t_end:
        raise ValueError(f"dt debe ser 0 < dt < t_end, recibido: {_dt}")

    t_eval = np.arange(0, _t_end, _dt)

    # Modo heterocigótico — Browne et al. (1994) y Servettini et al. (2023)
    # "EA1-affected individuals are heterozygous at the Kv1.1 locus,
    #  possessing a normal and a mutant allele."
    if heterozygous:
        gKv11_eff = (gKv11_WT + gKv11) / 2.0
    else:
        gKv11_eff = gKv11

    # Condiciones iniciales — estado de reposo
    V0  = V_rest
    m0  = m_inf(V0)
    h0  = h_inf(V0)
    n0  = n_inf(V0 + delta_V)
    hk0 = hk_inf(V0 + delta_V, mutant)
    y0  = [V0, m0, h0, n0, hk0]

    # Integracion con solve_ivp (API moderna de scipy, reemplaza odeint)
    sol = solve_ivp(
        fun=hh_odes,
        t_span=(0.0, _t_end),
        y0=y0,
        method='RK45',
        t_eval=t_eval,
        args=(gKv11_eff, mutant, stim_func,
              gamma_alpha, gamma_beta, delta_V, nfa),
        rtol=1e-6,
        atol=1e-9,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"Integracion fallida: {sol.message}")

    t_s  = sol.t
    V_s  = sol.y[0]
    m_s  = sol.y[1]
    h_s  = sol.y[2]
    n_s  = sol.y[3]
    hk_s = sol.y[4]

    # Recalcular corrientes desde el estado integrado
    INa_s  = gNa       * m_s**3 * h_s  * (V_s - ENa)
    IK_s   = gKv11_eff * n_s**4 * hk_s * (V_s - EK)
    IL_s   = gL * (V_s - EL)

    # Iext vectorizado — ahora incluido en el output
    Iext_s = np.array([stim_func(ti) for ti in t_s])

    # Registro de parametros usados en esta simulacion
    used_params = {
        'gKv11'        : gKv11_eff,
        'mutant'       : mutant,
        'heterozygous' : heterozygous,
        'gamma_alpha'  : gamma_alpha,
        'gamma_beta'   : gamma_beta,
        'delta_V'      : delta_V,
        'nfa'          : nfa,
        't_end'        : _t_end,
        'dt'           : _dt,
    }

    return {
        't'     : t_s,
        'V'     : V_s,
        'INa'   : INa_s,
        'IK'    : IK_s,
        'IL'    : IL_s,
        'Iext'  : Iext_s,
        'm'     : m_s,
        'h'     : h_s,
        'n'     : n_s,
        'hk'    : hk_s,
        'params': used_params,
    }


# ══════════════════════════════════════════════════════════════════════
# FUNCION DE ACCESO A TABLA DE CHANNELOPATIAS
# ══════════════════════════════════════════════════════════════════════

def run_channelopathy(variant_name, stim_func, t_end_=None, dt_=None,
                      nfa=False, heterozygous=False):
    """
    Ejecuta la simulacion para una channelopatia conocida de Kv1.1.

    Usa los parametros de CHANNELOPATHY_TABLE (Hafez & Gottschalk 2020).
    Es un wrapper conveniente sobre run_simulation().

    INPUT:
        variant_name : str — nombre de la variante, p. ej. 'V408A', 'F184C', 'WT'
        stim_func    : callable — estimulo I_ext(t)
        t_end_       : float opcional
        dt_          : float opcional
        nfa          : bool — aplicar NFA (Servettini 2023)
        heterozygous : bool — modo heterocigótico (Browne 1994)

    OUTPUT:
        dict — igual que run_simulation()

    EJEMPLO:
        from stimulus import I_single
        result     = run_channelopathy('V408A', I_single)
        result_nfa = run_channelopathy('V408A', I_single, nfa=True)
    """
    if variant_name not in CHANNELOPATHY_TABLE:
        available = list(CHANNELOPATHY_TABLE.keys())
        raise ValueError(
            f"Variante '{variant_name}' no encontrada en CHANNELOPATHY_TABLE.\n"
            f"Variantes disponibles: {available}"
        )

    gamma_alpha, gamma_beta, gamma_K, delta_V = CHANNELOPATHY_TABLE[variant_name]
    gKv11_variant = gKv11_WT * gamma_K
    is_mutant = (variant_name != 'WT')

    return run_simulation(
        gKv11=gKv11_variant,
        mutant=is_mutant,
        stim_func=stim_func,
        t_end_=t_end_,
        dt_=dt_,
        gamma_alpha=gamma_alpha,
        gamma_beta=gamma_beta,
        delta_V=delta_V,
        nfa=nfa,
        heterozygous=heterozygous,
    )


# ══════════════════════════════════════════════════════════════════════
# FUNCION UTILITARIA: ANCHO DEL POTENCIAL DE ACCION (FWHM)
# Resuelve el TODO pendiente en analysis.py
# ══════════════════════════════════════════════════════════════════════

def compute_ap_width(result, V_rest_ref=V_rest):
    """
    Calcula el ancho a mitad de amplitud (FWHM) del potencial de accion.

    El FWHM es una metrica estandar en electrofisiologia para comparar
    la duracion del PA entre condiciones WT y mutante. V408A tiene un PA
    mas ancho en terminales presinapticos de celulas basket, lo que
    aumenta la liberacion de GABA (Servettini et al. 2023, Fig. 3).

    INPUT:
        result      : dict — salida de run_simulation()
        V_rest_ref  : float — voltaje de referencia para calcular amplitud

    OUTPUT:
        float — ancho en ms, o None si no se detecto PA
    """
    t = result['t']
    V = result['V']

    idx_peak = np.argmax(V)
    V_peak   = V[idx_peak]

    if V_peak <= 0.0:
        return None

    V_half = (V_peak + V_rest_ref) / 2.0

    pre_peak  = V[:idx_peak]
    cross_up  = np.where(pre_peak >= V_half)[0]
    if len(cross_up) == 0:
        return None
    idx_up = cross_up[0]

    post_peak  = V[idx_peak:]
    cross_down = np.where(post_peak <= V_half)[0]
    if len(cross_down) == 0:
        return None
    idx_down = idx_peak + cross_down[0]

    return float(t[idx_down] - t[idx_up])


# ══════════════════════════════════════════════════════════════════════
# TESTS — correr con: python model.py
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from stimulus import I_single, I_zero

    print("=" * 65)
    print("  Tests de model.py — Kv1.1 WT vs V408A (HH extendido)")
    print("=" * 65)

    # Test 1: WT genera potencial de accion
    print("\n[1/7] WT — pulso unico...")
    result_wt = run_simulation(gKv11_WT, False, I_single, t_end_=200.0)
    assert np.max(result_wt['V']) > 0,   "FALLO: WT no genero potencial de accion"
    assert np.min(result_wt['V']) < -60, "FALLO: WT no mostro hiperpolarizacion"
    print(f"  OK WT: Vpico = {np.max(result_wt['V']):.1f} mV, "
          f"Vmin = {np.min(result_wt['V']):.1f} mV")

    # Test 2: V408A genera PA
    print("\n[2/7] V408A — pulso unico...")
    result_mut = run_simulation(gKv11_MUT, True, I_single, t_end_=200.0)
    assert np.max(result_mut['V']) > 0, "FALLO: V408A no genero potencial de accion"
    print(f"  OK V408A: Vpico = {np.max(result_mut['V']):.1f} mV")

    # Test 3: Keys correctas
    print("\n[3/7] Verificando keys del output dict...")
    expected_keys = ['t','V','INa','IK','IL','Iext','m','h','n','hk','params']
    for key in expected_keys:
        assert key in result_wt, f"FALLO: falta key '{key}'"
    print(f"  OK Todas las keys presentes")

    # Test 4: Iext en output
    print("\n[4/7] Verificando Iext en output...")
    assert len(result_wt['Iext']) == len(result_wt['t'])
    assert np.any(result_wt['Iext'] > 0)
    print(f"  OK Iext: max = {np.max(result_wt['Iext']):.1f} µA/cm²")

    # Test 5: run_channelopathy
    print("\n[5/7] run_channelopathy('V408A')...")
    result_v408a = run_channelopathy('V408A', I_single, t_end_=200.0)
    assert np.max(result_v408a['V']) > 0
    print(f"  OK V408A (tabla): Vpico = {np.max(result_v408a['V']):.1f} mV")

    # Test 6: Modo NFA
    print("\n[6/7] Modo NFA — Servettini et al. (2023)...")
    result_nfa = run_channelopathy('V408A', I_single, t_end_=200.0, nfa=True)
    assert np.max(result_nfa['V']) > 0
    print(f"  OK V408A + NFA: Vpico = {np.max(result_nfa['V']):.1f} mV")

    # Test 7: compute_ap_width
    print("\n[7/7] compute_ap_width — FWHM del PA...")
    width_wt  = compute_ap_width(result_wt)
    width_mut = compute_ap_width(result_mut)
    assert width_wt is not None and width_mut is not None
    assert width_wt > 0
    print(f"  OK AP width WT    = {width_wt:.2f} ms")
    print(f"  OK AP width V408A = {width_mut:.2f} ms")

    print("\n" + "=" * 65)
    print(f"  IK WT    rango: [{np.min(result_wt['IK']):.2f}, {np.max(result_wt['IK']):.2f}] µA/cm²")
    print(f"  IK V408A rango: [{np.min(result_mut['IK']):.2f}, {np.max(result_mut['IK']):.2f}] µA/cm²")
    print(f"  Variantes en tabla: {list(CHANNELOPATHY_TABLE.keys())}")
    print("\n  TODOS LOS TESTS PASARON. model.py listo.")