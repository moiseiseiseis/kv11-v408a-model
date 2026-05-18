# params.py — Constantes globales del modelo Kv1.1 V408A
# Responsable: Persona 1
# NADIE más modifica los valores numéricos de este archivo.
# Todos los demás módulos importan de aquí: from params import *

# ── Membrana ──────────────────────────────────────────────
Cm   = 1.0      # µF/cm² — capacitancia de membrana

# ── Corriente de sodio Nav (HH estándar) ──────────────────
gNa  = 120.0    # mS/cm²
ENa  = 50.0     # mV

# ── Corriente de fuga ──────────────────────────────────────
gL   = 0.3      # mS/cm²
EL   = -54.4    # mV

# ── Corriente de potasio Kv1.1 ────────────────────────────
EK            = -77.0   # mV — Nernst: [K]o=5 mM, [K]i=140 mM, T=310 K
gKv11_WT      = 36.0    # mS/cm² — Bhatt et al. 2020
gKv11_MUT     = 25.2    # mS/cm² — V408A: -30% Miceli et al. 2023

# Activación Kv1.1 (Boltzmann)
V_half_n      = -29.0   # mV — Bhatt et al. 2020
k_n           =  8.0    # mV

# Inactivación lenta Kv1.1 (C-type)
V_half_hk     = -46.0   # mV — WT y MUT no difieren en V½ (Miceli 2023)
k_hk          =  6.0    # mV

# Escala de tau de inactivación
tau_scale_WT  = 1.0
tau_scale_MUT = 0.6     # V408A inactiva 40% más rápido — Peters et al. 2011

# ── Simulación ────────────────────────────────────────────
V_rest  = -65.0   # mV — voltaje de reposo
dt      = 0.01    # ms — paso de integración
t_end   = 400.0   # ms — duración total

# ── Estímulo ──────────────────────────────────────────────
I_amp        = 10.0               # µA/cm²
t_stim_start = 50.0               # ms
t_stim_dur   = 1.0                # ms
train_times  = [50, 100, 150, 200, 250, 300]  # ms

if __name__ == '__main__':
    print("params.py cargado correctamente.")
    print(f"  EK = {EK} mV | ENa = {ENa} mV")
    print(f"  gKv11_WT = {gKv11_WT} | gKv11_MUT = {gKv11_MUT} mS/cm²")
    print(f"  tau_scale_MUT = {tau_scale_MUT} (V408A inactiva {(1-tau_scale_MUT)*100:.0f}% más rápido)")
