# stimulus.py — Generadores de corriente de estímulo
# Responsable: Persona 4
# Funciones puras: solo reciben t (float) y retornan µA/cm².
# Compatibles con scipy.odeint (reciben t como float en cada paso).

from params import I_amp, t_stim_start, t_stim_dur, train_times, t_end


def I_single(t):
    """
    Pulso único de corriente.
    Amplitud: I_amp µA/cm²
    Duración: t_stim_dur ms a partir de t_stim_start ms
    """
    if t_stim_start <= t <= t_stim_start + t_stim_dur:
        return I_amp
    return 0.0


def I_train(t):
    """
    Tren de 6 pulsos de corriente.
    Tiempos de inicio: train_times (definidos en params.py)
    """
    for ts in train_times:
        if ts <= t <= ts + t_stim_dur:
            return I_amp
    return 0.0


def I_ramp(t, I_start=0.0, I_end=20.0):
    """
    Rampa lineal de corriente de I_start a I_end a lo largo de t_end ms.
    Útil para encontrar umbral de disparo visualmente.
    """
    if t <= t_end:
        return I_start + (I_end - I_start) * (t / t_end)
    else:
        return I_end  # Mantiene el valor final si la simulación por alguna razón se extiende

def make_pulse(amp, t_start, duration):
    """
    Fábrica de estímulos — retorna una función lambda.

    USO:
        stim = make_pulse(10.0, 50.0, 1.0)
        I = stim(55.0)   # → 10.0 µA/cm²
        I = stim(49.0)   # → 0.0
    """
    return lambda t: amp if t_start <= t <= t_start + duration else 0.0


def I_zero(t):
    """Sin estímulo — útil para verificar estado de reposo."""
    return 0.0


# ══════════════════════════════════════════════════════════
# TESTS — correr con: python stimulus.py
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Corriendo tests de stimulus.py...\n")

    # Test 1: I_single
    assert I_single(49.9)  == 0.0,   "FALLO: I_single antes del estímulo"
    assert I_single(50.5)  == I_amp, "FALLO: I_single durante el estímulo"
    assert I_single(51.1)  == 0.0,   "FALLO: I_single después del estímulo"
    print(f"✓ I_single: pulso de {I_amp} µA/cm² en t=50–51 ms")

    # Test 2: I_train
    for ts in train_times:
        assert I_train(ts + 0.5) == I_amp, f"FALLO: I_train no activa en t={ts}"
    assert I_train(75.0) == 0.0, "FALLO: I_train activa entre pulsos"
    print(f"✓ I_train: {len(train_times)} pulsos en t={train_times} ms")

    # Test 3: make_pulse
    stim = make_pulse(15.0, 80.0, 2.0)
    assert stim(80.5) == 15.0, "FALLO: make_pulse no activa en rango"
    assert stim(83.0) == 0.0,  "FALLO: make_pulse activa fuera de rango"
    print("✓ make_pulse: fábrica de estímulos funciona correctamente")

    # Test 4: I_zero
    assert I_zero(100.0) == 0.0, "FALLO: I_zero no es cero"
    print("✓ I_zero: retorna 0 siempre")

    # Test 5: I_ramp
    assert I_ramp(0.0, 0.0, 20.0) == 0.0, "FALLO: I_ramp no inicia en I_start"
    assert I_ramp(t_end/2, 0.0, 20.0) == 10.0, "FALLO: I_ramp no es lineal a la mitad"
    assert I_ramp(t_end, 0.0, 20.0) == 20.0, "FALLO: I_ramp no llega a I_end en t_end"
    print("✓ I_ramp: interpolación lineal correcta")

    print("\n Todos los tests pasaron. stimulus.py listo.")
