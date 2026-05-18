# Kv1.1 V408A — Modelo de Hodgkin-Huxley

> **Electrofisiología Molecular I · 6.° Semestre**  
> Análisis computacional del impacto de la mutación V408A en el canal Kv1.1  
> sobre la excitabilidad neuronal mediante el modelo de Hodgkin-Huxley

---

## Descripción

Este repositorio contiene el código, documentación y secciones del manuscrito científico
para el trabajo final del curso. Simulamos en Python el comportamiento eléctrico de una
neurona con el canal de potasio **Kv1.1** en su forma **silvestre (WT)** y con la
**mutación V408A**, asociada a Ataxia Episódica Tipo 1 y epilepsia focal.

La mutación V408A acelera la inactivación del canal (~40%), reduce la corriente
macroscópica de K⁺ (~30%) y produce hiperexcitabilidad neuronal medible en el modelo.

---

## Estructura del repositorio

```
kv11-v408a-model/
├── src/                    # Código fuente — un módulo por persona
│   ├── params.py           # Constantes globales (Persona 1)
│   ├── gates.py            # Funciones de compuerta α, β, ∞, τ (Persona 2)
│   ├── model.py            # Sistema de ODEs + integrador (Persona 3)
│   ├── stimulus.py         # Generadores de estímulo (Persona 4)
│   ├── analysis.py         # Métricas y tabla de resultados (Persona 5)
│   ├── plots.py            # 5 figuras del manuscrito (Personas 6 y 7)
│   └── main.py             # Orquestador — corre el pipeline completo
│
├── outputs/                # Figuras y CSV generados (ignorados por Git)
│
├── docs/                   # Documentación y guías
│   ├── guia_global_kv11_proyecto.pdf
│   ├── pipeline_delegacion.md
│   └── literatura_kv11_v408a.md
│
├── manuscript/             # Secciones del manuscrito científico
│   └── secciones/
│       ├── 01_introduccion.md
│       ├── 02_metodologia.md
│       ├── 03_resultados.md
│       ├── 04_discusion.md
│       └── 05_conclusion.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/kv11-v408a-model.git
cd kv11-v408a-model

# 2. (Opcional pero recomendado) Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

**Dependencias:** `numpy`, `scipy`, `matplotlib`, `pandas`  
**Python:** 3.9 o superior

---

## Uso

### Correr el pipeline completo

```bash
cd src
python main.py
```

Genera automáticamente en `outputs/`:
- `figura_1_PA.png` — Potencial de acción WT vs V408A
- `figura_2_corrientes.png` — Corrientes iónicas INa, IK, IL
- `figura_3_tren.png` — Tren de estímulos (hiperexcitabilidad)
- `figura_4_boltzmann.png` — Curvas de estado estacionario
- `figura_5_umbral.png` — Comparación de umbral de disparo
- `tabla_parametros.csv` — Métricas cuantitativas WT vs V408A

### Probar un módulo individual

Cada módulo tiene sus propios tests. Correr desde `src/`:

```bash
python params.py      # verifica constantes
python gates.py       # verifica funciones de compuerta
python stimulus.py    # verifica generadores de estímulo
python model.py       # verifica simulación y potencial de acción
python analysis.py    # verifica métricas y umbral
python plots.py       # genera figura 4 de prueba (Boltzmann)
```

---

## Modelo

### Sistema de ODEs (Hodgkin-Huxley con Kv1.1)

```
dV/dt  = (I_ext − I_Na − I_K − I_L) / Cm

I_Na = gNa · m³ · h  · (V − ENa)
I_K  = gKv11 · n⁴ · hk · (V − EK)     ← canal Kv1.1
I_L  = gL · (V − EL)

dm/dt  = (m∞(V) − m)  / τm(V)
dh/dt  = (h∞(V) − h)  / τh(V)
dn/dt  = (n∞(V) − n)  / τn(V)
dhk/dt = (hk∞(V) − hk) / τhk(V)       ← inactivación lenta Kv1.1
```

### Parámetros clave de la mutación V408A

| Parámetro | WT | V408A | Fuente |
|---|---|---|---|
| g_Kv1.1 máx (mS/cm²) | 36.0 | 25.2 | Miceli et al. 2023 |
| Escala τ inactivación | 1.0 | 0.6 (−40%) | Peters et al. 2011 |
| V½ activación (mV) | −29 | −29 (sin cambio) | Miceli et al. 2023 |
| E_K (mV) | −77 | −77 | Nernst |

---

## Flujo de trabajo en equipo

### Reglas de Git para este proyecto

1. **Nunca trabajar directo en `main`**
2. Cada persona trabaja en su propia rama:

```bash
git checkout -b persona-2/gates
# ... trabajas en gates.py ...
git add src/gates.py
git commit -m "feat(gates): implementar alpha_m, beta_m, n_inf, hk_inf"
git push origin persona-2/gates
```

3. Cuando tu módulo pasa todos sus tests, abrir un **Pull Request** hacia `main`
4. Al menos otra persona revisa el PR antes de hacer merge

### Convención de commits

```
feat(modulo): descripción breve de lo que se implementó
fix(modulo): descripción del bug corregido
test(modulo): agregar o corregir tests
docs: actualizar documentación o sección del manuscrito
```

Ejemplos:
```
feat(gates): implementar tau_hk con forma de campana y escala WT/MUT
fix(model): manejar singularidad en alpha_m cuando V = -40 mV
docs: agregar borrador de introducción
test(analysis): agregar assert para umbral V408A < umbral WT
```

### Orden de dependencias (quién desbloquea a quién)

```
params.py
    └── gates.py + stimulus.py     (paralelo)
            └── model.py
                    ├── analysis.py
                    └── plots.py
                            └── main.py
```

**Cuello de botella crítico:** `gates.py` debe estar listo antes del día 5.

---

## Responsables

| Módulo | Archivo | Persona | Estado |
|---|---|---|---|
| Parámetros | `params.py` | Persona 1 | ⬜ En progreso |
| Compuertas | `gates.py` | Persona 2 | ⬜ En progreso |
| Modelo ODEs | `model.py` | Persona 3 | ⬜ Esperando gates.py |
| Estímulos | `stimulus.py` | Persona 4 | ⬜ En progreso |
| Análisis | `analysis.py` | Persona 5 | ⬜ Esperando model.py |
| Figuras | `plots.py` | Personas 6–7 | ⬜ Esperando model.py |
| Integración | `main.py` | Persona 3 | ⬜ Al final |

> Actualizar el estado a ✅ cuando el módulo pase todos sus tests.

---

## Literatura principal

| Referencia | Aporte al modelo |
|---|---|
| Browne et al. (1994) *Nat Genet* | Identificación original de V408A |
| Peters et al. (2011) *J Physiol* | τ inactivación −40% en V408A |
| Bhatt et al. (2020) *J Comput Neurosci* | Modelo HH con Kv1.1, parámetros base |
| Miceli et al. (2023) *PNAS* | Reducción de corriente −30%, NFA como terapia |
| Smart et al. (1998) *Neuron* | Validación in vivo: ratones Kv1.1 null con epilepsia |

---

## Licencia

Trabajo académico — Uso interno del equipo. No redistribuir sin autorización.