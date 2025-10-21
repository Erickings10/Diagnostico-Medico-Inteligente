# ---------------------------------------------
# Código base de ejemplo: Diagnóstico ECG simple
# Asistido por IA
# ---------------------------------------------

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# --- Paso 1: Cargar archivo CSV con datos ECG ---
# Se asume que el archivo tiene una sola columna de voltaje (en mV)
ruta = "ptbdb_normal.csv"     # <-- cambia por tu archivo
data = pd.read_csv(ruta, header=None)
ecg_signal = data.iloc[:, 0].values
fs = 125.0  # frecuencia de muestreo en Hz

# --- Paso 2: Filtro pasa banda para eliminar ruido ---
def bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, sig)

filt_ecg = bandpass_filter(ecg_signal, fs)

# --- Paso 3: Detección de picos R ---
# Se usa la señal integrada para resaltar los latidos
diff = np.ediff1d(filt_ecg, to_begin=0)
squared = diff ** 2
window = int(0.15 * fs)  # ventana de integración 150 ms
integrated = np.convolve(squared, np.ones(window)/window, mode='same')

# Detección de picos usando umbral y distancia mínima
min_distance = int(0.25 * fs)  # distancia mínima entre picos (≈240 bpm)
threshold = np.mean(integrated) + 0.5 * np.std(integrated)
peaks, _ = signal.find_peaks(integrated, distance=min_distance, height=threshold)

# --- Paso 4: Calcular frecuencia cardíaca promedio ---
if len(peaks) > 1:
    rr_intervals = np.diff(peaks) / fs
    hr_values = 60.0 / rr_intervals
    hr_mean = np.mean(hr_values)
else:
    hr_mean = np.nan

# --- Paso 5: Diagnóstico básico por reglas ---
if np.isnan(hr_mean):
    diagnostico = "No se pudo calcular frecuencia cardíaca (pocos picos detectados)."
elif hr_mean > 100:
    diagnostico = f"Frecuencia promedio {hr_mean:.1f} bpm → Posible TAQUICARDIA"
elif hr_mean < 60:
    diagnostico = f"Frecuencia promedio {hr_mean:.1f} bpm → Posible BRADICARDIA"
else:
    diagnostico = f"Frecuencia promedio {hr_mean:.1f} bpm → Ritmo normal"

# --- Paso 6: Mostrar resultados y gráficas ---
print(diagnostico)
print(f"Picos detectados: {len(peaks)}")

plt.figure(figsize=(10,5))
plt.plot(ecg_signal, label="Señal original")
plt.plot(peaks, ecg_signal[peaks], "ro", label="Picos R detectados")
plt.title("Detección de Picos R en Señal ECG")
plt.xlabel("Muestras")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.show()