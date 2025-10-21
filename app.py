import streamlit as st # Interfaz web (widgets, carga de archivos, display)
import numpy as np # Operaciones numéricas y arrays
import pandas as pd # Lectura y manejo de datos tabulares (CSV)
from scipy import signal # Procesamiento de señales (filtros, find_peaks)
import matplotlib.pyplot as plt # Graficación
from io import StringIO # Manejar archivos en memoria

st.set_page_config(page_title="Diagnóstico ECG - Sistema Experto", layout="wide")

def conjunto_segmentado(df):
    """
    Decide si el DataFrame es tipo 'segmentado por latido'
    Heurística: muchas columnas (>50) y pocas filas o forma (n_filas, n_cols) con n_cols ~ 188
    """
    filas, columnas = df.shape
    return columnas > 50 and filas >= 1 and columnas <= 1000 and (columnas == 188 or columnas > 50)

def leer_archivo(archivo):
    """Intenta leer CSV; devuelve DataFrame"""
    contenido = archivo.getvalue().decode("utf-8")
    df = pd.read_csv(StringIO(contenido), header=None)
    return df

def filtro_bandpass(señal, fs, corte_bajo=0.5, corte_alto=40.0, orden=3):
    nyq = 0.5*fs
    bajo = corte_bajo/nyq
    alto = corte_alto/nyq
    b, a = signal.butter(orden, [bajo, alto], btype='band')
    return signal.filtfilt(b, a, señal)

def mejorar_señal(ecg, fs, ventana_ms=150):
    derivada = np.ediff1d(ecg, to_begin=0)
    cuadrado = derivada**2
    muestras_ventana = max(1, int(ventana_ms/1000.0 * fs))
    integrada = np.convolve(cuadrado, np.ones(muestras_ventana)/muestras_ventana, mode='same')
    return derivada, cuadrado, integrada

def detectar_picos(integrada, fs, bpm_max=220, factor_umbral=0.5):
    distancia_minima = int(fs * 60.0 / bpm_max)
    umbral = np.mean(integrada) + factor_umbral * np.std(integrada)
    picos, props = signal.find_peaks(integrada, distance=distancia_minima, height=umbral)
    return picos, props

def calcular_fc_desde_picos(picos, fs):
    if len(picos) < 2:
        return np.array([]), float('nan')
    tiempos = picos / fs
    rr = np.diff(tiempos)
    fc_inst = 60.0 / rr
    return fc_inst, float(np.nanmean(fc_inst))

def diagnostico_por_reglas(fc_promedio, fc_inst=None):
    """
    Reglas simples:
     - fc_promedio > 100 -> taquicardia
     - fc_promedio < 60  -> bradicardia
     - sdnn alto -> posible arritmia (si hay fc_inst)
    Devuelve (etiqueta, lista de mensajes)
    """
    mensajes = []
    if np.isnan(fc_promedio):
        mensajes.append("No es posible calcular FC promedio (pocos picos detectados).")
        return "Indeterminado", mensajes
    if fc_promedio > 100:
        mensajes.append(f"Frecuencia promedio {fc_promedio:.1f} lpm → TAQUICARDIA.")
    elif fc_promedio < 60:
        mensajes.append(f"Frecuencia promedio {fc_promedio:.1f} lpm → BRADICARDIA.")
    else:
        mensajes.append(f"Frecuencia promedio {fc_promedio:.1f} lpm → RITMO DENTRO DE RANGO (normal).")
    if fc_inst is not None and len(fc_inst) >= 3:
        sdnn = float(np.std(fc_inst))
        if sdnn > 20.0:
            mensajes.append(f"Alta variabilidad (SDNN={sdnn:.1f} lpm) → posible arritmia.")
    return " / ".join([m.split("→")[0].strip() for m in mensajes]), mensajes

# --------------- Interfaz de usuario ----------------
st.title("Diagnóstico Médico Inteligente — ECG")
st.markdown("Sube un CSV: puede ser dataset segmentado por latidos o señal continua con 1 o 2 columnas.")

col1, col2 = st.columns([1,2])
with col1:
    archivo_subido = st.file_uploader("Archivo ECG", type=["csv"])
    fs_entrada = st.number_input("Frecuencia de muestreo (Hz) — para señal continua", value=125.0, step=1.0)
    corte_bajo = st.number_input("Corte bajo (Hz)", value=0.5)
    corte_alto = st.number_input("Corte alto (Hz)", value=40.0)
    ventana_ms = st.number_input("Ventana integración (ms)", value=150)
    factor_umbral = st.slider("Factor de umbral ", 0.0, 2.0, 0.5)
    boton = st.button("Procesar archivo")

with col2:
    st.info("Aquí se mostrarán los gráficos y resultados")

# --------------- Procesamiento (Asistido con IA) ----------------
if boton:
    if archivo_subido is None:
        st.error("Sube primero un archivo CSV.")
    else:
        df = leer_archivo(archivo_subido)
        st.write("Dimensiones del archivo:", df.shape)

        # Heurística para decidir tipo de archivo
        segmentado = conjunto_segmentado(df)
        st.write("Interpretando como dataset segmentado (una fila = un latido):", segmentado)

        if segmentado:
            # Cada fila = segmento. Se procesará una muestra de ejemplo y cálculo de R dentro de cada latido
            st.subheader("Dataset segmentado: análisis por latido")
            st.write("Número de latidos:", df.shape[0], "muestras por latido:", df.shape[1])

            # Mostrar primer latido
            muestra = df.iloc[0, :].values.astype(float)
            fig1, ax1 = plt.subplots(1,1, figsize=(8,3))
            ax1.plot(muestra); ax1.set_title("Primer latido (segmentado)")
            ax1.set_xlabel("Muestra"); ax1.set_ylabel("Amplitud")
            st.pyplot(fig1)

            # Detectar R dentro del primer latido
            fs = float(fs_entrada)
            filtrada = filtro_bandpass(muestra, fs, corte_bajo=max(0.1, corte_bajo), corte_alto=min(corte_alto, fs/2 - 1), orden=2)
            derivada, cuadrado, integrada = mejorar_señal(filtrada, fs, ventana_ms=ventana_ms)
            picos, props = detectar_picos(integrada, fs, bpm_max=220, factor_umbral=factor_umbral)

            fig2, ejes = plt.subplots(3,1, figsize=(8,6), sharex=True)
            ejes[0].plot(muestra); ejes[0].set_title("Original (segmento)")
            ejes[1].plot(filtrada); ejes[1].set_title("Filtrada")
            ejes[2].plot(integrada); ejes[2].plot(picos, integrada[picos], "rx"); ejes[2].set_title("Integrada y picos detectados")
            st.pyplot(fig2)

            st.write("Picos R detectados en el primer latido:", len(picos))
            if len(picos)>0:
                st.write("Índices de picos (primer latido):", picos)

            # Si el dataset tiene etiquetas en la ultima columna, mostrar conteo por clase
            if df.shape[1] > 1:
                # heurística: si el último valor es entero pequeño y los datos tienen más columnas
                ultima_columna = df.iloc[:, -1]
                if pd.api.types.is_integer_dtype(ultima_columna) or ultima_columna.dropna().apply(float.is_integer).all():
                    st.write("Posible columna de etiquetas detectada (última columna). Mostrando conteo:")
                    st.write(ultima_columna.value_counts())
            st.warning("Nota: en datasets segmentados no se puede calcular FC promedio por fila individual. Para FC promedio necesitas una señal continua con múltiples latidos en el tiempo.")

        else:
            # Interpretar como señal continua: dos columnas o una columna
            st.subheader("Señal continua: procesando para detección R y cálculo FC")
            arreglo = df.values
            if arreglo.ndim == 2 and arreglo.shape[1] == 1:
                voltaje = arreglo[:,0].astype(float)
                tiempo = np.arange(len(voltaje)) / float(fs_entrada)
            elif arreglo.ndim == 2 and arreglo.shape[1] >= 2:
                # interpretar primera como tiempo, segunda como voltaje
                tiempo = arreglo[:,0].astype(float)
                voltaje = arreglo[:,1].astype(float)
                # si tiempo no es monotónico o step, estimar fs desde tiempo
                dt = np.mean(np.diff(tiempo))
                if dt > 0:
                    fs = 1.0/dt
                else:
                    fs = float(fs_entrada)
            else:
                st.error("Formato de archivo no reconocido para señal continua.")
                st.stop()

            fs = float(fs_entrada)
            st.write(f"Usando fs = {fs} Hz (puedes ajustar el valor en el panel izquierdo).")
            # Filtrado
            filtrada = filtro_bandpass(voltaje, fs, corte_bajo=max(0.1, corte_bajo), corte_alto=min(corte_alto, fs/2 - 1), orden=2)
            derivada, cuadrado, integrada = mejorar_señal(filtrada, fs, ventana_ms=ventana_ms)
            picos, props = detectar_picos(integrada, fs, bpm_max=220, factor_umbral=factor_umbral)
            fc_inst, fc_promedio = calcular_fc_desde_picos(picos, fs)
            etiqueta, mensajes = diagnostico_por_reglas(fc_promedio, fc_inst)

            # Gráficos
            fig, ejes = plt.subplots(4,1, figsize=(10,7), sharex=True)
            ejes[0].plot(tiempo, voltaje); ejes[0].set_title("Señal original")
            ejes[1].plot(tiempo, filtrada); ejes[1].set_title("Señal filtrada")
            ejes[2].plot(tiempo, integrada); ejes[2].plot(tiempo[picos], integrada[picos], "rx"); ejes[2].set_title("Integrada (realce) y picos")
            ejes[3].plot(tiempo, voltaje); ejes[3].plot(tiempo[picos], voltaje[picos], "ro"); ejes[3].set_title("Detección de picos R sobre señal original")
            ejes[-1].set_xlabel("Tiempo (s)")
            st.pyplot(fig)

            st.subheader("Resultados")
            st.write("Picos R detectados:", len(picos))
            st.write("FC promedio (lpm):", fc_promedio if not np.isnan(fc_promedio) else "N/D")
            if len(fc_inst)>0:
                st.write("FC instantáneo (primeros 10):", np.round(fc_inst[:10],2))
                st.dataframe(pd.DataFrame({"FC_inst_lpm": np.round(fc_inst,2)}))

            st.subheader("Diagnóstico por reglas")
            st.write("Etiqueta:", etiqueta)
            for mensaje in mensajes:
                st.write("- " + mensaje)

# ---------------- Notas ----------------
st.markdown("---")
st.markdown("Notas: este sistema fue asistido con la generación de un código base mediante IA (ChatGPT-5) y apoyo en el procesamiento del DataFrame para la generación del diagnóstico.")