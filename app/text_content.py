# app/text_content.py

TEXT_TITLE ="""
            <div class="hero">
            <h1>BTC Predictor</h1>
            <p class="tagline">
                Predicción técnica + sentimiento para una señal de mercado más completa
            </p>
            <div class="cta">
        """
TEXT_DISCLAIMER = (
    "**Aviso: BTC Predictor es un proyecto demostrativo/educativo. "
    "La información mostrada es solo informativa y no constituye asesoramiento financiero.**"
)

TEXT_SUGGESTIONS="""
    Esta sección resume **extensiones naturales** de la solución, centradas en 
    refinar y escalar lo que ya funciona en **BTC Predictor**, a futuro.

    ---

    ### 📡 1. Integración con datos en tiempo real
    - Conectar una API de precios en vivo. Limitaciones actuales por rango.
    - Añadir el indicador *Fear & Greed Index* desde su API oficial.
    - Actualizar las noticias automáticamente para que la señal de sentimiento
      se alimente del mercado actual, de una manera más rápida.

    ### 🤖 2. Monetización de la predicción mediante un bot de trading
    - Conectar la predicción (basadda en precios en vivo) con un bot automatizado de trading.
    - Mandar órdenes de compra/venta en base a la predicción.

    ### 🔄 3. Automatizar el pipeline de predicción
    - Generar predicciones bajo demanda para fechas específicas.
    - Registrar entradas/salidas para evaluar históricamente el rendimiento del predictor.

    ### 📏 4. Evaluación y calibración continua
    - Implementar backtesting para comparar las predicciones con el precio real.
    - Analizar el rendimiento de cada régimen

    ### 💬 5. Extender el análisis de sentimiento
    - Incluir nuevas fuentes como redes sociales o el índice *Fear & Greed* diario.
    - Añadir ventanas ajustables por tipo de evento (últimas 4h, 24h, etc.)
    - Incorporar mayor granularidad en el peso de cada noticia, y mejorar el componente cualitativo sin cambiar el modelo predictivo.

    ### 🎛️ 6. Mejoras de experiencia de usuario
    - Agregar un modo “principiante” con explicaciones guiadas.
    - Resaltado visual cuando los modelos coinciden o divergen.
    - Exportación de vistas en PDF.

    ---

    En conjunto, estas recomendaciones **no cambian la lógica del predictor actual**, 
    sino que lo fortalecen y lo preparan para escenarios más dinámicos y escalables.
    """

TEXT_CONTEXT = """
## BTC Predictor

En un mercado altamente volátil como el de Bitcoin, las variaciones de precio suelen 
estar impulsadas no solo por factores técnicos, sino también por el contexto informativo 
y emocional del mercado.  
**BTC Predictor** combina **ambos mundos** para ofrecer una visión más completa del comportamiento del precio.

---

### 🔎 ¿Qué integra?

- **Datos cuantitativos:** precio en estructura OHLC (open, high, low, close) agrupados en series temporales ordenadas por datetimes (1h, 4h).  
- **Datos cualitativos:** noticias filtradas y un análisis de sentimiento basado en ventanas recientes.  

Esta combinación permite observar **qué está haciendo el precio** y **qué lo puede estar impulsando**.

---

### 🎯 ¿Qué hace el predictor?

BTC Predictor analiza un rango seleccionado por el usuario y construye:

- Precio histórico (1h / 4h) con resampling OHLC  
- Noticias filtradas por relevancia  
- Sentimiento agregado en ventanas recientes  
- **Predicción técnica t+1** (Random Forest y XGBoost)
- Señal final sobre la dirección del precio (subirá o bajará en t+1)


### 📌 Enfoque del proyecto

El objetivo es mostrar un **modelo predictivo funcional**, construido con algoritmos de Machine Learning con aprendizaje supervisado. 
Se basa en datasets históricos de precios Bitcoin, enriquecido con información contextual de noticias.  
La aplicación facilita la exploración visual y comparativa entre modos, 
permitiendo entender cómo reaccionan los modelos ante diferentes condiciones del mercado.

Este diseño permite extender el sistema fácilmente con nuevos modelos, fuentes de datos
o flujos automatizados en versiones futuras.
"""

TEXT_EDA_1='''# Time-Series EDA: GLOBAL vs REGÍMENES

## Link al EDA
[Time-series EDA](https://github.com/cbadil12/final-project/blob/main/notebooks/eda_timeseries_method_1.ipynb)

## Requisitos
- Python con: `numpy`, `pandas`, `matplotlib`, `seaborn`, `statsmodels`, `pmdarima`.

## Inputs
- `halving_dates`: fechas de regimenes (bordes) para segmentar la serie.
- `raw_data_input_path` y `raw_data_separator`: ruta y separador del CSV.
- `time_column` y `target_column`: columna de fecha y variable objetivo.
- `day_comes_first`: formato de fechas (day-first).
- `seasonal_period`: periodo estacional manual (o `None` para inferir).
- `accepted_alpha_dickey_fuller`: alpha del test ADF.
- `test_size`: proporcion de test para el split.
- `processed_data_output_path`: carpeta de salida para CSVs procesados.
- Parametros de graficas (tamanos de figura y fuentes).

---

## Paso 1) Explorar dataframe y construir la serie de tiempo
1) **Cargar el CSV** y mostrar un preview.
2) **Explorar columnas numericas** y estadisticos basicos (`describe`).
3) **Validar el eje temporal**:
   - Convertir a `datetime` (con `dayfirst` si aplica).
   - Eliminar fechas invalidas (NaT).
   - Ordenar por fecha.
   - Revisar si hay diffs negativos y duplicados (si existen, se detiene con error).
4) **Estimar frecuencia y granularidad**:
   - Calcular el delta mas comun.
   - Detectar gaps (saltos mayores al delta esperado).
   - Verificar consistencia de frecuencia (ratio >= 0.7).
   - Inferir granularidad (1h, 4h, 1d, etc.).
5) **Construir la serie**:
   - Asignar `DatetimeIndex`.
   - Extraer `target_column` como serie (`timeseries`).
6) **Definir regimenes por halving**:
   - Convertir `halving_dates` a fechas y filtrar las que esten dentro del rango.
   - Crear bordes `edges` y segmentos por regimen.
7) **Graficar** la serie por regimenes y marcar los halvings.

## Paso 2) Descomposicion y estacionalidad
1) **Inferir el periodo estacional (m)**:
   - Si hay `seasonal_period` manual, usarlo.
   - Si no, inferirlo por ACF entre candidatos segun granularidad.
2) **Evaluar estacionalidad global en retornos**:
   - Generar retornos (log-diff recomendado).
   - Descomponer con `seasonal_decompose` sobre retornos.
   - Medir fuerza estacional con:
     - `var_ratio = Var(seasonal) / Var(original)`
     - `acf_at_period`
   - Recomendar ARIMA vs SARIMA segun la fuerza.
3) **Evaluar estacionalidad por regimen** (misma logica que global).
4) **Chequear consistencia global vs regimen**:
   - Comparar var_ratio y acf con deltas absolutos.
   - Decidir estrategia: `global`, `per_regime` o `mixed`.
5) **Diagnostico de residuales (retornos)**:
   - Tendencia ~ 0, periodicidad baja, centrado en 0 y Ljung-Box (white noise).
6) **Descomposicion en precios (global)**:
   - Trend, seasonal y residual en la serie de precio.
   - Graficar los componentes.

## Paso 3) Estacionariedad (ADF) -> inferir d y D
1) **ADF global**:
   - Si es estacionaria, `d_global = 0`.
   - Si no, aplicar diferenciacion recursiva hasta estacionariedad (`d_global`).
2) **Diferenciacion estacional (D_global)**:
   - Solo si hay estacionalidad fuerte y `period` valido.
   - Comparar ADF sin y con diferencia estacional (`diff(period)`).
   - Si mejora (p < alpha), `D_global = 1`.
3) **ADF por regimen**:
   - Repetir la inferencia de `d_reg`.
   - Evaluar `D_reg` solo si la estacionalidad del regimen es fuerte.
4) **Registrar recomendaciones** de `d` y `D` para ARIMA/SARIMA.

## Paso 4) Diagnostico ACF/PACF y sugerencias de ordenes
1) **Preparar series de diagnostico**:
   - Aplicar `asfreq` segun granularidad si es posible.
   - Diferenciar con `d` (global y por regimen).
2) **ACF/PACF no estacional**:
   - Elegir un `safe_lag` segun granularidad y tamanio.
   - Sugerir `p` y `q` como primer lag significativo en PACF/ACF.
   - Graficar ACF/PACF (global y por regimen).
3) **ACF/PACF estacional**:
   - Si `period` valido, sugerir `P` y `Q` segun lags multiples de `m`.
   - Graficar ACF/PACF estacional (global y regimenes).
4) **Validar grillas con AIC/BIC (SARIMAX)**:
   - Probar combinaciones reducidas alrededor de los sugeridos.
   - Mantener top-N por AIC/BIC.
5) **Generar tabla de grillas** final para usar en modelado.

## Paso 5) Train/Test split y baseline naive
1) **Split global y por regimen** con `test_size`.
2) **Baseline naive** (predictor t-1) y metricas:
   - MAE, RMSE, MASE, sMAPE.
3) **Segundo split** (mismo tamano que test) para **estabilidad**.
4) **Rolling/expanding evaluation** (ventanas crecientes) para robustez.
5) **Guardar CSVs** de train/test con control de revision.

## Paso 6) Estabilidad / expanding adequacy + resumen
1) **Tabla de metricas** con todas las evaluaciones (global y regimenes).
2) **Umbrales**:
   - Estabilidad: percentil 75 de diferencias relativas.
   - Expanding: percentil 75 del coeficiente de variacion (CV) en folds.
3) **Tabla resumen** con `stability_check` y `expanding_check` por scope.

## Paso 7) Entrenamiento de los modelos + evaluación
1) Entrenamiento de modelos en base a las grids generadas + auto_arima.
2) Selección de mejores modelos.
3) Gráficas de modelos ARIMA/SARIMA
'''

TEXT_EDA_2='''# Time-Series EDA: RESAMPLING

## Link al EDA
[Time-series EDA](https://github.com/cbadil12/final-project/blob/main/notebooks/eda_timeseries_method_2.ipynb)

## Requisitos
- Python con: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`.

## Inputs
En la seccion **INPUTS** del notebook:
- `raw_data_input_path`: ruta del CSV con datos crudos.
- `processed_data_output_path`: carpeta de salida para CSVs.
- `raw_data_separator`: separador del CSV.
- `test_size`: proporcion de test.
- `accepted_alpha`: alpha del test ADF.
- `resample_key`: granularidad elegida para el split final (p.ej. `1-Hour`, `4-Hour`, `Daily`, etc.).
- Parametros de graficas (`figHeight_unit`, `figWidth_unit`).

---

## Paso 1) Explorar dataframe y graficar resamples
1) **Cargar el CSV** y mostrar preview.
2) **Convertir `time` a datetime**, ordenar y setear index temporal.
3) **Generar resamples**:
   - 1-Hour, 4-Hour, Daily, Monthly, Quarter, Yearly.
4) **Graficar** la serie `close` en cada resample para comparar patrones a distintas escalas.

## Paso 2) Stationary check + STL decomposition
1) **Definir periodo estacional** por resample:
   - 1-Hour → 24, 4-Hour → 6, Daily → 7, Monthly → 12, Quarter → 4, Yearly → 1.
2) **Descomponer cada serie (STL/seasonal_decompose)**:
   - Observed, Trend, Seasonal, Residual.
3) **Test ADF** sobre cada resample.
4) **Tabla resumen** de estacionariedad por resample (p-value y Yes/No).

## Paso 3) Box-Cox suitability summary
1) **Validar positividad** (Box-Cox requiere valores > 0).
2) **Aplicar Box-Cox** y obtener `lambda`.
3) **ADF en serie original** y recomendar si Box-Cox/log ayuda:
   - Si `lambda` cerca de 1 → probablemente no necesario.
   - Si `lambda` cerca de 0 → log/Box-Cox recomendado.
4) **Tabla resumen** con `lambda`, p-value y recomendacion.

## Paso 4) Seasonal differencing stationary check
1) **Aplicar diferenciacion estacional** (`diff(period)`).
2) **ADF** sobre serie diferenciada estacionalmente.
3) **Tabla resumen** con p-value y estacionariedad.

## Paso 5) Seasonal + regular differencing check
1) **Aplicar diff estacional** y luego **diff regular**.
2) **Descomposicion estacional** sobre la serie diferenciada.
3) **ADF** sobre la serie diferenciada final.
4) **Tabla resumen** con p-value y estacionariedad.

## Paso 6) Diferencing condicional + ACF/PACF (parametros iniciales)
1) **Elegir d y D** segun resultados de los pasos 4 y 5:
   - Si diff estacional basta → `D=1, d=0`.
   - Si requiere diff adicional → `D=1, d=1`.
   - Fallback: `D=1, d=1`.
2) **Aplicar differencing** a cada resample.
3) **Graficar ACF/PACF** con lags seguros.
4) **Estimar p y q iniciales** usando lags significativos.
5) **Definir rangos** de busqueda para SARIMA:
   - `p`, `q`, `P`, `Q` con limites pequenos.
6) **Tabla de parametros iniciales** por resample.

## Paso 7) Resample seleccionado + split + guardado
1) **Seleccionar resample** segun `resample_key`.
2) **Crear series train/test** por tiempo.
3) **Guardar CSVs** con sufijo del resample y numero de revision.

## Paso 8) Entrenamiento de los modelos + evaluación
1) Entrenamiento de modelos en base a las grids generadas mediante SARIMAX.
2) Selección de mejores modelos.
3) Gráficas de predicción
'''

TEXT_EDA_SENTIMENT='''# EDA Sentimientos

## Link al EDA
[EDA Sentimientos](https://github.com/cbadil12/final-project/blob/main/notebooks/eda_timeseries_method_2.ipynb)

Este EDA analiza el componente de **sentimiento** dentro del sistema de predicción de BTC.  
El objetivo **no** es entrenar modelos en este notebook, sino **validar el dataset**, justificar decisiones de preprocesamiento y evaluar si el sentimiento aporta señal sobre el retorno futuro.

## 1. Alcance y objetivos
Se busca:
- Justificar la selección y el preprocesamiento de datos.
- Evaluar calidad y consistencia temporal.
- Entender el comportamiento estadístico de features de sentimiento.
- Ver si existe relación entre sentimiento y retornos futuros de BTC.

No se incluye:
- Indicadores técnicos de precio.
- Modelos de series temporales (ARIMA/SARIMA).  
Eso se maneja en otros módulos y luego se integra en el ensemble.

## 2. Pipeline de datos (resumen)
La base del dataset de sentimiento se construye así:
1. **Ingesta de noticias crudas** (NewsAPI)
2. **Análisis de sentimiento** por ejes temáticos (BTC / MACRO / TECH)
3. **Agregación temporal** a 1h con features estadísticos (media, std, lags, shocks, etc.)
4. **Contexto externo** con Fear & Greed Index (forward-fill)
5. **Construcción del target** con precio BTC (retorno y dirección t+1h)

Este EDA analiza el output final de ese pipeline.

## 3. Cobertura temporal y régimen
Se detectaron períodos históricos con:
- Cobertura irregular de noticias.
- Frecuencias inconsistentes.
- Ejes temáticos incompletos.

Por eso se establece un **cutoff temporal (2025-11-12)** para garantizar coherencia semántica.  
El modelado solo considera datos **posteriores al cutoff**.

## 4. Calidad de noticias crudas
Tras el cutoff:
- Hay flujo continuo de artículos.
- Volumen estable.
- Faltantes concentrados en metadatos opcionales.

Esto confirma que valores 0 en features de sentimiento **representan horas sin noticias**, no datos faltantes.

## 5. Sanity check del dataset final
Se valida:
- Integridad de columnas y tipos.
- Consistencia temporal.
- Coherencia entre features y target.

## 6. Target
El target es la **dirección del precio a 1 hora** (t+1h).  
Se verifica estabilidad y correcta construcción antes de analizar features.

## 7. Distribuciones de features
Se inspeccionan:
- Variables degeneradas.
- Sesgos extremos.
- Escalas problemáticas.

Algunas features (por ejemplo, desviaciones estándar de sentimiento) resultan **degeneradas** y se descartan.

## 8. Comportamiento temporal
El sentimiento es **episódico** y **no estacionario**.  
Se analiza su evolución para ver patrones útiles y posibles ventanas informativas.

## 9. Relación feature–target
Se estudian correlaciones simples:
- La correlación lineal es **débil** en general.
- No se asume causalidad; es una primera señal de diagnóstico.

## 10. Análisis condicional
Se evalúa el target bajo distintos **regímenes de sentimiento**:
- Se observan efectos leves en contextos de alta intensidad noticiosa.
- La señal es dependiente del régimen.

## 11. Conclusiones
- Datos pre-2025-11-12 son heterogéneos → no aptos para entrenamiento.
- El sentimiento por sí solo **no predice bien** dirección a corto plazo.
- Su valor es **contextual** y funciona mejor combinado con modelos de precio.
- El EDA justifica usar sentimiento como **componente auxiliar** en un ensemble.

---
'''