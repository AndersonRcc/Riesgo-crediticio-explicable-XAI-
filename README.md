Riesgo Crediticio Explicable (XAI)
Descripción

Este proyecto analiza y predice el riesgo de incumplimiento de préstamos utilizando datos simulados de un buró de crédito.
El objetivo no es solo predecir si un cliente puede incumplir, sino entender qué factores influyen en cada decisión gracias a técnicas de interpretación como SHAP.

Exploración de datos (EDA)

Se analizaron variables personales, financieras y del préstamo: edad, ingreso, antigüedad laboral, tipo de vivienda, monto y tasa del préstamo, historial crediticio y estado del préstamo.

Se identificaron patrones de clientes cumplidores vs incumplidores, outliers y valores faltantes para preparar los datos.

Preprocesamiento

Limpieza de nulos y outliers.

Pipeline completo:

Escalado de variables numéricas

Codificación de variables categóricas

Uso de SMOTE para balancear clases desiguales.

Modelado y tuning

Se probaron varios modelos: Regresión Logística, Random Forest, XGBoost y LightGBM.

Cada modelo fue ajustado mediante tuning de hiperparámetros para optimizar desempeño.

LightGBM fue seleccionado por su mejor balance entre ROC-AUC, F1 y Recall (clase 1).

Modelo	ROC-AUC	F1 Score	Recall Clase 1
Regresión Logística	0.8652	0.6131	0.7913
Random Forest	0.9278	0.8145	0.7186
XGBoost	0.9449	0.8272	0.7243
LightGBM	0.9449	0.8295	0.7257

El pipeline final integra preprocesamiento + modelo para predicciones en tiempo real.

Interpretabilidad (XAI)

Técnica: SHAP

Permite explicar la contribución de cada variable a la predicción individual y global del modelo.

Hallazgos principales:

Capacidad de pago (ingreso y porcentaje del ingreso destinado al préstamo) → principal factor de riesgo

Condiciones del préstamo (calificación, monto, tasa) → también determinantes

Recomendaciones

Priorizar la relación ingreso/deuda y la calificación del préstamo al evaluar riesgos.

Ajustar políticas y productos según el perfil de cada cliente para reducir exposición al riesgo.

Despliegue

Interfaz web con Gradio para predicciones en tiempo real.

App disponible en Hugging Face Spaces: proyecto XAI

Funcionalidad: ingresar datos de un cliente, obtener predicción y explicación visual de SHAP.
