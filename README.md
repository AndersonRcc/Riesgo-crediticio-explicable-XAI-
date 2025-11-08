# 💳 Riesgo Crediticio Explicable (XAI)

## 🔹 Descripción
Proyecto para **predecir el riesgo de incumplimiento de préstamos** usando datos simulados de un buró de crédito, con explicaciones interpretables mediante **SHAP**.  
Permite conocer la probabilidad de default de un cliente y los factores que más influyen en cada decisión.

---

## 📊 Exploración de datos (EDA)
- Análisis de variables personales, financieras y del préstamo: edad, ingreso, antigüedad laboral, tipo de vivienda, monto, tasa del préstamo, historial crediticio y estado del préstamo.
- Comparación de clientes cumplidores vs incumplidores.
- Identificación de outliers y valores faltantes para limpieza y transformación.

---

## 🧹 Preprocesamiento
- Limpieza de nulos y outliers.  
- **Pipeline de transformación**:
  - Variables numéricas → escalado
  - Variables categóricas → codificación
- SMOTE para balancear clases desiguales.

---

## 🤖 Modelado y tuning
- Modelos probados: Regresión Logística, Random Forest, XGBoost y LightGBM.  
- **Tuning de hiperparámetros** aplicado a cada modelo para optimizar desempeño.  
- **LightGBM seleccionado** por mejor balance de métricas.

| Modelo                | ROC-AUC | F1 Score | Recall Clase 1 |
|-----------------------|---------|----------|----------------|
| Regresión Logística   | 0.8652  | 0.6131   | 0.7913         |
| Random Forest         | 0.9278  | 0.8145   | 0.7186         |
| XGBoost               | 0.9449  | 0.8272   | 0.7243         |
| **LightGBM**          | **0.9449** | **0.8295** | **0.7257** |

- Pipeline final: preprocesamiento + modelo → predicciones en tiempo real.

---

## 🧩 Interpretabilidad (XAI)
- Técnica: **SHAP**  
- Hallazgos clave:
  1. **Capacidad de pago** (ingreso y % destinado al préstamo) → principal factor de riesgo.
  2. **Condiciones del préstamo** (calificación, monto, tasa) → también determinantes.

---

## 💡 Recomendaciones
- Priorizar relación ingreso/deuda y calificación del préstamo al evaluar riesgos.  
- Ajustar políticas y productos según perfil de cliente para reducir exposición al riesgo.

---

## 🚀 Despliegue
- Interfaz web en **Gradio** para predicciones en tiempo real.  
- App disponible en Hugging Face Spaces: [proyecto XAI](https://huggingface.co/spaces/Ander21rcc/proyecto_xai)  
- Funcionalidad: ingresar datos de un cliente, obtener predicción y visualización SHAP.

---

## ⚙️ Uso local
```bash
git clone https://github.com/AndersonRcc/Riesgo-crediticio-explicable-XAI-.git
pip install -r requirements.txt
python app/app.py
