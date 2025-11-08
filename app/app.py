import gradio as gr
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from PIL import Image
import io
import warnings

# Suprimir advertencias de inconsistencia de versión de sklearn (si es necesario)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Carga de modelos ---
MODEL_PATH = "RiesgoCrediticio_XAI/models"

try:
    # Asegúrate de que esta ruta sea accesible en tu entorno
    transformer_pipeline = joblib.load(f"{MODEL_PATH}/transformador.pkl")
    model = joblib.load(f"{MODEL_PATH}/lightgbm_tuneado.pkl")
    MODEL_LOADED = True
except FileNotFoundError as e:
    print(f"⚠️ Error al cargar archivos ({e}). Usando modelos simulados.")

    class MockTransformer:
        def transform(self, df):
            features = [f'F{i}' for i in range(10)]
            return pd.DataFrame(np.random.rand(1, 10), columns=features)

        def get_feature_names_out(self, input_features=None):
            return [f'F{i}' for i in range(10)]

    class MockModel:
        def predict(self, X):
            return [0]
        def predict_proba(self, X):
            return [[0.92, 0.08]]

    transformer_pipeline = MockTransformer()
    model = MockModel()
    MODEL_LOADED = False

# --- Mapas y opciones ---
HOME_OWNERSHIP_MAP = {
    "Renta": "RENT",
    "Propia": "OWN",
    "Hipotecada": "MORTGAGE",
    "Otra/Desconocida": "Otros"
}

LOAN_INTENT_MAP = {
    "Educación": "EDUCATION",
    "Médico": "MEDICAL",
    "Personal": "PERSONAL",
    "Consolidación de Deuda": "DEBTCONSOLIDATION",
    "Emprendimiento/Venture": "VENTURE",
    "Mejora del Hogar": "HOMEIMPROVEMENT",
    "Otro Propósito": "Otros"
}

DEFAULT_ON_FILE_MAP = {
    "Sí (Tiene registro de impago)": "Y",
    "No (No tiene registro de impago)": "N"
}

HOME_OWNERSHIP_OPTIONS = list(HOME_OWNERSHIP_MAP.keys())
LOAN_INTENT_OPTIONS = list(LOAN_INTENT_MAP.keys())
DEFAULT_ON_FILE_OPTIONS = list(DEFAULT_ON_FILE_MAP.keys())
VALID_GRADES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# --- Función para obtener nombres de features de ColumnTransformer ---
def get_feature_names(transformer, input_df):
    try:
        return transformer.get_feature_names_out(input_df.columns)
    except:
        try:
            feature_names = []
            for name, trans, cols in transformer.transformers_:
                if name != 'remainder':
                    if hasattr(trans, 'get_feature_names_out'):
                        names = trans.get_feature_names_out(cols)
                    else:
                        names = cols
                    feature_names.extend(names)
            return feature_names
        except:
            return input_df.columns

# --- Función de predicción ---
def predict(person_age, person_income, person_home_ownership, person_emp_length,
            loan_intent, loan_grade, loan_amnt, loan_int_rate,
            cb_person_default_on_file, cb_person_cred_hist_length):

    # Mapear valores
    home_ownership_model = HOME_OWNERSHIP_MAP.get(person_home_ownership, "Otros")
    loan_intent_model = LOAN_INTENT_MAP.get(loan_intent, "Otros")
    default_on_file_model = DEFAULT_ON_FILE_MAP.get(cb_person_default_on_file, "N")

    # Validaciones
    if person_income is None or person_income <= 0:
        return "⚠️ Error: El Ingreso Anual debe ser un valor numérico positivo.", None
    if any(val is None or val < 0 for val in [person_age, loan_amnt, person_emp_length, cb_person_cred_hist_length]) or person_age < 18:
        return "⚠️ Error: Asegúrate que Edad, Antigüedad, Monto y Historial sean valores válidos y no vacíos (Edad ≥ 18).", None

    loan_int_rate = loan_int_rate if loan_int_rate is not None else 0.0
    loan_percent_income = loan_amnt / person_income

    new_df = pd.DataFrame([{
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': home_ownership_model,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent_model,
        'loan_grade': loan_grade if loan_grade in VALID_GRADES else 'G',
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': default_on_file_model,
        'cb_person_cred_hist_length': cb_person_cred_hist_length
    }])

    X_new = transformer_pipeline.transform(new_df)
    if isinstance(X_new, np.ndarray):
        X_new_df = pd.DataFrame(X_new, columns=get_feature_names(transformer_pipeline, new_df))
    else:
        X_new_df = X_new

    prediction = model.predict(X_new_df)[0]
    prob_default = model.predict_proba(X_new_df)[0][1]

    status = "**RIESGO ALTO (INCUMPLIMIENTO)** 🔴" if prediction == 1 else "**RIESGO BAJO (PAGO SEGURO)** 🟢"
    color = "red" if prediction == 1 else "green"

    markdown_output = f"""
##### {status}
**Probabilidad de Incumplimiento:** <span style='color:{color}; font-size:1.0em;'>**{prob_default:.2%}**</span>
"""
    return markdown_output, None 

# --- Función SHAP individual ---
def shap_individual(person_age, person_income, person_home_ownership, person_emp_length,
                    loan_intent, loan_grade, loan_amnt, loan_int_rate,
                    cb_person_default_on_file, cb_person_cred_hist_length):

    if not MODEL_LOADED:
        print("⚠️ Modelos simulados. SHAP real no disponible.")
        return None

    home_ownership_model = HOME_OWNERSHIP_MAP.get(person_home_ownership, "Otros")
    loan_intent_model = LOAN_INTENT_MAP.get(loan_intent, "Otros")
    default_on_file_model = DEFAULT_ON_FILE_MAP.get(cb_person_default_on_file, "N")
    loan_int_rate = loan_int_rate if loan_int_rate is not None else 0.0
    loan_percent_income = loan_amnt / person_income

    new_df = pd.DataFrame([{
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': home_ownership_model,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent_model,
        'loan_grade': loan_grade if loan_grade in VALID_GRADES else 'G',
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': default_on_file_model,
        'cb_person_cred_hist_length': cb_person_cred_hist_length
    }])

    X_new = transformer_pipeline.transform(new_df)
    if isinstance(X_new, np.ndarray):
        X_new_df = pd.DataFrame(X_new, columns=get_feature_names(transformer_pipeline, new_df))
    else:
        X_new_df = X_new

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_new_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf)
    return img

# ----------------------------------------------------------------------
# --- Interfaz Gradio ---
# ----------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Riesgo Crediticio App") as demo:

    gr.Markdown("<h1 style='text-align: center; margin-bottom: 20px;'>🏦 Predicción de Riesgo Crediticio con XAI (Explicabilidad)</h1>")

    if not MODEL_LOADED:
        gr.Warning("Los modelos reales no se pudieron cargar. Se usarán valores simulados.")

    # Inicialización de inputs
    person_age_input = None
    person_income_input = None
    person_home_ownership_input = None
    person_emp_length_input = None
    loan_intent_input = None
    loan_grade_input = None
    loan_amnt_input = None
    loan_int_rate_input = None
    cb_person_default_on_file_input = None
    cb_person_cred_hist_length_input = None
    output = None
    shap_output = None

    with gr.Row():
        with gr.Column(scale=1, min_width=450):
            gr.Markdown("### 📝 Ingrese los datos del solicitante")
            with gr.Tabs() as tabs:
                with gr.Tab("👤 Datos Personales"):
                    person_age_input = gr.Number(label="Edad (años)", minimum=18, maximum=100)
                    person_income_input = gr.Number(label="Ingreso Anual (USD)", minimum=1)
                    person_home_ownership_input = gr.Radio(HOME_OWNERSHIP_OPTIONS, label="Tipo de Vivienda")
                    person_emp_length_input = gr.Number(label="Antigüedad Laboral (años)", minimum=0)
                with gr.Tab("💸 Detalles del Préstamo"):
                    loan_intent_input = gr.Dropdown(LOAN_INTENT_OPTIONS, label="Motivo del Préstamo")
                    loan_amnt_input = gr.Number(label="Monto Solicitado (USD)", minimum=100)
                    loan_grade_input = gr.Radio(VALID_GRADES, label="Calificación del Préstamo")
                    loan_int_rate_input = gr.Number(label="Tasa de Interés (%)", minimum=0)
                with gr.Tab("🧾 Historial Crediticio"):
                    cb_person_cred_hist_length_input = gr.Number(label="Duración Historial Crediticio (años)", minimum=0)
                    cb_person_default_on_file_input = gr.Radio(DEFAULT_ON_FILE_OPTIONS, label="Historial de Incumplimiento")

        gr.Column(min_width=30, scale=0)

        with gr.Column(scale=1, min_width=350):
            gr.Markdown("### 🔮 Predicción de Riesgo")
            with gr.Row():
                output = gr.Markdown(value="Presiona 'Predecir' para evaluar el riesgo.")
                btn_predict = gr.Button("Predecir riesgo", variant="primary", scale=0)

            gr.Markdown("---")

            with gr.Column():
                gr.Markdown("### Análisis experto")
                shap_output = gr.Image(type="pil", label="Factores que influyen en el Riesgo", height=300)
                btn_shap = gr.Button("Mostrar Explicación SHAP", variant="primary")

    gr.Markdown("---")
    clear_btn = gr.ClearButton(
        value="❌ Reiniciar Formulario y Resultados",
        variant="primary",
        size="s",
        components=[
            person_age_input, person_income_input, person_home_ownership_input, person_emp_length_input,
            loan_intent_input, loan_grade_input, loan_amnt_input, loan_int_rate_input,
            cb_person_default_on_file_input, cb_person_cred_hist_length_input,
            output, shap_output
        ]
    )

    btn_predict.click(
        predict,
        inputs=[
            person_age_input, person_income_input, person_home_ownership_input, person_emp_length_input,
            loan_intent_input, loan_grade_input, loan_amnt_input, loan_int_rate_input,
            cb_person_default_on_file_input, cb_person_cred_hist_length_input
        ],
        outputs=[output, shap_output]
    )

    btn_shap.click(
        shap_individual,
        inputs=[
            person_age_input, person_income_input, person_home_ownership_input, person_emp_length_input,
            loan_intent_input, loan_grade_input, loan_amnt_input, loan_int_rate_input,
            cb_person_default_on_file_input, cb_person_cred_hist_length_input
        ],
        outputs=[shap_output]
    )

demo.launch()
