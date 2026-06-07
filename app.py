import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Detección de Fraude en Tarjetas de Crédito",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        padding: 10px 0 5px 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #4a4a6a;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
    }
    .fraud-badge {
        background-color: #e63946;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    .normal-badge {
        background-color: #2a9d8f;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🔐 Detección de Fraude en Tarjetas de Crédito</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Análisis Exploratorio · Series de Tiempo · Detección de Anomalías</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-card-front-side.png", width=80)
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Cargar dataset (.csv)",
        type=["csv"],
        help="Carga el archivo creditcard.csv de Kaggle"
    )

    st.markdown("---")
    st.markdown("**📊 Fuente de datos:**")
    st.markdown("[Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
    st.markdown("---")
    st.markdown("**🔍 Dataset:**")
    st.markdown("- 284,807 transacciones")
    st.markdown("- 492 fraudes (0.17%)")
    st.markdown("- 30 características (V1–V28, Time, Amount)")
    st.markdown("---")
    st.info("💡 Si no tienes el dataset, la app genera datos de demostración.")

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Synthetic demo data that mimics the real dataset structure
        np.random.seed(42)
        n = 10000
        n_fraud = 17  # ~0.17%

        # Normal transactions
        normal = pd.DataFrame({
            'Time': np.sort(np.random.uniform(0, 172800, n - n_fraud)),
            'Amount': np.abs(np.random.lognormal(3.5, 1.5, n - n_fraud)),
            'Class': 0
        })
        for i in range(1, 29):
            normal[f'V{i}'] = np.random.normal(0, 1, n - n_fraud)

        # Fraud transactions (higher amounts, anomalous features)
        fraud = pd.DataFrame({
            'Time': np.random.uniform(0, 172800, n_fraud),
            'Amount': np.abs(np.random.lognormal(5, 1.2, n_fraud)),
            'Class': 1
        })
        for i in range(1, 29):
            fraud[f'V{i}'] = np.random.normal(0, 3, n_fraud)

        df = pd.concat([normal, fraud], ignore_index=True)
        cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)] + ['Class']
        df = df[cols].sort_values('Time').reset_index(drop=True)
    return df

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
with st.spinner("⏳ Cargando datos..."):
    df = load_data(uploaded_file)

is_demo = uploaded_file is None
if is_demo:
    st.warning("⚠️ Usando **datos de demostración** (10,000 transacciones sintéticas). Carga el dataset real de Kaggle para resultados completos.")

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data
def feature_engineering(df):
    df = df.copy()
    df['Hour'] = (df['Time'] // 3600) % 24
    df['Day'] = (df['Time'] // 86400).astype(int) + 1
    df['Amount_log'] = np.log1p(df['Amount'])
    df['is_fraud'] = df['Class'].map({0: 'Normal', 1: 'Fraude'})
    return df

df = feature_engineering(df)

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filtros")
    class_filter = st.multiselect(
        "Tipo de transacción",
        options=['Normal', 'Fraude'],
        default=['Normal', 'Fraude']
    )
    amount_range = st.slider(
        "Rango de monto ($)",
        float(df['Amount'].min()),
        min(float(df['Amount'].max()), 5000.0),
        (0.0, min(float(df['Amount'].max()), 5000.0))
    )

df_filtered = df[
    (df['is_fraud'].isin(class_filter)) &
    (df['Amount'] >= amount_range[0]) &
    (df['Amount'] <= amount_range[1])
]

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
total = len(df_filtered)
n_fraud = (df_filtered['Class'] == 1).sum()
n_normal = (df_filtered['Class'] == 0).sum()
fraud_rate = n_fraud / total * 100 if total > 0 else 0
avg_amount_fraud = df_filtered[df_filtered['Class'] == 1]['Amount'].mean() if n_fraud > 0 else 0
avg_amount_normal = df_filtered[df_filtered['Class'] == 0]['Amount'].mean() if n_normal > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📦 Total Transacciones", f"{total:,}")
with col2:
    st.metric("✅ Normales", f"{n_normal:,}")
with col3:
    st.metric("🚨 Fraudes", f"{n_fraud:,}")
with col4:
    st.metric("⚠️ Tasa de Fraude", f"{fraud_rate:.3f}%")
with col5:
    st.metric("💰 Monto Prom. Fraude", f"${avg_amount_fraud:.2f}")

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Exploración General",
    "⏰ Series de Tiempo",
    "🔬 Análisis de Características",
    "🤖 Detección de Anomalías",
    "📋 Datos"
])

# ══════════════════════════════════════════════
# TAB 1 — EXPLORACIÓN GENERAL
# ══════════════════════════════════════════════
with tab1:
    st.subheader("📊 Exploración General del Dataset")

    col_a, col_b = st.columns(2)

    with col_a:
        # Distribución de clases
        class_counts = df_filtered['is_fraud'].value_counts().reset_index()
        class_counts.columns = ['Clase', 'Cantidad']
        fig_pie = px.pie(
            class_counts,
            names='Clase',
            values='Cantidad',
            title='Distribución de Clases',
            color='Clase',
            color_discrete_map={'Normal': '#2a9d8f', 'Fraude': '#e63946'},
            hole=0.45
        )
        fig_pie.update_traces(textposition='outside', textinfo='percent+label')
        fig_pie.update_layout(height=380, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        # Distribución de montos
        fig_amount = px.histogram(
            df_filtered[df_filtered['Amount'] <= 500],
            x='Amount',
            color='is_fraud',
            nbins=60,
            title='Distribución de Montos (≤ $500)',
            color_discrete_map={'Normal': '#2a9d8f', 'Fraude': '#e63946'},
            barmode='overlay',
            opacity=0.7,
            labels={'Amount': 'Monto ($)', 'is_fraud': 'Tipo'}
        )
        fig_amount.update_layout(height=380)
        st.plotly_chart(fig_amount, use_container_width=True)

    # Box plot montos por clase
    fig_box = px.box(
        df_filtered,
        x='is_fraud',
        y='Amount',
        color='is_fraud',
        title='Distribución de Montos por Clase (escala log)',
        color_discrete_map={'Normal': '#2a9d8f', 'Fraude': '#e63946'},
        log_y=True,
        labels={'is_fraud': 'Tipo de Transacción', 'Amount': 'Monto ($) — escala log'}
    )
    fig_box.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # Stats table
    st.markdown("#### 📋 Estadísticas Descriptivas por Clase")
    stats = df_filtered.groupby('is_fraud')['Amount'].describe().round(2)
    st.dataframe(stats, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — SERIES DE TIEMPO
# ══════════════════════════════════════════════
with tab2:
    st.subheader("⏰ Análisis Temporal de Transacciones")

    col_ts1, col_ts2 = st.columns([2, 1])
    with col_ts1:
        granularity = st.selectbox("Granularidad temporal", ["Por hora", "Por día"], index=0)
    with col_ts2:
        show_rolling = st.checkbox("Mostrar media móvil (fraudes)", value=True)

    time_col = 'Hour' if granularity == "Por hora" else 'Day'
    label = "Hora del día" if granularity == "Por hora" else "Día"

    # Transacciones por tiempo
    time_normal = df_filtered[df_filtered['Class'] == 0].groupby(time_col).size().reset_index(name='Normales')
    time_fraud = df_filtered[df_filtered['Class'] == 1].groupby(time_col).size().reset_index(name='Fraudes')
    time_df = time_normal.merge(time_fraud, on=time_col, how='left').fillna(0)

    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ts.add_trace(
        go.Bar(x=time_df[time_col], y=time_df['Normales'],
               name='Normales', marker_color='#2a9d8f', opacity=0.7),
        secondary_y=False
    )
    fig_ts.add_trace(
        go.Scatter(x=time_df[time_col], y=time_df['Fraudes'],
                   name='Fraudes', mode='lines+markers',
                   line=dict(color='#e63946', width=2.5),
                   marker=dict(size=6)),
        secondary_y=True
    )
    if show_rolling and len(time_df) >= 3:
        rolling = time_df['Fraudes'].rolling(3, min_periods=1).mean()
        fig_ts.add_trace(
            go.Scatter(x=time_df[time_col], y=rolling,
                       name='Media móvil fraudes', mode='lines',
                       line=dict(color='#ff9f1c', width=2, dash='dash')),
            secondary_y=True
        )
    fig_ts.update_xaxes(title_text=label)
    fig_ts.update_yaxes(title_text="# Transacciones Normales", secondary_y=False)
    fig_ts.update_yaxes(title_text="# Fraudes", secondary_y=True)
    fig_ts.update_layout(title=f"Transacciones {label.lower()}", height=420, hovermode="x unified")
    st.plotly_chart(fig_ts, use_container_width=True)

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        # Monto promedio por hora
        amt_by_time = df_filtered.groupby([time_col, 'is_fraud'])['Amount'].mean().reset_index()
        fig_amt = px.line(
            amt_by_time,
            x=time_col, y='Amount', color='is_fraud',
            title=f'Monto Promedio por {label}',
            color_discrete_map={'Normal': '#2a9d8f', 'Fraude': '#e63946'},
            markers=True,
            labels={time_col: label, 'Amount': 'Monto Prom. ($)', 'is_fraud': 'Tipo'}
        )
        fig_amt.update_layout(height=360)
        st.plotly_chart(fig_amt, use_container_width=True)

    with col_t2:
        # Tasa de fraude por hora
        fraud_rate_time = df_filtered.groupby(time_col).apply(
            lambda x: (x['Class'] == 1).sum() / len(x) * 100
        ).reset_index(name='Tasa Fraude (%)')
        fig_rate = px.area(
            fraud_rate_time,
            x=time_col, y='Tasa Fraude (%)',
            title=f'Tasa de Fraude (%) por {label}',
            color_discrete_sequence=['#e63946']
        )
        fig_rate.update_layout(height=360)
        st.plotly_chart(fig_rate, use_container_width=True)

    # Heatmap hora vs día (solo si hay datos reales con 2 días)
    if df_filtered['Day'].nunique() > 1:
        st.markdown("#### 🗓️ Mapa de Calor: Fraudes por Hora y Día")
        heatmap_data = df_filtered[df_filtered['Class'] == 1].groupby(['Day', 'Hour']).size().unstack(fill_value=0)
        fig_heat = px.imshow(
            heatmap_data,
            title="Concentración de Fraudes (Día × Hora)",
            color_continuous_scale='Reds',
            labels=dict(x="Hora", y="Día", color="# Fraudes"),
            aspect="auto"
        )
        fig_heat.update_layout(height=350)
        st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — ANÁLISIS DE CARACTERÍSTICAS
# ══════════════════════════════════════════════
with tab3:
    st.subheader("🔬 Análisis de Características (Features)")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        feat_x = st.selectbox("Característica Eje X", [f"V{i}" for i in range(1, 29)], index=0)
    with col_f2:
        feat_y = st.selectbox("Característica Eje Y", [f"V{i}" for i in range(1, 29)], index=1)

    # Scatter
    sample_df = df_filtered.sample(min(3000, len(df_filtered)), random_state=42)
    fig_scatter = px.scatter(
        sample_df, x=feat_x, y=feat_y, color='is_fraud',
        color_discrete_map={'Normal': '#2a9d8f', 'Fraude': '#e63946'},
        title=f'Dispersión: {feat_x} vs {feat_y}',
        opacity=0.5,
        labels={'is_fraud': 'Tipo'}
    )
    fig_scatter.update_layout(height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Distribución de feature seleccionada
    st.markdown("#### 📈 Distribución por Característica")
    feat_dist = st.selectbox("Seleccionar característica", [f"V{i}" for i in range(1, 29)], index=2)
    fig_dist = px.histogram(
        df_filtered, x=feat_dist, color='is_fraud',
        nbins=80, barmode='overlay', opacity=0.65,
        color_discrete_map={'Normal': '#2a9d8f', 'Fraude': '#e63946'},
        title=f'Distribución de {feat_dist} por clase',
        labels={'is_fraud': 'Tipo'}
    )
    fig_dist.update_layout(height=360)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Correlación top features
    st.markdown("#### 🔗 Correlación de Características con Clase")
    v_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
    corr_vals = df_filtered[v_cols + ['Class']].corr()['Class'].drop('Class').sort_values(key=abs, ascending=False)
    fig_corr = px.bar(
        x=corr_vals.values,
        y=corr_vals.index,
        orientation='h',
        title='Correlación de cada feature con la variable Class',
        color=corr_vals.values,
        color_continuous_scale='RdBu_r',
        labels={'x': 'Correlación de Pearson', 'y': 'Feature'}
    )
    fig_corr.update_layout(height=500, yaxis={'autorange': 'reversed'})
    st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — DETECCIÓN DE ANOMALÍAS
# ══════════════════════════════════════════════
with tab4:
    st.subheader("🤖 Detección de Anomalías con Isolation Forest")

    st.markdown("""
    **Isolation Forest** es un algoritmo no supervisado diseñado específicamente para detectar anomalías.
    Aísla observaciones construyendo árboles de decisión aleatorios: las anomalías requieren menos particiones para ser aisladas.
    """)

    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        contamination = st.slider(
            "Tasa de contaminación esperada",
            0.001, 0.05, 0.002, 0.001,
            help="Fracción esperada de anomalías en el dataset"
        )
        n_estimators = st.slider("Número de árboles", 50, 200, 100, 10)
        run_model = st.button("🚀 Ejecutar modelo", type="primary", use_container_width=True)

    with col_m2:
        st.info("""
        **¿Cómo funciona?**
        1. Se seleccionan aleatoriamente una característica y un punto de corte
        2. Se repite recursivamente hasta aislar cada punto
        3. Las anomalías se aíslan en menos pasos (puntuación alta)
        4. Se clasifican como fraude los puntos con mayor puntuación de anomalía
        """)

    if run_model:
        with st.spinner("🔄 Entrenando Isolation Forest..."):
            features = [f'V{i}' for i in range(1, 29)] + ['Amount_log']
            X = df_filtered[features].fillna(0)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            preds = model.fit_predict(X_scaled)
            scores = model.score_samples(X_scaled)

            df_result = df_filtered.copy()
            df_result['pred_label'] = np.where(preds == -1, 1, 0)
            df_result['anomaly_score'] = -scores  # Higher = more anomalous

        st.success("✅ Modelo entrenado exitosamente")

        # Métricas
        if df_result['Class'].nunique() > 1:
            from sklearn.metrics import precision_score, recall_score, f1_score
            y_true = df_result['Class']
            y_pred = df_result['pred_label']
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            m1, m2, m3 = st.columns(3)
            m1.metric("🎯 Precisión", f"{prec:.3f}")
            m2.metric("📡 Recall", f"{rec:.3f}")
            m3.metric("⚖️ F1-Score", f"{f1:.3f}")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            # Score distribution
            fig_score = px.histogram(
                df_result, x='anomaly_score', color='is_fraud',
                nbins=80, barmode='overlay', opacity=0.7,
                title='Distribución del Puntaje de Anomalía',
                color_discrete_map={'Normal': '#2a9d8f', 'Fraude': '#e63946'},
                labels={'anomaly_score': 'Puntaje de Anomalía', 'is_fraud': 'Tipo'}
            )
            fig_score.update_layout(height=380)
            st.plotly_chart(fig_score, use_container_width=True)

        with col_r2:
            # Confusion matrix
            if df_result['Class'].nunique() > 1:
                cm = confusion_matrix(df_result['Class'], df_result['pred_label'])
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    title='Matriz de Confusión',
                    labels=dict(x="Predicho", y="Real", color="Count"),
                    x=['Normal', 'Fraude'],
                    y=['Normal', 'Fraude'],
                    color_continuous_scale='Blues'
                )
                fig_cm.update_layout(height=380)
                st.plotly_chart(fig_cm, use_container_width=True)

        # Top anomalías detectadas
        st.markdown("#### 🚨 Top 20 Transacciones Más Anómalas Detectadas")
        top_anomalies = df_result.nlargest(20, 'anomaly_score')[
            ['Time', 'Amount', 'anomaly_score', 'Class', 'pred_label', 'Hour']
        ].copy()
        top_anomalies['Real'] = top_anomalies['Class'].map({0: '✅ Normal', 1: '🚨 Fraude'})
        top_anomalies['Predicción'] = top_anomalies['pred_label'].map({0: '✅ Normal', 1: '🚨 Fraude'})
        top_anomalies['Puntaje'] = top_anomalies['anomaly_score'].round(4)
        top_anomalies['Hora'] = top_anomalies['Hour'].astype(int)
        st.dataframe(
            top_anomalies[['Time', 'Amount', 'Hora', 'Puntaje', 'Real', 'Predicción']].reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.info("👆 Ajusta los parámetros y presiona **Ejecutar modelo** para ver los resultados.")


# ══════════════════════════════════════════════
# TAB 5 — DATOS
# ══════════════════════════════════════════════
with tab5:
    st.subheader("📋 Explorador de Datos")

    col_d1, col_d2 = st.columns([3, 1])
    with col_d2:
        show_fraud_only = st.checkbox("Solo mostrar fraudes", value=False)
        n_rows = st.slider("Filas a mostrar", 10, 200, 50)

    display_df = df_filtered[df_filtered['Class'] == 1] if show_fraud_only else df_filtered
    st.dataframe(display_df.head(n_rows), use_container_width=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### 📐 Forma del dataset")
        st.write(f"**Filas:** {len(df_filtered):,}  |  **Columnas:** {df_filtered.shape[1]}")

    with col_s2:
        st.markdown("#### 🔍 Valores nulos")
        nulls = df_filtered.isnull().sum()
        if nulls.sum() == 0:
            st.success("✅ No hay valores nulos")
        else:
            st.dataframe(nulls[nulls > 0])

    st.markdown("#### 📊 Estadísticas descriptivas")
    st.dataframe(display_df[['Time', 'Amount'] + [f'V{i}' for i in range(1, 6)]].describe().round(3),
                 use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "🔐 Credit Card Fraud Detection Dashboard · Proyecto Final de Maestría · Ingeniería / Tecnología"
    "</div>",
    unsafe_allow_html=True
)


