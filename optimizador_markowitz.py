# ============================================================
#   OPTIMIZADOR DE PORTAFOLIO - FRONTERA DE MARKOWITZ + CML
#   Versión Streamlit Cloud
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from scipy.optimize import minimize
import warnings
import time
warnings.filterwarnings("ignore")

# ============================================================
#   CONFIGURACIÓN DE PÁGINA
# ============================================================

st.set_page_config(
    page_title="Optimizador de Portafolio",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stApp { background-color: #0f0f1a; color: white; }
    .stSlider > div > div { color: white; }
    h1, h2, h3 { color: white; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Optimizador de Portafolio — Frontera de Markowitz")
st.markdown("---")

# ============================================================
#   PARÁMETROS EN SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Parámetros")

tickers_input = st.sidebar.text_area(
    "Tickers (uno por línea)",
    value="AAPL\nMSFT\nGOOGL\nAMZN\nJPM",
    height=150
)
TICKERS = [t.strip().upper() for t in tickers_input.strip().split("\n") if t.strip()]

AÑOS_HISTORIA       = st.sidebar.slider("Años de historia", 1, 10, 5)
TASA_LIBRE_RIESGO   = st.sidebar.number_input("Tasa libre de riesgo (%)", 0.0, 20.0, 3.70, 0.1) / 100
RETORNO_OBJETIVO    = st.sidebar.number_input("Retorno objetivo (%)", 1.0, 100.0, 15.0, 1.0) / 100
N_PORTFOLIOS        = st.sidebar.select_slider("Portafolios simulados", [1000, 2000, 3000, 5000], value=3000)
PESO_MINIMO         = st.sidebar.slider("Peso mínimo por activo (%)", 0, 20, 0) / 100

ejecutar = st.sidebar.button("🚀 Calcular", use_container_width=True)

# ============================================================
#   FUNCIONES
# ============================================================

@st.cache_data(ttl=3600)
def descargar_precios(tickers, años):
    import datetime
    end   = datetime.date.today()
    start = end.replace(year=end.year - años)
    intentos = 0
    while intentos < 3:
        try:
            df = yf.download(tickers, start=str(start), end=str(end),
                             auto_adjust=True, progress=False)["Close"]
            df.dropna(how="all", inplace=True)
            df.dropna(axis=1, how="all", inplace=True)
            return df, list(df.columns)
        except Exception as e:
            intentos += 1
            time.sleep(2)
    return None, []


def calcular_retornos(df):
    return np.log(df / df.shift(1)).dropna()


def estadisticas_portafolio(pesos, retornos_medios, cov_matrix):
    ret = np.dot(pesos, retornos_medios) * 252
    vol = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix * 252, pesos)))
    return ret, vol


def sharpe(pesos, retornos_medios, cov_matrix, rf):
    ret, vol = estadisticas_portafolio(pesos, retornos_medios, cov_matrix)
    return -(ret - rf) / vol


def volatilidad_fn(pesos, retornos_medios, cov_matrix):
    return estadisticas_portafolio(pesos, retornos_medios, cov_matrix)[1]


def retorno_neg(pesos, retornos_medios, cov_matrix):
    return -estadisticas_portafolio(pesos, retornos_medios, cov_matrix)[0]


def optimizar(objetivo, n, retornos_medios, cov_matrix, rf, peso_min=0.0, retorno_obj=None):
    bounds      = tuple((peso_min, 1.0) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if retorno_obj is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w: estadisticas_portafolio(w, retornos_medios, cov_matrix)[0] - retorno_obj
        })
    fns = {
        "sharpe":     lambda w: sharpe(w, retornos_medios, cov_matrix, rf),
        "volatilidad": lambda w: volatilidad_fn(w, retornos_medios, cov_matrix),
        "retorno":    lambda w: retorno_neg(w, retornos_medios, cov_matrix),
    }
    x0  = np.array([1 / n] * n)
    return minimize(fns[objetivo], x0, method="SLSQP", bounds=bounds, constraints=constraints)


def frontera_eficiente(retornos_medios, cov_matrix, rf, n, peso_min, n_puntos=60):
    res_min = optimizar("volatilidad", n, retornos_medios, cov_matrix, rf, peso_min)
    res_max = optimizar("retorno",     n, retornos_medios, cov_matrix, rf, peso_min)
    ret_min, _ = estadisticas_portafolio(res_min.x, retornos_medios, cov_matrix)
    ret_max, _ = estadisticas_portafolio(res_max.x, retornos_medios, cov_matrix)
    fe_vols, fe_rets = [], []
    for r_obj in np.linspace(ret_min, ret_max, n_puntos):
        res = optimizar("volatilidad", n, retornos_medios, cov_matrix, rf, peso_min, retorno_obj=r_obj)
        if res.success:
            r, v = estadisticas_portafolio(res.x, retornos_medios, cov_matrix)
            fe_vols.append(v)
            fe_rets.append(r)
    return np.array(fe_vols), np.array(fe_rets)

# ============================================================
#   EJECUCIÓN PRINCIPAL
# ============================================================

if ejecutar:
    with st.spinner("📥 Descargando datos de Yahoo Finance..."):
        df, tickers = descargar_precios(tuple(TICKERS), AÑOS_HISTORIA)

    if df is None or df.empty:
        st.error("❌ No se pudieron descargar los datos. Intentá de nuevo en unos segundos.")
        st.stop()

    tickers_fallidos = [t for t in TICKERS if t not in tickers]
    if tickers_fallidos:
        st.warning(f"⚠️ No se encontraron datos para: {', '.join(tickers_fallidos)}")

    st.success(f"✅ Datos descargados: {', '.join(tickers)} | {len(df)} días de historia")

    ret_log         = calcular_retornos(df)
    n               = len(tickers)
    retornos_medios = ret_log.mean().values
    cov_matrix      = ret_log.cov().values

    with st.spinner("⚙️ Simulando portafolios y calculando frontera..."):
        # Portafolios aleatorios
        port_rets, port_vols, port_sharpes, port_pesos = [], [], [], []
        for _ in range(N_PORTFOLIOS):
            w = np.random.dirichlet(np.ones(n))
            if PESO_MINIMO > 0:
                w = np.clip(w, PESO_MINIMO, 1)
                w /= w.sum()
            r, v = estadisticas_portafolio(w, retornos_medios, cov_matrix)
            port_rets.append(r)
            port_vols.append(v)
            port_sharpes.append((r - TASA_LIBRE_RIESGO) / v)
            port_pesos.append(w)

        port_rets    = np.array(port_rets)
        port_vols    = np.array(port_vols)
        port_sharpes = np.array(port_sharpes)

        # Frontera eficiente
        fe_vols, fe_rets = frontera_eficiente(retornos_medios, cov_matrix, TASA_LIBRE_RIESGO, n, PESO_MINIMO)

        # Portafolios óptimos
        res_sharpe = optimizar("sharpe",     n, retornos_medios, cov_matrix, TASA_LIBRE_RIESGO, PESO_MINIMO)
        res_minvol = optimizar("volatilidad", n, retornos_medios, cov_matrix, TASA_LIBRE_RIESGO, PESO_MINIMO)

        ret_s, vol_s   = estadisticas_portafolio(res_sharpe.x, retornos_medios, cov_matrix)
        ret_mv, vol_mv = estadisticas_portafolio(res_minvol.x, retornos_medios, cov_matrix)
        sr_s           = (ret_s - TASA_LIBRE_RIESGO) / vol_s

        res_obj = optimizar("volatilidad", n, retornos_medios, cov_matrix, TASA_LIBRE_RIESGO, PESO_MINIMO, retorno_obj=RETORNO_OBJETIVO)
        if res_obj.success:
            ret_o, vol_o = estadisticas_portafolio(res_obj.x, retornos_medios, cov_matrix)
        else:
            ret_o, vol_o = None, None

        # CML
        vol_cml = np.linspace(0, max(port_vols) * 1.1, 200)
        ret_cml = TASA_LIBRE_RIESGO + sr_s * vol_cml

    # ============================================================
    #   GRÁFICO PLOTLY
    # ============================================================

    fig = go.Figure()

    # Portafolios aleatorios
    hover_texts = []
    for i in range(len(port_rets)):
        pesos_str = "<br>".join([f"{tickers[j]}: {port_pesos[i][j]*100:.1f}%" for j in range(n)])
        hover_texts.append(
            f"Retorno: {port_rets[i]*100:.2f}%<br>Vol: {port_vols[i]*100:.2f}%<br>Sharpe: {port_sharpes[i]:.2f}<br><br>{pesos_str}"
        )

    fig.add_trace(go.Scatter(
        x=port_vols, y=port_rets,
        mode="markers",
        marker=dict(color=port_sharpes, colorscale="Viridis", size=4, opacity=0.5,
                    colorbar=dict(title="Sharpe", tickfont=dict(color="white"), titlefont=dict(color="white"))),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        name="Portafolios Aleatorios"
    ))

    # Frontera eficiente
    if len(fe_vols) > 1:
        fig.add_trace(go.Scatter(
            x=fe_vols, y=fe_rets,
            mode="lines",
            line=dict(color="#3a9eff", width=3),
            name="Frontera Eficiente"
        ))

    # CML
    fig.add_trace(go.Scatter(
        x=vol_cml, y=ret_cml,
        mode="lines",
        line=dict(color="#ff4d4d", width=2, dash="dash"),
        name="CML"
    ))

    # Sharpe Óptimo
    pesos_str_s = "<br>".join([f"{tickers[j]}: {res_sharpe.x[j]*100:.1f}%" for j in range(n)])
    fig.add_trace(go.Scatter(
        x=[vol_s], y=[ret_s],
        mode="markers",
        marker=dict(symbol="star", size=20, color="gold"),
        name=f"Sharpe Óptimo (SR={sr_s:.2f})",
        text=[f"Retorno: {ret_s*100:.2f}%<br>Vol: {vol_s*100:.2f}%<br>Sharpe: {sr_s:.2f}<br><br>{pesos_str_s}"],
        hovertemplate="%{text}<extra></extra>"
    ))

    # Mínima Volatilidad
    pesos_str_mv = "<br>".join([f"{tickers[j]}: {res_minvol.x[j]*100:.1f}%" for j in range(n)])
    fig.add_trace(go.Scatter(
        x=[vol_mv], y=[ret_mv],
        mode="markers",
        marker=dict(symbol="circle", size=14, color="#ff4d4d"),
        name="Mín. Volatilidad",
        text=[f"Retorno: {ret_mv*100:.2f}%<br>Vol: {vol_mv*100:.2f}%<br><br>{pesos_str_mv}"],
        hovertemplate="%{text}<extra></extra>"
    ))

    # Retorno Objetivo
    if ret_o is not None:
        pesos_str_o = "<br>".join([f"{tickers[j]}: {res_obj.x[j]*100:.1f}%" for j in range(n)])
        fig.add_trace(go.Scatter(
            x=[vol_o], y=[ret_o],
            mode="markers",
            marker=dict(symbol="x", size=16, color="#00ffaa"),
            name=f"Objetivo {RETORNO_OBJETIVO*100:.1f}%",
            text=[f"Retorno: {ret_o*100:.2f}%<br>Vol: {vol_o*100:.2f}%<br><br>{pesos_str_o}"],
            hovertemplate="%{text}<extra></extra>"
        ))

    fig.update_layout(
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        font=dict(color="white"),
        title=dict(
            text=f"Espacio de Portfolios — Frontera de Markowitz<br>"
                 f"<sub>Activos: {', '.join(tickers)} | Período: {AÑOS_HISTORIA} año(s) | Rf: {TASA_LIBRE_RIESGO*100:.2f}%</sub>",
            font=dict(size=16, color="white")
        ),
        xaxis=dict(title="Volatilidad Anual", tickformat=".0%", gridcolor="#222", color="white"),
        yaxis=dict(title="Retorno Anual",     tickformat=".0%", gridcolor="#222", color="white"),
        legend=dict(bgcolor="#1a1a2e", bordercolor="#555", font=dict(color="white")),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    #   MÉTRICAS Y TABLAS
    # ============================================================

    st.markdown("---")
    st.subheader("📊 Resultados de Optimización")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### ⭐ Sharpe Óptimo")
        st.metric("Retorno anual",  f"{ret_s*100:.2f}%")
        st.metric("Volatilidad",    f"{vol_s*100:.2f}%")
        st.metric("Sharpe ratio",   f"{sr_s:.3f}")
        df_s = pd.DataFrame({"Ticker": tickers, "Peso": [f"{w*100:.2f}%" for w in res_sharpe.x]})
        st.dataframe(df_s, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### 🔴 Mínima Volatilidad")
        sr_mv = (ret_mv - TASA_LIBRE_RIESGO) / vol_mv
        st.metric("Retorno anual",  f"{ret_mv*100:.2f}%")
        st.metric("Volatilidad",    f"{vol_mv*100:.2f}%")
        st.metric("Sharpe ratio",   f"{sr_mv:.3f}")
        df_mv = pd.DataFrame({"Ticker": tickers, "Peso": [f"{w*100:.2f}%" for w in res_minvol.x]})
        st.dataframe(df_mv, hide_index=True, use_container_width=True)

    with col3:
        st.markdown(f"### 🎯 Objetivo {RETORNO_OBJETIVO*100:.1f}%")
        if ret_o is not None:
            sr_o = (ret_o - TASA_LIBRE_RIESGO) / vol_o
            st.metric("Retorno anual",  f"{ret_o*100:.2f}%")
            st.metric("Volatilidad",    f"{vol_o*100:.2f}%")
            st.metric("Sharpe ratio",   f"{sr_o:.3f}")
            df_o = pd.DataFrame({"Ticker": tickers, "Peso": [f"{w*100:.2f}%" for w in res_obj.x]})
            st.dataframe(df_o, hide_index=True, use_container_width=True)
        else:
            st.warning("No se pudo alcanzar el retorno objetivo con los activos seleccionados.")

    st.markdown("---")
    st.caption(f"ℹ️ Tasa libre de riesgo: {TASA_LIBRE_RIESGO*100:.2f}% | Período: {AÑOS_HISTORIA} año(s) | Portafolios simulados: {N_PORTFOLIOS:,}")

else:
    st.info("👈 Configurá los parámetros en el panel izquierdo y hacé click en **Calcular**.")
    st.markdown("""
    ### ¿Qué hace esta app?
    - Descarga precios históricos de Yahoo Finance
    - Simula miles de portafolios aleatorios
    - Calcula la **Frontera Eficiente de Markowitz**
    - Encuentra el portafolio de **máximo Sharpe ratio**
    - Encuentra el portafolio de **mínima volatilidad**
    - Optimiza para un **retorno objetivo**
    - Grafica la **Capital Market Line (CML)**
    """)
