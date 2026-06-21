# ============================================================
#   OPTIMIZADOR DE PORTAFOLIO - FRONTERA DE MARKOWITZ + CML
#   Compatible con Google Colab
# ============================================================

# ---------- 1. INSTALACIÓN DE DEPENDENCIAS ----------
# Ejecutá esta celda primero en Colab:
# !pip install yfinance scipy matplotlib numpy pandas --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# ============================================================
#   PARÁMETROS DE ENTRADA — MODIFICÁ AQUÍ
# ============================================================

TICKERS          = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]   # Activos
AÑOS_HISTORIA    = 5          # Entre 1 y 20 años
TASA_LIBRE_RIESGO = 0.0370   # Tasa anual (ej: 0.037 = 3.70%)
RETORNO_OBJETIVO  = 0.15     # Retorno objetivo anual (ej: 0.15 = 15%)
N_PORTFOLIOS      = 5000      # Portafolios aleatorios a simular
PESO_MINIMO       = 0.00      # Peso mínimo por activo (ej: 0.05 = 5%)

# ============================================================
#   FUNCIONES AUXILIARES
# ============================================================

def descargar_precios(tickers, años):
    """Descarga precios de cierre ajustados desde Yahoo Finance."""
    import datetime
    end   = datetime.date.today()
    start = end.replace(year=end.year - años)
    print(f"\n📥 Descargando datos: {tickers}")
    print(f"   Período: {start} → {end} ({años} años)\n")
    df = yf.download(tickers, start=str(start), end=str(end),
                     auto_adjust=True, progress=False)["Close"]
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    tickers_ok = list(df.columns)
    print(f"✅ Activos disponibles: {tickers_ok}")
    print(f"   Observaciones: {len(df)} días\n")
    return df, tickers_ok


def calcular_retornos(df):
    """Retornos logarítmicos diarios."""
    return np.log(df / df.shift(1)).dropna()


def estadisticas_portafolio(pesos, retornos_medios, cov_matrix):
    """Retorno y volatilidad anualizados para un vector de pesos."""
    ret  = np.dot(pesos, retornos_medios) * 252
    vol  = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix * 252, pesos)))
    return ret, vol


def sharpe(pesos, retornos_medios, cov_matrix, rf):
    ret, vol = estadisticas_portafolio(pesos, retornos_medios, cov_matrix)
    return -(ret - rf) / vol   # negativo para minimizar


def volatilidad(pesos, retornos_medios, cov_matrix):
    return estadisticas_portafolio(pesos, retornos_medios, cov_matrix)[1]


def retorno_neg(pesos, retornos_medios, cov_matrix):
    return -estadisticas_portafolio(pesos, retornos_medios, cov_matrix)[0]


def optimizar(objetivo_fn, n, retornos_medios, cov_matrix, rf,
              peso_min=0.0, retorno_obj=None):
    """Optimización con restricciones de suma=1 y pesos mínimos."""
    bounds      = tuple((peso_min, 1.0) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if retorno_obj is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w: estadisticas_portafolio(w, retornos_medios, cov_matrix)[0] - retorno_obj
        })

    kwargs = dict(retornos_medios=retornos_medios,
                  cov_matrix=cov_matrix, rf=rf)

    if objetivo_fn == sharpe:
        fn = lambda w: sharpe(w, retornos_medios, cov_matrix, rf)
    elif objetivo_fn == volatilidad:
        fn = lambda w: volatilidad(w, retornos_medios, cov_matrix)
    else:
        fn = lambda w: retorno_neg(w, retornos_medios, cov_matrix)

    x0  = np.array([1 / n] * n)
    res = minimize(fn, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res


def frontera_eficiente(retornos_medios, cov_matrix, rf, n_activos,
                       peso_min, n_puntos=80):
    """Calcula puntos de la frontera eficiente."""
    # Rango: entre mín volatilidad y portafolio de máx retorno
    res_min = optimizar(volatilidad, n_activos, retornos_medios, cov_matrix, rf, peso_min)
    res_max = optimizar(retorno_neg,  n_activos, retornos_medios, cov_matrix, rf, peso_min)

    ret_min, _ = estadisticas_portafolio(res_min.x, retornos_medios, cov_matrix)
    ret_max, _ = estadisticas_portafolio(res_max.x, retornos_medios, cov_matrix)

    retornos_objetivo = np.linspace(ret_min, ret_max, n_puntos)
    frontera_vols, frontera_rets = [], []

    for r_obj in retornos_objetivo:
        res = optimizar(volatilidad, n_activos, retornos_medios, cov_matrix, rf,
                        peso_min, retorno_obj=r_obj)
        if res.success:
            ret, vol = estadisticas_portafolio(res.x, retornos_medios, cov_matrix)
            frontera_vols.append(vol)
            frontera_rets.append(ret)

    return np.array(frontera_vols), np.array(frontera_rets)


# ============================================================
#   MAIN
# ============================================================

def main():
    # --- Datos ---
    df, tickers = descargar_precios(TICKERS, AÑOS_HISTORIA)
    ret_log     = calcular_retornos(df)
    n           = len(tickers)

    retornos_medios = ret_log.mean().values
    cov_matrix      = ret_log.cov().values

    # --- Portafolios aleatorios ---
    print("⚙️  Simulando portafolios aleatorios...")
    port_rets, port_vols, port_sharpes = [], [], []

    for _ in range(N_PORTFOLIOS):
        w = np.random.dirichlet(np.ones(n))
        if PESO_MINIMO > 0:
            w = np.clip(w, PESO_MINIMO, 1)
            w /= w.sum()
        r, v = estadisticas_portafolio(w, retornos_medios, cov_matrix)
        port_rets.append(r)
        port_vols.append(v)
        port_sharpes.append((r - TASA_LIBRE_RIESGO) / v)

    port_rets    = np.array(port_rets)
    port_vols    = np.array(port_vols)
    port_sharpes = np.array(port_sharpes)

    # --- Frontera Eficiente ---
    print("📐 Calculando frontera eficiente...")
    fe_vols, fe_rets = frontera_eficiente(
        retornos_medios, cov_matrix, TASA_LIBRE_RIESGO, n, PESO_MINIMO)

    # --- Portafolio Sharpe Óptimo ---
    print("⭐ Optimizando Sharpe ratio...")
    res_sharpe = optimizar(sharpe, n, retornos_medios, cov_matrix,
                           TASA_LIBRE_RIESGO, PESO_MINIMO)
    ret_s, vol_s = estadisticas_portafolio(res_sharpe.x, retornos_medios, cov_matrix)
    sr_s         = (ret_s - TASA_LIBRE_RIESGO) / vol_s

    # --- Portafolio Mínima Volatilidad ---
    print("🔴 Optimizando mínima volatilidad...")
    res_minvol = optimizar(volatilidad, n, retornos_medios, cov_matrix,
                           TASA_LIBRE_RIESGO, PESO_MINIMO)
    ret_mv, vol_mv = estadisticas_portafolio(res_minvol.x, retornos_medios, cov_matrix)

    # --- Portafolio Retorno Objetivo ---
    print(f"🎯 Optimizando retorno objetivo ({RETORNO_OBJETIVO*100:.1f}%)...")
    res_obj = optimizar(volatilidad, n, retornos_medios, cov_matrix,
                        TASA_LIBRE_RIESGO, PESO_MINIMO,
                        retorno_obj=RETORNO_OBJETIVO)
    if res_obj.success:
        ret_o, vol_o = estadisticas_portafolio(res_obj.x, retornos_medios, cov_matrix)
    else:
        ret_o, vol_o = None, None
        print("   ⚠️  No se pudo alcanzar el retorno objetivo con los activos seleccionados.")

    # --- Capital Market Line (CML) ---
    vol_cml = np.linspace(0, max(port_vols) * 1.1, 200)
    ret_cml = TASA_LIBRE_RIESGO + sr_s * vol_cml

    # ============================================================
    #   GRÁFICO
    # ============================================================
    print("\n📊 Generando gráfico...\n")
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Portafolios aleatorios (coloreados por Sharpe)
    sc = ax.scatter(port_vols, port_rets,
                    c=port_sharpes, cmap="viridis",
                    alpha=0.4, s=8, zorder=1,
                    label="Portafolios Aleatorios")
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Sharpe Ratio", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Frontera Eficiente
    if len(fe_vols) > 1:
        ax.plot(fe_vols, fe_rets,
                color="#3a9eff", linewidth=2.5, zorder=3,
                label="Frontera Eficiente")

    # CML
    ax.plot(vol_cml, ret_cml,
            color="#ff4d4d", linewidth=1.8, linestyle="--", zorder=3,
            label="CML")

    # Punto de tangencia (Rf en eje Y)
    ax.axhline(y=TASA_LIBRE_RIESGO, color="#ff4d4d", linewidth=0.6,
               linestyle=":", alpha=0.5)
    ax.annotate(f"Rf = {TASA_LIBRE_RIESGO*100:.2f}%",
                xy=(0, TASA_LIBRE_RIESGO),
                color="#ff8080", fontsize=8, va="bottom")

    # Sharpe Óptimo
    ax.scatter(vol_s, ret_s, marker="*", color="gold", s=300, zorder=5,
               label=f"Sharpe Óptimo  (SR={sr_s:.2f})")

    # Mínima Volatilidad
    ax.scatter(vol_mv, ret_mv, marker="o", color="#ff4d4d", s=120, zorder=5,
               label="Mín. Volatilidad")

    # Retorno Objetivo
    if ret_o is not None:
        ax.scatter(vol_o, ret_o, marker="X", color="#00ffaa", s=200, zorder=5,
                   label=f"Objetivo {RETORNO_OBJETIVO*100:.1f}%")

    # Estética
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(colors="white", labelsize=10)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.set_xlabel("Volatilidad Anual", fontsize=13)
    ax.set_ylabel("Retorno Anual",     fontsize=13)
    ax.set_title(f"Espacio de Portfolios — Frontera de Markowitz\n"
                 f"Activos: {', '.join(tickers)}  |  Período: {AÑOS_HISTORIA} año(s)  |  "
                 f"Rf: {TASA_LIBRE_RIESGO*100:.2f}%",
                 color="white", fontsize=13, pad=12)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(color="#222", linewidth=0.5)

    legend = ax.legend(fontsize=9, facecolor="#1a1a2e",
                       edgecolor="#555", labelcolor="white",
                       loc="upper left")
    plt.tight_layout()
    plt.show()

    # ============================================================
    #   TABLA DE RESULTADOS
    # ============================================================
    print("=" * 60)
    print("  RESULTADOS DE OPTIMIZACIÓN")
    print("=" * 60)

    def tabla_portafolio(nombre, pesos, ret, vol, tickers):
        print(f"\n{'─'*60}")
        print(f"  {nombre}")
        print(f"{'─'*60}")
        print(f"  Retorno anual  : {ret*100:.2f}%")
        print(f"  Volatilidad    : {vol*100:.2f}%")
        sr = (ret - TASA_LIBRE_RIESGO) / vol
        print(f"  Sharpe ratio   : {sr:.3f}")
        print(f"  Asignación de pesos:")
        for t, w in zip(tickers, pesos):
            bar = "█" * int(w * 40)
            print(f"    {t:<8} {w*100:6.2f}%  {bar}")

    tabla_portafolio("⭐ SHARPE ÓPTIMO",
                     res_sharpe.x, ret_s, vol_s, tickers)
    tabla_portafolio("🔴 MÍNIMA VOLATILIDAD",
                     res_minvol.x, ret_mv, vol_mv, tickers)
    if ret_o is not None:
        tabla_portafolio(f"🎯 RETORNO OBJETIVO ({RETORNO_OBJETIVO*100:.1f}%)",
                         res_obj.x, ret_o, vol_o, tickers)

    print(f"\n{'─'*60}")
    print(f"  ℹ️  Tasa libre de riesgo: {TASA_LIBRE_RIESGO*100:.2f}%")
    print(f"  ℹ️  Período analizado   : {AÑOS_HISTORIA} año(s)")
    print(f"  ℹ️  Portafolios simulados: {N_PORTFOLIOS:,}")
    print("=" * 60)


# ============================================================
#   EJECUTAR
# ============================================================
main()
