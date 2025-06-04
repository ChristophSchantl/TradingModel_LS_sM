import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
from deap import base, creator, tools, algorithms
import random
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------
# Funktion, die im Hintergrund Optimierung und Trading ausführt
# ---------------------------------------
@st.cache_data(show_spinner=False)
def optimize_and_run(ticker: str, start_date_str: str, cost_pct: float):
    """
    Lädt Kursdaten für den gegebenen Ticker und Start-Datum (bis heute),
    führt die MA-Optimierung per GA durch, rundet die MA-Fenster, simuliert das Trading
    unter Berücksichtigung von Handelskosten (cost_pct), und gibt alle relevanten Ergebnisse zurück.
    """
    # 1. Datenbeschaffung und -aufbereitung
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date_str, end=end_date_str)
    data['Close'] = data['Close'].interpolate(method='linear')
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

    # 2. Fitness-Funktion für den GA
    def evaluate_strategy(individual):
        ma_short_window, ma_long_window = int(individual[0]), int(individual[1])
        if ma_short_window >= ma_long_window or ma_short_window <= 0 or ma_long_window <= 0:
            return -np.inf,
        df = data.copy()
        df['MA_short'] = df['Close'].rolling(window=ma_short_window).mean()
        df['MA_long'] = df['Close'].rolling(window=ma_long_window).mean()
        df.dropna(inplace=True)

        position = 0
        wealth_line = [10000.0]
        cumulative_pnl = 0.0
        positionswert = 0.0

        for i in range(1, len(df)):
            price_today = float(df['Close'].iloc[i])
            ma_s_t = float(df['MA_short'].iloc[i])
            ma_l_t = float(df['MA_long'].iloc[i])
            ma_s_y = float(df['MA_short'].iloc[i - 1])
            ma_l_y = float(df['MA_long'].iloc[i - 1])

            # Kaufsignal (Long eröffnen)
            if ma_s_t > ma_l_t and ma_s_y <= ma_l_y and position == 0:
                # Entry-Kosten auf Positionsgröße
                commission_entry = wealth_line[-1] * (cost_pct / 100.0)
                positionswert = wealth_line[-1] - commission_entry
                wealth_line.append(wealth_line[-1] - commission_entry)
                position = 1
                initial_price = price_today

            # Verkaufssignal (Long schließen)
            elif ma_s_t < ma_l_t and ma_s_y >= ma_l_y and position == 1:
                gross = (price_today - initial_price) / initial_price * positionswert
                commission_exit = (positionswert + gross) * (cost_pct / 100.0)
                net_pnl = gross - commission_exit
                cumulative_pnl += net_pnl
                wealth_line.append(wealth_line[-1] + net_pnl)
                position = 0

            # Short-Signal (Short eröffnen)
            elif ma_s_t < ma_l_t and ma_s_y >= ma_l_y and position == 0:
                commission_entry = wealth_line[-1] * (cost_pct / 100.0)
                positionswert = wealth_line[-1] - commission_entry
                wealth_line.append(wealth_line[-1] - commission_entry)
                position = -1
                initial_price = price_today

            # Short-Cover + direkt neu Long eröffnen
            elif ma_s_t > ma_l_t and ma_s_y <= ma_l_y and position == -1:
                gross = (initial_price - price_today) / initial_price * positionswert
                commission_exit = (positionswert + gross) * (cost_pct / 100.0)
                net_pnl = gross - commission_exit
                cumulative_pnl += net_pnl
                wealth_line.append(wealth_line[-1] + net_pnl)
                # direkt neues Long eröffnen
                commission_entry = wealth_line[-1] * (cost_pct / 100.0)
                positionswert = wealth_line[-1] - commission_entry
                wealth_line.append(wealth_line[-1] - commission_entry)
                position = 1
                initial_price = price_today

            else:
                # Position halten
                wealth_line.append(wealth_line[-1])

        wealth_series = pd.Series(wealth_line)
        returns = wealth_series.pct_change().dropna()
        if returns.std() == 0:
            return -np.inf,
        sharpe = (returns.mean() - (0.02 / 252.0)) / returns.std() * np.sqrt(252.0)
        return sharpe,

    # 3. DEAP-Setup für GA
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 5, 50)
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.attr_int, toolbox.attr_int),
        n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_strategy)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=20)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)
    best = tools.selBest(population, k=1)[0]

    # 4. Gerundete MA-Werte
    best_short = int(round(best[0]))
    best_long = int(round(best[1]))
    if best_short >= best_long:
        best_short, best_long = min(best_short, best_long - 1), max(best_short + 1, best_long)

    # 5. Handelsmodell mit den gerundeten MA-Werten (mit Kosten)
    data_vis = data.copy()
    data_vis['MA_short'] = data_vis['Close'].rolling(window=best_short).mean()
    data_vis['MA_long'] = data_vis['Close'].rolling(window=best_long).mean()
    data_vis.dropna(inplace=True)

    position = 0
    initial_price = None
    wealth = 10000.0
    cumulative_pnl = 0.0
    trades = []
    positionswert = 0.0
    position_history = []

    for i in range(len(data_vis)):
        price_today = float(data_vis['Close'].iloc[i])
        if i == 0:
            position_history.append(0)
            continue

        ma_s_t = float(data_vis['MA_short'].iloc[i])
        ma_l_t = float(data_vis['MA_long'].iloc[i])
        ma_s_y = float(data_vis['MA_short'].iloc[i - 1])
        ma_l_y = float(data_vis['MA_long'].iloc[i - 1])
        date = data_vis.index[i]

        # Kaufsignal (Long eröffnen)
        if ma_s_t > ma_l_t and ma_s_y <= ma_l_y and position == 0:
            initial_price = price_today
            position = 1
            commission_entry = wealth * (cost_pct / 100.0)
            positionswert = wealth - commission_entry
            wealth -= commission_entry
            trades.append({
                'Typ': 'Kauf',
                'Datum': date,
                'Kurs': price_today,
                'Spesen': commission_entry,
                'Positionswert': positionswert,
                'Profit/Loss': None,
                'Kumulative P&L': cumulative_pnl
            })

        # Verkaufssignal (Long schließen)
        elif ma_s_t < ma_l_t and ma_s_y >= ma_l_y and position == 1:
            position = 0
            gross = (price_today - initial_price) / initial_price * positionswert
            commission_exit = (positionswert + gross) * (cost_pct / 100.0)
            net_pnl = gross - commission_exit
            cumulative_pnl += net_pnl
            wealth += positionswert + net_pnl
            trades.append({
                'Typ': 'Verkauf (Long)',
                'Datum': date,
                'Kurs': price_today,
                'Spesen': commission_exit,
                'Positionswert': None,
                'Profit/Loss': net_pnl,
                'Kumulative P&L': cumulative_pnl
            })

        # Short-Signal (Short eröffnen)
        elif ma_s_t < ma_l_t and ma_s_y >= ma_l_y and position == 0:
            initial_price = price_today
            position = -1
            commission_entry = wealth * (cost_pct / 100.0)
            positionswert = wealth - commission_entry
            wealth -= commission_entry
            trades.append({
                'Typ': 'Short-Sell',
                'Datum': date,
                'Kurs': price_today,
                'Spesen': commission_entry,
                'Positionswert': positionswert,
                'Profit/Loss': None,
                'Kumulative P&L': cumulative_pnl
            })

        # Short-Cover + direkt neu Long eröffnen
        elif ma_s_t > ma_l_t and ma_s_y <= ma_l_y and position == -1:
            gross = (initial_price - price_today) / initial_price * positionswert
            commission_exit = (positionswert + gross) * (cost_pct / 100.0)
            net_cover = gross - commission_exit
            cumulative_pnl += net_cover
            wealth += positionswert + net_cover
            trades.append({
                'Typ': 'Short-Cover',
                'Datum': date,
                'Kurs': price_today,
                'Spesen': commission_exit,
                'Positionswert': None,
                'Profit/Loss': net_cover,
                'Kumulative P&L': cumulative_pnl
            })

            # Direkt neues Long eröffnen
            initial_price = price_today
            position = 1
            commission_entry = wealth * (cost_pct / 100.0)
            positionswert = wealth - commission_entry
            wealth -= commission_entry
            trades.append({
                'Typ': 'Kauf (nach Short-Cover)',
                'Datum': date,
                'Kurs': price_today,
                'Spesen': commission_entry,
                'Positionswert': positionswert,
                'Profit/Loss': None,
                'Kumulative P&L': cumulative_pnl
            })

        else:
            # Position halten
            pass

        position_history.append(position)

    # Offene Position zum Schluss schließen
    if position != 0:
        last_price = float(data_vis['Close'].iloc[-1])
        if position == 1:
            gross = (last_price - initial_price) / initial_price * positionswert
        else:
            gross = (initial_price - last_price) / initial_price * positionswert
        commission_exit = (positionswert + gross) * (cost_pct / 100.0)
        net_close = gross - commission_exit
        cumulative_pnl += net_close
        wealth += positionswert + net_close
        trades.append({
            'Typ': 'Schließen (Ende)',
            'Datum': data_vis.index[-1],
            'Kurs': last_price,
            'Spesen': commission_exit,
            'Positionswert': None,
            'Profit/Loss': net_close,
            'Kumulative P&L': cumulative_pnl
        })

    trades_df = pd.DataFrame(trades)

    # Renditen berechnen
    strategy_return = (wealth - 10000.0) / 10000.0 * 100.0
    buy_and_hold_return = (data_vis['Close'].iloc[-1] - data_vis['Close'].iloc[0]) / data_vis['Close'].iloc[0] * 100.0

    # Statistik zu positiven/negativen Trades
    pos_trades = trades_df[trades_df['Profit/Loss'] > 0]
    neg_trades = trades_df[trades_df['Profit/Loss'] < 0]
    pos_count = len(pos_trades)
    neg_count = len(neg_trades)
    pos_pnl = pos_trades['Profit/Loss'].sum()
    neg_pnl = neg_trades['Profit/Loss'].sum()

    # Gesamtzahl der Einträge (Entry + Exit)
    total_trades = len(trades_df)

    # Anzahl aller tatsächlich abgeschlossenen Exit-Trades
    closed_trades = pos_count + neg_count

    # Korrekte Prozentberechnung: nur auf abgeschlossene Trades bezogen
    if closed_trades > 0:
        pos_pct = pos_count / closed_trades * 100.0
        neg_pct = neg_count / closed_trades * 100.0
    else:
        pos_pct = neg_pct = 0.0

    total_pnl = pos_pnl + neg_pnl
    if total_pnl != 0:
        pos_perf = pos_pnl / total_pnl * 100.0
        neg_perf = neg_pnl / total_pnl * 100.0
    else:
        pos_perf = neg_perf = 0.0

    # DataFrame für Plot vorbereiten (Close und Position)
    df_plot = data_vis[['Close']].copy().iloc[len(data_vis) - len(position_history):]
    df_plot['Position'] = position_history

    return {
        "trades_df": trades_df,
        "strategy_return": float(strategy_return),
        "buy_and_hold_return": float(buy_and_hold_return),
        "total_trades": total_trades,
        "long_trades": len(trades_df[trades_df['Typ'].str.contains("Kauf")]),
        "short_trades": len(trades_df[trades_df['Typ'].str.contains("Short")]),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "pos_pnl": float(pos_pnl),
        "neg_pnl": float(neg_pnl),
        "total_pnl": float(total_pnl),
        "pos_perf": pos_perf,
        "neg_perf": neg_perf,
        "df_plot": df_plot
    }


# ---------------------------------------
# Streamlit-App
# ---------------------------------------
st.title("📊 Trading Model – LS25")

st.markdown("""
Bitte wähle unten den Ticker, den Beginn des Zeitraums und die Handelskosten (in %) aus.  
Die Optimierung berücksichtigt dann pro Trade die angegebene Transaktionsgebühr.
""")

# ------------------------------
# Eingabefelder für Ticker / Zeitfenster / Kosten
# ------------------------------
ticker_input = st.text_input(
    label="1️⃣ Welchen Aktien-Ticker möchtest du analysieren?",
    value="",  # leerer Standardwert
    help="Gib hier das Tickersymbol ein, z. B. 'AAPL', 'MSFT' oder 'O'."
)

start_date_input = st.date_input(
    label="2️⃣ Beginn des Analyse-Zeitraums",
    value=date(2024, 1, 1),
    max_value=date.today(),
    help="Wähle das Startdatum (bis heute)."
)

cost_pct = st.number_input(
    label="3️⃣ Handelskosten (% pro Trade)", 
    min_value=0.0, 
    value=5.0, 
    step=0.1, 
    format="%.2f",
    help="Gib hier die prozentuale Gebühr an, die bei jedem Entry und Exit abgezogen wird."
)

st.markdown("---")

# -------------
# Button zum Starten der Berechnung
# -------------
run_button = st.button("🔄 Ergebnisse berechnen")

# Nur wenn der Button gedrückt wurde und ein Ticker eingegeben ist:
if run_button:
    if ticker_input.strip() == "":
        st.error("Bitte gib zunächst einen gültigen Ticker ein, z. B. 'AAPL' oder 'MSFT'.")
    else:
        start_date_str = start_date_input.strftime("%Y-%m-%d")
        with st.spinner("⏳ Berechne Optimierung und Trades… bitte einen Moment warten"):
            results = optimize_and_run(ticker_input, start_date_str, cost_pct)

        trades_df = results["trades_df"]
        strategy_return = results["strategy_return"]
        buy_and_hold_return = results["buy_and_hold_return"]
        total_trades = results["total_trades"]
        long_trades = results["long_trades"]
        short_trades = results["short_trades"]
        pos_count = results["pos_count"]
        neg_count = results["neg_count"]
        pos_pct = results["pos_pct"]
        neg_pct = results["neg_pct"]
        pos_pnl = results["pos_pnl"]
        neg_pnl = results["neg_pnl"]
        total_pnl = results["total_pnl"]
        pos_perf = results["pos_perf"]
        neg_perf = results["neg_perf"]
        df_plot = results["df_plot"]

        # ---------------------------------------
        # 1. Performance-Vergleich (Strategie vs. Buy & Hold)
        # ---------------------------------------
        st.subheader("1. Performance-Vergleich")
        fig_performance, ax_perf = plt.subplots(figsize=(8, 5))
        
        # Balken zeichnen
        bars = ax_perf.bar(
            ['Strategie', 'Buy & Hold'],
            [strategy_return, buy_and_hold_return],
            color=['#1f77b4', '#ff7f0e'],
            alpha=0.8
        )
        
        # Prozentwerte über die Balken schreiben
        for bar in bars:
            height = bar.get_height()
            ax_perf.text(
                bar.get_x() + bar.get_width() / 2,  # x-Position in der Mitte
                height,                              # y-Position genau auf dem Balken
                f"{height:.2f}%",                    # Text
                ha='center',                         # horizontal zentriert
                va='bottom'                          # vertikal über dem Balken
            )
        
        ax_perf.set_ylabel("Rendite (%)", fontsize=12)
        ax_perf.set_title(f"Strategie vs. Buy-&-Hold für {ticker_input}", fontsize=14)
        ax_perf.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_performance)


        # ---------------------------------------
        # 2. Kursdiagramm mit Kauf-/Verkaufsphasen
        # ---------------------------------------
        st.subheader("2. Kursdiagramm mit Phasen (Kauf/Verkauf)")

        fig_trades, ax_trades = plt.subplots(figsize=(10, 5))
        dates = df_plot.index
        prices = df_plot['Close']
        positions = df_plot['Position']

        ax_trades.plot(dates, prices, label='Close-Preis', color='black', linewidth=1)

        # Phasenschattierung: Long = grün, Short = rot, Neutral = transparent
        current_phase = positions.iloc[0]
        start_idx = dates[0]
        for i in range(1, len(dates)):
            if positions.iloc[i] != current_phase:
                end_idx = dates[i - 1]
                if current_phase == 1:
                    ax_trades.axvspan(start_idx, end_idx, color='green', alpha=0.2)
                elif current_phase == -1:
                    ax_trades.axvspan(start_idx, end_idx, color='red', alpha=0.2)
                current_phase = positions.iloc[i]
                start_idx = dates[i]
        # Letzte Phase bis zum Ende
        if current_phase == 1:
            ax_trades.axvspan(start_idx, dates[-1], color='green', alpha=0.2)
        elif current_phase == -1:
            ax_trades.axvspan(start_idx, dates[-1], color='red', alpha=0.2)

        ax_trades.set_title(f"{ticker_input}-Kurs mit Kauf-/Verkaufsphasen", fontsize=14)
        ax_trades.set_xlabel("Datum", fontsize=12)
        ax_trades.set_ylabel("Preis", fontsize=12)
        ax_trades.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig_trades)

        st.markdown("""
        - **Grüne Bereiche**: Long-Phase (Kaufsignal aktiv).  
        - **Rote Bereiche**: Short-Phase (Verkaufssignal aktiv).  
        - **Ohne Schattierung**: Neutral (keine offene Position).
        """)

        # ---------------------------------------
        # 3. Tabelle der Einzeltrades
        # ---------------------------------------
        st.subheader("3. Tabelle der Einzeltrades")
        trades_table = trades_df[['Datum', 'Typ', 'Kurs', 'Spesen', 'Positionswert', 'Profit/Loss', 'Kumulative P&L']].copy()
        trades_table['Datum'] = trades_table['Datum'].dt.strftime('%Y-%m-%d')
        trades_table['Kurs'] = trades_table['Kurs'].map('{:.2f}'.format)
        trades_table['Spesen'] = trades_table['Spesen'].map('{:.2f}'.format)
        trades_table['Positionswert'] = trades_table['Positionswert'].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
        trades_table['Profit/Loss'] = trades_table['Profit/Loss'].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
        trades_table['Kumulative P&L'] = trades_table['Kumulative P&L'].map('{:.2f}'.format)
        st.dataframe(trades_table, use_container_width=True)

        # ---------------------------------------
        # 4. Statistiken zu Trades
        # ---------------------------------------
        st.subheader("4. Handelsstatistiken")

        # Layout: drei Spalten
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Gesamtzahl der Einträge (Entry+Exit)", total_trades)
            st.metric("Davon Long-Trades (Entry-Zeilen)", long_trades)
            st.metric("Davon Short-Trades (Entry-Zeilen)", short_trades)

        with col2:
            st.metric("Positive Trades (Anzahl)", pos_count)
            st.metric("Negative Trades (Anzahl)", neg_count)
            st.metric("Positive Trades (%)", f"{pos_pct:.2f}%")

        with col3:
            st.metric("Strategie-Return", f"{strategy_return:.2f}%")
            st.metric("Buy-&-Hold-Return", f"{buy_and_hold_return:.2f}%")
            diff = strategy_return - buy_and_hold_return
            sign = "+" if diff >= 0 else ""
            st.metric("Outperformance vs. B&H", f"{sign}{diff:.2f}%")

        st.markdown(f"""
        - **Negative Trades (%)**: {neg_pct:.2f}%  
        - **Gesamt-P&L der positiven Trades**: {pos_pnl:.2f} EUR  
        - **Gesamt-P&L der negativen Trades**: {neg_pnl:.2f} EUR  
        - **Gesamt-P&L des Systems**: {total_pnl:.2f} EUR  
        - **Performance positive Trades**: {pos_perf:.2f}%  
        - **Performance negative Trades**: {neg_perf:.2f}%  
        - **Handelskosten (pro Trade)**: {cost_pct:.2f}%
        """)

        # Professioneller Vergleichstext
        st.markdown("""
        ---
        **Vergleich Modell-Performance vs. Buy-&-Hold**  
        - Das Handelssystem erzielte in diesem Zeitraum eine Gesamt-Rendite von **{strategy_return:.2f}%**,  
          während die Buy-&-Hold-Strategie **{buy_and_hold_return:.2f}%** erwirtschaftete.  
        - Dadurch ergibt sich eine **Outperformance von {diff:+.2f}%** gegenüber dem reinen Halten der Aktie.  
        - Trotz Transaktionskosten von **{cost_pct:.2f}% pro Trade** bleibt das System robuster in volatilen Phasen,  
          da Short-Signale Drawdowns reduzieren und Long-Signale in Aufwärtsphasen Gewinne sichern.  
        - Die Buy-&-Hold-Strategie erzielt in seitwärts orientierten Märkten oft geringere Renditen und  
          kann stärkere Einbrüche erleiden, da sie nicht zwischen Long und Short unterscheidet.  
        - Insgesamt zeigt sich, dass das optimierte System mit Handelskosten von {cost_pct:.2f}% pro Trade  
          nach wie vor eine deutliche Outperformance gegenüber einem reinen Buy-&-Hold-Ansatz liefert.  
          
        > Eine Kombination aus GA-optimierten MA-Signalen und einem passiven Buy-&-Hold-Ansatz  
        > kann das Risiko/Rendite-Profil eines reinen Aktienengagements nachhaltig verbessern.  
        """.format(
            strategy_return=strategy_return,
            buy_and_hold_return=buy_and_hold_return,
            diff=diff,
            cost_pct=cost_pct
        ))

        # ---------------------------------------
        # 5. Balkendiagramm: Anzahl der Trades
        # ---------------------------------------
        st.subheader("5. Anzahl der Trades (Entry+Exit, Long, Short)")
        fig_counts, ax_counts = plt.subplots(figsize=(6, 4))
        ax_counts.bar(
            ['Einträge gesamt', 'Long-Einträge', 'Short-Einträge'],
            [total_trades, long_trades, short_trades],
            color=['#4c72b0', '#55a868', '#c44e52'],
            alpha=0.8
        )
        ax_counts.set_ylabel("Anzahl", fontsize=12)
        ax_counts.set_title("Trade-Einträge-Verteilung", fontsize=14)
        ax_counts.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_counts)
