from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

PAYMENT_LABELS = {
    1: "Cartao de Credito",
    2: "Dinheiro",
    3: "Sem Cobranca",
    4: "Disputa",
    5: "Desconhecido",
    6: "Cancelada",
}

DAY_LABELS = {
    "Monday": "Segunda-feira",
    "Tuesday": "Terca-feira",
    "Wednesday": "Quarta-feira",
    "Thursday": "Quinta-feira",
    "Friday": "Sexta-feira",
    "Saturday": "Sabado",
    "Sunday": "Domingo",
}

DAY_ORDER = [
    "Segunda-feira",
    "Terca-feira",
    "Quarta-feira",
    "Quinta-feira",
    "Sexta-feira",
    "Sabado",
    "Domingo",
]

PARQUET_PATH = Path(__file__).resolve().parent / "yellow_taxi_2016-03.parquet"


@st.cache_data(show_spinner="Lendo dados do Parquet...")
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["payment_label"] = df["payment_type"].map(PAYMENT_LABELS).fillna("Outro")
    df["day_of_week"] = df["day_of_week"].replace(DAY_LABELS)
    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")

    hour_range = st.sidebar.slider("Faixa de hora", 0, 23, (0, 23))

    vendor_options = sorted(df["VendorID"].unique().tolist())
    selected_vendors = st.sidebar.multiselect(
        "Fornecedores",
        options=vendor_options,
        default=vendor_options,
    )

    payment_options = sorted(df["payment_label"].unique().tolist())
    selected_payments = st.sidebar.multiselect(
        "Tipo de pagamento",
        options=payment_options,
        default=payment_options,
    )

    return df[
        df["hour_of_day"].between(hour_range[0], hour_range[1])
        & df["VendorID"].isin(selected_vendors)
        & df["payment_label"].isin(selected_payments)
    ]


def format_currency(value: float) -> str:
    value_str = f"{value:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")
    return f"US$ {value_str}"


def format_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


# ── Aba: Mapa de Calor ────────────────────────────────────────────────────────

def render_tab_map(filtered_df: pd.DataFrame) -> None:
    st.markdown("### 📍 Densidade de Embarques (Pickups)")
    st.markdown(
        "Cada coluna hexagonal representa a concentração de corridas naquela área. "
        "Quanto mais alta e clara, mais corridas partiram dali. "
        "Amostra de **50.000 viagens** para fluidez no navegador."
    )

    map_df = filtered_df.sample(n=min(50_000, len(filtered_df)), random_state=42)

    # Exibe métricas rápidas acima do mapa
    c1, c2, c3 = st.columns(3)
    c1.metric("Viagens na amostra", format_int(len(map_df)))
    c2.metric("Latitude central (aprox.)", "40.758°N")
    c3.metric("Longitude central (aprox.)", "73.985°O")

    layer = pdk.Layer(
        "HexagonLayer",
        data=map_df,
        get_position=["pickup_longitude", "pickup_latitude"],
        radius=150,               # raio de cada hexágono em metros
        elevation_scale=4,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,            # efeito 3D
        coverage=0.9,
        auto_highlight=True,
        color_range=[             # escala de cores: frio → quente
            [1,  152, 189],
            [73, 227, 206],
            [216, 254, 181],
            [254, 237, 177],
            [254, 173, 84],
            [209, 55,  78],
        ],
    )

    view_state = pdk.ViewState(
        longitude=-73.985,
        latitude=40.758,
        zoom=11,
        min_zoom=9,
        max_zoom=15,
        pitch=45,
        bearing=-15,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Corridas nesta área: {elevationValue}"},
            map_style="mapbox://styles/mapbox/dark-v10",
        ),
        use_container_width=True,
    )

    st.caption(
        "💡 Use o mouse para rotacionar (arrastar com botão direito) e o scroll para zoom. "
        "Passe o cursor sobre uma coluna para ver a contagem de corridas."
    )


# ── Aba: Predição de Tarifa (ML) ──────────────────────────────────────────────

def render_tab_ml(filtered_df: pd.DataFrame) -> None:
    st.markdown("### 🤖 Predição de Tarifa — Regressão Linear")

    if "predicted_fare" not in filtered_df.columns:
        st.info(
            "💡 As colunas de Machine Learning (`predicted_fare`, `fare_difference`) "
            "ainda não estão neste Parquet.\n\n"
            "**Como gerar:** adicione a seção ML ao `testando.py` e rode-o novamente "
            "para regravar o arquivo `.parquet` com as novas colunas."
        )
        return

    # ── Métricas do modelo ─────────────────────────────────────────────────────
    sample = filtered_df.sample(n=min(200_000, len(filtered_df)), random_state=42)

    from sklearn.metrics import mean_absolute_error, r2_score  # import local p/ não poluir globais

    mae = mean_absolute_error(sample["fare_amount"], sample["predicted_fare"])
    r2  = r2_score(sample["fare_amount"], sample["predicted_fare"])
    mean_diff = sample["fare_difference"].mean()

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "MAE — Erro Médio Absoluto",
        f"US$ {mae:.2f}",
        help="Quanto o modelo erra em média por corrida (menor = melhor).",
    )
    m2.metric(
        "R² — Coef. Determinação",
        f"{r2*100:.1f}%",
        help="Percentual da variação de tarifa explicado pelo modelo (maior = melhor).",
    )
    m3.metric(
        "Diferença Média (Real − Previsto)",
        f"US$ {mean_diff:+.2f}",
        help="Positivo: modelo subestima. Negativo: modelo superestima.",
    )

    st.markdown("---")

    col_scatter, col_outliers = st.columns([3, 2], gap="large")

    # ── Scatter: Real vs Previsto ──────────────────────────────────────────────
    with col_scatter:
        st.markdown("#### Tarifa Real vs Prevista")
        st.caption(
            "Cada ponto é uma corrida. A linha diagonal representa o modelo perfeito "
            "(previsto = real). Pontos acima da linha foram cobrados a mais que o esperado."
        )

        scatter_df = sample.sample(n=min(5_000, len(sample)), random_state=7)

        fig_scatter = px.scatter(
            scatter_df,
            x="fare_amount",
            y="predicted_fare",
            color="fare_difference",
            color_continuous_scale="RdYlGn_r",  # vermelho = cobrado a mais
            range_color=[-20, 20],
            labels={
                "fare_amount": "Tarifa Real (US$)",
                "predicted_fare": "Tarifa Prevista (US$)",
                "fare_difference": "Diferença (US$)",
            },
            opacity=0.5,
            height=420,
        )

        # linha de referência diagonal (modelo perfeito)
        max_val = float(scatter_df[["fare_amount", "predicted_fare"]].max().max())
        fig_scatter.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="white", width=1.5, dash="dash"),
                name="Modelo perfeito",
                showlegend=True,
            )
        )

        fig_scatter.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Tabela: corridas fora do padrão ───────────────────────────────────────
    with col_outliers:
        st.markdown("#### Top 100 — Cobrado Bem Acima do Previsto")
        st.caption(
            "Corridas onde `fare_amount − predicted_fare` é maior. "
            "Podem indicar tarifas especiais (aeroporto, feriados), "
            "trânsito intenso ou possíveis irregularidades."
        )

        outlier_cols = [
            "pickup_datetime",
            "trip_distance",
            "trip_duration_min",
            "fare_amount",
            "predicted_fare",
            "fare_difference",
        ]

        # filtra apenas colunas que existem (pickup_datetime pode não estar)
        existing_cols = [c for c in outlier_cols if c in filtered_df.columns]

        outliers = (
            filtered_df[existing_cols]
            .sort_values("fare_difference", ascending=False)
            .head(100)
            .round(2)
            .reset_index(drop=True)
        )

        st.dataframe(outliers, use_container_width=True, height=380)

    # ── Histograma das diferenças ──────────────────────────────────────────────
    st.markdown("#### Distribuição das Diferenças (Real − Previsto)")
    st.caption(
        "Um histograma centrado em 0 significa que o modelo não tem viés sistemático. "
        "Calda à direita indica corridas cobradas muito acima do esperado."
    )

    fig_hist = px.histogram(
        sample,
        x="fare_difference",
        nbins=100,
        range_x=[-30, 50],
        labels={"fare_difference": "Diferença: Real − Previsto (US$)"},
        color_discrete_sequence=["#F6C90E"],
        height=300,
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Zero")
    fig_hist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
        bargap=0.05,
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="NYC Yellow Taxi - Painel ETL",
        page_icon="🚕",
        layout="wide",
    )

    st.title("🚕 Dashboard ETL em Lote - NYC Yellow Taxi")
    st.markdown(
        """
        Projeto de Engenharia de Dados com fluxo ETL em lote (batch):
        **Extração → Validação → Tratamento → Transformação → Carga (Parquet) → Análise**.
        """
    )

    df = load_data(PARQUET_PATH)
    filtered_df = filter_data(df)

    if filtered_df.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
        st.stop()

    st.subheader("Resumo Geral")
    col1, col2, col3, col4, col5 = st.columns(5)

    total_trips   = len(filtered_df)
    total_revenue = float(filtered_df["total_amount"].sum())
    avg_fare      = float(filtered_df["fare_amount"].mean())
    avg_distance  = float(filtered_df["trip_distance"].mean())
    avg_tip_pct   = float(filtered_df["tip_pct"].mean())

    col1.metric("Viagens",              format_int(total_trips))
    col2.metric("Receita total",        format_currency(total_revenue))
    col3.metric("Tarifa média",         format_currency(avg_fare))
    col4.metric("Distância média (mi)", f"{avg_distance:.2f}")
    col5.metric("Gorjeta média (%)",    f"{avg_tip_pct:.2f}")

    tab_hour, tab_vendor, tab_payment, tab_weekday, tab_map, tab_ml = st.tabs(
        ["⏰ Por Hora", "🏢 Por Fornecedor", "💳 Por Pagamento", "📅 Por Dia da Semana",
         "📍 Mapa de Calor", "🤖 Predição de Tarifa"]
    )

    # ── Por Hora ──────────────────────────────────────────────────────────────
    with tab_hour:
        hourly = (
            filtered_df.groupby("hour_of_day", as_index=True)
            .agg(
                total_trips=("hour_of_day", "size"),
                avg_distance=("trip_distance", "mean"),
                avg_fare=("fare_amount", "mean"),
                avg_duration=("trip_duration_min", "mean"),
                avg_speed=("trip_speed_mph", "mean"),
                avg_tip_pct=("tip_pct", "mean"),
            )
            .sort_index()
            .round(2)
            .rename(columns={
                "total_trips":   "Total de viagens",
                "avg_distance":  "Distancia media (mi)",
                "avg_fare":      "Tarifa media (US$)",
                "avg_duration":  "Duracao media (min)",
                "avg_speed":     "Velocidade media (mph)",
                "avg_tip_pct":   "Gorjeta media (%)",
            })
        )
        chart_col1, chart_col2 = st.columns(2)
        chart_col1.line_chart(hourly["Total de viagens"],   use_container_width=True)
        chart_col2.bar_chart(hourly["Tarifa media (US$)"],  use_container_width=True)
        st.dataframe(hourly, use_container_width=True)

    # ── Por Fornecedor ────────────────────────────────────────────────────────
    with tab_vendor:
        vendor = (
            filtered_df.groupby("VendorID", as_index=True)
            .agg(
                total_trips=("VendorID", "size"),
                avg_distance=("trip_distance", "mean"),
                avg_fare=("fare_amount", "mean"),
                total_revenue=("total_amount", "sum"),
                avg_tip_pct=("tip_pct", "mean"),
                avg_speed=("trip_speed_mph", "mean"),
            )
            .round(2)
            .rename(columns={
                "total_trips":    "Total de viagens",
                "avg_distance":   "Distancia media (mi)",
                "avg_fare":       "Tarifa media (US$)",
                "total_revenue":  "Receita total (US$)",
                "avg_tip_pct":    "Gorjeta media (%)",
                "avg_speed":      "Velocidade media (mph)",
            })
        )
        st.bar_chart(vendor["Total de viagens"], use_container_width=True)
        st.dataframe(vendor, use_container_width=True)

    # ── Por Pagamento ─────────────────────────────────────────────────────────
    with tab_payment:
        payment = (
            filtered_df.groupby("payment_label", as_index=True)
            .agg(
                total_trips=("payment_label", "size"),
                avg_fare=("fare_amount", "mean"),
                avg_tip=("tip_amount", "mean"),
                avg_tip_pct=("tip_pct", "mean"),
                total_revenue=("total_amount", "sum"),
            )
            .round(2)
            .rename(columns={
                "total_trips":    "Total de viagens",
                "avg_fare":       "Tarifa media (US$)",
                "avg_tip":        "Gorjeta media (US$)",
                "avg_tip_pct":    "Gorjeta media (%)",
                "total_revenue":  "Receita total (US$)",
            })
        )
        payment["% de viagens"] = (
            (payment["Total de viagens"] / payment["Total de viagens"].sum()) * 100
        ).round(2)
        st.bar_chart(payment["Total de viagens"], use_container_width=True)
        st.dataframe(payment, use_container_width=True)

    # ── Por Dia da Semana ─────────────────────────────────────────────────────
    with tab_weekday:
        weekday = (
            filtered_df.groupby("day_of_week", as_index=True)
            .agg(
                total_trips=("day_of_week", "size"),
                avg_distance=("trip_distance", "mean"),
                avg_fare=("fare_amount", "mean"),
                avg_duration=("trip_duration_min", "mean"),
                avg_tip_pct=("tip_pct", "mean"),
            )
            .round(2)
            .rename(columns={
                "total_trips":   "Total de viagens",
                "avg_distance":  "Distancia media (mi)",
                "avg_fare":      "Tarifa media (US$)",
                "avg_duration":  "Duracao media (min)",
                "avg_tip_pct":   "Gorjeta media (%)",
            })
        )
        weekday = weekday.reindex(DAY_ORDER)
        st.bar_chart(weekday["Total de viagens"], use_container_width=True)
        st.dataframe(weekday, use_container_width=True)

    # ── Mapa de Calor ─────────────────────────────────────────────────────────
    with tab_map:
        render_tab_map(filtered_df)

    # ── Predição de Tarifa ────────────────────────────────────────────────────
    with tab_ml:
        render_tab_ml(filtered_df)


if __name__ == "__main__":
    main()