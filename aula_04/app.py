import os
import pandas as pd

import plotly.express as px
import streamlit as st

from get_transfers_list import get_extrato

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(120deg, #f0f8ff, #e6f7ff);
    }
    .metric-card { border-radius: 10px; padding: 15px; color: white; text-align: center; }
    .metric-green { background-color: #27ae60; }
    .metric-red   { background-color: #c0392b; }
    .metric-blue  { background-color: #2980b9; }
    .metric-gold  { background-color: #f39c12; }
    .stDataFrame tbody tr:nth-child(odd) { background-color: #f2f2f2; }
    </style>
    """,
    unsafe_allow_html=True,
)

os.makedirs("extrato", exist_ok=True)
st.set_page_config(page_title="Dashboard de Extrato", layout="wide")

with st.sidebar:
    st.header("📂 Extrato bancário")
    uploaded = st.file_uploader("Envie o PDF", type="pdf")

if not uploaded:
    st.sidebar.info("➕ Faça upload de um PDF para iniciar")
    st.stop()

save_path = os.path.join("extrato", uploaded.name)
with open(save_path, "wb") as f:
    f.write(uploaded.getbuffer())
try:
    transacoes = get_extrato(save_path)
except Exception as e:
    st.error(f"Erro ao processar extrato: {e}")
    st.stop()
finally:
    os.remove(save_path)

df = pd.DataFrame(
    [
        {
            "Valor": t.valor,
            "Origem": t.origem.value,
            "Categoria": t.categoria.value,
        }
        for t in transacoes
    ]
)

df["Categoria"] = df["Categoria"].str.title()
df["Origem"] = df["Origem"].str.title()
df["Gasto"] = df["Valor"].apply(lambda x: -x if x < 0 else 0)
df["Recebido"] = df["Valor"].apply(lambda x: x if x > 0 else 0)

total_recebido = df["Recebido"].sum()
total_gasto = df["Gasto"].sum()
saldo_final = total_recebido - total_gasto
num_categorias = df["Categoria"].nunique()

st.title("📊 Resumo Financeiro")
col1, col2, col3, col4 = st.columns(4)
col1.markdown(
    f'<div class="metric-card metric-green"><h4>Total Recebido</h4><h2>R$ {total_recebido:,.2f}</h2></div>',
    unsafe_allow_html=True,
)
col2.markdown(
    f'<div class="metric-card metric-red"><h4>Total Gasto</h4><h2>R$ {total_gasto:,.2f}</h2></div>',
    unsafe_allow_html=True,
)
col3.markdown(
    f'<div class="metric-card metric-blue"><h4>Saldo Final</h4><h2>R$ {saldo_final:,.2f}</h2></div>',
    unsafe_allow_html=True,
)
col4.markdown(
    f'<div class="metric-card metric-gold"><h4>Categorias</h4><h2>{num_categorias}</h2></div>',
    unsafe_allow_html=True,
)

st.subheader("Distribuição de Origem e Gastos por Categoria")

pie_col1, pie_col2 = st.columns(2)

with pie_col1:
    gastos_df = df[df["Valor"] < 0]
    origin_sums = (
        gastos_df.groupby("Origem")["Valor"].apply(lambda x: x.abs().sum()).sort_values(ascending=False)
    )
    fig1 = px.pie(
        names=origin_sums.index,
        values=origin_sums.values,
        title="Origem dos Gastos",
        hole=0.0,
    )
    fig1.update_traces(
        textinfo="none",
        hovertemplate="%{label}: %{value} (%{percent})",
    )
    fig1.update_layout(
        title_x=0.5,
        margin=dict(t=40, l=0, r=0, b=0),
        legend=dict(
            orientation="v",
            x=1.02,
            y=0.5,
            title_text="Origem",
            font=dict(size=11),
        ),
    )
    st.plotly_chart(fig1, use_container_width=True)

with pie_col2:
    gastos_cat = df[df["Gasto"] > 0].groupby("Categoria")["Gasto"].sum()
    fig2 = px.pie(
        names=gastos_cat.index,
        values=gastos_cat.values,
        title="Gastos por Categoria",
        hole=0.0,
    )
    fig2.update_traces(
        textinfo="none",
        hovertemplate="%{label}: R$ %{value:,.2f} (%{percent})",
    )
    fig2.update_layout(
        title_x=0.5,
        margin=dict(t=40, l=0, r=0, b=0),
        legend=dict(
            orientation="v",
            x=1.02,
            y=0.5,
            title_text="Categoria",
            font=dict(size=11),
        ),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Transações Detalhadas")
df_display = df[["Valor", "Origem", "Categoria"]].copy()
df_display["Valor"] = df_display["Valor"].map("R$ {:+,.2f}".format)
st.dataframe(df_display, use_container_width=True, height=300)