import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config (page_title = "My first dashboard", layout = "wide")

df = pd.read_csv('Transactions.csv')


st.sidebar.header ("Использование фильтров в данных. Выберите показатели")
year = st.sidebar.multiselect ( "Выберите год:", options = df["GJAHR"].unique(), default = df["GJAHR"].unique())
df_selection = df.query (" GJAHR == @year")
st.dataframe (df_selection)