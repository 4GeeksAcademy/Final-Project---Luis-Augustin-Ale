import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title = 'Data Results',
    page_icon = "🏂",
    layout = 'wide',
    initial_sidebar_state = 'expanded'
    )
# CSS styling
st.markdown("""
<style>
[data-testid="block-container"] {
padding-left: 2rem;
padding-right: 2rem;
padding-top: 1rem;
padding-bottom: 0rem;
margin-bottom: -7rem;
}
[data-testid="stVerticalBlock"] {
padding-left: 0rem;
padding-right: 0rem;
}
[data-testid="stMetric"] {
background-color: #393939;
text-align: center;
padding: 15px 0;
}
[data-testid="stMetricLabel"] {
display: flex;
justify-content: center;
align-items: center;
}
</style>
""", unsafe_allow_html=True)
# data
df = pd.read_csv('sample_para_dashboard_prueba.csv')
'''df['Ventas'] = df['Ventas'].apply(lambda x: float(x.replace('.', '').replace(',', '.')))
df['Costo_envio'] = df['Costo_envio'].apply(lambda x: float(x.replace('.', '').replace(',', '.')))
df['Descuento'] = df['Descuento'].apply(lambda x: float(x.replace('.', '').replace(',', '.')))
df['Margen'] = df['Margen'].apply(lambda x: float(x.replace('.', '').replace(',', '.')))
df['Utilidad'] = df['Utilidad'].apply(lambda x: float(x.replace('.', '').replace(',', '.')))
df['Precio_costo'] = df['Precio_costo'].apply(lambda x: float(x.replace('.', '').replace(',', '.')))
df['fecha_envio'] = pd.to_datetime(df['fecha_envio'], format='%d/%m/%Y')
df = df.sort_values('fecha_envio')'''
# Sidebar
with st.sidebar:
  st.title('🏂 Delivery Dashboard')
  years_unique = df['fecha_envio'].dt.year.unique()
  year_list = list(map(str, years_unique))
  selected_year = 2020
  selected_year = st.selectbox('Select a year', year_list)
  df_selected_year = df[df['fecha_envio'].dt.year == int(selected_year)]
# visualizaciones
#1
def make_time_line(data):
  fig = px.line(data, x='fecha_envio', y='Ventas', color ='Prioridad')
  fig.update_layout(
  title_text="", title_x=0.5,
  annotations=[dict(text="Gráfico de Línea para mostrar la cantidad de ventas por prioridad ", showarrow=False,
  x=0.5, y=-0.15, xref='paper', yref='paper')])
  print(selected_year)
  return fig
#2
def make_tree_map(input_df):
  fig = px.treemap(input_df, path=['Segmento', 'Region'], values ='Cantidad',
  color = 'Ventas', color_continuous_scale='RdBu',
  title='Relación entre Segmento y Región')
  fig.update_layout(
  title_text='', title_x=0.5,
  annotations=[dict(text='Esta gráfica de Tree map muestra la relacion en proporciones de los segmentos de mercado con las cantidades de ventas por Region', showarrow=False,
  x=0.5, y=-0.15, xref='paper', yref='paper')])
  return fig
#3
def make_sun(input_df):
  fig = px.sunburst(input_df, path=['Prioridad', 'Categoria'], title='',
  labels={'Categoria': 'Categoría'})
  fig.update_layout(
  title_text='', title_x=0.5,
  annotations=[dict(text='Gráfico de Tarta para la representación de la prioridad de envíos con respecto a su categoría', showarrow=False,
  x=0.5, y=-0.15, xref='paper', yref='paper')])
  return fig
#4
def make_pie(input_df):
  fig = px.pie(input_df, values='Cantidad', names='Region',
  title='',
  hover_data=['Costo_envio'], labels={'Costo_envio':'Costo_envio'})
  fig.update_traces(textposition='inside', textinfo='percent+label')
  fig.update_layout(
  title_text='', title_x=0.5,
  annotations=[dict(text='Gráfico de Tarta que representa las proporciones de envios y costos por envíos por Región', showarrow=False,
  x=0.5, y=-0.15, xref='paper', yref='paper')])
  return fig
# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')
with col[0]:
  st.metric(label='Records', value=df_selected_year.shape[0], delta=None)
with st.expander('', expanded=True):
  st.write('''
  - :orange[**About**]: Este conjunto de datos proporciona una visión detallada de las transacciones de ventas y comercio electrónico. Cada entrada en el dataset representa una transacción única, con información valiosa sobre productos, clientes y regiones.
  ''')
with col[1]:
  st.markdown('#### Cantidad de Ventas por Región')
  treemap = make_tree_map(df_selected_year)
  st.plotly_chart(treemap, use_container_width=True)
  st.markdown('#### Timeline de Ventas')
  time_line = make_time_line(df_selected_year)
  st.plotly_chart(time_line, use_container_width=True)
with col[2]:
  st.markdown('#### Porcentaje de Envios y Costos por Region')
  pie = make_pie(df_selected_year)
  st.plotly_chart(pie, use_container_width=True)
  st.markdown('#### Prioridad de envíos por Categoria')
  sun = make_sun(df_selected_year)
  st.plotly_chart(sun, use_container_width=True)

