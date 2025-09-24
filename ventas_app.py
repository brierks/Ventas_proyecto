
# ============================================
# 📊 Dashboard de Ventas con Streamlit
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# =============================
# 1. Configuración inicial
# =============================
st.set_page_config(
    page_title="Dashboard de Ventas",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard de Ventas")
st.markdown("Analiza datos de ventas con **costo, precio, utilidad y formas de pago**.")

# =============================
# 2. Cargar datos
# =============================
@st.cache_data
def cargar_datos(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    if "Utilidad" not in df.columns and "Precio_venta" in df.columns and "costo" in df.columns:
        df["Utilidad"] = (df["Precio_venta"] - df["costo"]) * df["cantidad"]
    return df

archivo = st.sidebar.file_uploader("📂 Cargar archivo CSV", type="csv")
if archivo is not None:
    df = cargar_datos(archivo)
else:
    st.stop()

# =============================
# 3. Filtros
# =============================
st.sidebar.header("Filtros")

departamentos = st.sidebar.multiselect("Departamento", df["departamento"].dropna().unique())
formas_pago = st.sidebar.multiselect("Forma de Pago", df["Forma_Pago"].dropna().unique())

df_filtered = df.copy()
if departamentos:
    df_filtered = df_filtered[df_filtered["departamento"].isin(departamentos)]
if formas_pago:
    df_filtered = df_filtered[df_filtered["Forma_Pago"].isin(formas_pago)]

# =============================
# 4. Páginas de navegación
# =============================
page = st.sidebar.radio(
    "Selecciona la sección",
    ["📊 Dashboard Principal", "🔮 Predicciones", "📈 Análisis Detallado", "🧑‍🤝‍🧑 Clientes"]
)

# =============================
# 5. Dashboard Principal
# =============================
if page == "📊 Dashboard Principal":
    st.header("📊 Dashboard Principal")

    # KPIs financieros
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_unidades = int(df_filtered["cantidad"].sum())
        st.metric("Total Unidades Vendidas", f"{total_unidades:,}")
    with col2:
        ingreso_total = (df_filtered["Precio_venta"].fillna(0) * df_filtered["cantidad"]).sum()
        st.metric("Ingreso Estimado", f"${ingreso_total:,.2f}")
    with col3:
        costo_total = (df_filtered["costo"].fillna(0) * df_filtered["cantidad"]).sum()
        st.metric("Costo Estimado", f"${costo_total:,.2f}")
    with col4:
        utilidad_total = (df_filtered["Utilidad"].fillna(0)).sum()
        st.metric("Utilidad Estimada", f"${utilidad_total:,.2f}")

    st.markdown("---")

    # === Nuevas gráficas del análisis exploratorio ===
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Top 10 Productos más Vendidos")
        top_prod = df_filtered.groupby("descripcion")["cantidad"].sum().sort_values(ascending=False).head(10)
        fig_top = px.bar(
            x=top_prod.values,
            y=top_prod.index,
            orientation="h",
            color=top_prod.values,
            labels={"x":"Cantidad","y":"Producto"},
            title="Top 10 Productos más Vendidos"
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        st.subheader("🏢 Ventas por Departamento")
        ventas_depto = df_filtered.groupby("departamento")["cantidad"].sum().sort_values(ascending=False).head(8)
        fig_depto = px.pie(
            values=ventas_depto.values,
            names=ventas_depto.index,
            title="Ventas por Departamento",
            hole=0.3
        )
        st.plotly_chart(fig_depto, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Ventas por Día de la Semana")
        orden_dias = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
        ventas_dia = df_filtered.groupby("nombre_dia_semana")["cantidad"].sum()
        ventas_dia = ventas_dia.reindex([d for d in orden_dias if d in ventas_dia.index])
        fig_dias = px.bar(
            x=ventas_dia.index,
            y=ventas_dia.values,
            labels={"x":"Día de la Semana","y":"Cantidad"},
            title="Ventas por Día de la Semana"
        )
        st.plotly_chart(fig_dias, use_container_width=True)

    with col2:
        st.subheader("📈 Tendencia de Ventas (Últimos 30 días)")
        if "fecha" in df_filtered.columns:
            ventas_diarias = df_filtered.groupby(df_filtered["fecha"].dt.date)["cantidad"].sum().tail(30)
            fig_trend = px.line(
                x=ventas_diarias.index,
                y=ventas_diarias.values,
                markers=True,
                labels={"x":"Fecha","y":"Cantidad"},
                title="Tendencia de Ventas (Últimos 30 días)"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

# =============================
# 6. Predicciones
# =============================
elif page == "🔮 Predicciones":
    st.header("🔮 Predicciones de Ventas, Ingresos y Utilidad")

    if os.path.exists("modelo_ventas.pkl"):
        modelo = joblib.load("modelo_ventas.pkl")

        descripcion = st.selectbox("Producto", df["descripcion"].unique())
        cantidad_input = st.number_input("Cantidad esperada", min_value=1, value=1)

        if st.button("Predecir"):
            pred_cantidad = modelo.predict([[cantidad_input]])[0]

            precio_unit = df[df["descripcion"] == descripcion]["Precio_venta"].mean()
            costo_unit = df[df["descripcion"] == descripcion]["costo"].mean()
            ingreso_pred = pred_cantidad * precio_unit
            utilidad_pred = (precio_unit - costo_unit) * pred_cantidad

            st.success(f"📦 Cantidad estimada: {pred_cantidad:.0f}")
            st.info(f"💵 Ingreso estimado: ${ingreso_pred:,.2f}")
            st.info(f"📈 Utilidad estimada: ${utilidad_pred:,.2f}")
    else:
        st.warning("⚠️ No se encontró un modelo entrenado (`modelo_ventas.pkl`).")

# =============================
# 7. Análisis Detallado
# =============================
elif page == "📈 Análisis Detallado":
    st.header("📈 Análisis Detallado")

    ventas_dia = df_filtered.groupby("Dia_sem")["Precio_venta"].sum().reset_index()
    fig_dia = px.line(ventas_dia, x="Dia_sem", y="Precio_venta", title="Ventas por día de la semana")
    st.plotly_chart(fig_dia, use_container_width=True)

# =============================
# 8. Clientes
# =============================
elif page == "🧑‍🤝‍🧑 Clientes":
    st.header("🧑‍🤝‍🧑 Clientes")

    cliente_utilidad = df_filtered.groupby("cliente")["Utilidad"].sum().sort_values(ascending=False).head(10)
    fig_clientes = px.bar(
        x=cliente_utilidad.index,
        y=cliente_utilidad.values,
        title="Top clientes por utilidad generada"
    )
    st.plotly_chart(fig_clientes, use_container_width=True)
