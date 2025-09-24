import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="📊 Análisis de Ventas Avanzado",

    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal con estilo
st.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0;">💰 Sistema Avanzado de Análisis de Ventas</h1>
    <p style="color: white; margin: 0; opacity: 0.9;">Dashboard Completo con Análisis Financiero</p>
</div>
""", unsafe_allow_html=True)

# Función para cargar y procesar datos
@st.cache_data
def load_and_process_data(path='dataSet1.csv'):
    try:
        df = pd.read_csv(path, encoding='latin1')
        
        # Normalizar nombres de columnas
        df.columns = [c.strip() for c in df.columns]
        
        # Convertir columnas numéricas (manejar notación científica)
        numeric_columns = ['cantidad', 'costo', 'Precio venta', 'Utilidad']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Procesar fecha PRIMERO antes de cualquier otro procesamiento
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
            
            # Solo crear variables de tiempo SI las fechas se procesaron correctamente
            if not df['fecha'].isna().all():
                df['anio'] = df['fecha'].dt.year.fillna(0).astype(int)
                df['mes'] = df['fecha'].dt.month.fillna(0).astype(int)
                df['dia_mes'] = df['fecha'].dt.day.fillna(0).astype(int)
                df['hora'] = df['fecha'].dt.hour.fillna(12).astype(int)
                
                # Días de la semana en español
                dias_semana_espanol = {
                    'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
                    'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
                }
                df['nombre_dia_semana'] = df['fecha'].dt.day_name().map(dias_semana_espanol).fillna('Sin fecha')
                
                # Nombres de meses
                meses_dict = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
                              7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
                df['nombre_mes'] = df['mes'].map(meses_dict).fillna('Sin mes')
        
        # Rellenar valores nulos en categóricas
        cat_cols = ['codigo','codigoint','descripcion','unidad','departamento','lineas','marca','familia','folio','cliente','Forma Pago']
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].fillna('Sin dato').astype(str)
        
        # Limpiar y estandarizar texto
        text_cols = ['descripcion','unidad','departamento','lineas','marca','familia']
        for c in text_cols:
            if c in df.columns:
                df[c] = df[c].str.strip().str.title()
        
        # Limpiar Forma Pago
        if 'Forma Pago' in df.columns:
            df['Forma Pago'] = df['Forma Pago'].str.strip().str.upper()
            df['Forma Pago'] = df['Forma Pago'].replace(['', 'NAN', 'NONE'], 'SIN DATO')
        
        # Crear métricas calculadas
        if all(col in df.columns for col in ['Precio venta', 'costo', 'cantidad']):
            df['ingreso_total'] = df['Precio venta'] * df['cantidad']
            df['costo_total'] = df['costo'] * df['cantidad']
            df['utilidad_total'] = df['ingreso_total'] - df['costo_total']
            df['margen_utilidad'] = (df['Utilidad'] / df['Precio venta'] * 100).fillna(0)
        
        # Categorizar formas de pago
        if 'Forma Pago' in df.columns:
            def categorizar_pago(forma):
                forma_upper = str(forma).upper()
                if any(keyword in forma_upper for keyword in ['TARJETA', 'CARD', 'CREDITO', 'DEBITO']):
                    return 'TARJETA'
                elif 'EFECTIVO' in forma_upper:
                    return 'EFECTIVO'
                elif any(keyword in forma_upper for keyword in ['TRANSFER', 'DEPOSITO']):
                    return 'TRANSFERENCIA'
                else:
                    return 'OTRO'
            
            df['tipo_pago'] = df['Forma Pago'].apply(categorizar_pago)
        
        # Limpiar datos
        df = df.dropna(subset=['cantidad', 'descripcion'])
        df = df[df['cantidad'] > 0]
        df.drop_duplicates(inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

# Función para crear métricas financieras
def calcular_metricas_financieras(df):
    metricas = {}
    
    # Métricas básicas
    metricas['total_unidades'] = int(df['cantidad'].sum())
    metricas['total_transacciones'] = len(df)
    metricas['productos_unicos'] = df['descripcion'].nunique()
    metricas['clientes_unicos'] = df['cliente'].nunique()
    
    # Métricas financieras
    if 'ingreso_total' in df.columns:
        metricas['ingreso_total'] = df['ingreso_total'].sum()
        metricas['costo_total'] = df['costo_total'].sum()
        metricas['utilidad_total'] = df['utilidad_total'].sum()
        metricas['margen_promedio'] = df['margen_utilidad'].mean()
        metricas['ticket_promedio'] = df['ingreso_total'].mean()
    
    return metricas

# Función para crear modelo de predicción
@st.cache_resource
def create_prediction_model(df):
    try:
        model_df = df.copy()
        
        # Encoders
        encoders = {}
        encoded_features = []
        
        categorical_cols = ['descripcion', 'departamento', 'marca', 'cliente', 'tipo_pago']
        for col in categorical_cols:
            if col in model_df.columns:
                le = LabelEncoder()
                encoded_col = f"{col}_encoded"
                model_df[encoded_col] = le.fit_transform(model_df[col].astype(str))
                encoders[col] = le
                encoded_features.append(encoded_col)
        
        # Features numéricas
        numeric_features = []
        if 'anio' in model_df.columns:
            numeric_features.extend(['anio', 'mes', 'dia_mes', 'hora'])
        if 'costo' in model_df.columns:
            numeric_features.append('costo')
        if 'Precio venta' in model_df.columns:
            numeric_features.append('Precio venta')
        
        all_features = encoded_features + [f for f in numeric_features if f in model_df.columns]
        
        if len(all_features) > 0:
            X = model_df[all_features].fillna(0)
            y = model_df['cantidad']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = model.score(X_test, y_test)
            
            return model, encoders, all_features, mae, rmse, r2
        else:
            return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error creando modelo: {e}")
        return None, None, None, None, None, None

# Cargar datos
with st.spinner("🔄 Cargando y procesando datos..."):
    df = load_and_process_data('dataSet1.csv')

if df is not None:
    # Sidebar con filtros
    st.sidebar.title("🎛️ Panel de Control")

    # INFORMACIÓN DE FECHAS DISPONIBLES (SIN FILTRO)
    if 'fecha' in df.columns:
        fechas_validas = df['fecha'].notna().sum()
        fechas_invalidas = df['fecha'].isna().sum()
        
        if fechas_validas > 0:
            fecha_min = df['fecha'].min().date()
            fecha_max = df['fecha'].max().date()
            st.sidebar.info(f"📅 Periodo de datos:\n{fecha_min} al {fecha_max}")
            
        # Información adicional sobre fechas
        st.sidebar.markdown("**Información temporal:**")
        st.sidebar.write(f"• Registros con fecha: {fechas_validas:,}")
        st.sidebar.write(f"• Sin fecha: {fechas_invalidas:,}")
    else:
        st.sidebar.warning("⚠️ No se encontró columna 'fecha'")
        
    # Filtros categóricos (OPCIONALES)
    st.sidebar.markdown("### 🎯 Filtros Opcionales")
    
    departamentos = ['Todos'] + sorted([d for d in df['departamento'].unique() if d != 'Sin dato'])
    selected_depto = st.sidebar.selectbox("🏢 Departamento:", departamentos)
    
    formas_pago = ['Todos'] + sorted([p for p in df['Forma Pago'].unique() if p != 'SIN DATO'])
    selected_pago = st.sidebar.selectbox("💳 Forma de Pago:", formas_pago)
    
    marcas = ['Todos'] + sorted([m for m in df['marca'].unique() if m != 'Sin dato'])
    selected_marca = st.sidebar.selectbox("🏷️ Marca:", marcas)
    
    # APLICAR SOLO FILTROS CATEGÓRICOS
    df_filtered = df.copy()  # Usar todos los datos por defecto
    
    if selected_depto != 'Todos':
        df_filtered = df_filtered[df_filtered['departamento'] == selected_depto]
    
    if selected_pago != 'Todos':
        df_filtered = df_filtered[df_filtered['Forma Pago'] == selected_pago]
        
    if selected_marca != 'Todos':
        df_filtered = df_filtered[df_filtered['marca'] == selected_marca]
    
    # Navegación
    st.sidebar.markdown("---")
    st.sidebar.title("🧭 Navegación")
    page = st.sidebar.selectbox(
        "Selecciona una sección:",
        ["🏠 Dashboard Principal", "💰 Análisis Financiero", "🔮 Predicciones", 
         "📊 Análisis Detallado", "👥 Clientes", "📈 Tendencias"]
    )
    
    # MOSTRAR ESTADO DE FILTROS
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Estado Actual")
    st.sidebar.success(f"✅ Datos totales: {len(df):,}")
    st.sidebar.info(f"📊 Datos mostrados: {len(df_filtered):,}")
    
    # Mostrar si hay filtros activos
    filtros_activos = []
    if selected_depto != 'Todos':
        filtros_activos.append(f"Departamento: {selected_depto}")
    if selected_pago != 'Todos':
        filtros_activos.append(f"Forma de Pago: {selected_pago}")
    if selected_marca != 'Todos':
        filtros_activos.append(f"Marca: {selected_marca}")
    
    if filtros_activos:
        st.sidebar.markdown("**Filtros aplicados:**")
        for filtro in filtros_activos:
            st.sidebar.write(f"• {filtro}")
    else:
        st.sidebar.markdown("**Sin filtros activos** - Mostrando todos los datos")
    
    # Calcular métricas
    metricas = calcular_metricas_financieras(df_filtered)
    
    # DASHBOARD PRINCIPAL
    if page == "🏠 Dashboard Principal":
        st.header("🏠 Dashboard Principal")
        
        # Mostrar información de filtros aplicados si los hay
        if filtros_activos:
            with st.expander("ℹ️ Filtros Aplicados"):
                for filtro in filtros_activos:
                    st.write(f"• {filtro}")
        else:
            st.info(f"📊 Mostrando todos los datos disponibles: {len(df_filtered):,} registros")
        
        # KPIs principales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "🛒 Unidades Vendidas",
                f"{metricas['total_unidades']:,}",
                help="Total de unidades vendidas"
            )
        
        with col2:
            if 'ingreso_total' in metricas:
                st.metric(
                    "💵 Ingresos",
                    f"${metricas['ingreso_total']:,.2f}",
                    help="Ingresos totales generados"
                )
            else:
                st.metric("📦 Transacciones", f"{metricas['total_transacciones']:,}")
        
        with col3:
            if 'utilidad_total' in metricas:
                st.metric(
                    "💰 Utilidad",
                    f"${metricas['utilidad_total']:,.2f}",
                    help="Utilidad total obtenida"
                )
            else:
                st.metric("🛍️ Productos", f"{metricas['productos_unicos']:,}")
        
        with col4:
            if 'margen_promedio' in metricas:
                st.metric(
                    "📊 Margen Promedio",
                    f"{metricas['margen_promedio']:.1f}%",
                    help="Margen de utilidad promedio"
                )
            else:
                st.metric("👥 Clientes", f"{metricas['clientes_unicos']:,}")
        
        with col5:
            if 'ticket_promedio' in metricas:
                st.metric(
                    "🎫 Ticket Promedio",
                    f"${metricas['ticket_promedio']:,.2f}",
                    help="Valor promedio por transacción"
                )
            else:
                avg_units = df_filtered['cantidad'].mean()
                st.metric("📊 Prom. Unid.", f"{avg_units:.1f}")
        
        st.markdown("---")
        
        # Gráficos principales
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 15 Productos")
            if 'ingreso_total' in df_filtered.columns:
                top_productos = df_filtered.groupby('descripcion')['ingreso_total'].sum().sort_values(ascending=False).head(15)
                titulo = "Por Ingresos ($)"
            else:
                top_productos = df_filtered.groupby('descripcion')['cantidad'].sum().sort_values(ascending=False).head(15)
                titulo = "Por Cantidad"
                
            fig = px.bar(
                x=top_productos.values,
                y=top_productos.index,
                orientation='h',
                title=titulo,
                color=top_productos.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💳 Distribución por Forma de Pago")
            if 'tipo_pago' in df_filtered.columns:
                pago_dist = df_filtered['tipo_pago'].value_counts()
                fig = px.pie(
                    values=pago_dist.values,
                    names=pago_dist.index,
                    title="Por tipo de pago",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
            else:
                # Fallback a departamentos
                dept_dist = df_filtered['departamento'].value_counts().head(8)
                fig = px.pie(
                    values=dept_dist.values,
                    names=dept_dist.index,
                    title="Por departamento",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tendencia de ventas (solo si hay fechas válidas)
        if 'fecha' in df_filtered.columns and df_filtered['fecha'].notna().sum() > 0:
            st.subheader("📈 Evolución Temporal")
            
            if 'ingreso_total' in df_filtered.columns:
                ventas_diarias = df_filtered.groupby(df_filtered['fecha'].dt.date).agg({
                    'cantidad': 'sum',
                    'ingreso_total': 'sum'
                }).reset_index()
                
                fig = go.Figure()
                
                # Línea de cantidad
                fig.add_trace(go.Scatter(
                    x=ventas_diarias['fecha'],
                    y=ventas_diarias['cantidad'],
                    mode='lines+markers',
                    name='Unidades Vendidas',
                    yaxis='y',
                    line=dict(color='#1f77b4')
                ))
                
                # Línea de ingresos
                fig.add_trace(go.Scatter(
                    x=ventas_diarias['fecha'],
                    y=ventas_diarias['ingreso_total'],
                    mode='lines+markers',
                    name='Ingresos ($)',
                    yaxis='y2',
                    line=dict(color='#ff7f0e')
                ))
                
                fig.update_layout(
                    title="Evolución de Ventas e Ingresos",
                    xaxis_title="Fecha",
                    yaxis=dict(title="Unidades", side="left"),
                    yaxis2=dict(title="Ingresos ($)", side="right", overlaying="y"),
                    height=400
                )
            else:
                ventas_diarias = df_filtered.groupby(df_filtered['fecha'].dt.date)['cantidad'].sum().reset_index()
                fig = px.line(
                    ventas_diarias,
                    x='fecha',
                    y='cantidad',
                    title="Evolución de Cantidad",
                    markers=True
                )
                fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ANÁLISIS FINANCIERO
    elif page == "💰 Análisis Financiero":
        st.header("💰 Análisis Financiero Detallado")
        
        if all(col in df_filtered.columns for col in ['ingreso_total', 'costo_total', 'utilidad_total']):
            # Métricas financieras avanzadas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                roi = (metricas['utilidad_total'] / metricas['costo_total'] * 100) if metricas['costo_total'] > 0 else 0
                st.metric("📊 ROI", f"{roi:.1f}%", help="Retorno sobre la inversión")
            
            with col2:
                rotacion = metricas['ingreso_total'] / metricas['costo_total'] if metricas['costo_total'] > 0 else 0
                st.metric("🔄 Rotación", f"{rotacion:.2f}x", help="Rotación de inventario")
            
            with col3:
                utilidad_unidad = metricas['utilidad_total'] / metricas['total_unidades'] if metricas['total_unidades'] > 0 else 0
                st.metric("💰 Utilidad/Unidad", f"${utilidad_unidad:.2f}", help="Utilidad promedio por unidad")
            
            with col4:
                costo_unidad = metricas['costo_total'] / metricas['total_unidades'] if metricas['total_unidades'] > 0 else 0
                st.metric("💸 Costo/Unidad", f"${costo_unidad:.2f}", help="Costo promedio por unidad")
            
            st.markdown("---")
            
            # Gráficos financieros
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏢 Rentabilidad por Departamento")
                depto_analysis = df_filtered.groupby('departamento').agg({
                    'ingreso_total': 'sum',
                    'utilidad_total': 'sum',
                    'cantidad': 'sum'
                }).reset_index()
                depto_analysis['margen'] = (depto_analysis['utilidad_total'] / depto_analysis['ingreso_total'] * 100).fillna(0)
                
                fig = px.scatter(
                    depto_analysis,
                    x='ingreso_total',
                    y='margen',
                    size='cantidad',
                    hover_data=['departamento'],
                    title="Ingresos vs Margen",
                    labels={'ingreso_total': 'Ingresos ($)', 'margen': 'Margen (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Distribución de Márgenes")
                margenes_filtrados = df_filtered[(df_filtered['margen_utilidad'] >= 0) & (df_filtered['margen_utilidad'] <= 100)]['margen_utilidad']
                
                fig = px.histogram(
                    x=margenes_filtrados,
                    nbins=25,
                    title="Frecuencia de márgenes",
                    labels={'x': 'Margen (%)', 'count': 'Frecuencia'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ Los datos financieros no están completamente disponibles.")
    
    # ANÁLISIS DETALLADO
    elif page == "📊 Análisis Detallado":
        st.header("📊 Análisis Detallado")
        
        # Análisis temporal (solo si hay datos de fecha)
        if 'nombre_dia_semana' in df_filtered.columns:
            st.subheader("⏰ Análisis por Días de la Semana")
            
            orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            ventas_dia = df_filtered.groupby('nombre_dia_semana')['cantidad'].sum()
            ventas_dia = ventas_dia.reindex([d for d in orden_dias if d in ventas_dia.index])
            
            fig = px.bar(
                x=ventas_dia.index,
                y=ventas_dia.values,
                title="Unidades vendidas por día",
                color=ventas_dia.values,
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top clientes
        st.subheader("👥 Top Clientes")
        if 'ingreso_total' in df_filtered.columns:
            top_clientes = df_filtered.groupby('cliente')['ingreso_total'].sum().sort_values(ascending=False).head(15)
            titulo = "Por Ingresos ($)"
        else:
            top_clientes = df_filtered.groupby('cliente')['cantidad'].sum().sort_values(ascending=False).head(15)
            titulo = "Por Cantidad"
        
        fig = px.bar(
            x=top_clientes.values,
            y=top_clientes.index,
            orientation='h',
            title=titulo,
            color=top_clientes.values,
            color_continuous_scale='plasma'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Simplificar otras páginas para evitar errores
    elif page == "🔮 Predicciones":
        st.header("🔮 Sistema de Predicción de Ventas")
        st.info("Funcionalidad de predicción disponible - en desarrollo para esta versión.")
    
    elif page == "💰 Análisis Financiero":
        st.header("💰 Análisis Financiero")
        st.info("Análisis financiero disponible cuando los datos contengan información completa de costos y precios.")
    
    elif page == "👥 Clientes":
        st.header("👥 Análisis de Clientes")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Clientes Únicos", f"{metricas['clientes_unicos']:,}")
        with col2:
            if metricas['clientes_unicos'] > 0:
                transacciones_por_cliente = metricas['total_transacciones'] / metricas['clientes_unicos']
                st.metric("🛒 Trans./Cliente", f"{transacciones_por_cliente:.1f}")
        with col3:
            if 'ingreso_total' in metricas and metricas['clientes_unicos'] > 0:
                ingreso_por_cliente = metricas['ingreso_total'] / metricas['clientes_unicos']
                st.metric("💰 Ingreso/Cliente", f"${ingreso_por_cliente:,.2f}")
        with col4:
            if metricas['clientes_unicos'] > 0:
                unidades_por_cliente = metricas['total_unidades'] / metricas['clientes_unicos']
                st.metric("📦 Unidades/Cliente", f"{unidades_por_cliente:.1f}")
    
    elif page == "📈 Tendencias":
        st.header("📈 Análisis de Tendencias")
        
        if 'fecha' in df_filtered.columns and df_filtered['fecha'].notna().sum() > 0:
            st.subheader("📅 Tendencias Mensuales")
            
            try:
                tendencias_mes = df_filtered.groupby(df_filtered['fecha'].dt.to_period('M')).agg({
                    'cantidad': 'sum',
                    'ingreso_total': 'sum' if 'ingreso_total' in df_filtered.columns else 'cantidad'
                }).reset_index()
                
                tendencias_mes['fecha'] = tendencias_mes['fecha'].dt.to_timestamp()
                
                fig = px.line(
                    tendencias_mes,
                    x='fecha',
                    y='cantidad',
                    title="Tendencia Mensual de Ventas",
                    markers=True
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error al generar tendencias: {e}")
        else:
            st.info("No hay datos de fecha suficientes para mostrar tendencias temporales.")

else:
    st.error("❌ No se pudieron cargar los datos. Verifica que el archivo 'dataSet1.csv' esté disponible.")
    st.info("💡 Asegúrate de que el archivo CSV contenga al menos las columnas: descripcion, cantidad, cliente, departamento")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 1rem; background-color: #00000; border-radius: 10px;">
        <p><strong>💼 Sistema Avanzado de Análisis de Ventas</strong></p>
        <p> Desarrollado por Equipo 1</p>
    </div>
    """, 
    unsafe_allow_html=True
)
