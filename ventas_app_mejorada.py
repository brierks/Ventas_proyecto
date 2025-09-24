
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
    page_icon="💰",
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
        
        # Procesar fecha
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        
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
        
        # Crear variables de tiempo
        if 'fecha' in df.columns and not df['fecha'].isna().all():
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
        numeric_features = ['anio', 'mes', 'dia_mes', 'hora']
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

    # Filtros de fecha
    if 'fecha' in df.columns and not df['fecha'].isna().all():
        min_date = df['fecha'].min().date()
        max_date = df['fecha'].max().date()
        
        
    
   
    
    # Filtros adicionales
    departamentos = ['Todos'] + sorted(df['departamento'].unique().tolist())
    selected_depto = st.sidebar.selectbox("🏢 Departamento:", departamentos)
    
    formas_pago = ['Todos'] + sorted(df['Forma Pago'].unique().tolist())
    selected_pago = st.sidebar.selectbox("💳 Forma de Pago:", formas_pago)
    
    marcas = ['Todos'] + sorted(df['marca'].unique().tolist())
    selected_marca = st.sidebar.selectbox("🏷️ Marca:", marcas)
    
    # Aplicar filtros
    df_filtered = df.copy()
    
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        if 'fecha' in df_filtered.columns:
            df_filtered = df_filtered[(df_filtered['fecha'] >= start_date) & (df_filtered['fecha'] <= end_date)]
    
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
    
    # Calcular métricas
    metricas = calcular_metricas_financieras(df_filtered)
    
    # DASHBOARD PRINCIPAL
    if page == "🏠 Dashboard Principal":
        st.header("🏠 Dashboard Principal")
        
        # Mostrar filtros aplicados
        with st.expander("ℹ️ Filtros Aplicados"):
            st.write(f"**Departamento:** {selected_depto}")
            st.write(f"**Forma de Pago:** {selected_pago}")
            st.write(f"**Marca:** {selected_marca}")
            if 'date_range' in locals():
                st.write(f"**Período:** {date_range[0]} - {date_range[1]}")
        
        # KPIs principales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "🛒 Unidades Vendidas",
                f"{metricas['total_unidades']:,}",
                help="Total de unidades vendidas en el período"
            )
        
        with col2:
            if 'ingreso_total' in metricas:
                st.metric(
                    "💵 Ingresos",
                    f"${metricas['ingreso_total']:,.2f}",
                    help="Ingresos totales generados"
                )
        
        with col3:
            if 'utilidad_total' in metricas:
                st.metric(
                    "💰 Utilidad",
                    f"${metricas['utilidad_total']:,.2f}",
                    help="Utilidad total obtenida"
                )
        
        with col4:
            if 'margen_promedio' in metricas:
                st.metric(
                    "📊 Margen Promedio",
                    f"{metricas['margen_promedio']:.1f}%",
                    help="Margen de utilidad promedio"
                )
        
        with col5:
            if 'ticket_promedio' in metricas:
                st.metric(
                    "🎫 Ticket Promedio",
                    f"${metricas['ticket_promedio']:,.2f}",
                    help="Valor promedio por transacción"
                )
        
        st.markdown("---")
        
        # Gráficos principales
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 15 Productos por Ingresos")
            if 'ingreso_total' in df_filtered.columns:
                top_productos_ingresos = df_filtered.groupby('descripcion')['ingreso_total'].sum().sort_values(ascending=False).head(15)
                fig = px.bar(
                    x=top_productos_ingresos.values,
                    y=top_productos_ingresos.index,
                    orientation='h',
                    title="Productos que más ingresos generan",
                    color=top_productos_ingresos.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Datos de ingresos no disponibles")
        
        with col2:
            st.subheader("💳 Distribución por Forma de Pago")
            if 'tipo_pago' in df_filtered.columns:
                pago_dist = df_filtered['tipo_pago'].value_counts()
                fig = px.pie(
                    values=pago_dist.values,
                    names=pago_dist.index,
                    title="Distribución de formas de pago",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Datos de forma de pago no disponibles")
        
        # Tendencia de ventas
        st.subheader("📈 Tendencia de Ventas e Ingresos")
        if 'fecha' in df_filtered.columns and 'ingreso_total' in df_filtered.columns:
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
            
            # Análisis por departamento
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
                    title="Ingresos vs Margen por Departamento",
                    labels={'ingreso_total': 'Ingresos ($)', 'margen': 'Margen (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🏆 Top 10 Productos Más Rentables")
                productos_rentables = df_filtered.groupby('descripcion').agg({
                    'utilidad_total': 'sum',
                    'cantidad': 'sum'
                }).reset_index()
                productos_rentables['utilidad_por_unidad'] = productos_rentables['utilidad_total'] / productos_rentables['cantidad']
                productos_rentables = productos_rentables.sort_values('utilidad_total', ascending=False).head(10)
                
                fig = px.bar(
                    productos_rentables,
                    x='utilidad_total',
                    y='descripcion',
                    orientation='h',
                    title="Productos con mayor utilidad total",
                    color='utilidad_por_unidad',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de márgenes
            st.subheader("📊 Análisis de Márgenes de Utilidad")
            
            # Histograma de márgenes
            fig = px.histogram(
                df_filtered,
                x='margen_utilidad',
                nbins=30,
                title="Distribución de Márgenes de Utilidad",
                labels={'margen_utilidad': 'Margen de Utilidad (%)', 'count': 'Frecuencia'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla resumen por marca
            st.subheader("🏷️ Análisis por Marca")
            marca_analysis = df_filtered.groupby('marca').agg({
                'ingreso_total': 'sum',
                'costo_total': 'sum',
                'utilidad_total': 'sum',
                'cantidad': 'sum',
                'margen_utilidad': 'mean'
            }).round(2)
            marca_analysis.columns = ['Ingresos', 'Costos', 'Utilidad', 'Unidades', 'Margen Promedio (%)']
            marca_analysis = marca_analysis.sort_values('Utilidad', ascending=False)
            st.dataframe(marca_analysis.head(20), use_container_width=True)
        
        else:
            st.warning("⚠️ Los datos financieros (costo, precio de venta, utilidad) no están completamente disponibles para realizar el análisis financiero.")
    
    # PREDICCIONES
    elif page == "🔮 Predicciones":
        st.header("🔮 Sistema de Predicción de Ventas")
        
        with st.spinner("🤖 Entrenando modelo de machine learning..."):
            model, encoders, features, mae, rmse, r2 = create_prediction_model(df_filtered)
        
        if model is not None:
            # Mostrar métricas del modelo
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{mae:.2f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}")
            with col3:
                st.metric("R² Score", f"{r2:.3f}")
            
            # Importancia de características
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(
                feature_importance.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Importancia de las Características en el Modelo"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🎯 Realizar Predicción")
            
            # Interface de predicción
            col1, col2 = st.columns(2)
            
            with col1:
                pred_producto = st.selectbox("Producto:", sorted(df['descripcion'].unique()))
                pred_departamento = st.selectbox("Departamento:", sorted(df['departamento'].unique()))
                pred_marca = st.selectbox("Marca:", sorted(df['marca'].unique()))
                pred_cliente = st.selectbox("Cliente:", sorted(df['cliente'].unique())[:50])  # Limitar opciones
            
            with col2:
                pred_tipo_pago = st.selectbox("Tipo de Pago:", sorted(df_filtered['tipo_pago'].unique()) if 'tipo_pago' in df_filtered.columns else ['EFECTIVO'])
                pred_fecha = st.date_input("Fecha:", value=datetime.now().date())
                pred_hora = st.slider("Hora:", 0, 23, 12)
                
                # Valores históricos promedio para el producto
                producto_hist = df[df['descripcion'] == pred_producto]
                if len(producto_hist) > 0 and 'costo' in df.columns:
                    costo_prom = producto_hist['costo'].mean()
                    precio_prom = producto_hist['Precio venta'].mean() if 'Precio venta' in df.columns else costo_prom * 1.3
                else:
                    costo_prom = 100
                    precio_prom = 130
                
                pred_costo = st.number_input("Costo estimado:", value=float(costo_prom), min_value=0.0)
                pred_precio = st.number_input("Precio de venta:", value=float(precio_prom), min_value=0.0)
            
            if st.button("🔮 Generar Predicción", type="primary"):
                try:
                    # Preparar datos para predicción
                    pred_data = {}
                    
                    # Codificar variables categóricas
                    categorical_cols = ['descripcion', 'departamento', 'marca', 'cliente', 'tipo_pago']
                    for col in categorical_cols:
                        if col in encoders:
                            if col == 'descripcion':
                                val = pred_producto
                            elif col == 'departamento':
                                val = pred_departamento
                            elif col == 'marca':
                                val = pred_marca
                            elif col == 'cliente':
                                val = pred_cliente
                            elif col == 'tipo_pago':
                                val = pred_tipo_pago
                            
                            try:
                                encoded_val = encoders[col].transform([val])[0]
                            except ValueError:
                                # Si el valor no existe en el encoder, usar el más común
                                encoded_val = 0
                            
                            pred_data[f"{col}_encoded"] = encoded_val
                    
                    # Agregar variables numéricas
                    pred_data['anio'] = pred_fecha.year
                    pred_data['mes'] = pred_fecha.month
                    pred_data['dia_mes'] = pred_fecha.day
                    pred_data['hora'] = pred_hora
                    
                    if 'costo' in features:
                        pred_data['costo'] = pred_costo
                    if 'Precio venta' in features:
                        pred_data['Precio venta'] = pred_precio
                    
                    # Crear vector de predicción
                    X_pred = np.array([[pred_data.get(f, 0) for f in features]])
                    
                    # Realizar predicción
                    cantidad_pred = model.predict(X_pred)[0]
                    
                    # Mostrar resultados
                    st.success(f"🎯 **Predicción de cantidad:** {cantidad_pred:.0f} unidades")
                    
                    # Estimaciones financieras
                    ingreso_estimado = cantidad_pred * pred_precio
                    costo_estimado = cantidad_pred * pred_costo
                    utilidad_estimada = ingreso_estimado - costo_estimado
                    margen_estimado = (utilidad_estimada / ingreso_estimado * 100) if ingreso_estimado > 0 else 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("💵 Ingreso Estimado", f"${ingreso_estimado:,.2f}")
                    with col2:
                        st.metric("💸 Costo Estimado", f"${costo_estimado:,.2f}")
                    with col3:
                        st.metric("💰 Utilidad Estimada", f"${utilidad_estimada:,.2f}")
                    with col4:
                        st.metric("📊 Margen Estimado", f"{margen_estimado:.1f}%")
                    
                    # Comparación con históricos
                    if len(producto_hist) > 0:
                        with st.expander("📊 Comparación con Históricos"):
                            st.write(f"**Promedio histórico:** {producto_hist['cantidad'].mean():.1f} unidades")
                            st.write(f"**Máximo histórico:** {producto_hist['cantidad'].max()} unidades")
                            st.write(f"**Mínimo histórico:** {producto_hist['cantidad'].min()} unidades")
                            
                            diferencia = cantidad_pred - producto_hist['cantidad'].mean()
                            porcentaje_dif = (diferencia / producto_hist['cantidad'].mean() * 100) if producto_hist['cantidad'].mean() > 0 else 0
                            
                            if diferencia > 0:
                                st.write(f"🔺 **La predicción es {diferencia:.1f} unidades ({porcentaje_dif:.1f}%) mayor al promedio histórico**")
                            else:
                                st.write(f"🔻 **La predicción es {abs(diferencia):.1f} unidades ({abs(porcentaje_dif):.1f}%) menor al promedio histórico**")
                
                except Exception as e:
                    st.error(f"Error en la predicción: {e}")
        else:
            st.error("No se pudo entrenar el modelo con los datos actuales.")
    
    # ANÁLISIS DETALLADO
    elif page == "📊 Análisis Detallado":
        st.header("📊 Análisis Detallado")
        
        # Análisis temporal
        if 'nombre_dia_semana' in df_filtered.columns:
            st.subheader("⏰ Análisis Temporal")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Ventas por Día de la Semana**")
                orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                ventas_dia = df_filtered.groupby('nombre_dia_semana').agg({
                    'cantidad': 'sum',
                    'ingreso_total': 'sum' if 'ingreso_total' in df_filtered.columns else 'cantidad'
                })
                ventas_dia = ventas_dia.reindex([d for d in orden_dias if d in ventas_dia.index])
                
                fig = px.bar(
                    x=ventas_dia.index,
                    y=ventas_dia['cantidad'],
                    title="Unidades vendidas por día",
                    color=ventas_dia['cantidad'],
                    color_continuous_scale='blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Ventas por Hora del Día**")
                if 'hora' in df_filtered.columns:
                    ventas_hora = df_filtered.groupby('hora')['cantidad'].sum()
                    fig = px.line(
                        x=ventas_hora.index,
                        y=ventas_hora.values,
                        title="Patrón de ventas por hora",
                        markers=True
                    )
                    fig.update_xaxis(title="Hora del día")
                    fig.update_yaxis(title="Unidades vendidas")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Datos de hora no disponibles")
        
        # Análisis de correlaciones
        st.subheader("🔗 Análisis de Correlaciones")
        if all(col in df_filtered.columns for col in ['cantidad', 'costo', 'Precio venta', 'Utilidad']):
            corr_data = df_filtered[['cantidad', 'costo', 'Precio venta', 'Utilidad']].corr()
            
            fig = px.imshow(
                corr_data,
                text_auto=True,
                aspect="auto",
                title="Matriz de Correlación",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Análisis ABC de productos
        st.subheader("📋 Análisis ABC de Productos")
        if 'ingreso_total' in df_filtered.columns:
            productos_abc = df_filtered.groupby('descripcion').agg({
                'ingreso_total': 'sum',
                'cantidad': 'sum'
            }).reset_index()
            productos_abc = productos_abc.sort_values('ingreso_total', ascending=False)
            productos_abc['acumulado_ingresos'] = productos_abc['ingreso_total'].cumsum()
            productos_abc['porcentaje_acumulado'] = productos_abc['acumulado_ingresos'] / productos_abc['ingreso_total'].sum() * 100
            
            # Clasificación ABC
            def clasificar_abc(porcentaje):
                if porcentaje <= 80:
                    return 'A'
                elif porcentaje <= 95:
                    return 'B'
                else:
                    return 'C'
            
            productos_abc['categoria_abc'] = productos_abc['porcentaje_acumulado'].apply(clasificar_abc)
            
            # Mostrar distribución ABC
            abc_dist = productos_abc['categoria_abc'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=abc_dist.values,
                    names=abc_dist.index,
                    title="Distribución de Productos ABC",
                    color_discrete_map={'A': '#ff9999', 'B': '#66b3ff', 'C': '#99ff99'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Resumen del Análisis ABC:**")
                for categoria in ['A', 'B', 'C']:
                    productos_cat = productos_abc[productos_abc['categoria_abc'] == categoria]
                    ingresos_cat = productos_cat['ingreso_total'].sum()
                    porcentaje_ingresos = ingresos_cat / productos_abc['ingreso_total'].sum() * 100
                    st.write(f"**Categoría {categoria}:** {len(productos_cat)} productos ({porcentaje_ingresos:.1f}% de ingresos)")
    
    # ANÁLISIS DE CLIENTES
    elif page == "👥 Clientes":
        st.header("👥 Análisis de Clientes")
        
        # Métricas de clientes
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
        
        st.markdown("---")
        
        # Análisis de clientes top
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 20 Clientes por Ingresos")
            if 'ingreso_total' in df_filtered.columns:
                top_clientes_ingresos = df_filtered.groupby('cliente').agg({
                    'ingreso_total': 'sum',
                    'cantidad': 'sum',
                    'folio': 'nunique' if 'folio' in df_filtered.columns else 'count'
                }).sort_values('ingreso_total', ascending=False).head(20)
                
                fig = px.bar(
                    x=top_clientes_ingresos['ingreso_total'],
                    y=top_clientes_ingresos.index,
                    orientation='h',
                    title="Clientes por ingresos generados",
                    color=top_clientes_ingresos['ingreso_total'],
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Segmentación RFM Simplificada")
            if 'fecha' in df_filtered.columns and 'ingreso_total' in df_filtered.columns:
                # Calcular métricas RFM
                fecha_actual = df_filtered['fecha'].max()
                rfm_data = df_filtered.groupby('cliente').agg({
                    'fecha': lambda x: (fecha_actual - x.max()).days,  # Recency
                    'folio': 'nunique' if 'folio' in df_filtered.columns else 'count',  # Frequency
                    'ingreso_total': 'sum'  # Monetary
                }).reset_index()
                
                rfm_data.columns = ['cliente', 'recency', 'frequency', 'monetary']
                
                # Segmentación simple
                def segmentar_cliente(row):
                    if row['monetary'] >= rfm_data['monetary'].quantile(0.8) and row['frequency'] >= rfm_data['frequency'].quantile(0.6):
                        return 'Champions'
                    elif row['monetary'] >= rfm_data['monetary'].quantile(0.6):
                        return 'Loyal Customers'
                    elif row['frequency'] >= rfm_data['frequency'].quantile(0.6):
                        return 'Potential Loyalists'
                    elif row['recency'] <= rfm_data['recency'].quantile(0.4):
                        return 'New Customers'
                    else:
                        return 'At Risk'
                
                rfm_data['segmento'] = rfm_data.apply(segmentar_cliente, axis=1)
                
                # Distribución de segmentos
                segmento_dist = rfm_data['segmento'].value_counts()
                fig = px.pie(
                    values=segmento_dist.values,
                    names=segmento_dist.index,
                    title="Segmentación de Clientes",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ANÁLISIS DE TENDENCIAS
    elif page == "📈 Tendencias":
        st.header("📈 Análisis de Tendencias")
        
        if 'fecha' in df_filtered.columns:
            # Tendencias mensuales
            st.subheader("📅 Tendencias Mensuales")
            
            tendencias_mes = df_filtered.groupby([df_filtered['fecha'].dt.to_period('M')]).agg({
                'cantidad': 'sum',
                'ingreso_total': 'sum' if 'ingreso_total' in df_filtered.columns else 'cantidad',
                'utilidad_total': 'sum' if 'utilidad_total' in df_filtered.columns else 'cantidad'
            }).reset_index()
            
            tendencias_mes['fecha'] = tendencias_mes['fecha'].dt.to_timestamp()
            
            # Gráfico de tendencias múltiples
            fig = go.Figure()
            
            # Cantidad
            fig.add_trace(go.Scatter(
                x=tendencias_mes['fecha'],
                y=tendencias_mes['cantidad'],
                mode='lines+markers',
                name='Unidades Vendidas',
                yaxis='y'
            ))
            
            if 'ingreso_total' in df_filtered.columns:
                # Ingresos (eje secundario)
                fig.add_trace(go.Scatter(
                    x=tendencias_mes['fecha'],
                    y=tendencias_mes['ingreso_total'],
                    mode='lines+markers',
                    name='Ingresos ($)',
                    yaxis='y2'
                ))
            
            fig.update_layout(
                title="Tendencias Mensuales de Ventas e Ingresos",
                xaxis_title="Mes",
                yaxis=dict(title="Unidades", side="left"),
                yaxis2=dict(title="Ingresos ($)", side="right", overlaying="y") if 'ingreso_total' in df_filtered.columns else None,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ No se pudieron cargar los datos. Verifica que el archivo 'dataSet1.csv' esté disponible.")
    st.info("💡 Asegúrate de que el archivo CSV contenga las columnas: codigo, descripcion, cantidad, costo, Precio venta, Utilidad, Forma Pago, fecha, cliente, departamento, marca")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;">
        <p><strong>💼 Sistema Avanzado de Análisis de Ventas</strong></p>
        <p>Desarrollado por <strong>Equipo</strong> | 📊 Dashboard con IA y Analytics</p>
    </div>
    """, 
    unsafe_allow_html=True
)
