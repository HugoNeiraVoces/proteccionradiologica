"""
DASHBOARD INTERACTIVO DE PROTECCIÓN RADIOLÓGICA
Streamlit app para simulación de blindaje
Autor: Estudiante de Física Nuclear
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import io
import base64

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Blindaje Radiológico",
    page_icon="☢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FUNCIONES DE CÁLCULO
# ============================================================================

def calcular_atenuacion(I0, mu, x):
    """Ley de atenuación exponencial"""
    return I0 * np.exp(-mu * x)

def obtener_coeficiente_atenuacion(elemento, energia_mev, tipo_radiacion):
    """Obtiene coeficiente de atenuación basado en parámetros"""
    # Base de datos simplificada de coeficientes (cm⁻¹)
    # Valores aproximados basados en NIST XCOM
    # NOTA: energía_mev siempre en MeV para todos los tipos de radiación

    coeficientes = {
        'Plomo': {
            'Gamma': {0.001: 150.0, 0.01: 26.0, 0.1: 59.7, 0.5: 1.71, 1.0: 0.776, 5.0: 0.202, 10.0: 0.102},
            'Beta': {0.1: 0.18, 1.0: 0.15, 5.0: 0.08, 10.0: 0.05},
            'Neutrones': {0.000001: 0.5, 0.000025: 0.3, 1.0: 0.11, 5.0: 0.06, 10.0: 0.04},
            'Rayos X': {0.001: 150.0, 0.01: 26.0, 0.1: 59.7}  # Mismo que Gamma (son fotones)
        },
        'Acero': {
            'Gamma': {0.001: 5.8, 0.01: 1.8, 0.1: 2.94, 0.5: 0.653, 1.0: 0.469, 5.0: 0.154, 10.0: 0.095},
            'Beta': {0.1: 0.15, 1.0: 0.12, 5.0: 0.06, 10.0: 0.04},
            'Neutrones': {0.000001: 0.4, 0.000025: 0.2, 1.0: 0.08, 5.0: 0.04, 10.0: 0.03},
            'Rayos X': {0.001: 5.8, 0.01: 1.8, 0.1: 2.94}
        },
        'Hormigón': {
            'Gamma': {0.001: 0.8, 0.01: 0.5, 0.1: 0.385, 0.5: 0.227, 1.0: 0.150, 5.0: 0.064, 10.0: 0.042},
            'Beta': {0.1: 0.12, 1.0: 0.09, 5.0: 0.05, 10.0: 0.03},
            'Neutrones': {0.000001: 0.9, 0.000025: 0.7, 1.0: 0.07, 5.0: 0.03, 10.0: 0.02},
            'Rayos X': {0.001: 0.8, 0.01: 0.5, 0.1: 0.385}
        },
        'Agua': {
            'Gamma': {0.001: 0.4, 0.01: 0.2, 0.1: 0.167, 0.5: 0.096, 1.0: 0.0706, 5.0: 0.030, 10.0: 0.022},
            'Beta': {0.1: 0.14, 1.0: 0.11, 5.0: 0.05, 10.0: 0.03},
            'Neutrones': {0.000001: 1.2, 0.000025: 1.0, 1.0: 0.12, 5.0: 0.05, 10.0: 0.03},
            'Rayos X': {0.001: 0.4, 0.01: 0.2, 0.1: 0.167}
        },
        'Wolframio': {
            'Gamma': {0.001: 80.0, 0.01: 15.0, 0.1: 30.4, 0.5: 1.45, 1.0: 0.648, 5.0: 0.181, 10.0: 0.095},
            'Beta': {0.1: 0.17, 1.0: 0.14, 5.0: 0.07, 10.0: 0.04},
            'Neutrones': {0.000001: 0.6, 0.000025: 0.4, 1.0: 0.09, 5.0: 0.04, 10.0: 0.03},
            'Rayos X': {0.001: 80.0, 0.01: 15.0, 0.1: 30.4}
        },
        'Uranio': {
            'Gamma': {0.001: 220.0, 0.01: 45.0, 0.1: 85.3, 0.5: 2.43, 1.0: 1.091, 5.0: 0.252, 10.0: 0.125},
            'Beta': {0.1: 0.19, 1.0: 0.16, 5.0: 0.09, 10.0: 0.06},
            'Neutrones': {0.000001: 0.7, 0.000025: 0.5, 1.0: 0.13, 5.0: 0.06, 10.0: 0.04},
            'Rayos X': {0.001: 220.0, 0.01: 45.0, 0.1: 85.3}
        }
    }

    # Interpolación lineal para energías no listadas
    if elemento in coeficientes and tipo_radiacion in coeficientes[elemento]:
        energias = list(coeficientes[elemento][tipo_radiacion].keys())
        valores = list(coeficientes[elemento][tipo_radiacion].values())
        
        # Ordenar por energía
        energias_ordenadas, valores_ordenados = zip(*sorted(zip(energias, valores)))
        
        # Para energía fuera del rango, usar el valor más cercano
        if energia_mev < min(energias_ordenadas):
            return valores_ordenados[0]
        elif energia_mev > max(energias_ordenadas):
            return valores_ordenados[-1]
        else:
            # Interpolación lineal en escala log-log para mejor aproximación
            log_energias = np.log10(energias_ordenadas)
            log_valores = np.log10(valores_ordenados)
            log_energia = np.log10(energia_mev)
            return 10**np.interp(log_energia, log_energias, log_valores)
    
    return 0.1  # Valor por defecto

def calcular_capas_hvl_tvl(mu):
    """Calcula capa de medio y décimo valor"""
    hvl = np.log(2) / mu if mu > 0 else 0
    tvl = np.log(10) / mu if mu > 0 else 0
    return hvl, tvl

def generar_tabla_periodica():
    """Genera DataFrame con información para tabla periódica interactiva"""
    elementos = [
        # Elementos comunes para blindaje
        {'Simbolo': 'Pb', 'Nombre': 'Plomo', 'Z': 82, 'Grupo': 'Metales',
         'Densidad': 11.34, 'Color': '#A0522D', 'Blindaje': 'Alto'},
        {'Simbolo': 'W', 'Nombre': 'Wolframio', 'Z': 74, 'Grupo': 'Metales',
         'Densidad': 19.25, 'Color': '#FFD700', 'Blindaje': 'Muy Alto'},
        {'Simbolo': 'U', 'Nombre': 'Uranio', 'Z': 92, 'Grupo': 'Actinidos',
         'Densidad': 19.10, 'Color': '#000000', 'Blindaje': 'Muy Alto'},
        {'Simbolo': 'Fe', 'Nombre': 'Hierro', 'Z': 26, 'Grupo': 'Metales',
         'Densidad': 7.87, 'Color': '#B0B0B0', 'Blindaje': 'Medio'},
        {'Simbolo': 'Ac', 'Nombre': 'Acero', 'Z': 'Mix', 'Grupo': 'Aleaciones',
         'Densidad': 7.85, 'Color': '#778899', 'Blindaje': 'Medio'},
        {'Simbolo': 'Cu', 'Nombre': 'Cobre', 'Z': 29, 'Grupo': 'Metales',
         'Densidad': 8.96, 'Color': '#B87333', 'Blindaje': 'Medio'},
        {'Simbolo': 'H', 'Nombre': 'Hidrógeno', 'Z': 1, 'Grupo': 'No Metales',
         'Densidad': 0.000089, 'Color': '#FF69B4', 'Blindaje': 'Bajo'},
        {'Simbolo': 'H2O', 'Nombre': 'Agua', 'Z': 'Mix', 'Grupo': 'Compuestos',
         'Densidad': 1.00, 'Color': '#1E90FF', 'Blindaje': 'Bajo'},
        {'Simbolo': 'Con', 'Nombre': 'Hormigón', 'Z': 'Mix', 'Grupo': 'Compuestos',
         'Densidad': 2.35, 'Color': '#A9A9A9', 'Blindaje': 'Medio'},
        {'Simbolo': 'B', 'Nombre': 'Boro', 'Z': 5, 'Grupo': 'Metaloides',
         'Densidad': 2.34, 'Color': '#FFA500', 'Blindaje': 'Neutrones'}
    ]

    return pd.DataFrame(elementos)

# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

def main():
    # Título principal
    st.title("☢️ Simulador Interactivo de Protección Radiológica")
    st.markdown("""
    ### Herramienta para el análisis y diseño de sistemas de blindaje
    *Trabajo de Física Nuclear - Protección Radiológica y Sistemas de Blindaje Avanzado*
    """)

    # Sidebar para controles
    with st.sidebar:
        st.header("⚙️ Parámetros de Simulación")

        # Selección de tipo de radiación
        tipo_radiacion = st.selectbox(
            "Tipo de radiación:",
            ["Gamma", "Beta", "Neutrones", "Rayos X"],
            index=0
        )

        # ENTRADA DE ENERGÍA FLEXIBLE
        st.markdown("### 🔋 Energía de la radiación")
        
        # Mostrar rangos típicos
        with st.expander("ℹ️ Rangos típicos por tipo de radiación"):
            st.markdown("""
            - **Rayos X**: 1-300 keV (diagnóstico: 20-150 keV)
            - **Radiación Gamma**: 0.01-10 MeV (⁶⁰Co: 1.17-1.33 MeV)
            - **Radiación Beta**: 0.1-10 MeV (³²P: 1.71 MeV máximo)
            - **Neutrones**: 0.001 eV - 20 MeV (térmicos: 0.025 eV, rápidos: >0.1 MeV)
            
            **Nota:** 1 MeV = 1000 keV = 1,000,000 eV
            """)

        # Seleccionar unidad según tipo de radiación
        if tipo_radiacion == "Rayos X":
            unidad = st.radio("Unidad de energía:", ["keV", "MeV"], horizontal=True)
            default_val = 50.0 if unidad == "keV" else 0.05
            min_val = 1.0 if unidad == "keV" else 0.001
            max_val = 300.0 if unidad == "keV" else 0.3
            step_val = 1.0 if unidad == "keV" else 0.001
            format_str = "%.0f" if unidad == "keV" else "%.3f"
        else:
            unidad = "MeV"
            if tipo_radiacion == "Gamma":
                default_val = 1.0
                min_val = 0.001  # 1 keV
                max_val = 10.0
                step_val = 0.01
                format_str = "%.3f"
            elif tipo_radiacion == "Beta":
                default_val = 2.0
                min_val = 0.01
                max_val = 10.0
                step_val = 0.01
                format_str = "%.2f"
            elif tipo_radiacion == "Neutrones":
                default_val = 1.0
                min_val = 0.000001  # 1 eV
                max_val = 20.0
                step_val = 0.000001
                format_str = "%.6f"

        # Input numérico con la unidad seleccionada
        energia = st.number_input(
            f"Energía ({unidad}):",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=float(step_val),
            format=format_str,
            help=f"Energía de la radiación {tipo_radiacion}"
        )

        # Convertir todo a MeV internamente (nuestra base de datos usa MeV)
        if unidad == "keV":
            energia_mev = energia / 1000.0
            energia_display = f"{energia} keV"
        else:
            energia_mev = energia
            if energia < 0.001:
                energia_display = f"{energia*1000:.3f} keV" if energia >= 0.000001 else f"{energia*1e6:.2f} eV"
            else:
                energia_display = f"{energia} MeV"

        # Intensidad inicial
        I0 = st.number_input(
            "Intensidad inicial (partículas/s·cm²):",
            min_value=1e3,
            max_value=1e15,
            value=1e9,
            step=1e6,
            format="%.0e"
        )

        # Espesor máximo para gráfica
        espesor_max = st.slider(
            "Espesor máximo a visualizar (cm):",
            min_value=1,
            max_value=500,
            value=100,
            step=10
        )

        st.divider()
        st.header("📊 Opciones de Visualización")
        mostrar_hvl = st.checkbox("Mostrar capa de medio valor (HVL)", value=True)
        mostrar_tvl = st.checkbox("Mostrar capa de décimo valor (TVL)", value=True)
        escala_log = st.checkbox("Escala logarítmica en Y", value=True)

    # Contenido principal en pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Inicio y Explicación",
        "🎯 Tabla Periódica Interactiva",
        "📈 Simulación de Atenuación",
        "🔍 Comparación de Materiales",
        "📚 Información Teórica"
    ])

    with tab1:
        st.header("🏠 Bienvenido al Simulador de Blindaje Radiológico")
        
        st.markdown("""
        ## 📋 ¿Qué puedes hacer con esta aplicación?
        
        Esta herramienta interactiva te permite simular la atenuación de diferentes tipos 
        de radiación a través de diversos materiales de blindaje, aplicando principios 
        fundamentales de física nuclear.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 **Funcionalidades principales:**")
            st.markdown("""
            1. **Tabla Periódica Interactiva**
               - Selecciona elementos y materiales de blindaje
               - Visualiza propiedades clave (densidad, efectividad)
            
            2. **Simulación de Atenuación**
               - Gráficas de atenuación exponencial
               - Cálculo de HVL (Capa de Medio Valor) y TVL (Capa de Décimo Valor)
               - Ajuste de parámetros en tiempo real
            
            3. **Comparación de Materiales**
               - Compara múltiples materiales simultáneamente
               - Análisis de efectividad relativa
            
            4. **Información Teórica**
               - Fundamentos físicos de la atenuación
               - Ejemplos prácticos de cálculo
            """)
        
        with col2:
            st.subheader("⚙️ **Cómo usar la aplicación:**")
            st.markdown("""
            ### Paso 1: Configura los parámetros
            - Usa la barra lateral para seleccionar:
              - **Tipo de radiación** (Gamma, Beta, Neutrones, Rayos X)
              - **Energía** (con unidad apropiada: keV o MeV)
              - **Intensidad inicial**
              - **Opciones de visualización**
            
            ### Paso 2: Selecciona un material
            - Ve a la pestaña "Tabla Periódica"
            - Haz clic en cualquier elemento/material
            
            ### Paso 3: Explora y compara
            - Observa la curva de atenuación
            - Compara con otros materiales
            - Ajusta espesores y parámetros
            """)
        
        st.divider()
        
        st.subheader("📊 **Parámetros técnicos incluidos:**")
        
        datos = {
            "Concepto": [
                "Coeficiente de atenuación lineal (μ)",
                "Capa de Medio Valor (HVL)", 
                "Capa de Décimo Valor (TVL)",
                "Ley de atenuación exponencial",
                "Base de datos de materiales"
            ],
            "Descripción": [
                "Probabilidad de interacción por unidad de longitud",
                "Espesor para reducir intensidad a la mitad",
                "Espesor para reducir intensidad al 10%",
                "I(x) = I₀·e^(-μx)",
                "Valores basados en NIST XCOM (aproximados)"
            ],
            "Unidad": [
                "cm⁻¹",
                "cm",
                "cm", 
                "Adimensional",
                "Coeficientes reales"
            ]
        }
        
        st.dataframe(pd.DataFrame(datos), use_container_width=True)
        
        st.info("""
        💡 **Tip:** Comienza seleccionando un tipo de radiación en la barra lateral, 
        luego ve a la pestaña de Tabla Periódica para elegir un material.
        """)

    with tab2:
        st.header("Tabla Periódica para Blindaje Radiológico")

        # Generar tabla periódica
        df_elementos = generar_tabla_periodica()

        # Mostrar tabla periódica como cuadrícula interactiva
        cols = st.columns(6)

        for idx, row in df_elementos.iterrows():
            col_idx = idx % 6
            with cols[col_idx]:
                # Botón para cada elemento con color personalizado
                if st.button(
                    f"**{row['Simbolo']}**\n{row['Nombre']}",
                    key=f"elem_{row['Simbolo']}",
                    help=f"Z={row['Z']}, ρ={row['Densidad']} g/cm³",
                ):
                    # Almacenar elemento seleccionado en session state
                    st.session_state['elemento_seleccionado'] = row['Simbolo']

                # Información adicional en tooltip (simulado con markdown)
                st.caption(f"ρ={row['Densidad']} g/cm³")

        st.divider()

        # Mostrar información del elemento seleccionado
        if 'elemento_seleccionado' in st.session_state:
            elem = st.session_state['elemento_seleccionado']
            info = df_elementos[df_elementos['Simbolo'] == elem].iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Elemento", info['Nombre'])
                st.metric("Densidad", f"{info['Densidad']} g/cm³")

            with col2:
                st.metric("Grupo", info['Grupo'])
                st.metric("Efectividad", info['Blindaje'])

            with col3:
                # Calcular coeficiente para este elemento
                mu = obtener_coeficiente_atenuacion(info['Nombre'], energia_mev, tipo_radiacion)
                hvl, tvl = calcular_capas_hvl_tvl(mu)

                st.metric("μ (cm⁻¹)", f"{mu:.4f}")
                st.metric("HVL", f"{hvl:.1f} cm")

        # Selector de espesor para el elemento seleccionado
        if 'elemento_seleccionado' in st.session_state:
            st.subheader("Configurar blindaje")

            col1, col2 = st.columns(2)

            with col1:
                espesor = st.slider(
                    f"Espesor de {st.session_state['elemento_seleccionado']} (cm):",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=0.5
                )

            with col2:
                if st.button("🔄 Calcular atenuación", type="primary"):
                    # Calcular atenuación
                    elem_nombre = df_elementos[
                        df_elementos['Simbolo'] == st.session_state['elemento_seleccionado']
                    ]['Nombre'].iloc[0]

                    mu = obtener_coeficiente_atenuacion(elem_nombre, energia_mev, tipo_radiacion)
                    I_final = calcular_atenuacion(I0, mu, espesor)
                    atenuacion = (1 - I_final/I0) * 100

                    st.success(f"""
                    **Resultados:**
                    - Intensidad final: {I_final:.2e} partículas/s·cm²
                    - Atenuación: {atenuacion:.2f}%
                    - Coeficiente μ: {mu:.4f} cm⁻¹
                    - HVL: {calcular_capas_hvl_tvl(mu)[0]:.2f} cm
                    """)

    with tab3:
        st.header("Simulación de Atenuación")

        # Si hay elemento seleccionado, mostrar gráfica
        if 'elemento_seleccionado' in st.session_state:
            elem = st.session_state['elemento_seleccionado']
            info = df_elementos[df_elementos['Simbolo'] == elem].iloc[0]

            # Calcular curva de atenuación
            espesores = np.linspace(0, espesor_max, 500)
            mu = obtener_coeficiente_atenuacion(info['Nombre'], energia_mev, tipo_radiacion)
            intensidades = calcular_atenuacion(I0, mu, espesores)

            # Crear gráfica con Plotly
            fig = go.Figure()

            # Curva principal
            fig.add_trace(go.Scatter(
                x=espesores,
                y=intensidades,
                mode='lines',
                name=f'{info["Nombre"]} (μ={mu:.3f} cm⁻¹)',
                line=dict(color=info['Color'], width=3),
                hovertemplate="Espesor: %{x:.1f} cm<br>Intensidad: %{y:.2e}<extra></extra>"
            ))

            # Líneas de HVL y TVL
            if mostrar_hvl:
                hvl, _ = calcular_capas_hvl_tvl(mu)
                if hvl > 0 and hvl <= espesor_max:
                    fig.add_vline(
                        x=hvl,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"HVL = {hvl:.1f} cm",
                        annotation_position="top right"
                    )
            
            if mostrar_tvl:
                _, tvl = calcular_capas_hvl_tvl(mu)
                if tvl > 0 and tvl <= espesor_max:
                    fig.add_vline(
                        x=tvl,
                        line_dash="dot",
                        line_color="blue",
                        annotation_text=f"TVL = {tvl:.1f} cm",
                        annotation_position="top right"
                    )

            # Configurar layout
            fig.update_layout(
                title=f'Atenuación de radiación {tipo_radiacion} ({energia_display}) en {info["Nombre"]}',
                xaxis_title='Espesor del blindaje (cm)',
                yaxis_title='Intensidad transmitida (partículas/s·cm²)',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )

            if escala_log:
                fig.update_yaxis(type="log", exponentformat='power')

            st.plotly_chart(fig, use_container_width=True)

            # Mostrar información adicional
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Coeficiente de atenuación μ", f"{mu:.4f} cm⁻¹")
                st.metric("Energía", energia_display)

            with col2:
                hvl, tvl = calcular_capas_hvl_tvl(mu)
                st.metric("Capa de medio valor (HVL)", f"{hvl:.2f} cm")
                st.metric("Número de HVLs en espesor máximo", f"{espesor_max/hvl:.1f}" if hvl > 0 else "N/A")

            with col3:
                st.metric("Capa de décimo valor (TVL)", f"{tvl:.2f} cm")
                st.metric("Reducción total (I/I₀)", f"{intensidades[-1]/I0:.2e}")

            # Calculadora rápida
            st.subheader("Calculadora de atenuación")
            col_calc1, col_calc2 = st.columns(2)

            with col_calc1:
                espesor_calc = st.number_input(
                    "Espesor para cálculo (cm):",
                    min_value=0.0,
                    max_value=500.0,
                    value=10.0,
                    step=1.0,
                    key="calc_espesor"
                )

            with col_calc2:
                if espesor_calc > 0:
                    I_calc = calcular_atenuacion(I0, mu, espesor_calc)
                    atenuacion = (1 - I_calc/I0) * 100
                    num_hvl = espesor_calc / hvl if hvl > 0 else 0
                    
                    st.metric("Intensidad transmitida", f"{I_calc:.2e}")
                    st.metric("% Atenuación", f"{atenuacion:.2f}%")
                    st.metric("Equivalente en HVLs", f"{num_hvl:.2f}")

    with tab4:
        st.header("Comparación de Materiales de Blindaje")

        # Generar tabla periódica si no está definida
        df_elementos = generar_tabla_periodica()
        
        # Selección múltiple de materiales
        materiales_seleccionados = st.multiselect(
            "Selecciona materiales para comparar:",
            df_elementos['Nombre'].tolist(),
            default=['Plomo', 'Hormigón', 'Agua', 'Acero']
        )

        if materiales_seleccionados:
            # Crear gráfica comparativa
            fig_comparativa = go.Figure()

            espesores = np.linspace(0, espesor_max, 300)

            for material in materiales_seleccionados:
                # Obtener color del elemento
                color = df_elementos[df_elementos['Nombre'] == material]['Color'].iloc[0]

                # Calcular curva
                mu = obtener_coeficiente_atenuacion(material, energia_mev, tipo_radiacion)
                intensidades = calcular_atenuacion(I0, mu, espesores)

                fig_comparativa.add_trace(go.Scatter(
                    x=espesores,
                    y=intensidades,
                    mode='lines',
                    name=f'{material} (μ={mu:.3f})',
                    line=dict(color=color, width=2),
                    hovertemplate=f"{material}<br>μ={mu:.3f} cm⁻¹<br>%{{x:.1f}} cm → %{{y:.2e}}<extra></extra>"
                ))

            # Configurar layout
            fig_comparativa.update_layout(
                title=f'Comparación de atenuación para {tipo_radiacion} ({energia_display})',
                xaxis_title='Espesor (cm)',
                yaxis_title='Intensidad transmitida (partículas/s·cm²)',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )

            if escala_log:
                fig_comparativa.update_yaxis(type="log", exponentformat='power')

            st.plotly_chart(fig_comparativa, use_container_width=True)

            # Tabla comparativa
            st.subheader("Tabla comparativa")

            datos_comparacion = []
            for material in materiales_seleccionados:
                mu = obtener_coeficiente_atenuacion(material, energia_mev, tipo_radiacion)
                hvl, tvl = calcular_capas_hvl_tvl(mu)
                densidad = df_elementos[df_elementos['Nombre'] == material]['Densidad'].iloc[0]
                
                # Calcular atenuación a espesor máximo
                I_final = calcular_atenuacion(I0, mu, espesor_max)
                atenuacion = (1 - I_final/I0) * 100

                datos_comparacion.append({
                    'Material': material,
                    'μ (cm⁻¹)': f"{mu:.4f}",
                    'HVL (cm)': f"{hvl:.2f}",
                    'TVL (cm)': f"{tvl:.2f}",
                    'Densidad (g/cm³)': densidad,
                    'Aten. a {espesor_max}cm': f"{atenuacion:.1f}%",
                    'Efectividad': df_elementos[df_elementos['Nombre'] == material]['Blindaje'].iloc[0]
                })

            st.dataframe(pd.DataFrame(datos_comparacion), use_container_width=True)

    with tab5:
        st.header("Fundamentos Teóricos")

        col_info1, col_info2 = st.columns(2)

        with col_info1:
            st.subheader("📖 Ley de atenuación exponencial")
            st.latex(r"I(x) = I_0 \cdot e^{-\mu \cdot x}")
            st.markdown("""
            Donde:
            - **I(x)**: Intensidad transmitida
            - **I₀**: Intensidad incidente
            - **μ**: Coeficiente de atenuación lineal (cm⁻¹)
            - **x**: Espesor del material (cm)
            
            **Válida para:** Fotones (Rayos X y Gamma) en condiciones ideales
            """)

            st.subheader("⚖️ Capa de medio valor (HVL)")
            st.latex(r"HVL = \frac{\ln(2)}{\mu} = \frac{0.693}{\mu}")
            st.markdown("Espesor necesario para reducir la intensidad a la mitad.")
            
            st.subheader("🔟 Capa de décimo valor (TVL)")
            st.latex(r"TVL = \frac{\ln(10)}{\mu} = \frac{2.303}{\mu}")
            st.markdown("Espesor necesario para reducir la intensidad al 10%.")

            st.subheader("📈 Coeficiente de atenuación másico")
            st.latex(r"\mu_m = \frac{\mu}{\rho}")
            st.markdown("""
            Donde:
            - **μ_m**: Coeficiente másico (cm²/g)
            - **ρ**: Densidad del material (g/cm³)
            
            Útil para comparar materiales independientemente de su densidad.
            """)

        with col_info2:
            st.subheader("🎯 Efectividad de materiales por tipo de radiación")

            efectividad = {
                "Fotones (Rayos X/Gamma)": ["Plomo (Pb)", "Wolframio (W)", "Uranio (U)", "Acero"],
                "Moderación neutrones": ["Agua (H₂O)", "Grafito (C)", "Hormigón", "Polietileno"],
                "Captura neutrones": ["Boro (B)", "Cadmio (Cd)", "Litio (Li)", "Gadolinio"],
                "Partículas Beta": ["Plástico", "Aluminio", "Vidrio", "Aire"],
                "Partículas Alfa": ["Papel", "Piel humana", "Aire (pocos cm)"]
            }

            for categoria, materiales in efectividad.items():
                with st.expander(f"📌 {categoria}"):
                    for mat in materiales:
                        st.write(f"- {mat}")

            st.subheader("📊 Factores a considerar en blindaje")
            st.markdown("""
            1. **Tipo de radiación**: Mecanismos de interacción diferentes
            2. **Energía**: Afecta significativamente la atenuación
            3. **Densidad del material**: Mayor densidad → Mayor atenuación (generalmente)
            4. **Número atómico (Z)**: Importante para radiación electromagnética
            5. **Espesor requerido**: Depende de HVL/TVL y nivel de protección
            6. **Costo y disponibilidad**
            7. **Propiedades mecánicas y térmicas**
            8. **Efectos secundarios**: Radiación de frenado, activación
            """)
        
        st.divider()
        
        st.subheader("⚠️ Limitaciones del modelo simplificado")
        st.markdown("""
        Esta aplicación utiliza modelos simplificados de atenuación:

        1. **Para fotones (Rayos X/Gamma):** Ley exponencial válida para haces monoenergéticos colimados
        2. **Para partículas Beta:** Modelo aproximado (el comportamiento real depende del alcance máximo)
        3. **Para neutrones:** Coeficientes promedio (las secciones eficaces reales tienen resonancias)
        4. **No incluye:** Dispersión múltiple, radiación de frenado, producción de rayos X característicos
        
        **Para cálculos precisos** en aplicaciones reales de protección radiológica, se utilizan:
        - Códigos Monte Carlo (MCNP, Geant4)
        - Bases de datos completas (NIST XCOM, ENDF/B)
        - Factores de dosis empíricos
        """)

        # Ejemplo de cálculo
        st.subheader("🧮 Ejemplo práctico")

        col_ej1, col_ej2 = st.columns(2)

        with col_ej1:
            st.markdown("**Problema:**")
            st.markdown("""
            Se tiene una fuente de ⁶⁰Co que emite rayos gamma de 1.25 MeV
            con una intensidad de 10⁶ partículas/s·cm².
            
            ¿Qué espesor de plomo se necesita para reducir la intensidad
            a 100 partículas/s·cm²?
            """)

        with col_ej2:
            st.markdown("**Solución:**")
            st.latex(r"x = -\frac{1}{\mu} \ln\left(\frac{I}{I_0}\right)")
            
            # Cálculo
            mu_plomo = 0.776  # cm⁻¹ para 1.25 MeV aproximado
            x_necesario = -np.log(100/1e6) / mu_plomo
            
            st.markdown(f"""
            - μ para Plomo a 1.25 MeV ≈ {mu_plomo} cm⁻¹
            - Cálculo: x = -1/{mu_plomo:.3f} · ln(100/10⁶)
            - **Resultado: x ≈ {x_necesario:.2f} cm de plomo**
            - Equivalente a {x_necesario/calcular_capas_hvl_tvl(mu_plomo)[0]:.2f} HVLs
            """)

    # Footer
    st.divider()
    st.caption("""
    **Simulador desarrollado para el trabajo de Física Nuclear** |
    Protección Radiológica y Sistemas de Blindaje Avanzado |
    Universidad [Tu Universidad] | Curso 2024 |
    Los valores de coeficientes de atenuación son aproximados basados en datos de NIST XCOM y referencias de protección radiológica
    """)

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    # Inicializar variables de sesión si no existen
    if 'elemento_seleccionado' not in st.session_state:
        st.session_state['elemento_seleccionado'] = 'Pb'

    main()
