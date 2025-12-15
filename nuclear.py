# -*- coding: utf-8 -*-
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
    if mu <= 0:
        return I0
    return I0 * np.exp(-mu * x)

def obtener_coeficiente_atenuacion(elemento, energia, tipo_radiacion):
    """Obtiene coeficiente de atenuación basado en parámetros"""
    # Base de datos simplificada de coeficientes (cm⁻¹)
    # Valores aproximados basados en NIST XCOM
    
    coeficientes = {
        'Plomo': {
            'Gamma': {0.1: 59.7, 0.5: 1.71, 1.0: 0.776, 5.0: 0.202},
            'Beta': {1.0: 0.15, 5.0: 0.08},
            'Neutrones': {1.0: 0.11, 5.0: 0.06},
            'Rayos X': {0.1: 59.7, 0.5: 1.71, 1.0: 0.776, 5.0: 0.202}
        },
        'Acero': {
            'Gamma': {0.1: 2.94, 0.5: 0.653, 1.0: 0.469, 5.0: 0.154},
            'Beta': {1.0: 0.12, 5.0: 0.06},
            'Neutrones': {1.0: 0.08, 5.0: 0.04},
            'Rayos X': {0.1: 2.94, 0.5: 0.653, 1.0: 0.469, 5.0: 0.154}
        },
        'Hormigón': {
            'Gamma': {0.1: 0.385, 0.5: 0.227, 1.0: 0.150, 5.0: 0.064},
            'Beta': {1.0: 0.09, 5.0: 0.05},
            'Neutrones': {1.0: 0.07, 5.0: 0.03},
            'Rayos X': {0.1: 0.385, 0.5: 0.227, 1.0: 0.150, 5.0: 0.064}
        },
        'Agua': {
            'Gamma': {0.1: 0.167, 0.5: 0.096, 1.0: 0.0706, 5.0: 0.030},
            'Beta': {1.0: 0.11, 5.0: 0.05},
            'Neutrones': {1.0: 0.12, 5.0: 0.05},
            'Rayos X': {0.1: 0.167, 0.5: 0.096, 1.0: 0.0706, 5.0: 0.030}
        },
        'Wolframio': {
            'Gamma': {0.1: 30.4, 0.5: 1.45, 1.0: 0.648, 5.0: 0.181},
            'Beta': {1.0: 0.14, 5.0: 0.07},
            'Neutrones': {1.0: 0.09, 5.0: 0.04},
            'Rayos X': {0.1: 30.4, 0.5: 1.45, 1.0: 0.648, 5.0: 0.181}
        },
        'Uranio': {
            'Gamma': {0.1: 85.3, 0.5: 2.43, 1.0: 1.091, 5.0: 0.252},
            'Beta': {1.0: 0.16, 5.0: 0.09},
            'Neutrones': {1.0: 0.13, 5.0: 0.06},
            'Rayos X': {0.1: 85.3, 0.5: 2.43, 1.0: 1.091, 5.0: 0.252}
        },
        'Hierro': {
            'Gamma': {0.1: 2.94, 0.5: 0.653, 1.0: 0.469, 5.0: 0.154},
            'Beta': {1.0: 0.12, 5.0: 0.06},
            'Neutrones': {1.0: 0.08, 5.0: 0.04},
            'Rayos X': {0.1: 2.94, 0.5: 0.653, 1.0: 0.469, 5.0: 0.154}
        },
        'Cobre': {
            'Gamma': {0.1: 4.65, 0.5: 1.02, 1.0: 0.693, 5.0: 0.198},
            'Beta': {1.0: 0.13, 5.0: 0.07},
            'Neutrones': {1.0: 0.09, 5.0: 0.04},
            'Rayos X': {0.1: 4.65, 0.5: 1.02, 1.0: 0.693, 5.0: 0.198}
        },
        'Boro': {
            'Gamma': {0.1: 0.02, 0.5: 0.015, 1.0: 0.012, 5.0: 0.008},
            'Beta': {1.0: 0.05, 5.0: 0.03},
            'Neutrones': {1.0: 1.50, 5.0: 0.80},  # Alto para neutrones
            'Rayos X': {0.1: 0.02, 0.5: 0.015, 1.0: 0.012, 5.0: 0.008}
        }
    }

    # Convertir energía para Rayos X (keV -> MeV)
    if tipo_radiacion == 'Rayos X':
        energia_mev = energia / 1000.0
    else:
        energia_mev = energia

    # Interpolación lineal para energías no listadas
    if elemento in coeficientes and tipo_radiacion in coeficientes[elemento]:
        energias = list(coeficientes[elemento][tipo_radiacion].keys())
        valores = list(coeficientes[elemento][tipo_radiacion].values())
        
        if len(energias) == 0:
            return 0.1
            
        # Asegurar que la energía esté dentro del rango
        if energia_mev <= min(energias):
            return valores[0]
        elif energia_mev >= max(energias):
            return valores[-1]
        else:
            # Interpolación lineal
            return np.interp(energia_mev, energias, valores)

    return 0.1  # Valor por defecto

def calcular_capas_hvl_tvl(mu):
    """Calcula capa de medio y décimo valor"""
    if mu <= 0:
        return 0, 0
    hvl = np.log(2) / mu
    tvl = np.log(10) / mu
    return hvl, tvl

def generar_tabla_periodica():
    """Genera DataFrame con información para tabla periódica interactiva"""
    elementos = [
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
    # Inicializar variables de sesión
    if 'elemento_seleccionado' not in st.session_state:
        st.session_state['elemento_seleccionado'] = 'Pb'
    
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
        
        # Configurar rango de energía según tipo de radiación
        if tipo_radiacion == "Rayos X":
            min_energy, max_energy, step, value = 1.0, 200.0, 1.0, 50.0
            unidad = "keV"
        else:
            min_energy, max_energy, step, value = 0.1, 10.0, 0.1, 1.0
            unidad = "MeV"
        
        # Slider para energía
        energia = st.slider(
            f"Energía ({unidad}):",
            min_value=min_energy,
            max_value=max_energy,
            value=value,
            step=step,
            help=f"Energía de la radiación incidente en {unidad}"
        )
        
        # Intensidad inicial
        I0 = st.number_input(
            "Intensidad inicial (partículas/s·cm²):",
            min_value=1e3,
            max_value=1e15,
            value=1e9,
            step=1e6,
            format="%e"
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Tabla Periódica Interactiva",
        "📈 Simulación de Atenuación",
        "🔍 Comparación de Materiales",
        "📚 Información Teórica"
    ])
    
    with tab1:
        st.header("Tabla Periódica para Blindaje Radiológico")
        
        # Generar tabla periódica
        df_elementos = generar_tabla_periodica()
        
        # Mostrar tabla periódica como cuadrícula interactiva
        cols = st.columns(3)  # Reducido de 6 a 3 columnas para mejor visualización
        
        for idx, row in df_elementos.iterrows():
            col_idx = idx % 3
            with cols[col_idx]:
                # Botón para cada elemento
                button_key = f"elem_btn_{row['Simbolo']}"
                if st.button(
                    f"**{row['Simbolo']}**\n{row['Nombre']}",
                    key=button_key,
                    help=f"Z={row['Z']}, ρ={row['Densidad']} g/cm³",
                    use_container_width=True
                ):
                    # Almacenar elemento seleccionado en session state
                    st.session_state['elemento_seleccionado'] = row['Simbolo']
                    st.rerun()
                
                # Información adicional
                st.caption(f"Densidad: {row['Densidad']} g/cm³")
        
        st.divider()
        
        # Mostrar información del elemento seleccionado
        if st.session_state['elemento_seleccionado']:
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
                mu = obtener_coeficiente_atenuacion(info['Nombre'], energia, tipo_radiacion)
                hvl, tvl = calcular_capas_hvl_tvl(mu)
                
                st.metric("μ (cm⁻¹)", f"{mu:.4f}")
                st.metric("HVL", f"{hvl:.1f} cm")
            
            # Selector de espesor para el elemento seleccionado
            st.subheader("Configurar blindaje")
            
            col1, col2 = st.columns(2)
            
            with col1:
                espesor = st.slider(
                    f"Espesor de {info['Nombre']} (cm):",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=0.5
                )
            
            with col2:
                # Calcular atenuación automáticamente
                mu = obtener_coeficiente_atenuacion(info['Nombre'], energia, tipo_radiacion)
                I_final = calcular_atenuacion(I0, mu, espesor)
                atenuacion = (1 - I_final/I0) * 100 if I0 > 0 else 0
                
                st.metric("Intensidad final", f"{I_final:.2e}", 
                         delta=f"Atenuación: {atenuacion:.1f}%")
                st.metric("Coeficiente μ", f"{mu:.4f} cm⁻¹")
    
    with tab2:
        st.header("Simulación de Atenuación")
        
        # Si hay elemento seleccionado, mostrar gráfica
        if st.session_state['elemento_seleccionado']:
            elem = st.session_state['elemento_seleccionado']
            info = df_elementos[df_elementos['Simbolo'] == elem].iloc[0]
            
            # Calcular curva de atenuación
            espesores = np.linspace(0, espesor_max, 500)
            mu = obtener_coeficiente_atenuacion(info['Nombre'], energia, tipo_radiacion)
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
            energia_str = f"{energia} {unidad}"
            fig.update_layout(
                title=f'Atenuación de radiación {tipo_radiacion} ({energia_str}) en {info["Nombre"]}',
                xaxis_title='Espesor del blindaje (cm)',
                yaxis_title='Intensidad transmitida (partículas/s·cm²)',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            if escala_log and I0 > 0:
                fig.update_layout(yaxis_type="log")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar información adicional
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Coeficiente de atenuación μ", f"{mu:.4f} cm⁻¹")
            
            with col2:
                hvl, tvl = calcular_capas_hvl_tvl(mu)
                st.metric("Capa de medio valor (HVL)", f"{hvl:.2f} cm")
            
            with col3:
                st.metric("Capa de décimo valor (TVL)", f"{tvl:.2f} cm")
            
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
                    key="espesor_calc"
                )
            
            with col_calc2:
                if espesor_calc >= 0:
                    I_calc = calcular_atenuacion(I0, mu, espesor_calc)
                    atenuacion = (1 - I_calc/I0) * 100 if I0 > 0 else 0
                    st.metric("Intensidad transmitida", f"{I_calc:.2e}")
                    st.metric("% Atenuación", f"{atenuacion:.2f}%")
    
    with tab3:
        st.header("Comparación de Materiales de Blindaje")
        
        # Selección múltiple de materiales
        materiales_disponibles = ['Plomo', 'Acero', 'Hormigón', 'Agua', 'Wolframio', 'Uranio', 'Hierro', 'Cobre', 'Boro']
        materiales_seleccionados = st.multiselect(
            "Selecciona materiales para comparar:",
            materiales_disponibles,
            default=['Plomo', 'Hormigón', 'Agua', 'Acero']
        )
        
        if materiales_seleccionados:
            # Crear gráfica comparativa
            fig_comparativa = go.Figure()
            
            espesores = np.linspace(0, espesor_max, 300)
            
            # Colores para cada material
            colores = {
                'Plomo': '#A0522D',
                'Acero': '#778899',
                'Hormigón': '#A9A9A9',
                'Agua': '#1E90FF',
                'Wolframio': '#FFD700',
                'Uranio': '#000000',
                'Hierro': '#B0B0B0',
                'Cobre': '#B87333',
                'Boro': '#FFA500'
            }
            
            for material in materiales_seleccionados:
                # Obtener color del elemento
                color = colores.get(material, '#808080')
                
                # Calcular curva
                mu = obtener_coeficiente_atenuacion(material, energia, tipo_radiacion)
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
                title=f'Comparación de atenuación para radiación {tipo_radiacion} ({energia} {unidad})',
                xaxis_title='Espesor (cm)',
                yaxis_title='Intensidad transmitida',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            if escala_log and I0 > 0:
                fig_comparativa.update_layout(yaxis_type="log")
            
            st.plotly_chart(fig_comparativa, use_container_width=True)
            
            # Tabla comparativa
            st.subheader("Tabla comparativa")
            
            datos_comparacion = []
            for material in materiales_seleccionados:
                mu = obtener_coeficiente_atenuacion(material, energia, tipo_radiacion)
                hvl, tvl = calcular_capas_hvl_tvl(mu)
                
                # Buscar densidad en la tabla periódica
                df_elem = generar_tabla_periodica()
                densidad_row = df_elem[df_elem['Nombre'] == material]
                if not densidad_row.empty:
                    densidad = densidad_row['Densidad'].iloc[0]
                    efectividad = densidad_row['Blindaje'].iloc[0]
                else:
                    densidad = 0
                    efectividad = "N/A"
                
                datos_comparacion.append({
                    'Material': material,
                    'μ (cm⁻¹)': f"{mu:.4f}",
                    'HVL (cm)': f"{hvl:.2f}",
                    'TVL (cm)': f"{tvl:.2f}",
                    'Densidad (g/cm³)': f"{densidad:.2f}",
                    'Efectividad': efectividad
                })
            
            st.dataframe(pd.DataFrame(datos_comparacion), use_container_width=True)
    
    with tab4:
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
            """)
            
            st.subheader("⚖️ Capa de medio valor (HVL)")
            st.latex(r"HVL = \frac{\ln(2)}{\mu}")
            st.markdown("Espesor necesario para reducir la intensidad a la mitad.")
            
            st.subheader("🔟 Capa de décimo valor (TVL)")
            st.latex(r"TVL = \frac{\ln(10)}{\mu}")
            st.markdown("Espesor necesario para reducir la intensidad al 10%.")
        
        with col_info2:
            st.subheader("🎯 Efectividad de materiales")
            
            efectividad = {
                "Alta densidad atómica": ["Plomo (Pb)", "Wolframio (W)", "Uranio (U)"],
                "Moderación neutrones": ["Agua (H₂O)", "Grafito (C)", "Hormigón"],
                "Captura neutrones": ["Boro (B)", "Cadmio (Cd)", "Litio (Li)"],
                "Propiedades mixtas": ["Acero", "Hormigón con boro", "Compuestos poliméricos"]
            }
            
            for categoria, materiales in efectividad.items():
                with st.expander(f"📌 {categoria}"):
                    for mat in materiales:
                        st.write(f"- {mat}")
            
            st.subheader("📊 Factores a considerar")
            st.markdown("""
            1. **Tipo de radiación**: Gamma, neutrones, beta, alfa
            2. **Energía**: Afecta significativamente la atenuación
            3. **Densidad del material**: Mayor densidad → Mayor atenuación
            4. **Número atómico (Z)**: Importante para radiación electromagnética
            5. **Costo y disponibilidad**
            6. **Propiedades mecánicas y térmicas**
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
            mu_plomo = 0.776  # cm⁻¹ para 1.25 MeV
            x_necesario = -np.log(100/1e6) / mu_plomo
            
            st.markdown(f"""
            - μ para Plomo a 1.25 MeV ≈ {mu_plomo} cm⁻¹
            - Cálculo: x = -1/{mu_plomo:.3f} · ln(100/10⁶)
            - **Resultado: x ≈ {x_necesario:.2f} cm de plomo**
            """)
    
    # Footer
    st.divider()
    st.caption("""
    **Simulador desarrollado para el trabajo de Física Nuclear** |
    Protección Radiológica y Sistemas de Blindaje Avanzado |
    Los valores de coeficientes de atenuación son aproximados basados en datos de NIST XCOM
    """)

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()
