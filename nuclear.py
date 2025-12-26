"""
DASHBOARD INTERACTIVO DE PROTECCIÓN RADIOLÓGICA
Streamlit app para simulación de blindaje - VERSIÓN FINAL
Autor: Estudiante de Física Nuclear
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Blindaje Radiológico",
    page_icon="☢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FUNCIONES DE CÁLCULO - MODELOS CIENTÍFICAMENTE CORRECTOS
# ============================================================================

def calcular_atenuacion_fotones(I0, mu, x):
    """Ley de atenuación exponencial - VÁLIDA SOLO PARA FOTONES"""
    return I0 * np.exp(-mu * x)

def calcular_atenuacion_beta(I0, energia_mev, densidad_material, x):
    """
    Modelo simplificado para partículas beta - alcance máximo
    """
    if energia_mev <= 0:
        return I0
    
    # Alcance aproximado en g/cm²
    if energia_mev < 0.8:
        alcance_gcm2 = 0.15 * energia_mev ** 1.5
    else:
        alcance_gcm2 = 0.5 * energia_mev
    
    # Convertir espesor x (cm) a espesor másico (g/cm²)
    espesor_masico = x * densidad_material
    
    # Si el espesor es mayor que el alcance, intensidad = 0
    if espesor_masico >= alcance_gcm2:
        return 0.0
    
    # Modelo simplificado
    fraccion_atenuada = espesor_masico / alcance_gcm2
    return I0 * (1 - fraccion_atenuada ** 2)

def calcular_atenuacion_neutrones(I0, sigma_total, densidad_atomica, x):
    """
    Modelo para neutrones - atenuación exponencial con sección eficaz
    """
    sigma_cm2 = sigma_total * 1e-24  # barns a cm²
    N = densidad_atomica
    return I0 * np.exp(-N * sigma_cm2 * x)

def calcular_atenuacion_alfa(I0, energia_mev, densidad_material, x):
    """
    Modelo para partículas alfa - solo pérdida de energía, no atenuación real
    Las partículas alfa rara vez se atenúan, solo pierden energía y se detienen
    """
    if energia_mev <= 0:
        return I0
    
    # Alcance en aire (cm) - fórmula aproximada
    if energia_mev < 4:
        alcance_aire = 0.56 * energia_mev ** 1.5  # Más preciso para bajas energías
    else:
        alcance_aire = 1.24 * energia_mev - 2.62  # Más preciso para altas energías
    
    densidad_aire = 0.001225
    alcance_material = alcance_aire * (densidad_aire / densidad_material)
    
    # Las partículas alfa prácticamente NO se atenúan hasta el final de su alcance
    # Solo consideramos que se detienen completamente al alcanzar el alcance
    if x >= alcance_material:
        return 0.0
    
    # Para x < alcance: prácticamente sin atenuación (pérdidas por ionización, no atenuación)
    # Aproximamos que la intensidad se mantiene constante hasta el alcance
    return I0

def obtener_parametros_material(elemento):
    """Obtiene parámetros físicos del material"""
    materiales = {
        'Aire': {
            'densidad': 0.001225,
            'Z_efectivo': 7.64,  # Promedio ponderado (78% N₂, 21% O₂, 1% Ar)
            'sigma_neutrones': 0.2,  # Baja sección eficaz
            'densidad_atomica': 5.0e19,  # Mucho menor que sólidos
            'Color': '#87CEEB'
        },
        'Plomo': {
            'densidad': 11.34,
            'Z_efectivo': 82,
            'sigma_neutrones': 5.0,
            'densidad_atomica': 3.3e22,
            'Color': '#A0522D'
        },
        'Acero': {
            'densidad': 7.85,
            'Z_efectivo': 26,
            'sigma_neutrones': 3.0,
            'densidad_atomica': 8.5e22,
            'Color': '#778899'
        },
        'Hormigón': {
            'densidad': 2.35,
            'Z_efectivo': 'mix',
            'sigma_neutrones': 8.0,
            'densidad_atomica': 1.0e23,
            'Color': '#A9A9A9'
        },
        'Agua': {
            'densidad': 1.00,
            'Z_efectivo': 'mix',
            'sigma_neutrones': 40.0,
            'densidad_atomica': 3.3e22,
            'Color': '#1E90FF'
        },
        'Wolframio': {
            'densidad': 19.25,
            'Z_efectivo': 74,
            'sigma_neutrones': 4.5,
            'densidad_atomica': 6.3e22,
            'Color': '#FFD700'
        },
        'Uranio': {
            'densidad': 19.10,
            'Z_efectivo': 92,
            'sigma_neutrones': 7.0,
            'densidad_atomica': 4.8e22,
            'Color': '#000000'
        },
        'Boro': {
            'densidad': 2.34,
            'Z_efectivo': 5,
            'sigma_neutrones': 100.0,
            'densidad_atomica': 1.3e23,
            'Color': '#FFA500'
        }
    }
    
    return materiales.get(elemento, {
        'densidad': 2.0,
        'Z_efectivo': 10,
        'sigma_neutrones': 5.0,
        'densidad_atomica': 5e22,
        'Color': '#808080'
    })

def obtener_coeficiente_atenuacion_fotones(elemento, energia_mev, tipo_radiacion):
    """Coeficiente de atenuación para fotones"""
    coeficientes = {       
        'Aire': {
            'Gamma': {0.001: 1.5e-5, 0.01: 1.2e-5, 0.1: 1.8e-4, 0.5: 7.5e-5, 1.0: 7.7e-5, 5.0: 3.5e-5, 10.0: 2.5e-5},
            'Rayos X': {0.001: 1.5e-5, 0.01: 1.2e-5, 0.1: 1.8e-4}
        },
        'Plomo': {
            'Gamma': {0.001: 150.0, 0.01: 26.0, 0.1: 59.7, 0.5: 1.71, 1.0: 0.776, 5.0: 0.202, 10.0: 0.102},
            'Rayos X': {0.001: 150.0, 0.01: 26.0, 0.1: 59.7}
        },
        'Acero': {
            'Gamma': {0.001: 5.8, 0.01: 1.8, 0.1: 2.94, 0.5: 0.653, 1.0: 0.469, 5.0: 0.154, 10.0: 0.095},
            'Rayos X': {0.001: 5.8, 0.01: 1.8, 0.1: 2.94}
        },
        'Hormigón': {
            'Gamma': {0.001: 0.8, 0.01: 0.5, 0.1: 0.385, 0.5: 0.227, 1.0: 0.150, 5.0: 0.064, 10.0: 0.042},
            'Rayos X': {0.001: 0.8, 0.01: 0.5, 0.1: 0.385}
        },
        'Agua': {
            'Gamma': {0.001: 0.4, 0.01: 0.2, 0.1: 0.167, 0.5: 0.096, 1.0: 0.0706, 5.0: 0.030, 10.0: 0.022},
            'Rayos X': {0.001: 0.4, 0.01: 0.2, 0.1: 0.167}
        },
        'Wolframio': {
            'Gamma': {0.001: 80.0, 0.01: 15.0, 0.1: 30.4, 0.5: 1.45, 1.0: 0.648, 5.0: 0.181, 10.0: 0.095},
            'Rayos X': {0.001: 80.0, 0.01: 15.0, 0.1: 30.4}
        },
        'Uranio': {
            'Gamma': {0.001: 220.0, 0.01: 45.0, 0.1: 85.3, 0.5: 2.43, 1.0: 1.091, 5.0: 0.252, 10.0: 0.125},
            'Rayos X': {0.001: 220.0, 0.01: 45.0, 0.1: 85.3}
        },
        'Boro': {
            'Gamma': {0.001: 1.2, 0.01: 0.8, 0.1: 0.5, 0.5: 0.15, 1.0: 0.08, 5.0: 0.02, 10.0: 0.01},
            'Rayos X': {0.001: 1.2, 0.01: 0.8, 0.1: 0.5}
        }
    }
    
    if elemento in coeficientes and tipo_radiacion in coeficientes[elemento]:
        energias = list(coeficientes[elemento][tipo_radiacion].keys())
        valores = list(coeficientes[elemento][tipo_radiacion].values())
        
        energias_ordenadas, valores_ordenados = zip(*sorted(zip(energias, valores)))
        
        if energia_mev < min(energias_ordenadas):
            return valores_ordenados[0]
        elif energia_mev > max(energias_ordenadas):
            return valores_ordenados[-1]
        else:
            log_energias = np.log10(energias_ordenadas)
            log_valores = np.log10(valores_ordenados)
            log_energia = np.log10(energia_mev)
            return 10**np.interp(log_energia, log_energias, log_valores)
    
    return 0.1  # Valor por defecto

def obtener_seccion_eficaz_neutrones(elemento, energia_mev):
    """Sección eficaz para neutrones (barns)"""
    secciones = {
        'Aire': {0.000025: 0.5, 0.001: 0.1, 1.0: 0.05, 10.0: 0.02},
        'Plomo': {0.000025: 0.17, 0.001: 0.3, 1.0: 5.0, 10.0: 3.0},
        'Acero': {0.000025: 2.5, 0.001: 2.8, 1.0: 3.0, 10.0: 2.0},
        'Hormigón': {0.000025: 4.0, 0.001: 5.0, 1.0: 8.0, 10.0: 6.0},
        'Agua': {0.000025: 40.0, 0.001: 20.0, 1.0: 5.0, 10.0: 3.0},
        'Wolframio': {0.000025: 2.0, 0.001: 2.5, 1.0: 4.5, 10.0: 3.0},
        'Uranio': {0.000025: 3.0, 0.001: 4.0, 1.0: 7.0, 10.0: 5.0},
        'Boro': {0.000025: 800.0, 0.001: 100.0, 1.0: 2.0, 10.0: 1.0}
    }
    
    if elemento in secciones:
        energias = list(secciones[elemento].keys())
        valores = list(secciones[elemento].values())
        
        energias_ordenadas, valores_ordenados = zip(*sorted(zip(energias, valores)))
        
        if energia_mev < min(energias_ordenadas):
            return valores_ordenados[0]
        elif energia_mev > max(energias_ordenadas):
            return valores_ordenados[-1]
        else:
            return np.interp(energia_mev, energias_ordenadas, valores_ordenados)
    
    return 5.0

def calcular_atenuacion_general(I0, elemento, energia_mev, tipo_radiacion, x):
    """Función principal que selecciona el modelo correcto"""
    params = obtener_parametros_material(elemento)
    
    if tipo_radiacion in ["Gamma", "Rayos X"]:
        mu = obtener_coeficiente_atenuacion_fotones(elemento, energia_mev, tipo_radiacion)
        return calcular_atenuacion_fotones(I0, mu, x)
    
    elif tipo_radiacion == "Beta":
        return calcular_atenuacion_beta(I0, energia_mev, params['densidad'], x)
    
    elif tipo_radiacion == "Neutrones":
        sigma = obtener_seccion_eficaz_neutrones(elemento, energia_mev)
        return calcular_atenuacion_neutrones(I0, sigma, params['densidad_atomica'], x)
    
    elif tipo_radiacion == "Alfa":
        return calcular_atenuacion_alfa(I0, energia_mev, params['densidad'], x)
    
    else:
        return I0  # Por defecto

def calcular_capas_hvl_tvl(mu):
    """Calcula HVL y TVL - SÓLO VÁLIDO PARA FOTONES"""
    if mu > 0:
        hvl = np.log(2) / mu
        tvl = np.log(10) / mu
        return hvl, tvl
    return 0, 0

def generar_tabla_periodica():
    """Genera DataFrame con información para tabla periódica interactiva"""
    elementos = [
        {'Simbolo': 'Air', 'Nombre': 'Aire', 'Z': 'Mix', 'Grupo': 'Gases',
         'Densidad': 0.001225, 'Color': '#87CEEB', 'Blindaje': 'Muy Bajo'},
        {'Simbolo': 'Pb', 'Nombre': 'Plomo', 'Z': 82, 'Grupo': 'Metales',
         'Densidad': 11.34, 'Color': '#A0522D', 'Blindaje': 'Alto'},
        {'Simbolo': 'W', 'Nombre': 'Wolframio', 'Z': 74, 'Grupo': 'Metales',
         'Densidad': 19.25, 'Color': '#FFD700', 'Blindaje': 'Muy Alto'},
        {'Simbolo': 'U', 'Nombre': 'Uranio', 'Z': 92, 'Grupo': 'Actinidos',
         'Densidad': 19.10, 'Color': '#000000', 'Blindaje': 'Muy Alto'},
        {'Simbolo': 'Ac', 'Nombre': 'Acero', 'Z': 'Mix', 'Grupo': 'Aleaciones',
         'Densidad': 7.85, 'Color': '#778899', 'Blindaje': 'Medio'},
        {'Simbolo': 'Con', 'Nombre': 'Hormigón', 'Z': 'Mix', 'Grupo': 'Compuestos',
         'Densidad': 2.35, 'Color': '#A9A9A9', 'Blindaje': 'Medio'},
        {'Simbolo': 'H2O', 'Nombre': 'Agua', 'Z': 'Mix', 'Grupo': 'Compuestos',
         'Densidad': 1.00, 'Color': '#1E90FF', 'Blindaje': 'Bajo'},
        {'Simbolo': 'B', 'Nombre': 'Boro', 'Z': 5, 'Grupo': 'Metaloides',
         'Densidad': 2.34, 'Color': '#FFA500', 'Blindaje': 'Neutrones'}
    ]
    return pd.DataFrame(elementos)

# ============================================================================
# INTERFAZ STREAMLIT - INTERFAZ ORIGINAL MEJORADA
# ============================================================================

def main():
    # Título principal
    st.title("☢️ Simulador Interactivo de Protección Radiológica")
    st.markdown("""
    ### Modelos científicos correctos para cada tipo de radiación
    *Trabajo de Física Nuclear - Protección Radiológica y Sistemas de Blindaje Avanzado*
    """)

    # Sidebar para controles
    with st.sidebar:
        st.header("⚙️ Parámetros de Simulación")

        # Selección de tipo de radiación (añadido Alfa)
        tipo_radiacion = st.selectbox(
            "Tipo de radiación:",
            ["Gamma", "Rayos X", "Beta", "Neutrones", "Alfa"],
            index=0
        )

        # Información sobre modelos
        with st.expander("📖 Modelo utilizado"):
            if tipo_radiacion in ["Gamma", "Rayos X"]:
                st.info("**Ley exponencial:** I(x) = I₀·e^(-μx)")
            elif tipo_radiacion == "Beta":
                st.info("**Modelo de alcance máximo**")
            elif tipo_radiacion == "Neutrones":
                st.info("**Atenuación por sección eficaz nuclear**")
            elif tipo_radiacion == "Alfa":
                st.info("**Modelo de alcance corto fijo**")

        # ENTRADA DE ENERGÍA FLEXIBLE
        st.markdown("### 🔋 Energía de la radiación")
        
        with st.expander("ℹ️ Rangos típicos"):
            st.markdown("""
            - **Rayos X**: 1-300 keV (diagnóstico)
            - **Gamma**: 0.01-10 MeV
            - **Beta**: 0.1-10 MeV  
            - **Neutrones**: 0.001 eV - 20 MeV
            - **Alfa**: 3-10 MeV
            """)

        # Seleccionar unidad según tipo de radiación
        if tipo_radiacion == "Rayos X":
            unidad = st.radio("Unidad:", ["keV", "MeV"], horizontal=True)
            default_val = 50.0 if unidad == "keV" else 0.05
            min_val = 1.0 if unidad == "keV" else 0.001
            max_val = 300.0 if unidad == "keV" else 0.3
            step_val = 1.0 if unidad == "keV" else 0.001
            format_str = "%.0f" if unidad == "keV" else "%.3f"
        else:
            unidad = "MeV"
            if tipo_radiacion == "Gamma":
                default_val = 1.0
                min_val = 0.001
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
                min_val = 0.000001
                max_val = 20.0
                step_val = 0.000001
                format_str = "%.6f"
            elif tipo_radiacion == "Alfa":
                default_val = 5.0
                min_val = 3.0
                max_val = 10.0
                step_val = 0.1
                format_str = "%.1f"

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

        # Convertir todo a MeV internamente
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

        # Espesor máximo para gráfica (ajustado por tipo de radiación)
        if tipo_radiacion == "Alfa":
            espesor_max = st.slider(
                "Espesor máximo (cm):",
                min_value=0.001,
                max_value=10.0,  # Aumentado de 1.0 a 10.0 cm
                value=1.0,
                step=0.001,
                help="Partículas alfa tienen alcance muy corto (normalmente <10 cm en aire)"
            )
        elif tipo_radiacion == "Beta":
            espesor_max = st.slider(
                "Espesor máximo (cm):",
                min_value=0.1,
                max_value=2000.0,  # Aumentado de 10.0 a 2000.0 cm (20 m)
                value=100.0,
                step=1.0,
                help="Partículas beta pueden viajar metros en aire"
            )
        elif tipo_radiacion == "Neutrones":
            espesor_max = st.slider(
                "Espesor máximo (cm):",
                min_value=1.0,
                max_value=10000.0,  # Aumentado a 100 m
                value=1000.0,
                step=10.0,
                help="Neutrones requieren grandes espesores para atenuación significativa"
            )
        else:  # Gamma y Rayos X
            espesor_max = st.slider(
                "Espesor máximo (cm):",
                min_value=1.0,
                max_value=5000.0,  # Aumentado a 50 m
                value=500.0,
                step=10.0,
                help="Fotones requieren espesores considerables para atenuación completa"
            )

        st.divider()
        st.header("📊 Opciones de Visualización")
        
        # Mostrar HVL/TVL para fotones y neutrones (pero con nombres adecuados)
        if tipo_radiacion in ["Gamma", "Rayos X", "Neutrones"]:
            if tipo_radiacion == "Neutrones":
                etiqueta_hvl = "Mostrar HVL equivalente"
                etiqueta_tvl = "Mostrar TVL equivalente"
                ayuda_hvl = "Capa de medio valor equivalente para neutrones (basado en sección eficaz)"
                ayuda_tvl = "Capa de décimo valor equivalente para neutrones (basado en sección eficaz)"
            else:
                etiqueta_hvl = "Mostrar capa de medio valor (HVL)"
                etiqueta_tvl = "Mostrar capa de décimo valor (TVL)"
                ayuda_hvl = "Ley exponencial exacta para fotones"
                ayuda_tvl = "Ley exponencial exacta para fotones"
            
            mostrar_hvl = st.checkbox(etiqueta_hvl, value=True, help=ayuda_hvl)
            mostrar_tvl = st.checkbox(etiqueta_tvl, value=True, help=ayuda_tvl)
        else:
            mostrar_hvl = False
            mostrar_tvl = False
            st.info("HVL/TVL solo aplican a fotones y su equivalene solo en neutrones")
        
        escala_log = st.checkbox("Escala logarítmica en Y", value=False)

    # Contenido principal en pestañas
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Inicio y Explicación",
        "🎯 Tabla Periódica Interactiva", 
        "🔍 Comparación de Materiales",
        "📚 Información Teórica"
    ])

    with tab1:
        st.header("🏠 Bienvenido al Simulador de Blindaje Radiológico")
        
        st.markdown("""
        ## 📋 ¿Qué puedes hacer con esta aplicación?
        
        Esta herramienta interactiva te permite simular la atenuación de diferentes tipos 
        de radiación a través de diversos materiales de blindaje, aplicando **modelos físicamente correctos** 
        para cada tipo de radiación.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 **Funcionalidades principales:**")
            st.markdown("""
            1. **Tabla Periódica Interactiva**
               - Selecciona elementos y materiales de blindaje
               - Visualiza propiedades clave (densidad, efectividad)
               - Gráficas automáticas al seleccionar
            
            2. **Simulación de Atenuación**
               - Modelos correctos para cada tipo de radiación
               - Para fotones: Ley exponencial con HVL/TVL
               - Para otras radiaciones: Modelos específicos
            
            3. **Comparación de Materiales**
               - Compara múltiples materiales simultáneamente
               - Análisis de efectividad relativa
            
            4. **Información Teórica**
               - Fundamentos físicos de la atenuación
               - Explicación de modelos matemáticos
            """)
        
        with col2:
            st.subheader("⚙️ **Cómo usar la aplicación:**")
            st.markdown("""
            ### Paso 1: Configura los parámetros
            - Usa la barra lateral para seleccionar:
              - **Tipo de radiación** (Gamma, Beta, Neutrones, Rayos X, Alfa)
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
        
        st.subheader("📊 **Modelos científicos implementados:**")
        
        datos_modelos = {
            "Tipo de radiación": ["Fotones (Gamma/Rayos X)", "Partículas Beta", "Neutrones", "Partículas Alfa"],
            "Modelo físico": [
                "Ley de atenuación exponencial: I(x) = I₀·e^(-μx)",
                "Modelo de alcance máximo (range)",
                "Atenuación por sección eficaz nuclear",
                "Modelo de alcance corto fijo"
            ],
            "Parámetros clave": [
                "μ (coeficiente de atenuación), HVL, TVL",
                "Energía máxima, densidad del material",
                "Sección eficaz σ, densidad atómica",
                "Energía, densidad del material"
            ]
        }
        
        st.dataframe(pd.DataFrame(datos_modelos), width='stretch')
        
        st.warning("""
        ⚠️ **Importante científico:** 
        - La ley exponencial **solo es válida** para fotones (Rayos X y Gamma)
        - Para otras radiaciones se utilizan modelos físicos específicos
        - Esta aplicación usa modelos simplificados para fines educativos
        """)

    with tab2:
        st.header("Tabla Periódica para Blindaje Radiológico")
    
        # Generar tabla periódica
        df_elementos = generar_tabla_periodica()
    
        # Mostrar tabla periódica como cuadrícula interactiva
        cols = st.columns(7)
    
        for idx, row in df_elementos.iterrows():
            col_idx = idx % 7
            with cols[col_idx]:
                # Botón para cada elemento con color personalizado
                if st.button(
                    f"**{row['Simbolo']}**\n{row['Nombre']}",
                    key=f"elem_{row['Simbolo']}",
                    help=f"Z={row['Z']}, ρ={row['Densidad']} g/cm³",
                ):
                    # Almacenar elemento seleccionado en session state
                    st.session_state['elemento_seleccionado'] = row['Simbolo']
    
                # Información adicional en tooltip
                st.caption(f"ρ={row['Densidad']} g/cm³")
    
        st.divider()
    
        # Si hay elemento seleccionado, mostrar gráfica y controles AUTOMÁTICAMENTE
        if 'elemento_seleccionado' in st.session_state:
            elem = st.session_state['elemento_seleccionado']
            info = df_elementos[df_elementos['Simbolo'] == elem].iloc[0]
            nombre_elemento = info['Nombre']
            color_elemento = info['Color']
            
            st.subheader(f"Simulación para {nombre_elemento}")
            
            # ============================================
            # TODA LA INFORMACIÓN UNIFICADA ANTES DE LA GRÁFICA
            # ============================================
            
            # Fila 1: Información básica del material
            col_fila1_1, col_fila1_2, col_fila1_3 = st.columns(3)
            
            with col_fila1_1:
                st.metric("Elemento", nombre_elemento)
                st.metric("Densidad", f"{info['Densidad']} g/cm³")
            
            with col_fila1_2:
                st.metric("Grupo", info['Grupo'])
                st.metric("Efectividad", info['Blindaje'])
            
            with col_fila1_3:
                params = obtener_parametros_material(nombre_elemento)
                
                if tipo_radiacion in ["Gamma", "Rayos X"]:
                    mu = obtener_coeficiente_atenuacion_fotones(nombre_elemento, energia_mev, tipo_radiacion)
                    hvl, tvl = calcular_capas_hvl_tvl(mu)
                    st.metric("Coeficiente μ", f"{mu:.4f} cm⁻¹")

                elif tipo_radiacion == "Beta":
                    if energia_mev < 0.8:
                        alcance_gcm2 = 0.15 * energia_mev ** 1.5
                    else:
                        alcance_gcm2 = 0.5 * energia_mev
                    alcance_cm = alcance_gcm2 / params['densidad']
                    st.metric("Alcance total", f"{alcance_cm:.2f} cm")
                    st.metric("Energía", f"{energia_mev:.2f} MeV")
                elif tipo_radiacion == "Alfa":
                    alcance_aire = 0.3 * energia_mev ** 1.5
                    alcance_material = alcance_aire * (0.001225 / params['densidad'])
                    st.metric("Alcance total", f"{alcance_material*1000:.1f} mm")
                    st.metric("Energía", f"{energia_mev:.2f} MeV")
                elif tipo_radiacion == "Neutrones":
                    sigma = obtener_seccion_eficaz_neutrones(nombre_elemento, energia_mev)
                    st.metric("Sección eficaz σ", f"{sigma:.1f} barns")
                    st.metric("Long. atenuación", f"{1/(params['densidad_atomica']*sigma*1e-24):.1f} cm")
            
            # Divider entre información y controles
            st.divider()
            
            # Fila 2: Parámetros de simulación y resultados (AHORA CON 3 COLUMNAS)
            col_fila2_1, col_fila2_2 = st.columns(2)
            
            with col_fila2_1:
                st.markdown("#### ⚙️ Parámetros entrada")
                st.metric("Energía", energia_display)
                st.metric("Intensidad inicial (I₀)", f"{I0:.2e}")

                # Mostrar sección eficaz solo para neutrones
                if tipo_radiacion == "Neutrones":
                    sigma = obtener_seccion_eficaz_neutrones(nombre_elemento, energia_mev)
                    st.metric("Sección eficaz σ", f"{sigma:.1f} barns")
                    st.caption("1 barn = 10⁻²⁴ cm²")
            
            with col_fila2_2:
                st.markdown("#### 📊 Resultados principales")
                # Esta información se actualizará después con el slider
                if tipo_radiacion in ["Gamma", "Rayos X"]:
                    mu = obtener_coeficiente_atenuacion_fotones(nombre_elemento, energia_mev, tipo_radiacion)
                    hvl, tvl = calcular_capas_hvl_tvl(mu)
                    st.metric("HVL", f"{hvl:.2f} cm")
                    st.metric("TVL", f"{tvl:.2f} cm")
                elif tipo_radiacion == "Beta":
                    if energia_mev < 0.8:
                        alcance_gcm2 = 0.15 * energia_mev ** 1.5
                    else:
                        alcance_gcm2 = 0.5 * energia_mev
                    alcance_cm = alcance_gcm2 / params['densidad']
                    st.metric("Alcance total", f"{alcance_cm:.2f} cm")
                elif tipo_radiacion == "Alfa":
                    alcance_aire = 0.3 * energia_mev ** 1.5
                    alcance_material = alcance_aire * (0.001225 / params['densidad'])
                    st.metric("Alcance total", f"{alcance_material*1000:.1f} mm")
                elif tipo_radiacion == "Neutrones":
                    sigma = obtener_seccion_eficaz_neutrones(nombre_elemento, energia_mev)
                    sigma_macroscopica = params['densidad_atomica'] * sigma * 1e-24  # cm⁻¹
                    
                    if sigma_macroscopica > 0:
                        # Calcular HVL y TVL equivalentes
                        hvl = np.log(2) / sigma_macroscopica
                        tvl = np.log(10) / sigma_macroscopica
                        
                        st.metric("HVL (equivalente)", f"{hvl:.1f} cm")
                        st.metric("TVL (equivalente)", f"{tvl:.1f} cm")
                        
                        # Aclaración como caption
                        st.caption("⚠️ HVL/TVL 'equivalentes' - σ varía con energía")
                    else:
                        st.metric("HVL (equivalente)", "∞ cm")
                        st.metric("TVL (equivalente)", "∞ cm")
            
            # ============================================
            # SLIDER DEL ESPESOR - AHORA SOLO Y CENTRADO
            # ============================================
            st.divider()
            
            # Espacio dedicado para el slider
            st.markdown(f"### 🎚️ Control de espesor para {nombre_elemento}")
            
            # Crear un contenedor centrado para el slider
            slider_container = st.container()
            
            with slider_container:
                # Valor inicial como porcentaje del máximo (1% para alfa, 5% para otros)
                porcentaje_inicial = 0.01 if tipo_radiacion == "Alfa" else 0.05
                espesor_default = min(float(espesor_max) * porcentaje_inicial, float(espesor_max))
                
                # Crear dos columnas: una para el slider y otra para el porcentaje
                col_slider, col_percent = st.columns([3, 1])
                
                with col_slider:
                    # Slider para espesor
                    espesor = st.slider(
                        f"**Espesor de {nombre_elemento} (cm):**",
                        min_value=0.0,
                        max_value=float(espesor_max),
                        value=espesor_default,
                        step=0.001 if tipo_radiacion == "Alfa" else 0.5,
                        key=f"espesor_{elem}"
                    )
                
                with col_percent:
                    # Calcular y mostrar solo el porcentaje de atenuación
                    I_final = calcular_atenuacion_general(I0, nombre_elemento, energia_mev, tipo_radiacion, espesor)
                    atenuacion = (1 - I_final/I0) * 100 if I0 > 0 else 0
                    st.metric("**Atenuación**", f"{atenuacion:.1f}%")
            
            # Divider antes de la gráfica
            st.divider()
            
            # ============================================
            # GRÁFICA DESPUÉS DEL SLIDER
            # ============================================
            
            # Calcular curva de atenuación para la gráfica
            espesores_grafica = np.linspace(0, espesor_max, 500)
            intensidades_grafica = [calcular_atenuacion_general(I0, nombre_elemento, energia_mev, tipo_radiacion, x) for x in espesores_grafica]
            
            # Crear gráfica con Plotly
            fig = go.Figure()
            
            # Curva principal
            fig.add_trace(go.Scatter(
                x=espesores_grafica,
                y=intensidades_grafica,
                mode='lines',
                name=f'{nombre_elemento}',
                line=dict(color=color_elemento, width=3),
                hovertemplate="Espesor: %{x:.3f} cm<br>Intensidad: %{y:.2e}<extra></extra>"
            ))
            
            # Línea vertical para el espesor seleccionado
            fig.add_vline(
                x=espesor,
                line_dash="solid",
                line_color="green",
                line_width=2,
                annotation_text=f"Espesor seleccionado: {espesor:.3f} cm",
                annotation_position="top left"
            )
            
            # Punto en la curva para el espesor seleccionado
            fig.add_trace(go.Scatter(
                x=[espesor],
                y=[I_final],
                mode='markers',
                name=f'I = {I_final:.2e}',
                marker=dict(size=12, color='green'),
                hovertemplate=f"Espesor: {espesor:.3f} cm<br>Intensidad: {I_final:.2e}<extra></extra>"
            ))
            
            # Líneas de HVL y TVL (para fotones y neutrones)
            if mostrar_hvl:
                if tipo_radiacion in ["Gamma", "Rayos X"]:
                    mu = obtener_coeficiente_atenuacion_fotones(nombre_elemento, energia_mev, tipo_radiacion)
                    hvl, _ = calcular_capas_hvl_tvl(mu)
                elif tipo_radiacion == "Neutrones":
                    sigma = obtener_seccion_eficaz_neutrones(nombre_elemento, energia_mev)
                    sigma_macroscopica = params['densidad_atomica'] * sigma * 1e-24
                    hvl = np.log(2) / sigma_macroscopica if sigma_macroscopica > 0 else 0
                else:
                    hvl = 0
                
                if hvl > 0 and hvl <= espesor_max:
                    fig.add_vline(
                        x=hvl,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"HVL{' (eq)' if tipo_radiacion=='Neutrones' else ''} = {hvl:.2f} cm",
                        annotation_position="top right"
                    )
            
            if mostrar_tvl:
                if tipo_radiacion in ["Gamma", "Rayos X"]:
                    mu = obtener_coeficiente_atenuacion_fotones(nombre_elemento, energia_mev, tipo_radiacion)
                    _, tvl = calcular_capas_hvl_tvl(mu)
                elif tipo_radiacion == "Neutrones":
                    sigma = obtener_seccion_eficaz_neutrones(nombre_elemento, energia_mev)
                    sigma_macroscopica = params['densidad_atomica'] * sigma * 1e-24
                    tvl = np.log(10) / sigma_macroscopica if sigma_macroscopica > 0 else 0
                else:
                    tvl = 0
                
                if tvl > 0 and tvl <= espesor_max:
                    fig.add_vline(
                        x=tvl,
                        line_dash="dot",
                        line_color="blue",
                        annotation_text=f"TVL{' (eq)' if tipo_radiacion=='Neutrones' else ''} = {tvl:.2f} cm",
                        annotation_position="top right"
                    )
            
            # Configurar layout
            fig.update_layout(
                title=f'📈 Gráfica de atenuación: {tipo_radiacion} ({energia_display}) en {nombre_elemento}',
                xaxis_title='Espesor del blindaje (cm)',
                yaxis_title='Intensidad transmitida (partículas/s·cm²)',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            if escala_log:
                fig.update_yaxes(type="log", exponentformat='power')
            
            st.plotly_chart(fig, width='stretch')

    with tab3:
        st.header("Comparación de Materiales de Blindaje")
        
        # Generar tabla periódica
        df_elementos = generar_tabla_periodica()
        
        # Selección múltiple de materiales
        materiales_seleccionados = st.multiselect(
            "Selecciona materiales para comparar:",
            df_elementos['Nombre'].tolist(),
            default=['Aire','Plomo', 'Acero', 'Hormigón', 'Agua']
        )

        if materiales_seleccionados:
            # Crear gráfica comparativa
            fig_comparativa = go.Figure()

            espesores = np.linspace(0, espesor_max, 300)

            for material in materiales_seleccionados:
                # Obtener color del elemento
                color = df_elementos[df_elementos['Nombre'] == material]['Color'].iloc[0]
                
                # Calcular curva para este material
                intensidades = [calcular_atenuacion_general(I0, material, energia_mev, tipo_radiacion, x) for x in espesores]
                
                # Información adicional para el tooltip
                params = obtener_parametros_material(material)
                info_extra = f"Densidad: {params['densidad']} g/cm³"
                
                if tipo_radiacion in ["Gamma", "Rayos X"]:
                    mu = obtener_coeficiente_atenuacion_fotones(material, energia_mev, tipo_radiacion)
                    info_extra += f"<br>μ={mu:.3f} cm⁻¹"
                elif tipo_radiacion == "Neutrones":
                    sigma = obtener_seccion_eficaz_neutrones(material, energia_mev)
                    info_extra += f"<br>σ={sigma:.1f} barns"

                fig_comparativa.add_trace(go.Scatter(
                    x=espesores,
                    y=intensidades,
                    mode='lines',
                    name=material,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{material}<br>{info_extra}<br>Espesor: %{{x:.3f}} cm → Intensidad: %{{y:.2e}}<extra></extra>"
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
                fig_comparativa.update_yaxes(type="log", exponentformat='power')

            st.plotly_chart(fig_comparativa, width='stretch')

            # Tabla comparativa
            st.subheader("📋 Tabla comparativa")

            datos_comparacion = []
            for material in materiales_seleccionados:
                params = obtener_parametros_material(material)
                
                # Calcular atenuación a espesor máximo
                I_final = calcular_atenuacion_general(I0, material, energia_mev, tipo_radiacion, espesor_max)
                atenuacion = (1 - I_final/I0) * 100 if I0 > 0 else 0
                
                # Información específica por tipo de radiación
                if tipo_radiacion in ["Gamma", "Rayos X"]:
                    mu = obtener_coeficiente_atenuacion_fotones(material, energia_mev, tipo_radiacion)
                    hvl, tvl = calcular_capas_hvl_tvl(mu)
                    info_especifico = f"μ={mu:.3f} cm⁻¹, HVL={hvl:.2f} cm"
                elif tipo_radiacion == "Beta":
                    if energia_mev < 0.8:
                        alcance_gcm2 = 0.15 * energia_mev ** 1.5
                    else:
                        alcance_gcm2 = 0.5 * energia_mev
                    alcance_cm = alcance_gcm2 / params['densidad']
                    info_especifico = f"Alcance≈{alcance_cm:.2f} cm"
                elif tipo_radiacion == "Neutrones":
                    sigma = obtener_seccion_eficaz_neutrones(material, energia_mev)
                    info_especifico = f"σ={sigma:.1f} barns"
                elif tipo_radiacion == "Alfa":
                    alcance_aire = 0.3 * energia_mev ** 1.5
                    alcance_material = alcance_aire * (0.001225 / params['densidad'])
                    info_especifico = f"Alcance≈{alcance_material*1000:.1f} mm"
                else:
                    info_especifico = "-"

                datos_comparacion.append({
                    'Material': material,
                    'Densidad (g/cm³)': params['densidad'],
                    'Aten. a {espesor_max}cm': f"{atenuacion:.1f}%",
                    'Info específica': info_especifico
                })

            df_comparacion = pd.DataFrame(datos_comparacion)
            st.dataframe(df_comparacion, width='stretch')

    with tab4:
        st.header("📚 Detalles de los Modelos Matemáticos")
        
        col_mod1, col_mod2 = st.columns(2)
        
        with col_mod1:
            st.subheader("1. Fotones (Gamma/Rayos X)")
            st.latex(r"I(x) = I_0 \cdot e^{-\mu \cdot x}")
            st.markdown("""
            Donde:
            - μ = coeficiente de atenuación lineal [cm⁻¹]
            - Depende de: Z (número atómico), ρ (densidad), E (energía)
            - HVL = ln(2)/μ, TVL = ln(10)/μ
            """)
            
            st.subheader("2. Partículas Beta")
            st.latex(r"R \approx 0.5 \cdot E_{\text{max}} \quad (\text{g/cm}^2)")
            st.markdown("""
            - R = alcance másico [g/cm²]
            - E_max = energía máxima [MeV]
            - En material: R_material = R / ρ
            - Modelo simplificado: I(x) = 0 si x ≥ R_material
            """)
        
        with col_mod2:
            st.subheader("3. Neutrones")
            st.latex(r"I(x) = I_0 \cdot e^{-N \cdot \sigma \cdot x}")
            st.markdown("""
            Donde:
            - N = densidad atómica [átomos/cm³]
            - σ = sección eficaz total [cm²]
            - σ varía mucho con energía (resonancias)
            - 1 barn = 10⁻²⁴ cm²
            """)
            
            st.subheader("4. Partículas Alfa")
            st.latex(r"R_{\text{aire}} \approx 0.3 \cdot E^{3/2} \quad (\text{cm})")
            st.markdown("""
            - R_aire = alcance en aire [cm]
            - En otros materiales: R_material = R_aire · (ρ_aire/ρ_material)
            - Atenuación casi completa al alcanzar R
            """)
        
        st.divider()
        
        st.subheader("⚠️ Limitaciones y Simplificaciones")
        st.markdown("""
        1. **Modelos reales son más complejos:** 
           - Betas: Curva de Bragg (pico de Bragg)
           - Neutrones: Moderación, secciones eficaces dependientes de energía
           - Alfa: Pérdida de energía por Bethe-Bloch
        
        2. **Esta simulación usa modelos simplificados** para fines educativos
        
        3. **Para cálculos precisos:** Usar códigos Monte Carlo (MCNP, Geant4)
        
        4. **Considerar siempre:**
           - Radiación secundaria (frenado, rayos X característicos)
           - Dispersión múltiple
           - Activación del material de blindaje
        """)

        st.subheader("ℹ️ Nota sobre neutrones")
        st.markdown("""
        Para neutrones, el concepto de **HVL y TVL es 'equivalente'** porque:
        
        1. **σ varía con energía**: La sección eficaz nuclear cambia drásticamente
        2. **Moderación**: Los neutrones pierden energía en colisiones
        3. **Dispersión múltiple**: No es un simple camino directo
        
        En esta simulación usamos:  
        **HVL(eq) = ln(2)/Σ** y **TVL(eq) = ln(10)/Σ**  
        donde **Σ = N·σ** (sección eficaz macroscópica)
        """)

if __name__ == "__main__":
    if 'elemento_seleccionado' not in st.session_state:
        st.session_state['elemento_seleccionado'] = 'Air'
    main()
