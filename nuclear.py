"""
DASHBOARD INTERACTIVO DE PROTECCIÓN RADIOLÓGICA
Streamlit app para simulación de blindaje
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
    
    # Alcance aproximado en g/cm² (fórmula simplificada común)
    # Para E > 0.8 MeV: R ≈ 0.5·E_max g/cm²
    # Para E < 0.8 MeV: R ≈ 0.15·E_max^{1.5} g/cm²
    if energia_mev < 0.8:
        alcance_gcm2 = 0.407 * energia_mev ** 1.38
    else:
        alcance_gcm2 = 0.542 * energia_mev - 0.133  # E_max en MeV
    
    # Convertir espesor x (cm) a espesor másico (g/cm²)
    espesor_masico = x * densidad_material
    
    # Si el espesor es mayor que el alcance, intensidad = 0
    if espesor_masico >= alcance_gcm2:
        return 0.0
    
    # Modelo simplificado de atenuación gradual
    # Podría mejorarse con una curva más realista, pero para fines educativos está bien
    fraccion_atenuada = espesor_masico / alcance_gcm2
    return I0 * (1 - fraccion_atenuada ** 2)

def calcular_atenuacion_neutrones(I0, sigma_total, densidad_atomica, x):
    """
    Modelo para neutrones - atenuación exponencial con sección eficaz
    """
    sigma_cm2 = sigma_total * 1e-24  # barns a cm²
    N = densidad_atomica
    return I0 * np.exp(-N * sigma_cm2 * x)

def calcular_alcance_alfa(energia_mev, material_nombre):
    """
    Calcula el alcance de partículas alfa en un material usando la regla de Bragg-Kleeman.

    Parámetros:
    - energia_mev: Energía de la partícula alfa en MeV.
    - material_nombre: Nombre del material (debe coincidir con las claves en obtener_parametros_material).

    Retorna:
    - alcance_material: Alcance lineal en el material (cm).

    Modelo:
    1. Alcance en aire de referencia: R_aire = 0.3 * E^(1.5) cm
    2. Corrección de Bragg-Kleeman para otros materiales:
       R_material = R_aire * (ρ_aire/ρ_material) * √(A_eff_material / A_eff_aire)

    Referencia para A_effective: A_eff = ( Σ (w_i / √A_i) )⁻¹
    (Analysis of ionizing charged-particle shielding and range, Eur. Phys. J. Plus, 2025)
    """
    # 1. Parámetros del aire (material de referencia)
    densidad_aire = 0.001225          # g/cm³
    masa_eff_aire = 3.82              # Valor A_effective para aire (calculado previamente)

    # 2. Obtener parámetros del material de blindaje
    params_material = obtener_parametros_material(material_nombre)
    densidad_material = params_material['densidad']
    masa_eff_material = params_material['masa_atomica_efectiva']

    # 3. Calcular alcance en aire (fórmula simplificada para fines educativos)
    # Nota: Para mayor precisión, se podrían usar fórmulas empíricas más complejas.
    if energia_mev <= 0:
        return 0.0
    alcance_aire_cm = 0.3 * (energia_mev ** 1.5)

    # 4. Aplicar la regla de Bragg-Kleeman para escalar al material
    factor_densidad = densidad_aire / densidad_material
    factor_masa = (masa_eff_material / masa_eff_aire) ** 0.5  # Raíz cuadrada
    alcance_material_cm = alcance_aire_cm * factor_densidad * factor_masa

    return alcance_material_cm


def calcular_atenuacion_alfa(I0, energia_mev, material_nombre, espesor_cm):
    """
    Calcula la intensidad transmitida de partículas alfa a través de un material.

    Parámetros:
    - I0: Intensidad inicial (partículas/s·cm²).
    - energia_mev: Energía de la partícula alfa en MeV.
    - material_nombre: Nombre del material del blindaje.
    - espesor_cm: Espesor del material (cm).

    Retorna:
    - I_transmitida: Intensidad transmitida (partículas/s·cm²).

    Modelo físico:
    Las partículas alfa tienen un alcance definido. No se atenúan gradualmente.
    - Si el espesor es MAYOR O IGUAL que el alcance total: intensidad transmitida = 0.
    - Si el espesor es MENOR que el alcance total: intensidad transmitida = I0 (sin atenuación).
    """
    # 1. Calcular el alcance total en el material
    alcance_total_cm = calcular_alcance_alfa(energia_mev, material_nombre)

    # 2. Aplicar el modelo de alcance fijo (sin atenuación gradual)
    if espesor_cm >= alcance_total_cm:
        return 0.0
    else:
        return I0

def obtener_parametros_material(elemento):
    """Obtiene parámetros físicos del material"""
    materiales = {
        'Aire': {
            'densidad': 0.001225,
            'masa_atomica_efectiva': 3.82,  # NUEVO: 1/(0.755/√14 + 0.232/√16 + 0.013/√40)
            'Z_efectivo': 7.64,  # Promedio ponderado (78% N₂, 21% O₂, 1% Ar)
            'sigma_neutrones': 0.2,  # Baja sección eficaz
            'densidad_atomica': 5.0e19,  # Mucho menor que sólidos
            'Color': '#87CEEB'
        },
        'Plomo': {
            'densidad': 11.34,
            'Z_efectivo': 82,
            'masa_atomica_efectiva': 207.2,  # Elemento puro: A_effective = A
            'sigma_neutrones': 5.0,
            'densidad_atomica': 3.3e22,
            'Color': '#A0522D'
        },
        'Acero': {
            'densidad': 7.85,
            'masa_atomica_efectiva': 7.43,  # NUEVO: 1/(0.995/√55.85 + 0.005/√12.01)
            'Z_efectivo': 26,
            'sigma_neutrones': 3.0,
            'densidad_atomica': 8.5e22,
            'Color': '#778899'
        },
        'Hormigón': {
            'densidad': 2.35,
            'masa_atomica_efectiva': 4.51,  # NUEVO: 1/(0.5/√16 + 0.25/√28.09 + 0.1/√40.08 + 0.15/√20)
            'Z_efectivo': 'mix',
            'sigma_neutrones': 8.0,
            'densidad_atomica': 1.0e23,
            'Color': '#A9A9A9'
        },
        'Agua': {
            'densidad': 1.00,
            'masa_atomica_efectiva': 3.00,  # NUEVO: 1/(0.1119/√1.008 + 0.8881/√16)
            'Z_efectivo': 'mix',
            'sigma_neutrones': 40.0,
            'densidad_atomica': 3.3e22,
            'Color': '#1E90FF'
        },
        'Wolframio': {
            'densidad': 19.25,
            'masa_atomica_efectiva': 183.84,
            'Z_efectivo': 74,
            'sigma_neutrones': 4.5,
            'densidad_atomica': 6.3e22,
            'Color': '#FFD700'
        },
        'Boro': {
            'densidad': 2.34,
            'masa_atomica_efectiva': 10.81,
            'Z_efectivo': 5,
            'sigma_neutrones': 100.0,
            'densidad_atomica': 1.3e23,
            'Color': '#FFA500'
        }
    }
    
    return materiales.get(elemento, {
        'densidad': 2.0,
        'masa_atomica_efectiva': 10.0,
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
        return calcular_atenuacion_alfa(I0, energia_mev, elemento, x)
    
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
    ### Trabajo de Física Nuclear - Protección Radiológica y Sistemas de Blindaje Avanzado
    *Hugo Neira, Alejandro González, David López y David Moyano*
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
            
            # VALORES POR DEFECTO Y PASOS SIMPLES
            if unidad == "keV":
                default_val = 50.0
                step_val = 1.0  # Saltos de 10 keV
                format_str = "%.0f"  # Sin decimales
            else:  # MeV
                default_val = 0.05
                step_val = 0.01  # Saltos de 0.01 MeV
                format_str = "%.3f"  # 3 decimales
        else:
            unidad = "MeV"
            if tipo_radiacion == "Gamma":
                default_val = 1.0
                step_val = 0.1  # Saltos de 0.1 MeV
                format_str = "%.2f"  # 2 decimales
            elif tipo_radiacion == "Beta":
                default_val = 2.0
                step_val = 0.1  # Saltos de 0.1 MeV
                format_str = "%.2f"  # 2 decimales
            elif tipo_radiacion == "Neutrones":
                default_val = 1.0
                step_val = 0.1  # Saltos de 0.1 MeV
                format_str = "%.2f"  # 2 decimales
            elif tipo_radiacion == "Alfa":
                default_val = 5.0
                step_val = 0.1  # Saltos de 0.1 MeV
                format_str = "%.1f"  # 1 decimal

        # Input numérico CON VALORES MÁS LÓGICOS
        energia = st.number_input(
            f"Energía ({unidad}):",
            min_value=0.0,  # Mínimo cero
            max_value=10000.0,  # Un límite alto pero razonable
            value=float(default_val),
            step=float(step_val),
            format=format_str,
            help=f"Energía de la radiación {tipo_radiacion}. Puedes introducir cualquier valor entre 0 y 10000 {unidad}."
        )

        # Validar que la energía sea positiva
        if energia <= 0:
            st.error("⚠️ La energía debe ser mayor que 0")
            energia = float(default_val)  # Valor por defecto si no es válido

        # Convertir todo a MeV internamente y formatear correctamente
        if unidad == "keV":
            energia_mev = energia / 1000.0
            
            # Formatear para mostrar sin decimales no necesarios
            if energia.is_integer():
                energia_display = f"{int(energia)} keV"
            else:
                # Mostrar con 1 decimal máximo
                energia_display = f"{energia:.1f} keV"
        else:
            energia_mev = energia
            
            # Formatear según el tipo de radiación para evitar decimales extraños
            if tipo_radiacion == "Rayos X":
                energia_display = f"{energia:.3f} MeV"
            elif tipo_radiacion == "Gamma":
                energia_display = f"{energia:.2f} MeV"
            elif tipo_radiacion == "Beta":
                energia_display = f"{energia:.2f} MeV"
            elif tipo_radiacion == "Neutrones":
                # Para neutrones, formatear según la magnitud
                if energia >= 10:
                    energia_display = f"{energia:.0f} MeV"
                elif energia >= 1:
                    energia_display = f"{energia:.1f} MeV"
                else:
                    energia_display = f"{energia:.2f} MeV"
            elif tipo_radiacion == "Alfa":
                energia_display = f"{energia:.1f} MeV"
            else:
                energia_display = f"{energia:.2f} MeV"

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
                "Grosor máximo (cm):",
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
            st.info("HVL/TVL solo aplican a fotones y su equivalente solo en neutrones")
        
        escala_log = st.checkbox("Escala logarítmica en Y", value=False)

    # Contenido principal en pestañas
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Inicio y Explicación",
        "🎯 Simulación", 
        "🔍 Comparación de Materiales",
        "📚 Información Teórica"
    ])

    with tab1:
        st.header("Bienvenido al Simulador de Blindaje Radiológico")
        
        st.markdown("""
        ### ¿Qué puedes hacer con esta aplicación?
        
        Esta herramienta interactiva te permite simular la atenuación de diferentes tipos 
        de radiación a través de diversos materiales de blindaje.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 **Funcionalidades principales:**")
            st.markdown("""
            1. **Simulación**
               - Selecciona materiales de blindaje.
               - Selecciona el tipo de radiación a la que estará expuesto y los parámetros de la misma.
               - Visualiza propiedades clave y gráfica de atenuación.
            
            2. **Comparación de Materiales**
               - Compara múltiples materiales simultáneamente
               - Análisis de efectividad relativa
            
            3. **Información Teórica**
               - Fundamentos físicos de la atenuación
               - Explicación de modelos simplificados
            """)
        
        with col2:
            st.subheader("⚙️ **Cómo usar la aplicación:**")
            st.markdown("""
            ### Paso 1: Configura los parámetros
            - Usa la barra lateral para seleccionar:
              - **Tipo de radiación** (Gamma, Beta, Neutrones, Rayos X, Alfa)
              - **Energía**
              - **Intensidad inicial**
              - **Grosor máximo del material** (esto marcará la escala de la gráfica)
              - **Opciones de visualización** (HVL, TVL y escala logarítmica)
            
            ### Paso 2: Selecciona un material
            - Ve a la pestaña "Simulación"
            - Haz clic en cualquier elemento/material
            
            ### Paso 3: Explora y compara
            - Observa la curva de atenuación
            - Compara con otros materiales
            - Ajusta grosor y observa la atenuación
            """)
        
        st.warning("""
        ⚠️ **Importante: Esta aplicación usa modelos simplificados para fines educativos**
        """)

    with tab2:
        st.header("Materiales de Blindaje")
    
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
            
            # Línea vertical para el espesor seleccionado (sin leyenda)
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
            
            # Líneas de HVL y TVL (para fotones y neutrones) - AHORA EN LEYENDA
            # Solo mostrar si están activados los checkboxes
            if tipo_radiacion in ["Gamma", "Rayos X", "Neutrones"]:
                # Calcular HVL y TVL si no están ya calculados
                if tipo_radiacion in ["Gamma", "Rayos X"]:
                    mu = obtener_coeficiente_atenuacion_fotones(nombre_elemento, energia_mev, tipo_radiacion)
                    hvl, tvl = calcular_capas_hvl_tvl(mu)
                elif tipo_radiacion == "Neutrones":
                    sigma = obtener_seccion_eficaz_neutrones(nombre_elemento, energia_mev)
                    sigma_macroscopica = params['densidad_atomica'] * sigma * 1e-24
                    if sigma_macroscopica > 0:
                        hvl = np.log(2) / sigma_macroscopica
                        tvl = np.log(10) / sigma_macroscopica
                    else:
                        hvl = 0
                        tvl = 0
                
                # Solo añadir a la leyenda si están habilitados los checkboxes
                if mostrar_hvl and hvl > 0 and hvl <= espesor_max:
                    # Crear una línea discontinua para HVL que aparezca en la leyenda
                    hvl_y = [min(intensidades_grafica), max(intensidades_grafica)]
                    fig.add_trace(go.Scatter(
                        x=[hvl, hvl],
                        y=hvl_y,
                        mode='lines',
                        name=f'HVL{" (eq)" if tipo_radiacion=="Neutrones" else ""} = {hvl:.2f} cm',
                        line=dict(color='red', dash='dash', width=2),
                        showlegend=True,
                        hovertemplate=f'HVL{" (equivalente)" if tipo_radiacion=="Neutrones" else ""}: {hvl:.2f} cm<extra></extra>'
                    ))
                
                if mostrar_tvl and tvl > 0 and tvl <= espesor_max:
                    # Crear una línea punteada para TVL que aparezca en la leyenda
                    tvl_y = [min(intensidades_grafica), max(intensidades_grafica)]
                    fig.add_trace(go.Scatter(
                        x=[tvl, tvl],
                        y=tvl_y,
                        mode='lines',
                        name=f'TVL{" (eq)" if tipo_radiacion=="Neutrones" else ""} = {tvl:.2f} cm',
                        line=dict(color='blue', dash='dot', width=2),
                        showlegend=True,
                        hovertemplate=f'TVL{" (equivalente)" if tipo_radiacion=="Neutrones" else ""}: {tvl:.2f} cm<extra></extra>'
                    ))
            
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

    with tab4:
        st.header("📚 Detalles de los Modelos Matemáticos")
        
        col_mod1, col_mod2 = st.columns(2)
        
        with col_mod1:
            st.subheader("1. Fotones (Gamma/Rayos X)")
            st.latex(r"I(x) = I_0 \cdot e^{-\mu \cdot x}")
            st.markdown(r"""
            - $\mu$ = coeficiente de atenuación lineal (cm⁻¹)
            - **Capa de medio valor:** Grosor en el cual un material reduce la intensidad de la radiación a la mitad $\text{HVL} = \dfrac{\ln(2)}{\mu}$
            - **Capa de décimo valor:** Grosor del material que reduce la intensidad de la radiación a la décima parte  $\text{TVL} = \dfrac{\ln(10)}{\mu}$
            """)
            
            st.subheader("2. Partículas Beta")
            st.latex(r"R_m \approx 0.542 \cdot E - 0.133 \quad (\text{g/cm}^2) \quad \text{para } E > 0.8 \text{ MeV}")
            st.latex(r"R_m \approx 0.407 \cdot E^{1.38} \quad (\text{g/cm}^2) \quad \text{para } E < 0.8 \text{ MeV}")
            st.latex(r"R = \frac{R_m}{\rho} \quad (\text{cm})")
            st.markdown("""
            - **Altas energías (>0.8 MeV):** Predomina la pérdida de energía por radiación
            - **Bajas energías (<0.8 MeV):** Predomina la pérdida por ionización
            - **Alcance másico ($R_m$):** Independiente del material
            - **Alcance lineal ($R$):** Alcance en cada material
            - **Atenuación:** Modelo experimental
            """)
        
        with col_mod2:
            st.subheader("3. Neutrones")
            st.latex(r"I(x) = I_0 \cdot e^{-N \cdot \sigma \cdot x}")
            st.markdown("""
            - N = densidad atómica (átomos/cm³)
            - σ = sección eficaz total (cm²)
            """)
            
            st.subheader("4. Partículas Alfa")
            st.latex(r"R_{\text{aire}} \approx 0.3 \cdot E^{1.5} \quad (\text{cm})")
            st.latex(r"R_{\text{material}} = R_{\text{aire}} \cdot \frac{\rho_{\text{aire}}}{\rho_{\text{material}}} \cdot \sqrt{\frac{A_{\text{eff, material}}}{A_{\text{eff, aire}}}}")
            st.markdown("**Cálculo de la masa atómica efectiva ($A_{eff}$) para compuestos:**")
            st.latex(r"A_{\text{eff}} = \left( \sum_i \frac{w_i}{\sqrt{A_i}} \right)^{-1}")
            st.markdown(""" siendo $w_i$ la fracción en peso del elemento $i$ en el compuesto y $A_i$ la masa atómica del elemento $i$. En los elementos se utiliza simplemente la masa atómica.
            """)
            st.markdown("""
            - **Comportamiento simplificado:** Las partículas alfa no se atenúan gradualmente, mantienen intensidad constante hasta su alcance total y luego se detienen abruptamente.
            """)

        st.warning("""
        ⚠️ **Importante** 
        - Esta aplicación usa modelos simplificados para fines educativos
        - Para neutrones, el concepto de HVL y TVL es 'equivalente' ya que los neutrones no siguen una atenuación exponencial simple.
        """)
        
        st.divider()
        
        st.subheader("Referencias")
        st.markdown(r"""
        ### **Rayos X y Fotones Gamma**
        
        **U.S. Nuclear Regulatory Commission (NRC)**  
        *"Radiation Protection and Shielding for Particle Accelerators"*  
        **Aportación:** Ley de atenuación exponencial, HVL y TVL.
        **Enlace:** https://www.nrc.gov/docs/ML1126/ML11262A163.pdf
        
        ### **Partículas Beta**
        
        **ScienceDirect - Engineering Topics**  
        *"Energy of Beta Particles"*  
        **Aportación:** Ecuaciones experimentales para la atenuación de la radiación beta.  
        **Enlace:** https://www.sciencedirect.com/topics/engineering/energy-beta-particle
        
        ### **Partículas Alfa y Cálculo de Masas Efectivas**
        
        **European Physical Journal Plus (Springer)**  
        *"A comprehensive study on alpha particle shielding properties of various materials"*  
        **Aportación:** Estudio detallado sobre propiedades de blindaje para partículas alfa, incluyendo métodos de cálculo de alcance y masas efectivas en diferentes materiales  
        **Enlace:** https://link.springer.com/article/10.1140/epjp/s13360-025-06345-6
        
        ### **Neutrones**
        
        **Universidad de Valencia - Física Nuclear Aplicada**  
        *"Tema 2: Interacción de la radiación con la materia"*  
        **Aportación:** Modelos de atenuación neutrónica 
        **Enlace:** https://www.uv.es/diazj/fna_tema2.pdf
        
        ### **Datos de Referencia y Herramientas**
        
        **NIST Physical Reference Data**  
        *Datos fundamentales de interacción radiación-materia*  
        **Aportación:** Coeficientes de atenuación, secciones eficaces y propiedades atómicas certificadas.  
        **Enlace:** https://physics.nist.gov/cgi-bin/Xcom/xcom2?Method=Comp&Output2=Hand
        
        **NIST Chemistry WebBook - Name Search**  
        *Base de datos de propiedades químicas y físicas*  
        **Aportación:** Información detallada sobre compuestos químicos, incluyendo composición elemental y masas atómicas.  
        **Enlace:** https://webbook.nist.gov/chemistry/name-ser/
        """)

if __name__ == "__main__":
    if 'elemento_seleccionado' not in st.session_state:
        st.session_state['elemento_seleccionado'] = 'Air'
    main()
