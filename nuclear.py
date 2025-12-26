"""
DASHBOARD INTERACTIVO DE PROTECCIÓN RADIOLÓGICA
Streamlit app para simulación de blindaje - MODELOS CORRECTOS
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
# FUNCIONES DE CÁLCULO - MODELOS CORRECTOS POR TIPO DE RADIACIÓN
# ============================================================================

def calcular_atenuacion_fotones(I0, mu, x):
    """Ley de atenuación exponencial - VÁLIDA SOLO PARA FOTONES"""
    return I0 * np.exp(-mu * x)

def calcular_atenuacion_beta(I0, energia_mev, densidad_material, x):
    """
    Modelo simplificado para partículas beta
    Basado en alcance máximo (range) aproximado
    """
    # Alcance aproximado en g/cm² (fórmula empírica para electrones)
    # R = 0.412 * E^(1.265-0.0954*ln(E)) para 0.01 < E < 2.5 MeV
    # Simplificado: R ≈ 0.5 * E_max para E > 0.8 MeV (en g/cm²)
    
    if energia_mev <= 0:
        return I0
    
    # Alcance en g/cm² (aproximación)
    if energia_mev < 0.8:
        alcance_gcm2 = 0.15 * energia_mev ** 1.5
    else:
        alcance_gcm2 = 0.5 * energia_mev
    
    # Convertir espesor x (cm) a espesor másico (g/cm²)
    espesor_masico = x * densidad_material
    
    # Si el espesor es mayor que el alcance, intensidad = 0
    if espesor_masico >= alcance_gcm2:
        return 0.0
    
    # Modelo simplificado: lineal hasta el alcance
    # En realidad es más complejo (curva de Bragg), pero simplificamos
    fraccion_atenuada = espesor_masico / alcance_gcm2
    return I0 * (1 - fraccion_atenuada ** 2)  # Aproximación cuadrática

def calcular_atenuacion_neutrones(I0, sigma_total, densidad_atomica, x):
    """
    Modelo para neutrones - atenucación exponencial PERO con sección eficaz
    I(x) = I0 * exp(-N * σ_total * x)
    donde N = densidad atómica (átomos/cm³)
    """
    # σ_total en barns (1 barn = 1e-24 cm²)
    sigma_cm2 = sigma_total * 1e-24
    # Densidad atómica aproximada (átomos/cm³)
    N = densidad_atomica
    
    return I0 * np.exp(-N * sigma_cm2 * x)

def calcular_atenuacion_alfa(I0, energia_mev, densidad_material, x):
    """
    Modelo para partículas alfa - alcance muy corto
    """
    # Alcance aproximado para alfa en aire: R ≈ 0.3 * E^(3/2) cm (en aire)
    # En otros materiales: R_material = R_aire * (ρ_aire/ρ_material)
    
    if energia_mev <= 0:
        return I0
    
    # Alcance en aire (cm)
    alcance_aire = 0.3 * energia_mev ** 1.5
    
    # Densidad del aire (g/cm³)
    densidad_aire = 0.001225
    
    # Alcance en el material (cm)
    alcance_material = alcance_aire * (densidad_aire / densidad_material)
    
    # Si el espesor es mayor que el alcance, intensidad = 0
    if x >= alcance_material:
        return 0.0
    
    # Modelo simplificado: caída brusca cerca del alcance
    fraccion = x / alcance_material
    return I0 * (1 - fraccion ** 3)

def obtener_parametros_material(elemento, energia_mev, tipo_radiacion):
    """Obtiene parámetros necesarios según tipo de radiación"""
    # Base de datos de materiales
    materiales = {
        'Plomo': {
            'densidad': 11.34,
            'Z_efectivo': 82,
            'sigma_neutrones': 5.0,  # barns (aproximado para 1 MeV)
            'densidad_atomica': 3.3e22  # átomos/cm³
        },
        'Acero': {
            'densidad': 7.85,
            'Z_efectivo': 26,
            'sigma_neutrones': 3.0,
            'densidad_atomica': 8.5e22
        },
        'Hormigón': {
            'densidad': 2.35,
            'Z_efectivo': 'mix',
            'sigma_neutrones': 8.0,
            'densidad_atomica': 1.0e23
        },
        'Agua': {
            'densidad': 1.00,
            'Z_efectivo': 'mix',
            'sigma_neutrones': 40.0,  # Alta para neutrones térmicos
            'densidad_atomica': 3.3e22
        },
        'Wolframio': {
            'densidad': 19.25,
            'Z_efectivo': 74,
            'sigma_neutrones': 4.5,
            'densidad_atomica': 6.3e22
        },
        'Uranio': {
            'densidad': 19.10,
            'Z_efectivo': 92,
            'sigma_neutrones': 7.0,
            'densidad_atomica': 4.8e22
        },
        'Boro': {
            'densidad': 2.34,
            'Z_efectivo': 5,
            'sigma_neutrones': 100.0,  # Muy alto para captura de neutrones
            'densidad_atomica': 1.3e23
        }
    }
    
    if elemento in materiales:
        return materiales[elemento]
    else:
        # Valores por defecto
        return {
            'densidad': 2.0,
            'Z_efectivo': 10,
            'sigma_neutrones': 5.0,
            'densidad_atomica': 5e22
        }

def calcular_atenuacion_general(I0, elemento, energia_mev, tipo_radiacion, x):
    """Función principal que selecciona el modelo correcto"""
    params = obtener_parametros_material(elemento, energia_mev, tipo_radiacion)
    
    if tipo_radiacion in ["Gamma", "Rayos X"]:
        # Para fotones, necesitamos coeficiente de atenuación lineal
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
        # Por defecto, modelo exponencial
        mu = 0.1
        return I0 * np.exp(-mu * x)

def obtener_coeficiente_atenuacion_fotones(elemento, energia_mev, tipo_radiacion):
    """Coeficiente de atenuación solo para fotones"""
    coeficientes = {
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
    
    return 0.1

def obtener_seccion_eficaz_neutrones(elemento, energia_mev):
    """Sección eficaz para neutrones (barns)"""
    # Valores aproximados
    secciones = {
        'Plomo': {0.000025: 0.17, 0.001: 0.3, 1.0: 5.0, 10.0: 3.0},
        'Acero': {0.000025: 2.5, 0.001: 2.8, 1.0: 3.0, 10.0: 2.0},
        'Hormigón': {0.000025: 4.0, 0.001: 5.0, 1.0: 8.0, 10.0: 6.0},
        'Agua': {0.000025: 40.0, 0.001: 20.0, 1.0: 5.0, 10.0: 3.0},
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
    
    return 5.0  # Valor por defecto

def calcular_capas_hvl_tvl(mu):
    """Calcula HVL y TVL - SÓLO VÁLIDO PARA FOTONES"""
    if mu > 0:
        hvl = np.log(2) / mu
        tvl = np.log(10) / mu
        return hvl, tvl
    return 0, 0

# ============================================================================
# INTERFAZ STREAMLIT - ACTUALIZADA CON MODELOS CORRECTOS
# ============================================================================

def main():
    st.title("☢️ Simulador de Blindaje Radiológico - Modelos Correctos")
    st.markdown("""
    ### **IMPORTANTE:** Diferentes modelos físicos para cada tipo de radiación
    *La ley exponencial solo es válida para fotones (Rayos X y Gamma)*
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Parámetros de Simulación")
        
        # Añadir partículas Alfa
        tipo_radiacion = st.selectbox(
            "Tipo de radiación:",
            ["Gamma", "Rayos X", "Beta", "Neutrones", "Alfa"],
            index=0
        )
        
        # Explicación de modelos
        with st.expander("📖 Modelos utilizados por tipo"):
            st.markdown("""
            - **Fotones (Gamma/Rayos X):** Ley exponencial I(x) = I₀·e^(-μx)
            - **Partículas Beta:** Modelo de alcance máximo (range)
            - **Neutrones:** Atenuación por sección eficaz nuclear
            - **Partículas Alfa:** Alcance corto fijo en material
            """)
        
        # ENTRADA DE ENERGÍA
        st.markdown("### 🔋 Energía de la radiación")
        
        with st.expander("ℹ️ Rangos típicos"):
            st.markdown("""
            - **Rayos X:** 1-300 keV
            - **Gamma:** 0.01-10 MeV
            - **Beta:** 0.1-10 MeV
            - **Neutrones:** 0.001 eV - 20 MeV
            - **Alfa:** 3-10 MeV
            """)
        
        # Parámetros según tipo
        if tipo_radiacion == "Rayos X":
            unidad = st.radio("Unidad:", ["keV", "MeV"], horizontal=True)
            default_val = 50.0 if unidad == "keV" else 0.05
            min_val = 1.0 if unidad == "keV" else 0.001
            max_val = 300.0 if unidad == "keV" else 0.3
        elif tipo_radiacion == "Gamma":
            unidad = "MeV"
            default_val = 1.0
            min_val = 0.001
            max_val = 10.0
        elif tipo_radiacion == "Beta":
            unidad = "MeV"
            default_val = 2.0
            min_val = 0.01
            max_val = 10.0
        elif tipo_radiacion == "Neutrones":
            unidad = "MeV"
            default_val = 1.0
            min_val = 0.000001
            max_val = 20.0
        elif tipo_radiacion == "Alfa":
            unidad = "MeV"
            default_val = 5.0
            min_val = 3.0
            max_val = 10.0
        
        energia = st.number_input(
            f"Energía ({unidad}):",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=0.01,
            format="%.3f"
        )
        
        # Convertir a MeV
        if unidad == "keV":
            energia_mev = energia / 1000.0
            energia_display = f"{energia} keV"
        else:
            energia_mev = energia
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
        
        # Espesor máximo
        if tipo_radiacion == "Alfa":
            espesor_max = st.slider("Espesor máximo (cm):", 0.001, 1.0, 0.1, 0.001)
        elif tipo_radiacion == "Beta":
            espesor_max = st.slider("Espesor máximo (cm):", 0.1, 10.0, 2.0, 0.1)
        else:
            espesor_max = st.slider("Espesor máximo (cm):", 1, 500, 100, 10)
        
        st.divider()
        st.header("📊 Opciones de Visualización")
        
        # Solo mostrar HVL/TVL para fotones
        if tipo_radiacion in ["Gamma", "Rayos X"]:
            mostrar_hvl = st.checkbox("Mostrar HVL/TVL", value=True)
        else:
            mostrar_hvl = False
            st.info("HVL/TVL solo aplican a fotones")
        
        escala_log = st.checkbox("Escala logarítmica en Y", value=True)
    
    # Pestañas principales
    tab1, tab2, tab3 = st.tabs(["🏠 Explicación", "🎯 Simulación", "📚 Modelos"])
    
    with tab1:
        st.header("Modelos Correctos de Atenuación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📖 Por qué diferentes modelos?")
            st.markdown("""
            Cada tipo de radiación interactúa de manera diferente con la materia:
            
            **Fotones (X/Gamma):**
            - Interacción por efecto fotoeléctrico, Compton, producción de pares
            - Cada fotón tiene probabilidad constante de ser absorbido
            - ✅ **Ley exponencial:** I(x) = I₀·e^(-μx)
            
            **Partículas Beta (e⁻/e⁺):**
            - Pérdida continua de energía por ionización
            - Alcance máximo definido (range)
            - ✗ **NO exponencial** - Modelo de alcance
            
            **Neutrones:**
            - Dispersión elástica/inelástica + captura nuclear
            - Depende de sección eficaz σ(E)
            - ✗ **NO exponencial simple** - Modelo nuclear
            
            **Partículas Alfa (α):**
            - Pérdida densa de energía por ionización
            - Alcance muy corto y fijo
            - ✗ **NO exponencial** - Modelo de alcance corto
            """)
        
        with col2:
            st.subheader("🎯 Implicaciones para blindaje")
            st.markdown("""
            **Materiales efectivos por tipo:**
            
            1. **Fotones:** Materiales densos con alto Z (Pb, W, U)
            2. **Beta:** Materiales ligeros (plástico, Al) para minimizar radiación de frenado
            3. **Neutrones:** Materiales con H (agua) para moderación + B/Cd para captura
            4. **Alfa:** Cualquier material (incluso papel o aire)
            
            **Espesores típicos:**
            - Alfa: µm a mm
            - Beta: mm a cm  
            - Neutrones: cm a m
            - Fotones: cm a m (dependiendo de energía)
            """)
        
        st.warning("""
        ⚠️ **Importante:** Las simulaciones anteriores usaban modelo exponencial para todo. 
        Esta versión usa modelos físicamente correctos para cada tipo de radiación.
        """)
    
    with tab2:
        st.header(f"Simulación para {tipo_radiacion}")
        
        # Tabla periódica simplificada
        elementos = ["Plomo", "Acero", "Hormigón", "Agua", "Wolframio", "Uranio", "Boro"]
        
        col_sel1, col_sel2 = st.columns([3, 1])
        
        with col_sel1:
            elemento = st.selectbox("Selecciona material:", elementos, index=0)
        
        with col_sel2:
            espesor = st.number_input(
                "Espesor (cm):",
                min_value=0.0,
                max_value=float(espesor_max),
                value=min(1.0, float(espesor_max)),
                step=0.01,
                key="espesor_sim"
            )
        
        # Calcular atenuación
        I_final = calcular_atenuacion_general(I0, elemento, energia_mev, tipo_radiacion, espesor)
        atenuacion = (1 - I_final/I0) * 100 if I0 > 0 else 0
        
        # Gráfica
        espesores = np.linspace(0, espesor_max, 300)
        intensidades = [calcular_atenuacion_general(I0, elemento, energia_mev, tipo_radiacion, x) for x in espesores]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=espesores,
            y=intensidades,
            mode='lines',
            name=f'{elemento}',
            line=dict(width=3),
            hovertemplate="Espesor: %{x:.3f} cm<br>Intensidad: %{y:.2e}<extra></extra>"
        ))
        
        # Línea para espesor seleccionado
        fig.add_vline(
            x=espesor,
            line_dash="solid",
            line_color="green",
            line_width=2,
            annotation_text=f"{espesor} cm",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title=f"Atenuación de {tipo_radiacion} ({energia_display}) en {elemento}",
            xaxis_title="Espesor (cm)",
            yaxis_title="Intensidad transmitida (partículas/s·cm²)",
            template='plotly_white',
            height=500
        )
        
        if escala_log:
            fig.update_yaxes(type="log", exponentformat='power')
        
        st.plotly_chart(fig, width='stretch')
        
        # Resultados
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric("Intensidad inicial", f"{I0:.2e}")
            st.metric("Energía", energia_display)
        
        with col_res2:
            st.metric("Intensidad final", f"{I_final:.2e}")
            st.metric("Atenuación", f"{atenuacion:.1f}%")
        
        with col_res3:
            params = obtener_parametros_material(elemento, energia_mev, tipo_radiacion)
            st.metric("Densidad", f"{params['densidad']} g/cm³")
            
            if tipo_radiacion in ["Gamma", "Rayos X"]:
                mu = obtener_coeficiente_atenuacion_fotones(elemento, energia_mev, tipo_radiacion)
                hvl, tvl = calcular_capas_hvl_tvl(mu)
                st.metric("HVL", f"{hvl:.2f} cm")
            elif tipo_radiacion == "Neutrones":
                sigma = obtener_seccion_eficaz_neutrones(elemento, energia_mev)
                st.metric("σ (barns)", f"{sigma:.1f}")
        
        # Información específica del modelo
        st.subheader("📊 Información del modelo utilizado")
        
        if tipo_radiacion in ["Gamma", "Rayos X"]:
            mu = obtener_coeficiente_atenuacion_fotones(elemento, energia_mev, tipo_radiacion)
            st.markdown(f"""
            **Modelo exponencial:** I(x) = I₀·e^(-μx)
            - μ = {mu:.4f} cm⁻¹
            - HVL = {calcular_capas_hvl_tvl(mu)[0]:.2f} cm
            - TVL = {calcular_capas_hvl_tvl(mu)[1]:.2f} cm
            """)
        
        elif tipo_radiacion == "Beta":
            params = obtener_parametros_material(elemento, energia_mev, tipo_radiacion)
            # Calcular alcance aproximado
            if energia_mev < 0.8:
                alcance_gcm2 = 0.15 * energia_mev ** 1.5
            else:
                alcance_gcm2 = 0.5 * energia_mev
            
            alcance_cm = alcance_gcm2 / params['densidad']
            
            st.markdown(f"""
            **Modelo de alcance para beta:**
            - Energía máxima: {energia_mev:.3f} MeV
            - Alcance aproximado: {alcance_cm:.3f} cm
            - Densidad material: {params['densidad']} g/cm³
            - Atenuación completa a {alcance_cm:.3f} cm
            """)
            
            if espesor >= alcance_cm:
                st.success("✅ Atenuación completa alcanzada")
            else:
                st.info(f"ℹ️ {((alcance_cm - espesor)/alcance_cm*100):.1f}% del alcance restante")
        
        elif tipo_radiacion == "Neutrones":
            sigma = obtener_seccion_eficaz_neutrones(elemento, energia_mev)
            params = obtener_parametros_material(elemento, energia_mev, tipo_radiacion)
            
            st.markdown(f"""
            **Modelo de sección eficaz:**
            - Sección eficaz total: σ = {sigma:.1f} barns
            - Densidad atómica: N ≈ {params['densidad_atomica']:.1e} átomos/cm³
            - Longitud de atenuación: λ = 1/(Nσ) ≈ {1/(params['densidad_atomica']*sigma*1e-24):.2f} cm
            """)
        
        elif tipo_radiacion == "Alfa":
            params = obtener_parametros_material(elemento, energia_mev, tipo_radiacion)
            alcance_aire = 0.3 * energia_mev ** 1.5
            alcance_material = alcance_aire * (0.001225 / params['densidad'])
            
            st.markdown(f"""
            **Modelo de alcance para alfa:**
            - Energía: {energia_mev:.2f} MeV
            - Alcance en aire: {alcance_aire:.3f} cm
            - Alcance en {elemento}: {alcance_material:.5f} cm
            - Densidad material: {params['densidad']} g/cm³
            """)
            
            if espesor >= alcance_material:
                st.success("✅ Atenuación completa alcanzada")
            else:
                st.info(f"ℹ️ {((alcance_material - espesor)/alcance_material*100):.1f}% del alcance restante")
    
    with tab3:
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

if __name__ == "__main__":
    main()
