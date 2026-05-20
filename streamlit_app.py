import streamlit as st
import joblib
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression

# Calcular VPD
def calcular_VPD(T, RH):
    # NOTA: T en ºC, RH en % -> VPD en mbar
    return 6.11 * np.exp((17.625 * T) / (T + 243.04)) * (1.0 - RH/100.0)


# Obtener Lags y Rolling
def build_strict_matrices(df_group, target_col, input_cols, time_col='t', lag=3, max_step=1):
    X_rows, indices = [], []

    # ---- PADDING ----
    # Copiamos exógenas del prmier día hacia atrás, y hacemo zero-padding con WL y DI
    # Esto nos permite predecir desde el primer día (t=0)
    pad_size = lag

    # Copiamos la primera fila como base para el pasado "inexistente"
    first_row = df_group.iloc[0].copy()
    pad_df = pd.DataFrame([first_row] * pad_size)
    
    # Ajustamos los valores biológicos lógicos para el relleno: antes del día 0, asumimos que no había pérdida de peso ni podridos
    if 'WL' in pad_df.columns: pad_df['WL'] = 0.0
    if 'DI' in pad_df.columns: pad_df['DI'] = 0.0
    
    # Unimos el relleno al principio del dataframe original
    df_padded = pd.concat([pad_df, df_group], ignore_index=True)


    # ---- MATRIZ ----
    # Iteramos asumiendo que el índice 'i' es "HOY" (el presente)
    for step1_idx in range(len(df_group) - max_step + 1):
        # "Hoy" es el día justo anterior a Step_1
        hoy_idx_real = step1_idx - 1
        hoy_idx_pad = hoy_idx_real + pad_size   # Índice de "hoy" en la matriz con padding
            
        x_dict = {}

        # ---- EL PASADO Y EL PRESENTE (Conocido) ----
        # a) LAGS
        for k in range(lag):
            idx_past = hoy_idx_pad - k
            
            # Lags de las Exógenas (Ambiental y Tiempo)
            for inp in input_cols:
                x_dict[f'{inp}_lag{k+1}'] = df_padded[inp].iloc[idx_past]
        
        # b) ROLLING (estadísticas de la ventana de Lags)
        hist_idx_start = hoy_idx_pad - lag + 1
        hist_idx_end = hoy_idx_pad + 1
        
        # Rolling de las Exógenas
        for inp in input_cols:
            window_inp = df_padded[inp].iloc[hist_idx_start:hist_idx_end]
            x_dict[f'{inp}_roll_mean_{lag}'] = window_inp.mean()
            x_dict[f'{inp}_roll_std_{lag}']  = window_inp.std() if lag > 1 else 0
            x_dict[f'{inp}_roll_min_{lag}']  = window_inp.min()
            x_dict[f'{inp}_roll_max_{lag}']  = window_inp.max()
            x_dict[f'{inp}_roll_sum_{lag}']  = window_inp.sum()
            x_dict[f'{inp}_roll_slope_{lag}'] = np.polyfit(np.arange(len(window_inp)), window_inp.values, 1)[0] if len(window_inp) > 1 else 0

                
        # ---- EL FUTURO CONOCIDO (STEPS) ----
        for s in range(1, max_step + 1):
            target_idx_futuro = step1_idx + s - 1
            # Tiempo
            x_dict[f'{time_col}_step_{s}'] = df_group[time_col].iloc[target_idx_futuro]

            # Datos de los sensores el día de la predicción (t+1) (T/RH y/o WL/DI)
            if s == 1:
                for inp in input_cols:
                    x_dict[f'{inp}_step_{s}'] = df_group[inp].iloc[target_idx_futuro]
            
        X_rows.append(x_dict)
        indices.append(df_group.index[step1_idx])
        
    return pd.DataFrame(X_rows, index=indices)

# Aplicar monotonía a las predicciones
def apply_global_monotony(df):
    df_clean = []
    
    # Agrupamos por cada configuración única de curva
    # Un mismo target
    for name, group in df.groupby(['Target']):
        group = group.sort_values('t_target').copy()
        target_name = name   # El nombre del Target actual
        
        # Configuramos si la serie debe subir (True) o bajar (False)
        is_increasing = True if target_name in ['WL', 'DI'] else False
        ir = IsotonicRegression(increasing=is_increasing, out_of_bounds='clip')
        
        # Aplicamos el filtro a la curva predicha
        group['Pred'] = ir.fit_transform(group['t_target'].values, group['Pred'].values)
        df_clean.append(group)
        
    return pd.concat(df_clean)



# =======================
# CONFIGURACIÓN DE LA WEB
# =======================
st.set_page_config(page_title="Cherry Tomato Digital Twin", layout="wide")
st.title("🍅 Simulator for the Quality and Shelf Life of Cherry Tomatoes: Digital Twin")

# Cargar los modelos (se hace cache para no recargar en cada clic)
MODELO = 'XGB'      # MODELO que utiliza la web
@st.cache_resource
def load_models():
    model_wl = joblib.load(f'models/modelo_WL_web_{MODELO}.joblib')
    model_di = joblib.load(f'models/modelo_DI_web_{MODELO}.joblib')
    model_rsl = joblib.load(f'models/modelo_RSL_web_{MODELO}.joblib')
    return model_wl, model_di, model_rsl

model_wl, model_di, model_rsl = load_models()

inputs = ['t', 'T', 'RH', 'VPD']
targets = ['WL', 'DI', 'RSL']
time_col = 't'


# DATOS METIDOS POR EL USUARIO
st.write("Introduce environmental data of the test (day 0 represents initial state).")

# Variable para almacenar el dataframe final independientemente del método elegido
df_final = None

# --- Pestañas de Interfaz ---
tab1, tab2, tab3 = st.tabs([
    "🌡️ 1. Constant Values", 
    "✍️ 2. Day-by-Day Schedule", 
    "📁 3. Upload CSV File"
])


# --- MÉTOD0 1: Valores Constantes ---
with tab1:
    st.subheader("Constant Values")
    dias_const = st.slider("Duration of the Test (days)", min_value=1, max_value=30, value=15, key="dias_t1")
    T_const = st.slider("Constant Temperature (°C)", min_value=0.0, max_value=40.0, value=20.0)
    RH_const = st.slider("Constant Relative Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
    
    if st.button("Generate Constant Values and Simulate Scenario", type="primary"):
        # t va del 0 al número de días (incluyendo el 0)
        t_array = np.arange(0, dias_const + 1)
        df_final = pd.DataFrame({
            't': t_array,
            'T': [T_const] * len(t_array),
            'RH': [RH_const] * len(t_array)
        })

# --- MÉTOD0 2: Tabla Día a Día ---
with tab2:
    st.subheader("Day-by-Day Schedule")
    dias_tabla = st.number_input("Duration of the Test (days)", min_value=1, max_value=60, value=15, key="dias_t2")
    
    # Creamos un dataframe por defecto para que el usuario lo edite
    t_array_def = np.arange(0, dias_tabla + 1)
    df_default = pd.DataFrame({
        't': t_array_def,
        'T': [20.0] * len(t_array_def), # Valores por defecto sugeridos
        'RH': [80.0] * len(t_array_def)
    })
    
    st.info("Edit the cells in columns T and RH directly in the table")
    # st.data_editor crea un excel interactivo en la web
    # Bloqueamos la columna 't' para que el usuario no rompa la serie temporal
    df_editado = st.data_editor(df_default, disabled=["t"], hide_index=True, use_container_width=True)
    
    if st.button("Confirm Table Values and Simulate Scenario", type="primary"):
        df_final = df_editado.copy()

# --- MÉTOD0 3: Subir CSV ---
with tab3:
    st.subheader("Upload CSV File")
    st.markdown("""
    Upload a **CSV** file. It must contain two columns named `T` and `RH`, with daily data on temperature (T) and relative humidity (RH) starting from day 0.
    """)
    archivo_csv = st.file_uploader("Select your CSV file", type=['csv'])
    
    if archivo_csv is not None:
        try:
            df_csv = pd.read_csv(archivo_csv)
            # Validar que las columnas existan
            if 'T' in df_csv.columns and 'RH' in df_csv.columns:
                # Si el usuario no puso columna de tiempo, se la creamos (lo hacemos sí o sí)
                if 't' not in df_csv.columns:
                    df_csv.insert(0, 't', np.arange(0, len(df_csv)))
                
                # Seleccionamos solo las columnas que nos importan
                df_final = df_csv[['t', 'T', 'RH']].copy()
                st.success("File uploaded successfully. Click below to proceed.")
                
            else:
                st.error("⚠️ The CSV file must contain the exact columns: ‘T’ and ‘RH’ (case-sensitive).")
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            
    if df_final is not None and tab3:
        if st.button("Process CSV File and Simulate Scenario", type="primary"):
             pass   # El df_final ya está cargado, el flujo sigue abajo



# --- PROCESAMIENTO FINAL (Común para los 3 métodos) ---
# Si el usuario ha generado el dataframe por cualquiera de las 3 vías:
if df_final is not None:
    st.divider()        # Línea separadora
    st.subheader("📊 Generated Dataset")
    
    # Calculamos la variable VPD aplicando la función vectorizada a las columnas
    df_final['VPD'] = calcular_VPD(df_final['T'], df_final['RH'])
    
    # Mostramos los primeros resultados al usuario
    st.dataframe(df_final.style.format({
        'T': '{:.2f} °C',
        'RH': '{:.2f} %',
        'VPD': '{:.4f} mbar'
    }), use_container_width=True)
    
    # Graficamos el perfil ambiental para que el usuario visualice lo que ha introducido
    st.write("**Environmental profile of the test:**")

    col1, col2, col3 = st.columns(3)
    with col1:
        df_t = df_final.rename(columns={'t': 'Time (days)', 'T': 'Temperature (°C)'})
        st.line_chart(df_t, x='Time (days)', y='Temperature (°C)', color="#FF4B4B")

    with col2:
        df_t = df_final.rename(columns={'t': 'Time (days)', 'RH': 'Relative Humidity (%)'})
        st.line_chart(df_t, x='Time (days)', y='Relative Humidity (%)', color="#00CC96")

    with col3:
        df_t = df_final.rename(columns={'t': 'Time (days)', 'VPD': 'VPD (mbar)'})
        st.line_chart(df_t, x='Time (days)', y='VPD (mbar)', color="#AB63FA")
    


    # ====================
    # === PREDICCIONES ===
    # ====================

    # Ya está creado el Dataframe df_S4 = df_final con t, T, RH, VPD
    # Aplicar build_strict_matrices + Limpieza para obtener X_test
    # Finalmente hacer y_pred = model.predict(X_test) para cada modelo (WL, DI y RSL, en ese orden), y aplicar monotonía a las predicciones al final
    
    list_predictions = []
    for target in targets:
        # Generamos X_test
        inputs_in = inputs

        # Generacicón de matrices
        X_list = []
            
        X_tr = build_strict_matrices(df_final, target, inputs_in)
        if not X_tr.empty:
            X_list.append(X_tr)

        if not X_list:
            print("  -> (!) Datos insuficientes para esta combinación temporal.")
            continue
        
        X_full = pd.concat(X_list)

        # Eliminamos variables futuras
        cols_to_keep = []
        for col in X_full.columns:
            if '_step_' not in col:
                cols_to_keep.append(col)        # Es un Lag o Rolling, se queda
            else:
                var_name = col.split('_step_')[0]
                step_val = int(col.split('_step_')[1])
                # Para el tiempo, permitimos todos los steps
                if var_name == time_col:
                    if step_val <= 1:
                        cols_to_keep.append(col)
                # Para cualquier otra exógena, solo hasta step 1 (sensores, incluyendo virtuales, miden solo hasta hoy, que es step 1)
                else:
                    if step_val <= 1:
                        cols_to_keep.append(col)
                    
        X_step_limpio = X_full[cols_to_keep]

        # --- OBTENEMOS X_test ---
        X_test = X_step_limpio.copy()


        # --- PREDICCIONES ---
        # 1) WEIGHT LOSS
        if target == 'WL':
            # SUMA ACUMULATIVA DE DELTAS PREDICHOS
            y_pred_deltas = model_wl.predict(X_test)        # Predicción (VAL) -> Son Deltas
            y_pred_deltas = np.where(y_pred_deltas <= 0, 0.6, y_pred_deltas)     # Deltas positivos

            # Recuperamos el t_target para ordenar cronológicamente, en un Dataframe temporal
            t_target = X_step_limpio['t_step_1'].values
            df_reconstruccion = pd.DataFrame({
                't_target': t_target,
                'Delta_Pred': y_pred_deltas,
            })
            df_reconstruccion = df_reconstruccion.sort_values(['t_target'])
            
            # ---> Suma acumulada secuencial (inicialmente, valen 0)
            df_reconstruccion['Pred_Absoluta'] = df_reconstruccion['Delta_Pred'].cumsum()
            y_pred = df_reconstruccion['Pred_Absoluta'].values

        # 2) DECAY INCIDENCE
        elif target == 'DI':
            # SUMA ACUMULATIVA DE DELTAS PREDICHOS
            y_pred_deltas = model_di.predict(X_test)        # Predicción (VAL) -> Son Deltas
            y_pred_deltas = np.where(y_pred_deltas <= 0, 0.3, y_pred_deltas)     # Deltas positivos

            # Recuperamos el t_target para ordenar cronológicamente, en un Dataframe temporal
            t_target = X_step_limpio['t_step_1'].values
            df_reconstruccion = pd.DataFrame({
                't_target': t_target,
                'Delta_Pred': y_pred_deltas,
            })
            df_reconstruccion = df_reconstruccion.sort_values(['t_target'])
            
            # ---> Suma acumulada secuencial (inicialmente, valen 0)
            df_reconstruccion['Pred_Absoluta'] = df_reconstruccion['Delta_Pred'].cumsum()
            y_pred = df_reconstruccion['Pred_Absoluta'].values

        # 3) REMAINING SHELF LIFE
        elif target == 'RSL':
            # Hacemos una copia de df_final (todos los datos son Test)
            df_S4_test_simulado = df_final.copy()
            
            # Buscamos las predicciones de WL y DI
            df_pred_WL = next(df for df in list_predictions if (df['Target'] == 'WL').all())
            df_pred_DI = next(df for df in list_predictions if (df['Target'] == 'DI').all())
            
            # Sobrescribimos la realidad con las predicciones en nuestro DataFrame simulado
            for _, row in df_pred_WL.iterrows():
                mask = (df_S4_test_simulado[time_col] == row['t_target'])
                if mask.any(): df_S4_test_simulado.loc[mask, 'WL'] = row['Pred']
                    
            for _, row in df_pred_DI.iterrows():
                mask = (df_S4_test_simulado[time_col] == row['t_target'])
                if mask.any(): df_S4_test_simulado.loc[mask, 'DI'] = row['Pred']
            
            # Reconstruimos X_test usando el DataFrame simulado, con las predicciones de WL y DI
            X_test_sim_list = []
            df_S4_test_simulado = df_S4_test_simulado.sort_values(time_col).reset_index(drop=True)
            X_sim = build_strict_matrices(df_S4_test_simulado, target, inputs_in + ['WL', 'DI'])
            if not X_sim.empty: X_test_sim_list.append(X_sim)
                    
            X_test_sim_full = pd.concat(X_test_sim_list)

            # Poda Dinámica: filtramos para quedarnos solo con el Step actual y sus lags
            # Del tiempo cogemos todos los steps hasta el step actual, y de las exógenas solo hasta step 1 (sensores + sensores virtuales)
            cols_to_keep_sim = [c for c in X_test_sim_full.columns if '_step_' not in c or (int(c.split('_step_')[1]) <= 1 if c.split('_step_')[0] == time_col else int(c.split('_step_')[1]) <= 1)]
            X_test_sim_limpio = X_test_sim_full[cols_to_keep_sim]
            
            # PREDECIMOS USANDO LA MATRIZ CON PREDICCIONES DE WL Y DI
            y_pred = model_rsl.predict(X_test_sim_limpio)
            for i in range(1, len(y_pred)): y_pred[i] = y_pred[i-1] - 0.5 if y_pred[i] >= y_pred[i-1] else y_pred[i]    # Monotonía decreciente


        # --- GUARDAR PREDICCIONES ---
        # Recuperamos el tiempo exacto al que corresponde la predicción, y guardamos las predicciones
        t_future_values = X_step_limpio['t_step_1'].values
        df_preds = pd.DataFrame({
            'Target': target,
            't_target': t_future_values,
            'Pred': y_pred
        })
        
        # Actualizamos lista de predicciones
        list_predictions.append(df_preds)


    # Agrupamos todas las predicciones
    df_all_predictions = pd.concat(list_predictions, ignore_index=True)

    # Aplicamos monotonía con regresión isotónica
    #df_all_predictions = apply_global_monotony(df_all_predictions)     # Opcional


    # ==== MOSTRAR RESULTADOS ====
    st.subheader("Remaining Shelf Life and Quality Evolution of Cherry Tomatoes")

    orden_targets = ['RSL', 'WL', 'DI']

    fig = px.area(df_all_predictions, x='t_target', y='Pred', color='Target', facet_col='Target',
        facet_col_spacing=0.07,
        category_orders={'Target': orden_targets},
        markers=True,
        labels={'t_target': 'Time (days)', 'Pred': ''})
    
    fig.update_traces(
        line=dict(width=2)       # Grosor de línea
    )

    # Ejes Y individuales
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_layout(
        yaxis_title="Remaining Shelf Life (days)",
        yaxis2_title="Weight Loss (%)",
        yaxis3_title="Decay Incidence (%)",

        showlegend=False
    )

    # Quitar títulos individuales
    fig.for_each_annotation(lambda a: a.update(text=""))
    
    # Graficar en la web
    st.plotly_chart(fig, use_container_width=True)
