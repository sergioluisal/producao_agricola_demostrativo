import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

# ===================================================================
# 1. DEFINIÇÃO DE TODAS AS FUNÇÕES NO NÍVEL SUPERIOR
# ===================================================================

@st.cache_data
def optimize_dataframe(df):
    """Otimiza os tipos de dados de um DataFrame para reduzir o uso de memória."""
    for col in df.columns:
        if df[col].dtype == 'object':
            if len(df[col].unique()) / len(df[col]) < 0.5: df[col] = df[col].astype('category')
        elif 'int' in str(df[col].dtype): df[col] = pd.to_numeric(df[col], downcast='integer')
        elif 'float' in str(df[col].dtype): df[col] = pd.to_numeric(df[col], downcast='float')
    return df
    
#carregamento do dataset direto do github
#==========================
@st.cache_data
def train_models(data, knn_k, rf_n, test_split_size):
    """Função cacheada e segura para treinar os modelos de ML."""
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    df_ml = data.copy()
    label_encoder_soil = LabelEncoder(); label_encoder_crop = LabelEncoder()
    df_ml["Soil_Type"] = label_encoder_soil.fit_transform(df_ml["Soil_Type"])
    df_ml["Crop"] = label_encoder_crop.fit_transform(df_ml["Crop"])
    X = df_ml.drop("Yield_tons_per_hectare", axis=1); y = df_ml["Yield_tons_per_hectare"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_size, random_state=42)
    scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsRegressor(n_neighbors=knn_k); knn.fit(X_train_scaled, y_train); y_pred_knn = knn.predict(X_test_scaled)
    rf = RandomForestRegressor(n_estimators=rf_n, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1); rf.fit(X_train, y_train); y_pred_rf = rf.predict(X_test)
    metrics = {'KNN': {'r2': r2_score(y_test, y_pred_knn), 'rmse': np.sqrt(mean_squared_error(y_test, y_pred_knn)), 'mae': mean_absolute_error(y_test, y_pred_knn), 'y_pred': y_pred_knn},'Random Forest': {'r2': r2_score(y_test, y_pred_rf), 'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)), 'mae': mean_absolute_error(y_test, y_pred_rf), 'y_pred': y_pred_rf}}
    return metrics, y_test

def get_plotly_style_transparent():
    """Configura o estilo visual para gráficos Plotly com fundo transparente"""
    return {'plot_bgcolor': 'rgba(0,0,0,0)', 'paper_bgcolor': 'rgba(0,0,0,0)', 'font': {'color': '#FFFFFF'}}

def create_interactive_comparison_chart_transparent(metrics):
    """Cria gráfico de comparação interativo melhorado com fundo transparente"""
    colors = ['#4080FF', '#57A9FB', '#37D4CF']
    modelos = list(metrics.keys())
    r2_scores = [metrics[modelo]['r2'] for modelo in modelos]
    rmse_scores = [metrics[modelo]['rmse'] for modelo in modelos]
    mae_scores = [metrics[modelo]['mae'] for modelo in modelos]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='R² (Coeficiente de Determinação)', x=modelos, y=r2_scores, marker_color=colors[0], text=[f'{s:.3f}' for s in r2_scores], textposition='auto', textfont=dict(color='white'), hovertemplate='''<b>%{x}</b>  
R²: %{y:.3f}  
<i>Explica %{customdata:.1f}% da variância</i><extra></extra>''', customdata=[s * 100 for s in r2_scores], offsetgroup=1))
    fig.add_trace(go.Bar(name='RMSE (Erro Quadrático Médio)', x=modelos, y=rmse_scores, marker_color=colors[1], text=[f'{s:.2f}' for s in rmse_scores], textposition='auto', textfont=dict(color='white'), hovertemplate='''<b>%{x}</b>  
RMSE: %{y:.2f} ton/ha  
<i>Erro médio quadrático</i><extra></extra>''', offsetgroup=2))
    fig.add_trace(go.Bar(name='MAE (Erro Absoluto Médio)', x=modelos, y=mae_scores, marker_color=colors[2], text=[f'{s:.2f}' for s in mae_scores], textposition='auto', textfont=dict(color='white'), hovertemplate='''<b>%{x}</b>  
MAE: %{y:.2f} ton/ha  
<i>Erro médio absoluto</i><extra></extra>''', offsetgroup=3))
    annotations = []
    if r2_scores:
        melhor_r2_idx = np.argmax(r2_scores)
        annotations.append(dict(x=modelos[melhor_r2_idx], y=r2_scores[melhor_r2_idx] + max(r2_scores) * 0.1, text="🏆 Melhor R²", showarrow=True, arrowhead=2, arrowcolor=colors[0], font=dict(color=colors[0], size=12)))
    if rmse_scores:
        melhor_rmse_idx = np.argmin(rmse_scores)
        annotations.append(dict(x=modelos[melhor_rmse_idx], y=rmse_scores[melhor_rmse_idx] + max(rmse_scores) * 0.1, text="🎯 Menor RMSE", showarrow=True, arrowhead=2, arrowcolor=colors[1], font=dict(color=colors[1], size=12)))
    fig.update_layout(title={'text': '''<b>🤖 Comparação Interativa: KNN vs Random Forest</b>  
<sub>Análise de Desempenho para Predição de Produtividade Agrícola</sub>''', 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Modelos de Machine Learning', yaxis_title='Valores das Métricas', barmode='group', legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5), height=600, annotations=annotations, **get_plotly_style_transparent())
    fig.update_layout(updatemenus=[dict(type="buttons", direction="left", buttons=[dict(args=[{"visible": [True, True, True]}], label="Todas", method="restyle"), dict(args=[{"visible": [True, False, False]}], label="R²", method="restyle"), dict(args=[{"visible": [False, True, False]}], label="RMSE", method="restyle"), dict(args=[{"visible": [False, False, True]}], label="MAE", method="restyle")], pad={"r": 10, "t": 10}, showactive=True, x=0.01, xanchor="left", y=1.15, yanchor="top")])
    return fig

def create_dashboard_comparison_transparent(metrics, y_test):
    """Cria dashboard completo com múltiplos gráficos com fundo transparente"""
    colors = {'KNN': '#4080FF', 'Random Forest': '#23C343'}
    fig = make_subplots(rows=2, cols=2, subplot_titles=('<b>Comparação de Métricas R²</b>', '<b>Distribuição de Erros</b>', '<b>Real vs Predito - KNN</b>', '<b>Real vs Predito - Random Forest</b>'), vertical_spacing=0.15, horizontal_spacing=0.1)
    modelos = list(metrics.keys())
    r2_scores = [metrics[modelo]['r2'] for modelo in modelos]
    fig.add_trace(go.Bar(name='R²', x=modelos, y=r2_scores, marker_color=[colors.get(m, '#cccccc') for m in modelos], text=[f'{s:.3f}' for s in r2_scores], textposition='auto'), row=1, col=1)
    erros_knn = np.abs(y_test - metrics['KNN']['y_pred']); erros_rf = np.abs(y_test - metrics['Random Forest']['y_pred'])
    fig.add_trace(go.Histogram(x=erros_knn, name='Erros KNN', marker_color=colors['KNN'], opacity=0.7, nbinsx=20), row=1, col=2)
    fig.add_trace(go.Histogram(x=erros_rf, name='Erros RF', marker_color=colors['Random Forest'], opacity=0.7, nbinsx=20), row=1, col=2)
    for i, modelo in enumerate(modelos):
        row, col = 2, i + 1
        fig.add_trace(go.Scatter(x=y_test, y=metrics[modelo]['y_pred'], mode='markers', name=modelo, marker=dict(color=colors.get(modelo, '#cccccc'), size=4, opacity=0.6), hovertemplate='''<b>Real</b>: %{x:.2f}  
<b>Predito</b>: %{y:.2f}<extra></extra>'''), row=row, col=col)
        min_val = min(y_test.min(), metrics[modelo]['y_pred'].min()); max_val = max(y_test.max(), metrics[modelo]['y_pred'].max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Ideal', line=dict(dash='dash', color='red', width=2), showlegend=(i==0)), row=row, col=col)
    fig.update_layout(title={'text': '📊 Dashboard Completo: Análise KNN vs Random Forest', 'x': 0.5, 'xanchor': 'center'}, height=800, barmode='overlay', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), **get_plotly_style_transparent())
    return fig

# ===================================================================
# 2. INÍCIO DA EXECUÇÃO DO SCRIPT E INTERFACE DO APP
# ===================================================================

st.set_page_config(page_title="Dashboard de Análise Agrícola", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

# (Seu CSS original vai aqui)
st.markdown("""<style>...</style>""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌾 Análise Agrícola</h1>', unsafe_allow_html=True)
GITHUB_CSV_URL = "https://raw.githubusercontent.com/sergioluisal/producao_agricola_demostrativo/main/crop_yiel.csv"

@st.cache_data
def load_data_auto():
    try:
        df = pd.read_csv(GITHUB_CSV_URL)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

df = load_data_auto()

# Renomeação e validação de colunas
col_renames = {
    "Região": "Region", "Tipo_de_Solo": "Soil_Type", "Cultura": "Crop",
    "Condição_Climática": "Weather_Condition", "Fertilizante_Utilizado": "Fertilizer_Used",
    "Irrigação_Utilizada": "Irrigation_Used", "Produtividade_ton_ha": "Yield_tons_per_hectare",
    "Chuva_mm": "Rainfall_mm", "Temperatura_Celsius": "Temperature_Celsius"
}
df.rename(columns=col_renames, inplace=True)
required_columns = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
if any(col not in df.columns for col in required_columns):
    st.error(f"Colunas essenciais ausentes. Verifique se o arquivo contém: {', '.join(required_columns)}")
    st.stop()

# Bloco de filtros que cria o `filtered_df`
st.sidebar.markdown('<div class="filter-header">🔍 Filtros</div>', unsafe_allow_html=True)
selected_region = st.sidebar.selectbox("Região", ["Todas as Regiões"] + sorted(df["Region"].dropna().unique().tolist()))
selected_soil_type = st.sidebar.selectbox("Tipo de Solo", ["Todos os Tipos"] + sorted(df["Soil_Type"].dropna().unique().tolist()))
selected_crop = st.sidebar.selectbox("Cultura", ["Todas as Culturas"] + sorted(df["Crop"].dropna().unique().tolist()))
selected_weather_condition = st.sidebar.selectbox("Condição Climática", ["Todas as Condições"] + sorted(df["Weather_Condition"].dropna().unique().tolist()))
selected_fertilizer_used = st.sidebar.selectbox("Uso de Fertilizante", ["Todos", "Sim", "Não"])
selected_irrigation_used = st.sidebar.selectbox("Uso de Irrigação", ["Todos", "Sim", "Não"])

if st.sidebar.button("🗑️ Limpar Filtros"):
    st.rerun()

filtered_df = df.copy()
if selected_region != "Todas as Regiões": filtered_df = filtered_df[filtered_df["Region"] == selected_region]
if selected_soil_type != "Todos os Tipos": filtered_df = filtered_df[filtered_df["Soil_Type"] == selected_soil_type]
if selected_crop != "Todas as Culturas": filtered_df = filtered_df[filtered_df["Crop"] == selected_crop]
if selected_weather_condition != "Todas as Condições": filtered_df = filtered_df[filtered_df["Weather_Condition"] == selected_weather_condition]
if "Fertilizer_Used" in filtered_df.columns and selected_fertilizer_used != "Todos": filtered_df = filtered_df[filtered_df["Fertilizer_Used"] == (selected_fertilizer_used == "Sim")]
if "Irrigation_Used" in filtered_df.columns and selected_irrigation_used != "Todos": filtered_df = filtered_df[filtered_df["Irrigation_Used"] == (selected_irrigation_used == "Sim")]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**📊 Total de Registros:** {len(filtered_df)}")
st.sidebar.markdown(f"**📈 Total Original:** {len(df)}")

# (O resto do seu código de visualização de dados vai aqui)
# ...

# ===================================================================
# 3. SEÇÃO DE MACHINE LEARNING COM GRÁFICOS INCREMENTADOS
# ===================================================================
st.markdown('<div class="ml-section">', unsafe_allow_html=True)
st.markdown('<div class="comparison-header">🤖 Machine Learning: Comparação Interativa KNN vs Random Forest</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Configurações ML")
knn_neighbors = st.sidebar.slider("KNN - Número de Vizinhos", 3, 15, 5, 1)
rf_estimators = st.sidebar.slider("Random Forest - Número de Árvores", 10, 200, 50, 10)
test_size = st.sidebar.slider("Tamanho do Conjunto de Teste (%)", 10, 40, 20, 5) / 100

knn_required = ["Rainfall_mm", "Temperature_Celsius", "Soil_Type", "Crop", "Yield_tons_per_hectare"]

if any(col not in filtered_df.columns for col in knn_required):
    st.warning(f"Colunas faltando para análise de ML: {', '.join(knn_required)}")
else:
    df_ml = filtered_df[knn_required].dropna()
    if len(df_ml) < 50:
        st.warning("⚠️ Dados insuficientes para treinamento (necessário >50 registros após filtros).")
    else:
        if st.button("🚀 Treinar Modelos e Gerar Análise", use_container_width=True):
            with st.spinner("🔄 Treinando modelos..."):
                metrics, y_test = train_models(df_ml, knn_neighbors, rf_estimators, test_size)
                st.session_state['ml_metrics'] = metrics
                st.session_state['ml_y_test'] = y_test
        
        if 'ml_metrics' in st.session_state:
            metrics = st.session_state['ml_metrics']
            y_test = st.session_state['ml_y_test']

            st.subheader("📊 Resultados dos Modelos")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔍 K-Nearest Neighbors (KNN)")
                knn_col1, knn_col2, knn_col3 = st.columns(3)
                knn_col1.metric("📈 R²", f"{metrics['KNN']['r2']:.3f}")
                knn_col2.metric("📉 RMSE", f"{metrics['KNN']['rmse']:.2f} ton/ha")
                knn_col3.metric("📏 MAE", f"{metrics['KNN']['mae']:.2f} ton/ha")
            with col2:
                st.markdown("#### 🌲 Random Forest")
                rf_col1, rf_col2, rf_col3 = st.columns(3)
                rf_col1.metric("📈 R²", f"{metrics['Random Forest']['r2']:.3f}")
                rf_col2.metric("📉 RMSE", f"{metrics['Random Forest']['rmse']:.2f} ton/ha")
                rf_col3.metric("📏 MAE", f"{metrics['Random Forest']['mae']:.2f} ton/ha")

            st.subheader("🏆 Análise Comparativa")
            melhor_r2 = max(metrics, key=lambda x: metrics[x]['r2'])
            melhor_rmse = min(metrics, key=lambda x: metrics[x]['rmse'])
            melhor_mae = min(metrics, key=lambda x: metrics[x]['mae'])
            col1, col2, col3 = st.columns(3)
            col1.success(f"🏆 Melhor R²: **{melhor_r2}** ({metrics[melhor_r2]['r2']:.3f})")
            col2.success(f"🎯 Menor RMSE: **{melhor_rmse}** ({metrics[melhor_rmse]['rmse']:.2f})")
            col3.success(f"📏 Menor MAE: **{melhor_mae}** ({metrics[melhor_mae]['mae']:.2f})")

            st.subheader("🎨 Gráfico de Comparação Interativo")
            fig_comparison = create_interactive_comparison_chart_transparent(metrics)
            st.plotly_chart(fig_comparison, use_container_width=True)

            st.subheader("📋 Dashboard Completo de Análise")
            fig_dashboard = create_dashboard_comparison_transparent(metrics, y_test)
            st.plotly_chart(fig_dashboard, use_container_width=True)

            # **CORREÇÃO: Conteúdo dos expanders restaurado**
            with st.expander("💡 Interpretação dos Resultados"):
                st.markdown("""
                **Explicação das Métricas:**
                - **R² (Coeficiente de Determinação)**: Indica a proporção da variância explicada pelo modelo (0-1, quanto maior melhor)
                - **RMSE (Erro Quadrático Médio)**: Penaliza mais os erros grandes (quanto menor melhor)
                - **MAE (Erro Absoluto Médio)**: Média dos erros absolutos (quanto menor melhor)
                
                **Como Interpretar os Gráficos:**
                - **Gráfico de Comparação**: Permite filtrar métricas específicas usando os botões
                - **Real vs Predito**: Pontos próximos à linha vermelha indicam predições mais precisas
                - **Dashboard Completo**: Visão abrangente com múltiplas perspectivas dos resultados
                """)

            with st.expander("💾 Download dos Resultados"):
                results_df = pd.DataFrame({
                    'Modelo': list(metrics.keys()),
                    'R²': [metrics[modelo]['r2'] for modelo in metrics.keys()],
                    'RMSE': [metrics[modelo]['rmse'] for modelo in metrics.keys()],
                    'MAE': [metrics[modelo]['mae'] for modelo in metrics.keys()]
                })
                st.dataframe(results_df, use_container_width=True)
                csv_results = results_df.to_csv(index=False)
                st.download_button(label="📥 Baixar Métricas (CSV)", data=csv_results, file_name="metricas_ml_comparacao.csv", mime="text/csv")
                
                predictions_df = pd.DataFrame({'Real': y_test, 'KNN_Predito': metrics['KNN']['y_pred'], 'RF_Predito': metrics['Random Forest']['y_pred']})
                csv_predictions = predictions_df.to_csv(index=False)
                st.download_button(label="📥 Baixar Predições (CSV)", data=csv_predictions, file_name="predicoes_ml_comparacao.csv", mime="text/csv")

st.markdown('</div>', unsafe_allow_html=True)
# ... (Seu rodapé)


# Rodapé
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'><p>🌾 Dashboard de Análise Agrícola | Desenvolvido por Sérgio</p></div>", unsafe_allow_html=True)
