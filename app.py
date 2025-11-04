import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import logging

# Desativar logs informativos do Prophet
logging.getLogger('prophet').setLevel(logging.WARNING)

# --- Configurações Iniciais ---
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'

# Parâmetros de Estoque (Valores Fictícios para Demonstração)
CUSTO_MANUTENCAO_UNITARIO = 0.50  # Custo por unidade por mês
CUSTO_FALTA_UNITARIO = 5.00       # Custo de falta por unidade (perda de venda)
LEAD_TIME_DIAS = 7                # Tempo de espera do pedido em dias

# --- Funções de Pré-processamento e Modelagem (Reutilizadas da Parte 1) ---

@st.cache_data
def load_data():
    """Carrega e pré-processa os dados de treino e teste."""
    try:
        df_train = pd.read_csv(TRAIN_PATH, parse_dates=['date'])
        df_test = pd.read_csv(TEST_PATH, parse_dates=['date'])
    except FileNotFoundError:
        st.error("Arquivos de dados (train.csv ou test.csv) não encontrados.")
        return None, None

    # Preparação para o Prophet
    df_prophet = df_train.rename(columns={'date': 'ds', 'sales': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet['y'] = df_prophet['y'].astype(float)
    df_prophet['store_item'] = df_prophet['store'].astype(str) + '_' + df_prophet['item'].astype(str)

    df_test['store_item'] = df_test['store'].astype(str) + '_' + df_test['item'].astype(str)
    df_test = df_test.rename(columns={'date': 'ds'})
    df_test['ds'] = pd.to_datetime(df_test['ds'])

    return df_prophet, df_test

def get_holidays():
    """Define o dataframe de feriados."""
    holidays = pd.DataFrame({
        'holiday': 'US_Holiday',
        'ds': pd.to_datetime([
            '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01',
            '2013-05-27', '2014-05-26', '2015-05-25', '2016-05-30', '2017-05-29',
            '2013-07-04', '2014-07-04', '2015-07-04', '2016-07-04', '2017-07-04',
            '2013-09-02', '2014-09-01', '2015-09-07', '2016-09-05', '2017-09-04',
            '2013-11-28', '2014-11-27', '2015-11-26', '2016-11-24', '2017-11-23',
            '2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25', '2017-12-25'
        ]),
        'lower_window': 0,
        'upper_window': 1
    })
    black_friday = pd.DataFrame({
        'holiday': 'Black_Friday',
        'ds': pd.to_datetime([
            '2013-11-29', '2014-11-28', '2015-11-27', '2016-11-25', '2017-11-24'
        ]),
        'lower_window': -3,
        'upper_window': 0
    })
    return pd.concat([holidays, black_friday]).sort_values('ds').reset_index(drop=True)

@st.cache_resource
def train_and_forecast(series_id, df_prophet, df_test, holidays):
    """Treina o modelo Prophet e gera previsões para uma série específica."""
    df_series = df_prophet[df_prophet['store_item'] == series_id].copy()
    
    model = Prophet(
        growth='linear',
        seasonality_mode='multiplicative',
        weekly_seasonality=True,
        yearly_seasonality=True,
        holidays=holidays,
        changepoint_prior_scale=0.05,
        interval_width=0.95
    )
    model.add_seasonality(name='daily', period=1, fourier_order=5)
    model.fit(df_series)
    
    future = df_test[df_test['store_item'] == series_id][['ds']]
    forecast = model.predict(future)
    
    return forecast, model

def calculate_rmse(df_prophet, series_id, holidays):
    """Calcula o RMSE no último ano de treino para estimar o desvio padrão do erro."""
    df_eval = df_prophet[df_prophet['store_item'] == series_id].copy()
    
    # Backtesting no último ano
    cutoff_date = df_eval['ds'].max() - pd.Timedelta(days=365)
    df_train_eval = df_eval[df_eval['ds'] <= cutoff_date]
    df_test_eval = df_eval[df_eval['ds'] > cutoff_date]
    
    if len(df_train_eval) == 0 or len(df_test_eval) == 0:
        # Se não houver dados suficientes para backtesting, usa um valor padrão
        return 10.0 

    model_eval = Prophet(
        growth='linear',
        seasonality_mode='multiplicative',
        weekly_seasonality=True,
        yearly_seasonality=True,
        holidays=holidays,
        changepoint_prior_scale=0.05
    )
    model_eval.add_seasonality(name='daily', period=1, fourier_order=5)
    model_eval.fit(df_train_eval)
    
    future_eval = df_test_eval[['ds']]
    forecast_eval = model_eval.predict(future_eval)
    
    y_true = df_test_eval['y'].values
    y_pred = forecast_eval['yhat'].values
    y_pred[y_pred < 0] = 0
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# --- Lógica de Recomendação de Estoque (Parte 2) ---

def calculate_optimal_order(forecast_df, service_level, rmse, lead_time_days):
    """
    Calcula a quantidade ideal de pedido (Q) usando o conceito de Estoque de Segurança.
    
    Q = Demanda Média no Lead Time + Estoque de Segurança
    Estoque de Segurança = Z * Desvio Padrão da Demanda no Lead Time
    """
    
    # 1. Demanda Média no Lead Time (D_LT)
    D_LT = forecast_df['yhat'].sum()
    
    # 2. Desvio Padrão da Demanda no Lead Time (Sigma_LT)
    period_days = len(forecast_df)
    sigma_LT = rmse * np.sqrt(period_days)
    
    # 3. Fator Z (Z-score)
    Z = norm.ppf(service_level)
    
    # 4. Estoque de Segurança (SS)
    SS = Z * sigma_LT
    
    # 5. Quantidade Ideal de Pedido (Q)
    Q = np.ceil(D_LT + SS)
    
    # 6. Cálculo do Custo (Exemplo Simples)
    custo_manutencao = SS * CUSTO_MANUTENCAO_UNITARIO
    
    return {
        'demanda_media_prevista': D_LT,
        'estoque_seguranca': SS,
        'fator_z': Z,
        'desvio_padrao_demanda': sigma_LT,
        'quantidade_ideal_pedido': Q,
        'custo_manutencao_estimado': custo_manutencao
    }

# --- Aplicação Streamlit ---

def main():
    st.set_page_config(page_title="Ferramenta de Recomendação de Estoque", layout="wide")
    
    # Título e Descrição
    st.title("📦 Ferramenta de Recomendação de Gestão de Estoque")
    st.markdown("**Parte 2: Otimização de Inventário com Previsão de Demanda**")
    st.markdown("---")

    df_prophet, df_test = load_data()
    if df_prophet is None:
        return

    holidays = get_holidays()
    unique_series = df_prophet['store_item'].unique()
    
    # Sidebar para seleção de parâmetros
    st.sidebar.header("⚙️ Parâmetros de Recomendação")
    
    # Seleção de Loja e Item
    stores = sorted(df_prophet['store'].unique())
    items = sorted(df_prophet['item'].unique())
    
    selected_store = st.sidebar.selectbox("🏪 Selecione a Loja (Store)", stores)
    selected_item = st.sidebar.selectbox("📦 Selecione o Item (Item)", items)
    
    series_id = f"{selected_store}_{selected_item}"
    
    # Parâmetros de Estoque
    service_level = st.sidebar.slider(
        "📊 Nível de Serviço Desejado (%)",
        min_value=50, max_value=99, value=95, step=1
    )
    service_level = service_level / 100
    
    # Exibir parâmetros fixos (para contexto)
    st.sidebar.markdown("### 📋 Parâmetros Fixos")
    st.sidebar.info(f"**Lead Time:** {LEAD_TIME_DIAS} dias")
    st.sidebar.info(f"**Custo de Manutenção:** R$ {CUSTO_MANUTENCAO_UNITARIO:.2f}/unidade/mês")

    st.markdown(f"## 🎯 Recomendação para Loja {selected_store}, Item {selected_item}")

    if series_id not in unique_series:
        st.warning("⚠️ Combinação de Loja/Item não encontrada nos dados de treino.")
        return

    with st.spinner("⏳ Treinando modelo e calculando previsão..."):
        # 1. Treinar e Prever
        forecast_df, model = train_and_forecast(series_id, df_prophet, df_test, holidays)
        
        # 2. Estimar o Erro (RMSE)
        rmse = calculate_rmse(df_prophet, series_id, holidays)
        
        # 3. Calcular a Recomendação
        recommendation = calculate_optimal_order(
            forecast_df, 
            service_level, 
            rmse, 
            LEAD_TIME_DIAS
        )

    # --- Exibição dos Resultados Principais ---
    
    st.markdown("### 📈 Resultado da Recomendação")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Demanda Prevista", 
            value=f"{recommendation['demanda_media_prevista']:.0f}",
            delta="unidades (90 dias)"
        )

    with col2:
        st.metric(
            label="Estoque de Segurança", 
            value=f"{recommendation['estoque_seguranca']:.0f}",
            delta=f"Nível de Serviço: {service_level*100:.0f}%"
        )

    with col3:
        st.metric(
            label="Quantidade Ideal de Pedido", 
            value=f"{recommendation['quantidade_ideal_pedido']:.0f}",
            delta="unidades"
        )
    
    with col4:
        st.metric(
            label="Custo de Manutenção", 
            value=f"R$ {recommendation['custo_manutencao_estimado']:.2f}",
            delta="estimado"
        )

    st.success("✅ Recomendação Acionável para o Centro de Distribuição")

    st.markdown("---")

    # --- Gráficos e Visualizações ---
    
    st.markdown("### 📊 Visualizações do Modelo")
    
    # Abas para diferentes visualizações
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Previsão", "🔄 Componentes", "📉 Histórico", "🧮 Detalhes"]
    )
    
    with tab1:
        st.subheader("Previsão de Demanda (Prophet)")
        st.markdown("Gráfico da previsão de demanda com intervalo de confiança (95%).")
        
        # Plotar a previsão
        fig_forecast = model.plot(forecast_df)
        fig_forecast.set_figwidth(12)
        fig_forecast.set_figheight(6)
        st.pyplot(fig_forecast)
        
    with tab2:
        st.subheader("Componentes do Modelo (Tendência, Sazonalidade e Feriados)")
        st.markdown("Decomposição da série temporal em seus componentes principais.")
        
        # Plotar componentes
        fig_components = model.plot_components(forecast_df, yearly_start=0)
        fig_components.set_figwidth(12)
        fig_components.set_figheight(10)
        st.pyplot(fig_components)
        
    with tab3:
        st.subheader("Histórico de Vendas (Dados de Treino)")
        st.markdown("Série temporal histórica da demanda para a combinação selecionada.")
        
        # Obter dados históricos
        df_hist = df_prophet[df_prophet['store_item'] == series_id].copy()
        
        # Criar gráfico de histórico
        fig_hist, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_hist['ds'], df_hist['y'], linewidth=2, color='#1f77b4', label='Vendas Reais')
        ax.set_xlabel('Data', fontsize=12)
        ax.set_ylabel('Quantidade Vendida', fontsize=12)
        ax.set_title(f'Histórico de Vendas - Loja {selected_store}, Item {selected_item}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_hist)
        
    with tab4:
        st.subheader("Detalhes do Cálculo")
        st.markdown("Parâmetros e métricas utilizadas na recomendação.")
        
        details = pd.DataFrame({
            'Métrica': [
                'Nível de Serviço Desejado',
                'Fator Z (Z-score)',
                'RMSE (Erro Diário)',
                'Desvio Padrão (90 dias)',
                'Demanda Média Prevista',
                'Estoque de Segurança',
                'Quantidade Ideal de Pedido'
            ],
            'Valor': [
                f"{service_level*100:.1f}%",
                f"{recommendation['fator_z']:.4f}",
                f"{rmse:.2f} unidades",
                f"{recommendation['desvio_padrao_demanda']:.2f} unidades",
                f"{recommendation['demanda_media_prevista']:.2f} unidades",
                f"{recommendation['estoque_seguranca']:.2f} unidades",
                f"{recommendation['quantidade_ideal_pedido']:.0f} unidades"
            ]
        })
        st.dataframe(details, use_container_width=True)
        
        # Explicação da fórmula
        st.markdown("### 📐 Fórmula de Cálculo")
        st.markdown("""
        A quantidade ideal de pedido é calculada como:
        
        **Q = D + SS**
        
        Onde:
        - **D**: Demanda Média Prevista (soma das previsões diárias)
        - **SS**: Estoque de Segurança = Z × σ
        - **Z**: Fator Z (obtido do nível de serviço desejado)
        - **σ**: Desvio Padrão da Demanda = RMSE × √(dias de previsão)
        
        O **Nível de Serviço** representa a probabilidade de não faltar estoque durante o período.
        """)

    st.markdown("---")
    
    # Rodapé
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px; margin-top: 30px;">
        <p>Ferramenta de Recomendação de Estoque | Parte 2 do Desafio de Data Science</p>
        <p>Desenvolvida com Streamlit e Prophet</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
