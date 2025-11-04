# **Projeto de Previsão de Demanda e Otimização de Estoque**

## **Visão Geral**

Este projeto é uma solução de Data Science de ponta a ponta que aborda o desafio de gestão de inventário (com base no desafio "Store Item Demand Forecasting"). O objetivo é ir além de uma simples previsão de vendas, entregando uma ferramenta de recomendação acionável que otimiza os níveis de estoque.

A solução é dividida em duas partes principais:

1. **Modelo Preditivo:** Um modelo de séries temporais que prevê com precisão a demanda futura de itens.  
2. **Ferramenta de Recomendação:** Uma aplicação web interativa que traduz a previsão em uma recomendação de pedido de estoque, com base em metas de negócio (Nível de Serviço).

## **🚀 Tecnologias Utilizadas**

* **Python 3.x**  
* **Pandas:** Para manipulação e análise de dados.  
* **Prophet:** Para modelagem e previsão de séries temporais.  
* **Scikit-learn:** Para cálculo de métricas de erro (RMSE).  
* **Streamlit:** Para a construção da ferramenta web interativa.  
* **Matplotlib:** Para visualização de dados.

## **📂 Estrutura dos Arquivos**

.  
├── 📁 dados/  
│   ├── train.csv         \# 5 anos de dados históricos de vendas (2013-2017)  
│   ├── test.csv          \# 3 meses de dados para previsão (Jan-Mar 2018\)  
│   └── sample\_submission.csv \# Exemplo de submissão do desafio original  
│  
├── 📜 o desafio sera beseado em datasets.txt \# Descrição oficial do desafio  
├── 📜 requirements.txt     \# Lista de bibliotecas Python necessárias  
│  
├──  notebooks/  
│   └── parte\_1\_modelo\_preditivo.ipynb \# Notebook com a análise e modelagem (Prophet)  
│  
└── 🚀 app.py                 \# Aplicação web Streamlit (Ferramenta de Recomendação)

*(Nota: Os arquivos .csv estão na raiz neste projeto, mas um README ideal sugere uma pasta dados/)*

### **Descrição dos Arquivos Principais**

* **parte\_1\_modelo\_preditivo.ipynb**: Notebook Jupyter que detalha a construção do modelo. Inclui:  
  * Análise exploratória dos dados.  
  * Justificativa da escolha do modelo (Prophet).  
  * Treinamento do modelo, capturando tendências e sazonalidades (anual e semanal).  
  * Visualização dos componentes do modelo (gráficos de decomposição).  
  * Avaliação do modelo (cálculo do RMSE).  
* **app.py**: O "produto de dados" final. Esta é uma aplicação web interativa onde o usuário (gerente) pode:  
  1. Selecionar uma Loja e um Item específico.  
  2. Definir um **Nível de Serviço** desejado (ex: 95%).  
  3. Receber uma recomendação clara de "Quantas unidades pedir" para o próximo período.  
  4. Visualizar a previsão de demanda futura para o item selecionado.  
  5. Ver em detalhes como o cálculo foi feito (Demanda Prevista \+ Estoque de Segurança).  
* **requirements.txt**: Define todas as dependências do projeto.

## **⚙️ Como Executar o Projeto**

Siga os passos abaixo para executar a ferramenta de recomendação localmente.

### **1\. Pré-requisitos**

* Python 3.8 ou superior  
* pip (gerenciador de pacotes Python)

### **2\. Instalação**

Clone este repositório e instale as dependências:

\# Clone o repositório (se aplicável)  
\# git clone https://...

\# Navegue até a pasta do projeto  
\# cd seu-projeto

\# Crie um ambiente virtual (Recomendado)  
python \-m venv venv  
source venv/bin/activate  \# No Windows: venv\\Scripts\\activate

\# Instale as bibliotecas necessárias  
pip install \-r requirements.txt

### **3\. Executando a Ferramenta**

Após a instalação, inicie a aplicação Streamlit com o seguinte comando no seu terminal:

streamlit run app.py

Seu navegador web será aberto automaticamente no endereço http://localhost:8501, exibindo a ferramenta de recomendação.

## **📈 Metodologia**

### **Parte 1: Modelo Preditivo (O Oráculo)**

Utilizamos o **Prophet** por sua habilidade superior em lidar com as características deste dataset:

* **Múltiplas Sazonalidades:** O modelo captura automaticamente os padrões semanais (vendas mais altas nos fins de semana) e anuais (picos no Natal).  
* **Tendência (Trend):** O modelo identifica se um produto está em crescimento ou declínio a longo prazo.  
* **Robustez:** Lida bem com dados faltantes ou outliers.

A incerteza do modelo é medida usando o **RMSE (Root Mean Squared Error)**, que nos diz, em média, quantas unidades o modelo tende a errar.

### **Parte 2: Recomendação de Estoque (O Estrategista)**

A ferramenta não apenas prevê a média, mas calcula o estoque ideal usando uma fórmula clássica de gestão de inventário:

$$\\text{Quantidade de Pedido (Q)} \= \\text{Demanda Prevista (D)} \+ \\text{Estoque de Segurança (SS)} $$Onde: \* \*\*D (Demanda Prevista):\*\* É a soma das previsões diárias (\`yhat\`) do Prophet para o período futuro. \* \*\*SS (Estoque de Segurança):\*\* É o "colchão" para proteger contra a incerteza da demanda. Ele é calculado como: $$SS \= Z \\times \\sigma $$ \\\* \*\*Z (Fator Z):\*\* Um valor estatístico que representa o \*\*Nível de Serviço\*\* desejado (ex: 95% \= Z de 1.645). $$\\text{Quantidade de Pedido (Q)} \= \\text{Demanda Prevista (D)} \+ \\text{Estoque de Segurança (SS)} $$Onde: \* \*\*D (Demanda Prevista):\*\* É a soma das previsões diárias (\`yhat\`) do Prophet para o período futuro. \* \*\*SS (Estoque de Segurança):\*\* É o "colchão" para proteger contra a incerteza da demanda. Ele é calculado como: $$SS \= Z \\times \\sigma $$ \\\* \*\*Z (Fator Z):\*\* Um valor estatístico que representa o \*\*Nível de Serviço\*\* desejado (ex: 95% \= Z de 1.645). \* \*\*$\\sigma$ (Incerteza):\*\* O desvio padrão da demanda durante o período. Estimamos isso usando o RMSE do nosso modelo. Esta abordagem transforma uma previsão estatística em uma decisão de negócios que equilibra o risco de falta de estoque com o custo de manutenção de inventário.$$ \* \*\*$\\sigma$ (Incerteza):\*\* O desvio padrão da demanda durante o período. Estimamos isso usando o RMSE do nosso modelo. Est$$\\text{Quantidade de Pedido (Q)} \= \\text{Demanda Prevista (D)} \+ \\text{Estoque de Segurança (SS)} $$Onde: \* \*\*D (Demanda Prevista):\*\* É a soma das previsões diárias (\`yhat\`) do Prophet para o período futuro. \* \*\*SS (Estoque de Segurança):\*\* É o "colchão" para proteger contra a incerteza da demanda. Ele é calculado como: $$SS \= Z \\times \\sigma $$ \\\* \*\*Z (Fator Z):\*\* Um valor estatístico que representa o \*\*Nível de Serviço\*\* desejado (ex: 95% \= Z de 1.645). \* \*\*$\\sigma$ (Incerteza):\*\* O desvio padrão da demanda durante o período. Estimamos isso usando o RMSE do nosso modelo. Esta abordagem transforma uma previsão estatística em uma decisão de negócios que equilibra o risco de falta de estoque com o custo de manutenção de inventário.$$a abordagem transforma uma previsão estatística em uma decisão de negócios que equilibra o risco de falta de estoque com o custo de manutenção de inventário.$$
