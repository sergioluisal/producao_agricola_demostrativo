# 🌾 Dashboard de Análise Agrícola

Este projeto é um **Dashboard interativo** desenvolvido com [Streamlit](https://streamlit.io/) para análise de dados agrícolas. Ele permite **carregar arquivos CSV/XLSX**, aplicar **filtros interativos**, visualizar **métricas estatísticas** e **gráficos dinâmicos** sobre produtividade, clima e solo.

## 🚀 Funcionalidades

- Upload de arquivos (.csv, .xls, .xlsx)
- Renomeação automática de colunas para padronização
- Filtros por:
  - Região
  - Tipo de Solo
  - Cultura
  - Condição Climática
  - Uso de Fertilizante
  - Uso de Irrigação
- Exibição de métricas:
  - Produtividade média
  - Chuva média
  - Temperatura média
  - Correlação entre chuva e produtividade
- Visualizações com Plotly:
  - Barras por Cultura, Região e Solo
  - Dispersão Chuva x Produtividade com linha de tendência
  - Histograma de produtividade
- Exportação dos dados filtrados

## 🧪 Tecnologias Utilizadas

- Python
- Streamlit
- Pandas
- Numpy
- Plotly (Express + GraphObjects)

## 🛠️ Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/seunome/dashboard-analise-agricola.git
   cd dashboard-analise-agricola
