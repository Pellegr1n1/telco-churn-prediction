# Projeto Final de Ciência de Dados: Previsão de Churn de Clientes de Telecomunicações

### **Alunos:** Leandro Pellegrini e Vítor Celestino

---

## Visão Geral

O objetivo deste trabalho é consolidar o aprendizado do ciclo completo de um projeto de dados, desde a concepção do problema até a entrega de um modelo funcional. O foco desta avaliação é demonstrar a construção de um modelo preditivo de classificação para prever o churn (cancelamento) de clientes em uma empresa de telecomunicações, justificando cada etapa do processo.

## Estrutura do Projeto

O repositório está organizado com a seguinte estrutura:

```
/telco-churn-prediction
├── README.md
├── notebooks/
│   ├── 01_data_pipeline.ipynb    # Notebook de exploração e pipeline de dados
│   ├── 02_modelagem_comparativa.ipynb # Notebook de modelagem e avaliação
│   └── 03_deploy_modelo.ipynb    # Notebook de deploy e uso do modelo
├── data/
│   ├── Telco-Customer-Churn.csv # Dataset bruto original
│   └── processed/                 # Dados processados (legado)
├── lakehouse/                # Data Lakehouse (DuckDB + Parquet)
│   ├── churn_lakehouse.duckdb    # Banco de dados DuckDB
│   ├── bronze/                   # 🥉 Camada Bronze (dados brutos)
│   ├── silver/                   # 🥈 Camada Silver (dados limpos)
│   └── gold/                     # 🥇 Camada Gold (pronto para ML)
├── scripts/
│   ├── train_model.py            # Script de treinamento
│   └── setup_lakehouse.py        # Setup do Data Lakehouse
├── requirements.txt          # Dependências
├── modelo_final.pkl          # Modelo salvo com pickle
├── modelo_final_joblib.pkl   # Modelo salvo com joblib
├── DATA_LAKEHOUSE.md         # Documentação do Lakehouse
└── PICKLE_VS_JOBLIB.md       # Comparação pickle vs joblib
```### Como Rodar o Projeto

1.  **Clone o repositório:**

```bash
git clone https://github.com/Pellegr1n1/telco-churn-prediction.git
cd telco-churn-prediction
```

2.  **Instale as dependências:**

```bash
pip install -r requirements.txt
```

3.  **Configure o Data Lakehouse:**

```bash
cd scripts
python3 setup_lakehouse.py
```

Este script criará a arquitetura Medallion (Bronze → Silver → Gold) com DuckDB.

4.  **Execute os notebooks Jupyter:**
    Abra o Jupyter Lab ou Jupyter Notebook e execute os notebooks na ordem numérica:
    - `01_data_pipeline.ipynb`
    - `02_modelagem_comparativa.ipynb`
    - `03_deploy_modelo.ipynb`

---

## Parte 1: O Problema de Negócio

Nosso projeto se insere no contexto do **mercado de telecomunicações**, um setor altamente competitivo onde a retenção de clientes é um desafio estratégico. A perda de clientes (churn) representa não apenas a perda de receita recorrente, mas também custos elevados para adquirir novos clientes. Portanto, prever quais clientes estão em risco de cancelar seus serviços permite que a empresa tome ações proativas para retê-los, como ofertas personalizadas, descontos ou melhorias no serviço.

### 1.2. Pergunta de Negócio

A pergunta central que guiou toda a nossa análise foi:

> **"É possível construir um modelo de machine learning que preveja com alta precisão se um cliente de telecomunicações irá cancelar seu serviço (churn) com base em suas características demográficas, de contrato e de uso do serviço?"**

### 1.3. Objetivo do Modelo

O objetivo do nosso modelo preditivo de classificação é:

> **"Construir um modelo de classificação que identifique clientes com alta probabilidade de churn, fornecendo uma ferramenta de apoio para que a equipe de retenção possa agir de forma proativa e direcionada, otimizando os esforços e reduzindo a perda de receita."**

---

## Parte 2: Pipeline e Arquitetura

O pipeline completo de dados, desde a coleta até a preparação para modelagem, está detalhado no notebook `01_data_pipeline.ipynb`. As principais etapas foram:

1.  **Origem e Repositório:** O dataset "Telco Customer Churn" foi obtido do Kaggle. Implementamos um **Data Lakehouse** usando **DuckDB** com **Arquitetura Medallion** (Bronze → Silver → Gold). Veja `DATA_LAKEHOUSE.md` para detalhes completos.

2.  **Ingestão:** Carregamento do arquivo `Telco-Customer-Churn.csv`.

3.  **Limpeza e Transformação (ETL):**
    -   Conversão da coluna `TotalCharges` para formato numérico, tratando valores inválidos.
    -   Remoção de 11 linhas com valores ausentes (0.15% do total).
    -   Remoção da coluna `customerID`, que não possui valor preditivo.

4.  **Análise Exploratória (EDA):**
    -   Análise da distribuição da variável alvo (`Churn`), identificando um desbalanceamento de classes (73.5% "Não" vs. 26.5% "Sim").
    -   Identificação de variáveis com forte correlação com o churn, como `Contract` (contrato mensal), `tenure` (baixo tempo de permanência) e `InternetService` (fibra óptica).

5.  **Preparação para Modelagem:**
    -   **Codificação de Variáveis Categóricas:** Utilização de One-Hot Encoding para transformar variáveis textuais em formato numérico.
    -   **Divisão dos Dados:** Separação do dataset em conjuntos de treino (80%) e teste (20%), utilizando amostragem estratificada para manter a proporção de churn em ambos.

---

## Parte 3: Modelagem e Avaliação Comparativa

A etapa de modelagem e avaliação, detalhada no notebook `02_modelagem_comparativa.ipynb`, consistiu em:

### 3.1. Treinamento de Três Modelos

Foram treinados e avaliados 6 algoritmos de classificação:
1.  Regressão Logística
2.  Árvore de Decisão
3.  Random Forest
4.  **Gradient Boosting (Modelo Escolhido)**
5.  SVM
6.  KNN

### 3.2. Avaliação com Três Métricas

As métricas escolhidas foram:
-   **Acurácia:** Desempenho geral.
-   **Precisão:** Importante para evitar custos com ações de retenção desnecessárias.
-   **Recall (Sensibilidade):** **Métrica mais importante para este problema**, pois nosso objetivo é identificar o máximo possível de clientes que irão cancelar.
-   **F1-Score:** Média harmônica entre precisão e recall, útil para dados desbalanceados.

### 3.3. Análise Comparativa dos Resultados

A tabela abaixo resume o desempenho dos modelos no conjunto de teste:

| Modelo                | Acurácia | Precisão | Recall | F1-Score |
|-----------------------|----------|----------|--------|----------|
| **Gradient Boosting** | 0.814    | 0.665    | 0.587  | 0.624    |
| Random Forest         | 0.806    | 0.667    | 0.523  | 0.586    |
| Regressão Logística   | 0.810    | 0.658    | 0.579  | 0.616    |
| SVM                   | 0.795    | 0.673    | 0.437  | 0.530    |
| Árvore de Decisão     | 0.780    | 0.591    | 0.566  | 0.578    |
| KNN                   | 0.767    | 0.561    | 0.512  | 0.535    |

**Discussão:** O modelo **Gradient Boosting** foi escolhido como o melhor modelo. Embora a Regressão Logística tenha um F1-Score similar, o Gradient Boosting apresentou um **Recall superior**, que é a métrica prioritária para o nosso problema de negócio. Ele consegue identificar uma maior proporção de clientes que realmente irão cancelar, permitindo uma ação de retenção mais eficaz.

---

## Parte 4: Deploy

O processo de deploy e uso prático do modelo está documentado no notebook `03_deploy_modelo.ipynb`.

### 4.1. Salvando o Modelo Treinado

O modelo final (`GradientBoostingClassifier`) foi treinado com todos os dados de treino e salvo usando **duas bibliotecas diferentes** para demonstrar ambos os métodos:

**Método 1: Usando Pickle**
```python
import pickle

# Salvar o modelo
with open('modelo_final.pkl', 'wb') as f:
    pickle.dump(final_model, f)
```

**Método 2: Usando Joblib (recomendado para scikit-learn)**
```python
import joblib

# Salvar o modelo
joblib.dump(final_model, 'modelo_final_joblib.pkl')
```

**Arquivos gerados:**
- `modelo_final.pkl` (433 KB) - Versão pickle
- `modelo_final_joblib.pkl` (436 KB) - Versão joblib

### 4.2. Carregando e Utilizando o Modelo

Demonstramos o uso do modelo carregando-o e fazendo uma previsão para um **novo cliente fictício** com alto risco de churn.

**Exemplo de Novo Dado:**
-   Contrato: Mensal
-   Tempo de permanência: 3 meses
-   Serviço de internet: Fibra óptica

**Resultado da Previsão:**

```
 Previsão de Churn: Sim
 Probabilidade de Churn: 68.45%
```

**Explicação:** O modelo previu corretamente que este cliente tem uma alta probabilidade de cancelar o serviço. Com essa informação, a equipe de retenção pode entrar em contato proativamente para oferecer benefícios e evitar o churn, validando a utilidade prática do nosso trabalho.
