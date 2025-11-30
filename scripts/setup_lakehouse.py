#!/usr/bin/env python3
"""
Script para configurar Data Lakehouse usando DuckDB
Arquitetura Medallion: Bronze → Silver → Gold
"""

import duckdb
import pandas as pd
import os
from pathlib import Path

print("="*70)
print("CONFIGURAÇÃO DO DATA LAKEHOUSE")
print("Arquitetura Medallion: Bronze → Silver → Gold")
print("="*70)

# Caminhos
BASE_DIR = Path(__file__).parent.parent
LAKEHOUSE_DIR = BASE_DIR / "lakehouse"
BRONZE_DIR = LAKEHOUSE_DIR / "bronze"
SILVER_DIR = LAKEHOUSE_DIR / "silver"
GOLD_DIR = LAKEHOUSE_DIR / "gold"
DATA_DIR = BASE_DIR / "data"

# Criar conexão com DuckDB
db_path = LAKEHOUSE_DIR / "churn_lakehouse.duckdb"
conn = duckdb.connect(str(db_path))

print(f"\n✓ Conexão com DuckDB estabelecida: {db_path}")

# =============================================================================
# CAMADA BRONZE: Dados Brutos
# =============================================================================
print("\n" + "="*70)
print("CAMADA BRONZE: Ingestão de Dados Brutos")
print("="*70)

# Carregar CSV original
csv_path = DATA_DIR / "Telco-Customer-Churn.csv"
df_raw = pd.read_csv(csv_path)

print(f"\n[1/3] Carregando dados brutos de: {csv_path}")
print(f"      Dimensões: {df_raw.shape}")

# Salvar em formato Parquet (mais eficiente que CSV)
bronze_parquet = BRONZE_DIR / "churn_raw.parquet"
df_raw.to_parquet(bronze_parquet, index=False)

print(f"[2/3] Dados salvos em Parquet: {bronze_parquet}")
print(f"      Tamanho: {os.path.getsize(bronze_parquet) / 1024:.2f} KB")

# Criar tabela externa no DuckDB apontando para o Parquet
conn.execute(f"""
    CREATE OR REPLACE TABLE bronze_churn AS 
    SELECT * FROM read_parquet('{bronze_parquet}')
""")

print(f"[3/3] Tabela 'bronze_churn' criada no DuckDB")

# Verificar dados
row_count = conn.execute("SELECT COUNT(*) FROM bronze_churn").fetchone()[0]
print(f"      Registros: {row_count}")

# =============================================================================
# CAMADA SILVER: Dados Limpos e Transformados
# =============================================================================
print("\n" + "="*70)
print("CAMADA SILVER: Limpeza e Transformação")
print("="*70)

print("\n[1/4] Aplicando transformações...")

# Criar tabela Silver com dados limpos
conn.execute("""
    CREATE OR REPLACE TABLE silver_churn AS
    SELECT 
        -- Remover customerID (não é feature preditiva)
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        -- Converter TotalCharges para numérico (alguns valores são strings vazias ou espaços)
        TRY_CAST(NULLIF(TRIM(TotalCharges), '') AS DOUBLE) as TotalCharges,
        Churn
    FROM bronze_churn
    WHERE TRY_CAST(NULLIF(TRIM(TotalCharges), '') AS DOUBLE) IS NOT NULL  -- Remover NULLs
""")

print("      ✓ Conversão de TotalCharges para numérico")
print("      ✓ Remoção de valores ausentes")
print("      ✓ Remoção de customerID")

# Salvar em Parquet
silver_parquet = SILVER_DIR / "churn_clean.parquet"
conn.execute(f"""
    COPY silver_churn TO '{silver_parquet}' (FORMAT PARQUET)
""")

print(f"\n[2/4] Dados limpos salvos: {silver_parquet}")
print(f"      Tamanho: {os.path.getsize(silver_parquet) / 1024:.2f} KB")

# Verificar dados
silver_count = conn.execute("SELECT COUNT(*) FROM silver_churn").fetchone()[0]
removed = row_count - silver_count
print(f"\n[3/4] Registros na camada Silver: {silver_count}")
print(f"      Registros removidos: {removed} ({removed/row_count*100:.2f}%)")

# Estatísticas básicas
print("\n[4/4] Estatísticas da camada Silver:")
stats = conn.execute("""
    SELECT 
        COUNT(*) as total_clientes,
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churn_count,
        ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as churn_rate,
        ROUND(AVG(tenure), 2) as avg_tenure,
        ROUND(AVG(MonthlyCharges), 2) as avg_monthly_charges
    FROM silver_churn
""").fetchdf()

print(stats.to_string(index=False))

# =============================================================================
# CAMADA GOLD: Dados Prontos para Modelagem
# =============================================================================
print("\n" + "="*70)
print("CAMADA GOLD: Preparação para Modelagem")
print("="*70)

print("\n[1/3] Criando features para modelagem...")

# Criar tabela Gold com features prontas para ML
conn.execute("""
    CREATE OR REPLACE TABLE gold_churn_features AS
    SELECT 
        -- Features numéricas
        tenure,
        MonthlyCharges,
        TotalCharges,
        SeniorCitizen,
        
        -- Features categóricas (mantidas como string para posterior encoding)
        gender,
        Partner,
        Dependents,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        
        -- Target
        CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END as Churn_Binary
        
    FROM silver_churn
""")

print("      ✓ Features numéricas selecionadas")
print("      ✓ Features categóricas mantidas")
print("      ✓ Target convertido para binário (0/1)")

# Salvar em Parquet
gold_parquet = GOLD_DIR / "churn_features.parquet"
conn.execute(f"""
    COPY gold_churn_features TO '{gold_parquet}' (FORMAT PARQUET)
""")

print(f"\n[2/3] Features prontas salvas: {gold_parquet}")
print(f"      Tamanho: {os.path.getsize(gold_parquet) / 1024:.2f} KB")

# Exportar para CSV (para compatibilidade com notebooks existentes)
gold_csv = GOLD_DIR / "churn_features.csv"
conn.execute(f"""
    COPY gold_churn_features TO '{gold_csv}' (HEADER, DELIMITER ',')
""")

print(f"[3/3] Features exportadas para CSV: {gold_csv}")

# =============================================================================
# CRIAR VIEWS ANALÍTICAS
# =============================================================================
print("\n" + "="*70)
print("CRIANDO VIEWS ANALÍTICAS")
print("="*70)

# View: Análise de Churn por Contrato
conn.execute("""
    CREATE OR REPLACE VIEW view_churn_by_contract AS
    SELECT 
        Contract,
        COUNT(*) as total_clientes,
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churn_count,
        ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as churn_rate
    FROM silver_churn
    GROUP BY Contract
    ORDER BY churn_rate DESC
""")

print("\n✓ View criada: view_churn_by_contract")
print(conn.execute("SELECT * FROM view_churn_by_contract").fetchdf().to_string(index=False))

# View: Análise de Churn por Serviço de Internet
conn.execute("""
    CREATE OR REPLACE VIEW view_churn_by_internet AS
    SELECT 
        InternetService,
        COUNT(*) as total_clientes,
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churn_count,
        ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as churn_rate
    FROM silver_churn
    GROUP BY InternetService
    ORDER BY churn_rate DESC
""")

print("\n✓ View criada: view_churn_by_internet")
print(conn.execute("SELECT * FROM view_churn_by_internet").fetchdf().to_string(index=False))

# =============================================================================
# RESUMO FINAL
# =============================================================================
print("\n" + "="*70)
print("RESUMO DO DATA LAKEHOUSE")
print("="*70)

print(f"""
📊 Arquitetura Medallion Implementada:

🥉 BRONZE (Dados Brutos):
   - Tabela: bronze_churn
   - Arquivo: {bronze_parquet}
   - Registros: {row_count}

🥈 SILVER (Dados Limpos):
   - Tabela: silver_churn
   - Arquivo: {silver_parquet}
   - Registros: {silver_count}

🥇 GOLD (Pronto para ML):
   - Tabela: gold_churn_features
   - Arquivo: {gold_parquet}
   - Registros: {silver_count}

📈 Views Analíticas:
   - view_churn_by_contract
   - view_churn_by_internet

💾 Banco de Dados DuckDB:
   - Localização: {db_path}
   - Tamanho: {os.path.getsize(db_path) / 1024:.2f} KB
""")

# Listar todas as tabelas
print("\n📋 Tabelas disponíveis no Data Lakehouse:")
tables = conn.execute("SHOW TABLES").fetchdf()
print(tables.to_string(index=False))

# Fechar conexão
conn.close()

print("\n" + "="*70)
print("✅ DATA LAKEHOUSE CONFIGURADO COM SUCESSO!")
print("="*70)
print("\nPróximos passos:")
print("1. Use 'gold_churn_features.csv' para treinar modelos")
print("2. Consulte o DuckDB para análises SQL")
print("3. Acesse as camadas Bronze/Silver/Gold conforme necessário")
