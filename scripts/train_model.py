#!/usr/bin/env python3
"""
Script para processar dados e treinar o modelo de previsão de churn
"""

import pandas as pd
import numpy as np
import os
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("="*70)
print("PROCESSAMENTO DE DADOS E TREINAMENTO DO MODELO")
print("="*70)

# 1. Carregar dados brutos
print("\n[1/5] Carregando dados brutos...")
df = pd.read_csv('../data/Telco-Customer-Churn.csv')
print(f"✓ Dados carregados: {df.shape}")

# 2. Limpeza de dados
print("\n[2/5] Limpando dados...")
# Converter TotalCharges para numérico
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Remover valores ausentes
df_clean = df.dropna()
# Remover customerID
df_clean = df_clean.drop('customerID', axis=1)
print(f"✓ Dados limpos: {df_clean.shape}")

# 3. Preparar features e target
print("\n[3/5] Preparando features e target...")
selected_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

X = df_clean[selected_features]
y = df_clean['Churn']

# Identificar variáveis categóricas
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Converter target para numérico
y_encoded = y.map({'No': 0, 'Yes': 1})

print(f"✓ Features codificadas: {X_encoded.shape}")
print(f"✓ Target codificado: {y_encoded.shape}")

# 4. Dividir em treino e teste
print("\n[4/5] Dividindo em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

# Criar diretório processed se não existir
os.makedirs('../data/processed', exist_ok=True)

# Salvar datasets processados
X_train.to_csv('../data/processed/X_train.csv', index=False)
X_test.to_csv('../data/processed/X_test.csv', index=False)
y_train.to_csv('../data/processed/y_train.csv', index=False, header=['Churn'])
y_test.to_csv('../data/processed/y_test.csv', index=False, header=['Churn'])

print(f"✓ Treino: {X_train.shape}, Teste: {X_test.shape}")
print(f"✓ Dados salvos em: ../data/processed/")

# 5. Treinar modelo final
print("\n[5/5] Treinando modelo Gradient Boosting...")
model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_train, y_train)
print("✓ Modelo treinado!")

# Avaliar no conjunto de teste
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*70)
print("MÉTRICAS DO MODELO NO CONJUNTO DE TESTE")
print("="*70)
print(f"Acurácia:  {accuracy:.4f}")
print(f"Precisão:  {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Salvar modelo com pickle
model_path_pickle = '../modelo_final.pkl'
with open(model_path_pickle, 'wb') as f:
    pickle.dump(model, f)
print(f"\n✓ Modelo salvo com pickle em: {model_path_pickle}")

# Salvar modelo com joblib
model_path_joblib = '../modelo_final_joblib.pkl'
joblib.dump(model, model_path_joblib)
print(f"✓ Modelo salvo com joblib em: {model_path_joblib}")

print("\n" + "="*70)
print("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
print("="*70)
