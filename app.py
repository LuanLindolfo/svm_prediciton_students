import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Carregar o seu dataset
# Certifique-se de que o arquivo está na mesma pasta ou ajuste o caminho
df = pd.read_csv('student_dataset_10000_rows(1).csv')

# =====================================================================
# IMPORTANTÍSSIMO: Ajuste os nomes das colunas abaixo de acordo com o seu CSV!
# A ordem aqui DEVE ser a mesma ordem que organizamos no arquivo app.py:
# [study_hours, attendance, sleep_hours, internet_usage, assignments_completed, previous_score]
# =====================================================================
colunas_features = ['study_hours', 'attendance', 'sleep_hours', 'internet_usage', 'assignments_completed', 'previous_score']
coluna_alvo = 'performance_label' # Nome da coluna que tem "Alto", "Médio", "Baixo"

X = df[colunas_features]
y = df[coluna_alvo]

# 2. Pré-processamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Treinar o modelo SVM
# Usando C e gamma padrão, altere se fez tuning de hiperparâmetros
modelo_svm = SVC(kernel='rbf', C=1.0, random_state=42)
modelo_svm.fit(X_scaled, y_encoded)

print("✅ Modelo SVM treinado com sucesso!")

# 4. Criar a estrutura de pastas e salvar o dicionário com os objetos
os.makedirs('data/results', exist_ok=True)
caminho_salvamento = 'data/results/modelo_svm_alunos.pkl'

dados_para_salvar = {
    'model': modelo_svm,
    'scaler': scaler,
    'label_encoder': label_encoder
}

with open(caminho_salvamento, 'wb') as f:
    pickle.dump(dados_para_salvar, f)

print(f"🎉 Arquivo salvo com sucesso em: {caminho_salvamento}")
print("Agora é só enviar esse arquivo para o seu repositório no GitHub!")
