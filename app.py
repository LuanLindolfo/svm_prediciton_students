import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Configuração da Página
st.set_page_config(
    page_title="Previsão de Rendimento Escolar",
    page_icon="📊",
    layout="centered"
)

CAMINHO_CSV = 'student_dataset_10000_rows(1).csv'

# 2. Treinamento Dinâmico (Garante adaptação total ao arquivo do repositório)
@st.cache_resource
def carregar_e_treinar_svm():
    if not os.path.exists(CAMINHO_CSV):
        st.error(f"❌ Erro: O arquivo de dados `{CAMINHO_CSV}` não foi encontrado no repositório.")
        st.stop()
        
    try:
        # Detecta automaticamente o separador (vírgula ou ponto e vírgula)
        df = pd.read_csv(CAMINHO_CSV, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        
        # Mapeamento robusto das colunas por aproximação de nome
        colunas_mapeadas = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'stud' in col_lower or 'estudo' in col_lower:
                colunas_mapeadas['study_hours'] = col
            elif 'attend' in col_lower or 'frequen' in col_lower:
                colunas_mapeadas['attendance'] = col
            elif 'sleep' in col_lower or 'sono' in col_lower:
                colunas_mapeadas['sleep_hours'] = col
            elif 'inter' in col_lower or 'net' in col_lower:
                colunas_mapeadas['internet_usage'] = col
            elif 'assign' in col_lower or 'taref' in col_lower or 'entreg' in col_lower:
                colunas_mapeadas['assignments_completed'] = col
            elif 'prev' in col_lower or 'nota' in col_lower or 'score' in col_lower:
                colunas_mapeadas['previous_score'] = col

        # Identifica a coluna alvo (Target)
        coluna_alvo = None
        for col in df.columns:
            if col not in colunas_mapeadas.values():
                if any(k in col.lower() for k in ['perf', 'label', 'class', 'res', 'status', 'alvo', 'place']):
                    coluna_alvo = col
                    break
        
        if not coluna_alvo:
            sobras = [c for c in df.columns if c not in colunas_mapeadas.values()]
            if sobras:
                coluna_alvo = sobras[-1]

        features_obrigatorias = ['study_hours', 'attendance', 'sleep_hours', 'internet_usage', 'assignments_completed', 'previous_score']
        
        if len(colunas_mapeadas) < len(features_obrigatorias) or not coluna_alvo:
            st.error("❌ Erro ao alinhar as colunas do arquivo CSV.")
            st.stop()

        # Organiza os dados na ordem correta
        X = df[[colunas_mapeadas[f] for f in features_obrigatorias]]
        y = df[coluna_alvo]

        # Pré-processamento
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # SOLUÇÃO: Kernel linear + balanced impede o modelo de ficar preso em uma única previsão
        svm_model = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)
        svm_model.fit(X_scaled, y_encoded)
        
        return svm_model, scaler, label_encoder

    except Exception as e:
        st.error(f"Erro ao processar e treinar o modelo: {e}")
        st.stop()

# Inicializa o modelo dinamicamente e guarda em cache na memória do app
svm_model, scaler, labelencoder = carregar_e_treinar_svm()

# 3. Cabeçalho da Interface
st.title("📊 Previsão de Rendimento Escolar")
st.markdown("### Support Vector Machine (SVM) — preencha os dados do aluno")
st.divider()

# 4. Criando o Layout (Grid 2x3)
col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("Horas de estudo/dia", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
    sleep_hours = st.number_input("Horas de sono/dia", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    assignments_completed = st.number_input("Tarefas entregues (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

with col2:
    attendance = st.number_input("Frequência (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0)
    internet_usage = st.number_input("Uso de internet (h/dia)", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
    previous_score = st.number_input("Nota anterior (0–100)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

st.write("") 

# 5. Botão de Previsão e Lógica
if st.button("Prever Rendimento", use_container_width=True, type="primary"):
    dados = [study_hours, attendance, sleep_hours, internet_usage, assignments_completed, previous_score]
    entrada = np.array(dados).reshape(1, -1)
    
    entrada_scaled = scaler.transform(entrada)
    pred_num = svm_model.predict(entrada_scaled)[0]
    pred_label = labelencoder.inverse_transform([pred_num])[0]
    
    # 6. Exibição Dinâmica dos Resultados em Português
    st.markdown("#### Resultado da Análise:")
    label_str = str(pred_label).strip().lower()
    
    if label_str in ["alto", "high", "placed"]:
        st.success("🎉 Rendimento Alto")
    elif label_str in ["médio", "medio", "medium"]:
        st.warning("📘 Rendimento Médio")
    elif label_str in ["baixo", "low", "not placed"]:
        st.error("⚠️ Rendimento Baixo")
    else:
        st.info(f"Resultado: {pred_label}")
