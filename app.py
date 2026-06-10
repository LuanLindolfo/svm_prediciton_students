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

CAMINHO_MODELO = 'data/results/modelo_svm_alunos.pkl'
CAMINHO_CSV = 'student_dataset_10000_rows(1).csv'

# Torna o treinamento inteligente e imune a pequenos erros de nomes de colunas
def treinar_e_salvar_modelo_automatico():
    if not os.path.exists(CAMINHO_CSV):
        st.error(f"❌ Erro: Nem o modelo `{CAMINHO_MODELO}` nem o CSV `{CAMINHO_CSV}` foram encontrados no repositório.")
        st.stop()
        
    try:
        # sep=None e engine='python' detectam automaticamente se é vírgula ou ponto e vírgula
        df = pd.read_csv(CAMINHO_CSV, sep=None, engine='python')
        
        # Corrige o problema de espaços invisíveis nos nomes das colunas
        df.columns = df.columns.str.strip()
        
        # Mapeamento inteligente por aproximação de texto (ignora maiúsculas/minúsculas)
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

        # Identifica a coluna alvo dinamicamente
        coluna_alvo = None
        for col in df.columns:
            if col not in colunas_mapeadas.values():
                if any(k in col.lower() for k in ['perf', 'label', 'class', 'res', 'status', 'alvo', 'index']):
                    coluna_alvo = col
                    break
        
        if not coluna_alvo:
            sobras = [c for c in df.columns if c not in colunas_mapeadas.values()]
            if sobras:
                coluna_alvo = sobras[-1]

        features_obrigatorias = ['study_hours', 'attendance', 'sleep_hours', 'internet_usage', 'assignments_completed', 'previous_score']
        
        # Se falhar em mapear, mostra as colunas reais para o usuário
        if len(colunas_mapeadas) < len(features_obrigatorias) or not coluna_alvo:
            st.error("❌ Não foi possível alinhar as colunas do seu CSV automaticamente.")
            st.write("Colunas encontradas no seu arquivo:", df.columns.tolist())
            st.stop()

        # Garante a ordem exata das colunas
        X = df[[colunas_mapeadas[f] for f in features_obrigatorias]]
        y = df[coluna_alvo]

        # Pré-processamento
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Treinamento do SVM
        svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
        svm_model.fit(X_scaled, y_encoded)

        # Salva o arquivo final para não precisar treinar toda vez que a página recarregar
        os.makedirs(os.path.dirname(CAMINHO_MODELO), exist_ok=True)
        dados_para_salvar = {
            'model': svm_model,
            'scaler': scaler,
            'label_encoder': label_encoder
        }
        with open(CAMINHO_MODELO, 'wb') as f:
            pickle.dump(dados_para_salvar, f)
            
        return svm_model, scaler, label_encoder

    except Exception as e:
        st.error(f"Erro ao processar o arquivo de dados: {e}")
        st.stop()

# 2. Carregar ou Inicializar os Componentes do Modelo
@st.cache_resource
def carregar_ou_treinar():
    if os.path.exists(CAMINHO_MODELO):
        with open(CAMINHO_MODELO, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['scaler'], data['label_encoder']
    else:
        return treinar_e_salvar_modelo_automatico()

svm_model, scaler, labelencoder = carregar_ou_treinar()

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
    
    # 6. Exibição Dinâmica dos Resultados (compatível com inglês ou português no CSV)
    st.markdown("#### Resultado da Análise:")
    label_str = str(pred_label).strip().lower()
    
    if label_str in ["alto", "high"]:
        st.success("🎉 Rendimento Alto")
    elif label_str in ["médio", "medio", "medium"]:
        st.warning("📘 Rendimento Médio")
    elif label_str in ["baixo", "low"]:
        st.error("⚠️ Rendimento Baixo")
    else:
        st.info(f"Resultado: {pred_label}")
