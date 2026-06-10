import streamlit as st
import pickle
import numpy as np
import os

# 1. Configuração da Página
st.set_page_config(
    page_title="Previsão de Rendimento Escolar",
    page_icon="📊",
    layout="centered"
)

# 2. Carregar o Modelo e Objetos de Pré-processamento
@st.cache_resource
def carregar_modelo():
    # Caminho atualizado para apontar para o modelo SVM
    caminho_modelo = 'data/results/modelo_svm_alunos.pkl'
    
    if not os.path.exists(caminho_modelo):
        st.error(f"⚠️ O arquivo '{caminho_modelo}' não foi encontrado. Verifique se o push para o GitHub funcionou e se o nome do arquivo está correto.")
        st.stop()
        
    with open(caminho_modelo, 'rb') as f:
        data = pickle.load(f)
        
    return data['model'], data['scaler'], data['label_encoder']

# Inicializando o modelo (variável renomeada para svm_model)
try:
    svm_model, scaler, labelencoder = carregar_modelo()
except Exception as e:
    st.error(f"Erro ao carregar os componentes do modelo: {e}")
    st.stop()

# 3. Cabeçalho da Interface
st.title("📊 Previsão de Rendimento Escolar")
# Texto atualizado para refletir o uso do SVM
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

st.write("") # Espaçamento

# 5. Botão de Previsão e Lógica
if st.button("Prever Rendimento", use_container_width=True, type="primary"):
    # Organizando a entrada (Verifique se a ordem bate com X.columns do seu treino)
    dados = [study_hours, attendance, sleep_hours, internet_usage, assignments_completed, previous_score]
    entrada = np.array(dados).reshape(1, -1)
    
    # Transformação com o Scaler e Previsão (chamando svm_model)
    entrada_scaled = scaler.transform(entrada)
    pred_num = svm_model.predict(entrada_scaled)[0]
    pred_label = labelencoder.inverse_transform([pred_num])[0]
    
    # 6. Exibição Dinâmica dos Resultados
    st.markdown("#### Resultado da Análise:")
    if pred_label == "Alto":
        st.success("🎉 Rendimento Alto")
    elif pred_label == "Médio":
        st.warning("📘 Rendimento Médio")
    elif pred_label == "Baixo":
        st.error("⚠️ Rendimento Baixo")
    else:
        st.info(f"Resultado: {pred_label}")
