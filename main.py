from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

app = FastAPI(
    title="Credit Analysis API",
    description="API para previsão de score de crédito usando Random Forest/KNN",
    version="1.1"
)

# Variáveis globais para armazenar o modelo e metadados
model = None
encoders = None
target_encoder = None
model_features = None

# Carregar modelo ao iniciar
try:
    model = joblib.load('modelo_final.pkl')
    encoders = joblib.load('encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    model_features = joblib.load('model_features.pkl') # Lista com a ordem correta das colunas
    print("✅ Modelo e artefatos carregados com sucesso!")
except Exception as e:
    print(f"⚠️ Aviso: Não foi possível carregar o modelo: {e}")
    print("Certifique-se de rodar o 'train_model.py' primeiro.")

class CustomerData(BaseModel):
    # Features Numéricas
    idade: float
    salario_anual: float
    num_contas: float
    num_cartoes: float
    juros_emprestimo: float
    num_emprestimos: float
    dias_atraso: float
    num_pagamentos_atrasados: float
    num_verificacoes_credito: float
    divida_total: float
    taxa_uso_credito: float
    idade_historico_credito: float
    investimento_mensal: float
    saldo_final_mes: float
    
    # Features Binárias (0 ou 1)
    emprestimo_carro: int
    emprestimo_casa: int
    emprestimo_pessoal: int
    emprestimo_credito: int
    emprestimo_estudantil: int
    
    # Features de Texto (Categoricas)
    profissao: str
    mix_credito: str
    comportamento_pagamento: str

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
def predict_score(data: CustomerData):
    if not model:
        raise HTTPException(status_code=503, detail="Modelo de IA não está carregado no servidor.")

    try:
        # 1. Converter entrada para DataFrame
        input_dict = data.dict()
        df_input = pd.DataFrame([input_dict])

        # 2. Aplicar os Encoders (Texto -> Número)
        for col, encoder in encoders.items():
            if col in df_input.columns:
                val = df_input[col].iloc[0]
                # Se o valor existe no encoder (ex: 'Engenheiro'), usa o código
                if val in encoder.classes_:
                    df_input[col] = encoder.transform([val])
                else:
                    # Valor desconhecido -> 0
                    df_input[col] = 0

        # 3. CRUCIAL: Reordenar as colunas para bater com o treinamento
        # Isso corrige o erro "Feature names should match"
        if model_features:
            df_input = df_input[model_features]

        # 4. Fazer a previsão
        prediction_index = model.predict(df_input)[0]
        
        # 5. Decodificar o resultado
        prediction_label = target_encoder.inverse_transform([prediction_index])[0]

        return {
            "prediction": prediction_label,
            "status": "success"
        }

    except Exception as e:
        # Imprime o erro no terminal para ajudar a debugar
        print(f"Erro na previsão: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar previsão: {str(e)}")