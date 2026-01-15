import pandas as pd
import os
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Configuração do Banco
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://admin:admin@localhost:5433/credit_score_db')

def train_and_evaluate():
    print("--- 1. Carregando dados do Banco de Dados ---")
    engine = create_engine(DATABASE_URL)
    # Lendo a tabela inteira
    df = pd.read_sql("SELECT * FROM clientes", engine)
    print(f"Dados carregados: {df.shape[0]} linhas e {df.shape[1]} colunas.")

    print("\n--- 2. Limpeza e Pré-processamento ---")
    
    # Removendo colunas que não ajudam na previsão (IDs, nomes, etc)
    # id_cliente: é apenas um identificador
    # nome: não temos aqui, mas se tivesse removeríamos
    df = df.drop(columns=['id_cliente', 'mes'], errors='ignore')

    # Tratamento de valores vazios (simples: preencher numéricos com média, texto com 'Desconhecido')
    # Na vida real faríamos uma análise mais profunda, mas para o protótipo isso garante que não quebre.
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Desconhecido')
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Codificação (Transformar Texto em Números)
    # Ex: Profissão "Engenheiro" vira 1, "Advogado" vira 2
    encoders = {}
    label_encoder_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'score_credito':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            label_encoder_cols.append(col)

    # A coluna ALVO (Target) também precisa ser numérica
    # Good, Standard, Poor
    target_encoder = LabelEncoder()
    df['score_credito'] = target_encoder.fit_transform(df['score_credito'])
    
    print("Colunas de texto convertidas para números com sucesso.")

    print("\n--- 3. Separação Treino vs Teste ---")
    # X são as características (tudo menos o score)
    # y é o alvo (só o score)
    X = df.drop(columns=['score_credito'])
    y = df['score_credito']

    # 70% para treinar a IA, 30% para testar se ela aprendeu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Treinando com {len(X_train)} clientes. Testando com {len(X_test)} clientes.")

    print("\n--- 4. Treinando Modelos ---")
    
    # Modelo 1: Random Forest
    print("Treinando Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Acurácia Random Forest: {rf_acc:.2%}")

    # Modelo 2: KNN (K-Nearest Neighbors)
    print("Treinando KNN...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_pred)
    print(f"Acurácia KNN: {knn_acc:.2%}")

    print("\n--- 5. Conclusão e Importância das Variáveis ---")
    best_model = rf_model if rf_acc > knn_acc else knn_model
    best_name = "Random Forest" if rf_acc > knn_acc else "KNN"
    
    print(f"O melhor modelo foi: {best_name}")

    if best_name == "Random Forest":
        importances = pd.Series(data=rf_model.feature_importances_, index=X.columns)
        print("\nTop 5 características mais importantes para definir o score:")
        print(importances.sort_values(ascending=False).head(5))

    # Salvando o melhor modelo para usar na API depois
    joblib.dump(best_model, 'modelo_final.pkl')
    joblib.dump(encoders, 'encoders.pkl') # Precisamos disso para converter os dados dos novos clientes
    joblib.dump(target_encoder, 'target_encoder.pkl') # Para saber o que é 0, 1, 2 (Good, Standard, Poor)
    print("\nModelo salvo como 'modelo_final.pkl'!")

if __name__ == "__main__":
    train_and_evaluate()
