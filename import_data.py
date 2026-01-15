import pandas as pd
from sqlalchemy import create_engine
import os

# Configuração da conexão (pega do ambiente ou usa o default do Docker)
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://admin:admin@localhost:5432/credit_score_db')

def import_data():
    print("Iniciando leitura do CSV...")
    # Lendo o CSV
    try:
        df = pd.read_csv('clientes.csv')
        print(f"CSV lido com sucesso! {len(df)} registros encontrados.")
    except FileNotFoundError:
        print("Erro: Arquivo 'clientes.csv' não encontrado.")
        return

    # Criando engine do SQLAlchemy
    engine = create_engine(DATABASE_URL)

    print("Conectando ao banco de dados e enviando dados...")
    # Salvando no banco (substitui a tabela se existir)
    try:
        df.to_sql('clientes', engine, if_exists='replace', index=False)
        print("Sucesso! Dados importados para a tabela 'clientes'.")
    except Exception as e:
        print(f"Erro ao salvar no banco: {e}")

if __name__ == "__main__":
    import_data()
