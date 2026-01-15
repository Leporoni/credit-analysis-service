# 🏦 Credit Analysis Service (AI Powered)

Este projeto é um microserviço de análise de crédito que utiliza Inteligência Artificial (Machine Learning) para prever o score de crédito de clientes bancários. O sistema consome dados históricos de um banco PostgreSQL, treina modelos de classificação e disponibiliza as previsões via API.

## 🚀 Tecnologias Utilizadas

*   **Linguagem:** Python 3.9
*   **Banco de Dados:** PostgreSQL 15
*   **Machine Learning:** Scikit-Learn (Random Forest & KNN)
*   **Manipulação de Dados:** Pandas
*   **API:** FastAPI
*   **Infraestrutura:** Docker & Docker Compose

---

## 🛠️ Como Executar o Projeto

### 1. Pré-requisitos
Certifique-se de ter o **Docker** e o **Docker Compose** instalados na sua máquina.

### 2. Iniciando o Ambiente
Para subir o banco de dados e o container da aplicação:

```bash
# Na pasta do projeto
sudo docker-compose up -d
```
> **O que isso faz?** Constrói a imagem Docker do Python, baixa a imagem do PostgreSQL e inicia ambos em rede.
> *   **API:** Acessível em `http://localhost:8000`
> *   **Banco de Dados:** Acessível na porta `5433` (externa)

### 3. Pipeline de Dados e IA

Como o ambiente é containerizado, executamos os comandos Python **dentro** do container `credit_app`.

#### Passo A: Importar Base de Dados
Carrega os dados do arquivo `clientes.csv` para o PostgreSQL.

```bash
sudo docker exec credit_app python import_data.py
```
*   **Resultado:** Cria a tabela `clientes` e insere ~100.000 registros.

#### Passo B: Treinar a Inteligência Artificial
Lê os dados do banco, trata valores nulos, converte textos em números e treina os modelos.

```bash
sudo docker exec credit_app python train_model.py
```
*   **Resultado:**
    1.  Compara Random Forest vs KNN.
    2.  Salva o "cérebro" da IA (`modelo_final.pkl`) e os metadados (`encoders.pkl`, `model_features.pkl`).
    3.  **Necessário reiniciar a API após o treino:** `sudo docker-compose restart app`

---

## 🔮 Como Usar a API (Previsões)

A API possui uma documentação interativa automática (Swagger UI).

1.  Acesse no seu navegador: **[http://localhost:8000/docs](http://localhost:8000/docs)**
2.  Clique no endpoint **`POST /predict`**.
3.  Clique no botão **Try it out**.
4.  Cole o JSON abaixo no campo de texto e clique em **Execute**.

### JSON de Exemplo (Teste de Bom Cliente)

```json
{
  "idade": 25,
  "salario_anual": 50000,
  "num_contas": 2,
  "num_cartoes": 3,
  "juros_emprestimo": 5,
  "num_emprestimos": 1,
  "dias_atraso": 2,
  "num_pagamentos_atrasados": 1,
  "num_verificacoes_credito": 2,
  "divida_total": 800,
  "taxa_uso_credito": 25,
  "idade_historico_credito": 200,
  "investimento_mensal": 500,
  "saldo_final_mes": 1200,
  "emprestimo_carro": 1,
  "emprestimo_casa": 0,
  "emprestimo_pessoal": 0,
  "emprestimo_credito": 0,
  "emprestimo_estudantil": 0,
  "profissao": "cientista",
  "mix_credito": "Bom",
  "comportamento_pagamento": "baixo_gasto_pagamento_alto"
}
```

### Resposta Esperada
```json
{
  "prediction": "Good",
  "status": "success"
}
```

---

## 📂 Estrutura do Projeto

*   `docker-compose.yml`: Orquestração dos containers.
*   `Dockerfile`: Definição do ambiente Python.
*   `main.py`: Código da API (FastAPI).
*   `import_data.py`: Script ETL (Extração e Carga) do CSV para o SQL.
*   `train_model.py`: Script de ML (Limpeza, Treino e Avaliação).
*   `clientes.csv`: Base de dados histórica (Raw Data).

## 📊 Modelos Analisados

O sistema avalia automaticamente dois algoritmos clássicos:
1.  **Random Forest (Floresta Aleatória):** Cria múltiplas árvores de decisão. Geralmente mais preciso para dados tabulares complexos.
2.  **KNN (K-Nearest Neighbors):** Baseia-se na similaridade com vizinhos próximos.

O script `train_model.py` seleciona automaticamente o vencedor e o salva como `modelo_final.pkl`.