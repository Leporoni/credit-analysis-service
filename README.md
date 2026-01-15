# 🏦 Credit Analysis Service (AI Powered)

Este projeto é um microserviço de análise de crédito que utiliza Inteligência Artificial (Machine Learning) para prever o score de crédito de clientes bancários. O sistema consome dados históricos de um banco PostgreSQL, treina modelos de classificação e disponibiliza as previsões.

## 🚀 Tecnologias Utilizadas

*   **Linguagem:** Python 3.9
*   **Banco de Dados:** PostgreSQL 15
*   **Machine Learning:** Scikit-Learn (Random Forest & KNN)
*   **Manipulação de Dados:** Pandas
*   **API:** FastAPI (Em construção)
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
> **O que isso faz?** Constrói a imagem Docker do Python, baixa a imagem do PostgreSQL e inicia ambos em rede. O banco ficará acessível na porta `5433` do seu host.

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
    2.  Exibe a acurácia (Ex: ~81%).
    3.  Mostra as variáveis mais importantes (Ex: Dívida Total).
    4.  Salva o "cérebro" da IA em arquivos `.pkl` para uso na API.

---

## 📂 Estrutura do Projeto

*   `docker-compose.yml`: Orquestração dos containers.
*   `Dockerfile`: Definição do ambiente Python.
*   `import_data.py`: Script ETL (Extração e Carga) do CSV para o SQL.
*   `train_model.py`: Script de ML (Limpeza, Treino e Avaliação).
*   `clientes.csv`: Base de dados histórica (Raw Data).

## 📊 Modelos Analisados

O sistema avalia automaticamente dois algoritmos clássicos:
1.  **Random Forest (Floresta Aleatória):** Cria múltiplas árvores de decisão. Geralmente mais preciso para dados tabulares complexos.
2.  **KNN (K-Nearest Neighbors):** Baseia-se na similaridade com vizinhos próximos.

O script `train_model.py` seleciona automaticamente o vencedor e o salva como `modelo_final.pkl`.
