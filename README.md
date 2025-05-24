# Analisador de Imagens com Deep Learning - Frontend React e App Streamlit

Este repositório contém um projeto para análise de imagens usando Deep Learning, dividido em duas partes principais:

1.  **`frontend/`**: Uma aplicação React interativa (existente) que simula e permite a exploração de conceitos de classificação e clusterização de imagens. Ideal para demonstrações e design de UI.
    *   **Chat com IA - Marcelo Claro**: Inclui um chat interativo com um assistente de IA (Marcelo Claro) que utiliza a API Gemini. Marcelo Claro pode:
        *   Perguntar sobre o tipo de classificação que está sendo realizada para contextualizar suas respostas.
        *   Analisar os resultados da simulação do modelo.
        *   Simular a consulta a "agentes de pesquisa especializados" para perguntas complexas, exibindo um log dessa atividade simulada.
        *   As respostas do Marcelo Claro podem ser vocalizadas em Português do Brasil usando a API de Síntese de Fala do navegador.
2.  **`streamlit_app/`**: Uma aplicação Python usando Streamlit, destinada a demonstrar e fornecer uma estrutura para a implementação de processamento real de Machine Learning (treinamento de modelos, inferência, XAI) e uma simulação rica de análise avançada com agentes e chat interativo.
    *   **Chat com IA - Marcelo Claro (Streamlit Edition)**: Um chat totalmente funcional integrado com a API Gemini.
        *   **Contextualização Automática**: Tenta carregar e utilizar os resultados (simulados) gerados pela seção "Configurações e Upload" do app Streamlit para fornecer respostas mais relevantes.
        *   **Interação Proativa**: Marcelo Claro pergunta sobre o "tipo de classificação" para personalizar a análise, similar ao app React.
        *   **Simulação de Agentes de Pesquisa**: Para perguntas complexas, simula a ativação de agentes especializados e exibe um log dessa atividade.
        *   **Vocalização de Respostas da IA**: As respostas da IA no chat do Streamlit podem ser ouvidas usando a biblioteca `gTTS` (Google Text-to-Speech), que gera um áudio para ser tocado diretamente no navegador.

## Estrutura do Projeto

*   **`frontend/`**: Contém a aplicação React.
    *   `index.html`, `index.tsx`, `App.tsx`, `components/`, `services/`, `types.ts`, `constants.ts`, `metadata.json`
*   **`streamlit_app/`**: Contém a aplicação Streamlit para o backend e processamento de ML.
    *   `app.py`: Ponto de entrada principal da aplicação Streamlit, incluindo toda a lógica do Chat com IA.
    *   `utils/`: Módulos auxiliares para a lógica de ML e IA.
        *   `data_loader.py`: Esqueleto para carregar e pré-processar dados.
        *   `model_trainer.py`: Esqueleto para treinamento e avaliação de modelos.
        *   `explainability.py`: Esqueleto para interpretabilidade (ex: CAM).
        *   `gemini_utils.py`: Funções para interagir com a API Gemini (Python SDK).
        *   `crewai_handler.py`: Esqueleto para definir e orquestrar agentes CrewAI (simulado).
        *   `mcp_tools_sim.py`: Esqueleto para ferramentas simuladas inspiradas no MCP que os agentes CrewAI podem usar.
        *   `tts_utils.py`: Utilitário para converter texto em fala usando `gTTS`.
    *   `assets/`: Para quaisquer arquivos estáticos que o app Streamlit possa precisar.
*   **`README.md`**: Este arquivo.
*   **`.gitignore`**: Especifica arquivos e pastas a serem ignorados pelo Git.
*   **`requirements.txt`**: Lista as dependências Python para a aplicação Streamlit e o backend, incluindo `google-generativeai`, `crewai` e `gTTS`.

## Configuração e Execução

### Frontend React (Demonstração)

A aplicação React (`frontend/`) é projetada para ser uma demonstração interativa e opera principalmente no navegador, simulando muitas das operações de Deep Learning.

**Para executar o frontend React localmente usando VSCode:**

1.  **Abra a Pasta do Frontend no VSCode:**
    *   Abra o Visual Studio Code.
    *   Vá em "File" > "Open Folder..." e selecione a pasta `frontend/` deste repositório.

2.  **Servidor HTTP Local:**
    Como a aplicação utiliza ES Modules diretamente via `esm.sh` (configurado no `index.html`), você precisa de um servidor HTTP local para servir os arquivos. Existem várias maneiras simples de fazer isso:
    *   **Opção 1: Usando a extensão "Live Server" (Recomendado para facilidade):**
        1.  Se ainda não tiver, instale a extensão "Live Server" de Ritwick Dey no VSCode Marketplace.
        2.  No explorador de arquivos do VSCode, clique com o botão direito no arquivo `frontend/index.html`.
        3.  Selecione "Open with Live Server". Seu navegador padrão deve abrir com a aplicação rodando.
    *   **Opção 2: Usando Python (se você tiver Python 3 instalado):**
        1.  Abra um terminal no VSCode (Terminal > New Terminal).
        2.  Navegue até a pasta `frontend/`: `cd frontend`
        3.  Execute o comando: `python -m http.server 8000`
        4.  Abra seu navegador e acesse `http://localhost:8000`.
    *   **Opção 3: Usando Node.js (se você tiver Node.js e npx):**
        1.  Abra um terminal no VSCode.
        2.  Navegue até a pasta `frontend/`: `cd frontend`
        3.  Execute o comando: `npx serve`
        4.  O terminal indicará o endereço local (geralmente `http://localhost:3000` ou similar). Abra este endereço no seu navegador.

3.  **Chave da API Gemini para o Chat com IA:**
    *   Para que o "Chat com IA - Marcelo Claro" funcione no frontend local, a variável de ambiente `API_KEY` precisa estar acessível.
    *   No ambiente de desenvolvimento e prototipagem da Google, esta chave é injetada automaticamente (`process.env.API_KEY`).
    *   **Localmente:** Você precisará garantir que esta variável de ambiente esteja definida no seu sistema operacional antes de iniciar o VSCode ou o servidor HTTP. Uma alternativa (menos segura e **não recomendada para commits no Git**) seria substituir `process.env.API_KEY` diretamente no código do `App.tsx` pela sua chave durante o teste local. A maneira correta, se você adicionar um bundler (como Vite ou Webpack) ao projeto frontend, seria usar um arquivo `.env`.

### Streamlit App (Processamento Real de ML e Simulação de Agentes)

1.  **Crie um Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```
2.  **Instale as Dependências:**
    Navegue até a pasta raiz do repositório e execute:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure a Chave da API Gemini:**
    Defina a variável de ambiente `API_KEY` com sua chave da API Google Gemini.
    ```bash
    export API_KEY="SUA_CHAVE_API_AQUI" # Linux/macOS
    # set API_KEY="SUA_CHAVE_API_AQUI" # Windows Command Prompt
    # $env:API_KEY="SUA_CHAVE_API_AQUI" # Windows PowerShell
    ```
    A aplicação Streamlit verificará esta chave. Sem ela, as funcionalidades do Chat com IA não operarão.
4.  **Execute a Aplicação Streamlit:**
    Navegue até a pasta raiz do repositório e execute:
    ```bash
    streamlit run streamlit_app/app.py
    ```
    Isso deve abrir a aplicação Streamlit no seu navegador. Explore a seção "Configurações e Upload" para simular a geração de resultados, e então interaja com "Chat com IA - Marcelo Claro".

## Desenvolvimento

### Distinção entre Frontend (Simulação) e Backend (Processamento Real)

*   **Frontend React (`frontend/`):**
    *   Como descrito, esta parte da aplicação **simula** as operações de Deep Learning (treinamento, geração de métricas, XAI).
    *   É excelente para prototipar a interface do usuário, testar fluxos de interação e demonstrar conceitos.
    *   O chat com a API Gemini é funcional e se conecta à API real se a `API_KEY` estiver configurada.

*   **Uso de PyTorch/TensorFlow (Processamento Real com Python):**
    *   PyTorch e TensorFlow são bibliotecas Python e **não rodam diretamente no navegador** como parte do código JavaScript/TypeScript do React.
    *   Para realizar treinamento de modelos, inferências e cálculos de XAI **reais** usando essas bibliotecas, é necessário um **backend Python separado**.
    *   **Arquitetura Cliente-Servidor:**
        *   **Frontend (React):** Continuaria sendo sua interface do usuário.
        *   **Backend (Python):** Você criaria um servidor API em Python (usando, por exemplo, Flask ou FastAPI). Este servidor:
            *   Conteria seu código PyTorch/TensorFlow.
            *   Receberia dados do frontend (ex: configurações, arquivos de imagem).
            *   Executaria o processamento de Deep Learning.
            *   Enviaria os resultados de volta para o frontend React para visualização.
        *   **Comunicação:** O frontend React e o backend Python se comunicariam através de requisições HTTP (APIs RESTful).
    *   **VSCode para Desenvolvimento Backend:** Você pode usar o VSCode para desenvolver este backend Python, gerenciando-o no mesmo workspace do frontend, se desejar.

### Backend de Machine Learning (em `streamlit_app/utils/`)

Os arquivos em `streamlit_app/utils/` (`data_loader.py`, `model_trainer.py`, `explainability.py`) contêm esqueletos de funções. Você precisará implementar a lógica de Machine Learning real usando PyTorch ou TensorFlow dentro dessas funções. Após o processamento simulado na aba "Configurações e Upload", os dados de resultados (mock) são armazenados em `st.session_state` para serem usados pelo Chat com IA.

### Análise Avançada com Agentes (CrewAI & MCP)

*   **`crewai_handler.py`**: Contém um esqueleto para definir agentes e tarefas usando o framework CrewAI. A execução da "crew" é simulada para mostrar o fluxo de interação entre agentes e ferramentas.
*   **`mcp_tools_sim.py`**: Define ferramentas simuladas (ex: busca na web, leitura de arquivos) que os agentes CrewAI utilizariam.
*   **Importante:** A implementação de um sistema MCP real com servidores Dockerizados e a integração completa com agentes CrewAI é um trabalho de backend significativo. A presente estrutura no Streamlit serve como uma **demonstração conceitual e um ponto de partida para tal desenvolvimento**.

A funcionalidade TTS no Streamlit com `gTTS` requer conexão com a internet. A simulação de agentes e o chat contextual visam fornecer uma experiência rica e demonstrar o potencial de assistentes de IA mais avançados.
```