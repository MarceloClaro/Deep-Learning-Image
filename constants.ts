
import type { SidebarConfig, ExplanationTopic } from './types';

export enum AppSection {
  DATA_CONFIG = "Dados & Configuração",
  TRAINING = "Monitor de Treinamento",
  EVALUATION = "Resultados da Avaliação",
  CLUSTERING = "Análise de Clusterização",
  INSPECTOR = "Inspetor de Imagem",
  CHAT_IA = "Chat com IA - Marcelo Claro",
  EXPLANATIONS = "Aprender Conceitos",
}

export const DEFAULT_CONFIG: SidebarConfig = {
  numClasses: 2,
  modelName: 'ResNet18',
  fineTune: false,
  epochs: 20,
  learningRate: 0.001,
  batchSize: 16,
  trainSplit: 0.7,
  validSplit: 0.15,
  useWeightedLoss: false,
  l2Lambda: 0.01,
  patience: 3,
  optimizerName: 'Adam',
  lrSchedulerName: 'Nenhum',
  dataAugmentationMethod: 'Padrão',
  camMethod: 'Grad-CAM',
  validationStrategy: 'Hold-out (Treino/Validação/Teste)',
  simulatedUncertainty: true,
};

export const MODEL_OPTIONS: SidebarConfig['modelName'][] = ['ResNet18', 'ResNet50', 'DenseNet121', 'VisionTransformer (ViT)', 'EfficientNetB0'];
export const LEARNING_RATE_OPTIONS: number[] = [0.1, 0.01, 0.001, 0.0001];
export const BATCH_SIZE_OPTIONS: number[] = [4, 8, 16, 32, 64];
export const OPTIMIZER_OPTIONS: SidebarConfig['optimizerName'][] = ['Adam', 'AdamW', 'Ranger'];
export const LR_SCHEDULER_OPTIONS: SidebarConfig['lrSchedulerName'][] = ['Nenhum', 'Recozimento por Cosseno', 'Política de Um Ciclo'];
export const DATA_AUGMENTATION_OPTIONS: SidebarConfig['dataAugmentationMethod'][] = ['Padrão', 'Mixup', 'Cutmix'];
export const CAM_METHOD_OPTIONS: SidebarConfig['camMethod'][] = ['Grad-CAM', 'Grad-CAM++', 'Score-CAM', 'LayerCAM', 'LIME', 'SHAP'];
export const VALIDATION_STRATEGY_OPTIONS: SidebarConfig['validationStrategy'][] = ['Hold-out (Treino/Validação/Teste)', 'K-Fold Cross-Validation'];


export const ALL_EXPLANATIONS: ExplanationTopic[] = [
  {
    id: 'architecture_real_vs_demo',
    title: 'Arquitetura da Aplicação: Demonstração vs. Implementação Real',
    summary: 'Esta aplicação, em seu estado atual, é uma ferramenta de demonstração interativa que opera inteiramente no seu navegador (frontend). Ela simula o fluxo de trabalho de um projeto de Deep Learning, desde o carregamento de dados até a análise de resultados. As operações de "treinamento", "cálculo de métricas", "geração de CAM" e "clusterização" são realizadas por meio de simulações no frontend para fins educacionais e de prototipagem rápida de interface. O Chat com IA (Marcelo Claro) é uma exceção, pois utiliza a API Gemini para fornecer respostas reais.',
    details: `Para realizar treinamento e inferência **reais** com bibliotecas como PyTorch ou TensorFlow (versões Python) e processar conjuntos de dados, especialmente os de pesquisa ou diagnóstico, seria necessária uma arquitetura cliente-servidor (frontend-backend):\n\n1.  **Frontend (Esta Aplicação):** Continuaria sendo a interface do usuário para carregar dados, configurar parâmetros e visualizar resultados.\n\n2.  **Backend (Servidor Dedicado):** Um servidor Python com PyTorch/TensorFlow instalados. Este backend seria responsável por:\n    *   Receber o arquivo ZIP de dados.\n    *   Realizar o pré-processamento real das imagens.\n    *   Construir e treinar os modelos de Deep Learning (ResNet, ViT, etc.) em CPU ou GPU.\n    *   Calcular métricas de avaliação com base nos resultados reais do modelo.\n    *   Gerar visualizações XAI (como Grad-CAM) a partir das ativações reais do modelo treinado.\n    *   Executar algoritmos de clusterização sobre embeddings reais.\n    *   Armazenar modelos treinados e resultados.\n\n3.  **API (Interface de Programação de Aplicativos):** O frontend se comunicaria com o backend através de chamadas de API para:\n    *   Enviar os dados e a configuração.\n    *   Iniciar o processo de treinamento.\n    *   Receber atualizações de progresso e logs.\n    *   Buscar os resultados finais (métricas, visualizações, etc.) para exibição.\n\n**Por que esta distinção é crucial?**\n*   **Recursos Computacionais:** Treinar modelos de Deep Learning é intensivo. Navegadores têm limitações de memória e poder de processamento. Backends em servidores dedicados podem acessar CPUs potentes e GPUs, essenciais para treinar em tempo hábil.\n*   **Ambiente e Dependências:** PyTorch e TensorFlow (Python) e suas vastas bibliotecas associadas são projetadas para ambientes de servidor ou desktop, não diretamente para execução em navegador (com exceção do TensorFlow.js, que tem suas próprias considerações para modelos complexos).\n*   **Persistência de Dados e Modelos:** Um backend permite armazenar de forma confiável os datasets, modelos treinados e resultados de experimentos.\n\nEsta demonstração visa educar sobre o workflow e permitir o design interativo da interface. Para transformá-la em uma ferramenta de pesquisa "estado da arte" para diagnóstico, a implementação de um backend robusto com as capacidades descritas seria o próximo passo fundamental.`,
    keywords: ['arquitetura de software', 'frontend', 'backend', 'PyTorch', 'TensorFlow', 'treinamento de modelo real', 'API', 'cliente-servidor', 'CPU', 'GPU', 'Gemini'],
    promptSuggestion: "Explique a diferença entre uma aplicação de IA puramente frontend (que simula operações, como esta demonstração) e uma aplicação com um backend dedicado para treinamento real de modelos de Deep Learning. Quais são as vantagens de uma arquitetura backend para tarefas de IA complexas?"
  },
  {
    id: 'chat_ia_marcelo_claro',
    title: 'Chat com IA: Marcelo Claro',
    summary: 'Converse com Marcelo Claro, seu assistente de IA especializado, para obter insights e análises sobre os resultados do seu modelo. Marcelo poderá perguntar sobre o tipo de classificação que você está realizando para contextualizar melhor as respostas e, se necessário, simular uma consulta a "agentes de pesquisa" para informações mais aprofundadas, utilizando a API Gemini.',
    details: 'Ao interagir com Marcelo Claro, você pode fazer perguntas como:\n- "Quais são os pontos fortes do meu modelo com base nestes resultados?"\n- "Há alguma classe com desempenho consistentemente baixo?"\n- "Explique a Matriz de Confusão em termos simples referente a estes dados."\n- "Sugira próximos passos com base nesta avaliação."\n\nMarcelo Claro poderá perguntar: "Qual o tipo de classificação de imagens que você está realizando?" para adaptar suas respostas. Se você fizer perguntas que exijam pesquisa (ex: "Quais os últimos avanços em IA para [seu tipo de classificação]?"), ele poderá simular a ativação de "agentes especializados" e exibir um log dessa atividade, antes de fornecer uma resposta com base na API Gemini.\n\nPara melhores resultados, certifique-se de que o treinamento foi concluído e os resultados estão disponíveis. A qualidade das respostas dependerá da clareza da sua pergunta e dos dados de contexto fornecidos.',
    keywords: ['chat IA', 'assistente virtual', 'análise de resultados', 'Gemini', 'interpretação de dados', 'Marcelo Claro', 'diagnóstico assistido por IA', 'agentes de IA'],
    promptSuggestion: "Aja como Marcelo Claro, um especialista em IA e ciência de dados. Eu informarei o tipo de classificação de imagens que estou realizando. Analise os resultados do modelo (fornecidos no contexto) e responda às minhas perguntas. Se necessário, simule a consulta a agentes de pesquisa para obter informações mais detalhadas e multidisciplinares, e então forneça uma resposta abrangente."
  },
   {
    id: 'mcp_crewai_analysis',
    title: 'Análise Avançada com Agentes (MCP & CrewAI)',
    summary: 'Simulação de como uma arquitetura multi-agente (inspirada no MCP - Multi-Component Platform e utilizando um framework como CrewAI) poderia ser usada para análises mais profundas. Agentes especializados poderiam pesquisar artigos, analisar dados e correlacionar informações multidisciplinares.',
    details: 'Esta seção conceitual descreve uma futura capacidade. Em uma implementação real (requer backend):\n\n1.  **Plataforma Multi-Componente (MCP):** Baseada na ideia de ter microsserviços especializados (Servidor de Arquivos, Servidor de Execução de Comandos) que os agentes podem usar como ferramentas. Estes servidores seriam dockerizados e se comunicariam através de volumes compartilhados e APIs, como descrito no artigo "Manual do engenheiro de IA para arquitetura MCP".\n\n2.  **CrewAI:** Um framework para orquestrar agentes de IA autônomos. Poderíamos definir:\n    *   **Agentes:** Ex: `PesquisadorCientifico`, `AnalistaDeDadosDeImagem`, `CorrelacionadorMultidisciplinar`.\n    *   **Ferramentas (MCP):** Os agentes usariam as ferramentas expostas pelos servidores MCP (ex: `ler_arquivo_csv_resultados`, `buscar_artigos_pubmed`, `executar_script_analise_estatistica`).\n    *   **Tarefas:** Ex: "Pesquisar os últimos 5 artigos sobre o uso de IA em [tipo de classificação]", "Analisar o arquivo de resultados `results.csv` e identificar pontos fracos do modelo", "Correlacionar os achados com informações de [área multidisciplinar relevante]".\n    *   **Processo (Crew):** Uma equipe de agentes colaborando para produzir um relatório de análise detalhado.\n\n3.  **Fluxo no Streamlit (Simulação):**\n    *   O usuário informa o tipo de classificação.\n    *   Ao clicar em "Iniciar Análise Avançada", o Streamlit simularia a orquestração:\n        *   Mostraria um log de "Agente X usando ferramenta Y".\n        *   "Agente Z sintetizando informações".\n    *   O resultado final seria um resumo gerado (nesta demo, potencialmente pelo Gemini, mas em um sistema real, pela colaboração dos agentes).\n\nEste é um conceito avançado que combina a robustez de microsserviços (MCP) com a inteligência adaptativa de agentes autônomos (CrewAI) para investigações complexas.',
    keywords: ['multi-agente', 'CrewAI', 'MCP', 'IA autônoma', 'microsserviços', 'pesquisa científica', 'análise de dados avançada', 'raspagem de dados', 'FastAPI', 'Docker'],
    promptSuggestion: "Explique como uma arquitetura multi-agente, usando conceitos do MCP (servidores de ferramentas) e um framework como CrewAI, poderia realizar uma análise aprofundada e multidisciplinar dos resultados de um modelo de classificação de imagens. Descreva os papéis dos agentes, tipos de ferramentas e o processo colaborativo."
  },
  {
    id: 'data_augmentation',
    title: 'Aumento de Dados & Transformações',
    summary: 'Técnicas para aumentar sinteticamente a diversidade dos dados de treinamento, melhorando a generalização do modelo e reduzindo o overfitting pela aplicação de transformações aleatórias como inversões, rotações e ajustes de cor.',
    keywords: ['aumento de dados', 'transformação de imagem', 'overfitting', 'generalização', 'pytorch transforms'],
    promptSuggestion: "Explique a importância do aumento de dados no treinamento de modelos de aprendizado profundo para classificação de imagens, incluindo técnicas comuns e seus benefícios, de uma forma que um iniciante possa entender."
  },
  {
    id: 'num_classes',
    title: 'Número de Classes',
    summary: 'Define o número total de categorias distintas que o modelo deve prever. Isso impacta a estrutura da camada de saída (ex: unidades Softmax) e a complexidade da tarefa de classificação.',
    keywords: ['tarefa de classificação', 'camada de saída', 'softmax', 'classificação binária', 'classificação multiclasse'],
    promptSuggestion: "Descreva como o número de classes afeta a arquitetura e o treinamento de uma rede neural para classificação de imagens. Quais são as diferenças entre problemas binários e multiclasse?"
  },
  {
    id: 'sota_models',
    title: 'Modelos SOTA (ResNet, DenseNet, ViT, EfficientNet)',
    summary: 'Arquiteturas de ponta (State-Of-The-Art) como ResNets, DenseNets, Vision Transformers (ViTs) e EfficientNets são pré-treinadas em grandes datasets (ex: ImageNet). Elas servem como extratores de características poderosos, permitindo aprendizado por transferência eficaz para novas tarefas, muitas vezes com menos dados.',
    keywords: ['aprendizado por transferência', 'ResNet', 'DenseNet', 'Vision Transformer', 'ViT', 'EfficientNet', 'ImageNet', 'redes neurais convolucionais', 'transformers'],
    promptSuggestion: "Compare brevemente as arquiteturas ResNet, Vision Transformer (ViT) e EfficientNet em termos de seus princípios de design e aplicações típicas em classificação de imagens."
  },
  {
    id: 'fine_tuning',
    title: 'Ajuste Fino (Fine-Tuning)',
    summary: 'Ajusta os pesos de um modelo pré-treinado em um novo conjunto de dados específico. O "fine-tuning completo" desbloqueia todas as camadas, permitindo que se adaptem, enquanto o fine-tuning parcial pode treinar apenas as camadas finais.',
    keywords: ['fine-tuning', 'aprendizado por transferência', 'adaptação de modelo', 'treinamento de rede neural'],
    promptSuggestion: "O que é fine-tuning em aprendizado profundo? Compare o fine-tuning completo versus o congelamento de algumas camadas ao adaptar um modelo pré-treinado para uma nova tarefa."
  },
  {
    id: 'l2_regularization',
    title: 'Regularização L2 (Weight Decay)',
    summary: 'Técnica para prevenir overfitting adicionando um termo de penalidade à função de perda, proporcional ao quadrado dos pesos. Isso incentiva pesos menores, levando a modelos mais simples.',
    keywords: ['regularização L2', 'weight decay', 'overfitting', 'função de perda', 'complexidade do modelo'],
    promptSuggestion: "Explique a regularização L2 (weight decay) e como ela ajuda a prevenir o overfitting em redes neurais. Qual é o papel do hiperparâmetro lambda?"
  },
  {
    id: 'early_stopping',
    title: 'Parada Antecipada (Early Stopping)',
    summary: 'Técnica de regularização onde o treinamento é interrompido quando o desempenho em um conjunto de validação para de melhorar (ou começa a piorar), prevenindo overfitting. "Paciência" determina quantas épocas esperar.',
    keywords: ['parada antecipada', 'overfitting', 'conjunto de validação', 'treinamento de modelo', 'parâmetro de paciência'],
    promptSuggestion: "Descreva a técnica de parada antecipada em aprendizado de máquina. Como funciona e qual a importância do parâmetro 'paciência'?"
  },
  {
    id: 'weighted_loss',
    title: 'Perda Ponderada para Classes Desbalanceadas',
    summary: 'Aborda o desbalanceamento de classes atribuindo pesos maiores na função de perda às classes minoritárias. Isso força o modelo a prestar mais atenção às classes sub-representadas durante o treinamento. Técnicas como oversampling (SMOTE) e undersampling também podem ser usadas.',
    keywords: ['desbalanceamento de classes', 'perda ponderada', 'classe minoritária', 'SMOTE', 'oversampling', 'undersampling'],
    promptSuggestion: "Como o uso de uma função de perda ponderada ajuda no treinamento de modelos em conjuntos de dados desbalanceados? Mencione outras técnicas como SMOTE."
  },
  {
    id: 'validation_strategy',
    title: 'Estratégias de Validação (Hold-out, Cross-Validation)',
    summary: 'Métodos para avaliar o desempenho do modelo de forma robusta. Hold-out divide os dados em treino, validação e teste. K-Fold Cross-Validation divide os dados em K partes, treinando e testando K vezes, usando uma parte diferente para teste a cada vez, o que é mais robusto para datasets menores.',
    keywords: ['validação', 'hold-out', 'validação cruzada', 'k-fold', 'generalização', 'avaliação de modelo'],
    promptSuggestion: "Compare a estratégia de validação hold-out com a k-fold cross-validation. Quais são as vantagens da validação cruzada, especialmente para conjuntos de dados limitados?"
  },
  {
    id: 'advanced_metrics',
    title: 'Métricas de Avaliação Avançadas (AUC-PR, Sensibilidade, Especificidade)',
    summary: 'Além da acurácia, métricas como Sensibilidade (taxa de verdadeiros positivos), Especificidade (taxa de verdadeiros negativos) e AUC-PR (Area Under Precision-Recall Curve) são cruciais, especialmente para dados desbalanceados e em contextos médicos.',
    keywords: ['AUC-PR', 'sensibilidade', 'recall', 'especificidade', 'precisão', 'recall', 'métricas de classificação', 'dados desbalanceados'],
    promptSuggestion: "Explique a importância da Sensibilidade, Especificidade e da curva AUC-PR na avaliação de classificadores, especialmente em cenários com classes desbalanceadas como diagnóstico médico."
  },
  {
    id: 'xai_methods',
    title: 'Interpretabilidade (XAI - Grad-CAM, LIME, SHAP)',
    summary: 'Métodos de IA Explicável (XAI) ajudam a entender como os modelos tomam decisões. Grad-CAM e variantes destacam regiões importantes na imagem. LIME e SHAP oferecem explicações locais aproximando o modelo ou baseadas em teoria dos jogos, respectivamente.',
    keywords: ['XAI', 'interpretabilidade', 'Grad-CAM', 'LIME', 'SHAP', 'mapas de ativação', 'confiança no modelo'],
    promptSuggestion: "O que é XAI (IA Explicável)? Descreva brevemente como Grad-CAM, LIME e SHAP ajudam a interpretar as previsões de modelos de imagem."
  },
  {
    id: 'uncertainty_quantification',
    title: 'Quantificação de Incerteza',
    summary: 'Essencial para aplicações críticas, como diagnóstico médico. Permite que o modelo indique o quão confiante está em sua previsão. Métodos incluem dropout na inferência (Monte Carlo Dropout), ensembles de modelos ou redes neurais bayesianas.',
    keywords: ['incerteza', 'confiança do modelo', 'Monte Carlo Dropout', 'ensembles', 'redes neurais bayesianas', 'tomada de decisão'],
    promptSuggestion: "Por que a quantificação de incerteza é importante em modelos de aprendizado de máquina para diagnóstico? Mencione algumas abordagens para estimar a incerteza."
  },
  {
    id: 'dicom_data_versioning',
    title: 'Dados Médicos (DICOM) e Versionamento (DVC, MLflow)',
    summary: 'O formato DICOM é padrão para imagens médicas, contendo metadados ricos. Ferramentas como DVC (Data Version Control) e MLflow são cruciais para a reprodutibilidade em projetos de IA, permitindo versionar dados, código e modelos.',
    keywords: ['DICOM', 'metadados médicos', 'DVC', 'MLflow', 'reprodutibilidade', 'gerenciamento de experimentos', 'pipeline de IA'],
    promptSuggestion: "Qual a importância do formato DICOM em imagens médicas e como ferramentas como DVC e MLflow contribuem para a reprodutibilidade em projetos de IA?"
  },
];


export const SIDEBAR_EXPLANATIONS: Record<keyof SidebarConfig | 'numClassesEffectiveNote' | 'dicomSupport' | 'segmentationMethod', string> = {
  numClasses: "O que é: Quantas categorias o modelo deve distinguir. \nComo afeta: Se 'Fine-Tuning Completo' ativo e ZIP carregado, é definido pelas pastas no ZIP. Caso contrário, defina manualmente. Mais classes podem aumentar a complexidade.",
  numClassesEffectiveNote: "Nota: Com 'Fine-Tuning Completo' ativo, o número de classes é determinado pelas pastas dentro do arquivo ZIP carregado.",
  modelName: "O que é: São arquiteturas de inteligência artificial, muitas pré-treinadas em milhões de imagens (ex: ImageNet). Usá-las é como dar um 'cérebro' já experiente para o seu projeto.\nComo afeta: Modelos maiores ou mais recentes (como ViT, EfficientNet) podem ser mais precisos, mas exigem mais do computador. ResNet18 é mais leve. A escolha depende da complexidade da sua tarefa e dos recursos disponíveis.",
  fineTune: "O que é: Se marcado, o 'cérebro' do modelo pré-treinado (todos os seus conhecimentos) será reajustado especificamente para as suas imagens. As classes serão determinadas pelas pastas no seu arquivo ZIP.\nComo afeta: Marcado, o treinamento pode ser mais demorado, mas tem potencial para aprender melhor as características únicas das suas imagens. Se desmarcado, apenas uma pequena parte final do modelo é treinada (transfer learning tradicional), o que é mais rápido mas pode ser menos adaptado se suas imagens forem muito diferentes das que o modelo viu originalmente.",
  epochs: "O que é: Uma 'época' significa que o modelo viu todas as suas imagens de treinamento uma vez. Mais épocas dão ao modelo mais chances de aprender.\nComo afeta: Muitas épocas podem levar a um bom aprendizado, mas também podem causar 'decoreba' (overfitting), onde o modelo fica bom demais nas imagens de treino mas ruim em imagens novas. Poucas épocas podem significar que o modelo não aprendeu o suficiente.",
  learningRate: "O que é: Controla o quão 'rápido' o modelo ajusta seus conhecimentos a cada imagem que vê. É como o tamanho do passo que o modelo dá ao aprender.\nComo afeta: Uma taxa muito alta pode fazer o modelo 'pular' a solução ideal, causando instabilidade. Uma taxa muito baixa torna o aprendizado excessivamente lento. Valores pequenos, como 0.001 ou 0.0001, são comuns.",
  batchSize: "O que é: Quantas imagens o modelo analisa de uma vez antes de atualizar seus conhecimentos. Pense em 'grupos' de estudo.\nComo afeta: Lotes maiores podem acelerar o treinamento e dar uma estimativa mais estável de como ajustar o modelo, mas exigem mais memória do computador. Lotes menores podem ajudar o modelo a não ficar preso em soluções ruins e generalizar melhor, mas o progresso do treinamento pode ser mais 'barulhento' (variável).",
  trainSplit: "O que é: A porcentagem das suas imagens que será usada para ensinar o modelo.\nComo afeta: Quanto mais imagens para treino, melhor o modelo pode aprender os padrões. Geralmente, um valor alto como 70% (0.7) ou 80% (0.8) é usado. O restante das imagens é dividido entre validação e teste.",
  validSplit: "O que é: A porcentagem das suas imagens que será usada para verificar o aprendizado do modelo durante o treinamento, como se fossem 'provinhas' periódicas. Essas imagens não são usadas para ensinar diretamente.\nComo afeta: Ajuda a monitorar se o modelo está aprendendo de forma generalizada ou apenas 'decorando' os dados de treino. Essencial para técnicas como 'Parada Antecipada'. O que sobra após treino e validação torna-se o conjunto de teste.",
  validationStrategy: "O que é: Como o modelo será avaliado durante e após o treinamento para garantir que está generalizando bem.\nComo afeta: 'Hold-out' é simples (treino/validação/teste). 'K-Fold Cross-Validation' é mais robusto, especialmente para datasets menores, pois treina e valida o modelo K vezes em diferentes subconjuntos de dados, dando uma estimativa de desempenho mais confiável. (Nota: K-Fold é conceitualmente representado aqui; a implementação real requer um backend).",
  useWeightedLoss: "O que é: Se marcado, e se algumas das suas classes tiverem muito menos imagens que outras (conjunto de dados desbalanceado), esta opção faz o modelo dar 'mais atenção' e peso às classes com poucas imagens durante o cálculo do erro.\nComo afeta: Ajuda a evitar que o modelo ignore as classes raras e melhora o desempenho nessas classes. Útil se, por exemplo, você tem 900 imagens de 'Cachorros' e só 100 de 'Gatos'.",
  l2Lambda: "O que é: Uma técnica de regularização para evitar que o modelo fique complexo demais e 'decore' os dados (overfitting). Funciona adicionando uma penalidade à função de erro do modelo, baseada na soma dos quadrados dos 'pesos' (parâmetros internos) do modelo.\nComo afeta: Um valor pequeno de lambda (ex: 0.01) incentiva o modelo a usar pesos menores, resultando num modelo mais simples que generaliza melhor para imagens novas. Um valor muito alto pode simplificar demais e impedir um bom aprendizado (underfitting). λ=0 desativa esta regularização.",
  patience: "O que é: Parâmetro da 'Parada Antecipada'. Indica quantas épocas o treinamento continua mesmo que o desempenho na 'provinha' (validação) não melhore. Se a paciência acabar e não houver melhora, o treino para antes de completar todas as épocas definidas.\nComo afeta: Evita gastar tempo treinando um modelo que não está mais melhorando e é uma forma crucial de prevenir o overfitting. Uma paciência de 3-10 épocas é comum, dependendo da tarefa.",
  optimizerName: "O que é: O 'método de estudo' ou algoritmo que o modelo usa para aprender e ajustar seus conhecimentos (pesos internos) com base nos erros que comete. Adam, AdamW, Ranger são algoritmos diferentes para essa otimização.\nComo afeta: Diferentes otimizadores podem levar a velocidades de aprendizado e resultados finais ligeiramente diferentes. Adam é uma escolha popular e geralmente robusta para muitas tarefas.",
  lrSchedulerName: "O que é: Um 'Agendador de Taxa de Aprendizagem'. Uma forma de ajustar automaticamente a 'Taxa de Aprendizagem' durante o treinamento. Por exemplo, pode começar com uma taxa maior e diminuí-la gradualmente conforme o treino avança.\nComo afeta: Pode ajudar o modelo a convergir para uma boa solução mais rapidamente e de forma mais estável. 'Recozimento por Cosseno' (Cosine Annealing) e 'Política de Um Ciclo' (One Cycle Policy) são estratégias comuns que variam a taxa de forma cíclica ou decrescente.",
  dataAugmentationMethod: "O que é: Técnicas para criar variações sintéticas das suas imagens de treino (girando, mudando cores, cortando, etc.) para que o modelo veja mais exemplos 'diferentes' sem precisar de mais imagens reais.\nComo afeta: Ajuda muito a evitar 'decoreba' (overfitting) e faz o modelo aprender a reconhecer objetos de formas variadas, tornando-o mais robusto. 'Padrão' aplica um conjunto de transformações comuns. 'Mixup' e 'Cutmix' são técnicas mais avançadas que misturam imagens e seus rótulos para criar exemplos de treinamento mais desafiadores e regularizadores.",
  camMethod: "O que é: Método de IA Explicável (XAI) para visualização. Grad-CAM e suas variantes (Score-CAM, LayerCAM) destacam as regiões da imagem que foram mais importantes para a decisão do modelo. LIME e SHAP são outros métodos XAI que oferecem diferentes perspectivas sobre a decisão do modelo.\nComo afeta: Ajuda a entender 'onde' o modelo olhou para classificar uma imagem. Diferentes métodos podem fornecer insights complementares. (Nota: A visualização é representativa; a geração real requer um backend com o modelo treinado).",
  simulatedUncertainty: "O que é: Representa a capacidade do modelo de expressar confiança em suas previsões. Se marcado, um 'score de incerteza' será gerado.\nComo afeta: Em aplicações reais, alta incerteza pode sinalizar a necessidade de revisão humana. Aqui, demonstra o conceito. (Nota: A quantificação real de incerteza requer um backend).",
  dicomSupport: "O que é: (Demonstrativo) Indica se o sistema está preparado para processar arquivos no formato DICOM, padrão em imagens médicas, que contêm metadados importantes além da imagem.\nComo afeta: Em um sistema real, habilitar isso permitiria carregar e extrair informações de arquivos DICOM. Aqui é apenas um indicador.",
  segmentationMethod: "O que é: (Demonstrativo) Escolha de um método para segmentar (isolar) regiões de interesse nas imagens antes da classificação, como uma lesão em uma imagem dermatológica.\nComo afeta: Em um sistema real, a segmentação pode melhorar drasticamente o desempenho ao focar o modelo na área relevante. Aqui, demonstra a etapa no pipeline.",
};
