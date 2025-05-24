
import React, { useState, useCallback, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { DataUpload } from './components/DataUpload';
import { TrainingMonitor } from './components/TrainingMonitor';
import { EvaluationResults } from './components/EvaluationResults';
import { ClusteringAnalysis } from './components/ClusteringAnalysis';
import { ImageInspector } from './components/ImageInspector';
import { ChatIA } from './components/ChatIA';
import { ExplanationHub } from './components/ExplanationHub';
import { DEFAULT_CONFIG, AppSection, ALL_EXPLANATIONS, SIDEBAR_EXPLANATIONS } from './constants';
import type { TrainingMetrics, ClassificationReportData, ConfusionMatrixData, ClusterVisualizationData, IndividualEvaluationData, ExplanationTopic, TrainingStatus, ConfigDataEntry, SidebarConfig, SampleImage, ErrorAnalysisItem, ROCCurveData, ClusterPoint, ChatMessage, UserOrModelMessage, SystemLogMessage } from './types';
import { generateMockMetrics, generateMockClassificationReport, generateMockConfusionMatrix, generateMockClusterData, generateMockErrorAnalysis, generateMockCAMImage, generateMockROCCurveData, generateMockAugmentedEmbeddings, generateMockPRCurveData, generateMockUncertaintyScore } from './services/mockDataService';
import { exportResultsToCSV, exportTrainingMetricsToCSV } from './services/csvExporter'; 
import JSZip from 'jszip';
import { GoogleGenAI, Chat } from "@google/genai";

const App: React.FC = () => {
  const [config, setConfig] = useState<SidebarConfig>(DEFAULT_CONFIG);
  const [currentSection, setCurrentSection] = useState<AppSection>(AppSection.DATA_CONFIG);
  const [zipFile, setZipFile] = useState<File | null>(null);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics | null>(null);
  const [evaluationReport, setEvaluationReport] = useState<ClassificationReportData | null>(null);
  const [confusionMatrix, setConfusionMatrix] = useState<ConfusionMatrixData | null>(null);
  const [errorAnalysisData, setErrorAnalysisData] = useState<ErrorAnalysisItem[] | null>(null);
  const [clusterData, setClusterData] = useState<ClusterVisualizationData | null>(null);
  const [augmentedEmbeddingsData, setAugmentedEmbeddingsData] = useState<ClusterPoint[] | null>(null);
  const [individualEval, setIndividualEval] = useState<IndividualEvaluationData | null>(null);
  const [camImage, setCamImage] = useState<string | null>(null);
  const [uploadedImageForEval, setUploadedImageForEval] = useState<File | null>(null);
  const [rocCurveData, setRocCurveData] = useState<ROCCurveData | null>(null);
  const [prCurveData, setPrCurveData] = useState<ROCCurveData | null>(null); 
  
  const [numClassesInData, setNumClassesInData] = useState<number>(DEFAULT_CONFIG.numClasses);
  const [classNamesInData, setClassNamesInData] = useState<string[]>(Array.from({ length: DEFAULT_CONFIG.numClasses }, (_, i) => `Classe ${String.fromCharCode(65 + i)}`));
  const [sampleImagesFromZip, setSampleImagesFromZip] = useState<SampleImage[]>([]);
  const [isProcessingZip, setIsProcessingZip] = useState<boolean>(false);
  const [areResultsAvailable, setAreResultsAvailable] = useState<boolean>(false); 
  const [isTrainingComplete, setIsTrainingComplete] = useState<boolean>(false);
  const [trainingLog, setTrainingLog] = useState<string[]>([]);

  // ChatIA State
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState<string>('');
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  const [geminiChatInstance, setGeminiChatInstance] = useState<Chat | null>(null);
  const [userClassificationType, setUserClassificationType] = useState<string | null>(null);
  const [simulatedAgentLog, setSimulatedAgentLog] = useState<string[]>([]);


  const handleConfigChange = useCallback((newConfig: SidebarConfig) => {
    if (newConfig.fineTune && zipFile && newConfig.numClasses !== numClassesInData) {
      setConfig({ ...newConfig, numClasses: numClassesInData });
    } else {
      setConfig(newConfig);
    }
  }, [zipFile, numClassesInData]);


  const handleZipUpload = async (file: File) => {
    setZipFile(file);
    setIsProcessingZip(true);
    setSampleImagesFromZip([]); 
    setAreResultsAvailable(false);
    setIsTrainingComplete(false);
    setTrainingLog([]);
    setEvaluationReport(null);
    setConfusionMatrix(null);
    setErrorAnalysisData(null);
    setClusterData(null);
    setAugmentedEmbeddingsData(null);
    setRocCurveData(null);
    setPrCurveData(null);
    setTrainingMetrics(null);
    setGeminiChatInstance(null); 
    setChatMessages([]);
    setUserClassificationType(null);
    setSimulatedAgentLog([]);


    try {
      const zip = await JSZip.loadAsync(file);
      const classFolders: { [key: string]: JSZip.JSZipObject[] } = {};
      const imageFileExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp'];
      
      zip.forEach((relativePath, zipEntry) => {
        if (!zipEntry.dir && relativePath.includes('/')) {
          const parts = relativePath.split('/');
          const topLevelFolder = parts[0];
          
          if (parts.length > 1 && !zipEntry.name.startsWith('__MACOSX')) { 
             if (parts.length === 2 || (parts.length > 2 && parts[1] !== '' && imageFileExtensions.some(ext => parts[parts.length -1].toLowerCase().endsWith(ext)))) {
                 if (!classFolders[topLevelFolder]) {
                    classFolders[topLevelFolder] = [];
                 }
                 if (imageFileExtensions.some(ext => zipEntry.name.toLowerCase().endsWith(ext))) {
                    classFolders[topLevelFolder].push(zipEntry);
                 }
             }
          }
        }
      });

      const detectedClassNames = Object.keys(classFolders).filter(name => classFolders[name].length > 0);
      const detectedNumClasses = detectedClassNames.length;

      if (detectedNumClasses === 0) {
        alert("Nenhuma pasta de classe com imagens válidas encontrada no arquivo ZIP. Certifique-se de que o ZIP contenha pastas de primeiro nível, cada uma representando uma classe com arquivos de imagem (jpg, png, gif, bmp) dentro.");
        setNumClassesInData(DEFAULT_CONFIG.numClasses);
        setClassNamesInData(Array.from({ length: DEFAULT_CONFIG.numClasses }, (_, i) => `Classe Padrão ${String.fromCharCode(65 + i)}`));
        setIsProcessingZip(false);
        return;
      }
      
      setNumClassesInData(detectedNumClasses);
      setClassNamesInData(detectedClassNames);

      const extractedSamples: SampleImage[] = [];
      const maxSamplesPerClass = 3; 
      const totalMaxSamples = 10; 

      for (const className of detectedClassNames) {
        if (extractedSamples.length >= totalMaxSamples) break;
        const imageFiles = classFolders[className];
        if (imageFiles && imageFiles.length > 0) {
          for (let i = 0; i < Math.min(imageFiles.length, maxSamplesPerClass); i++) {
            if (extractedSamples.length >= totalMaxSamples) break;
            try {
                const imageFile = imageFiles[i];
                const blob = await imageFile.async('blob');
                const dataUrl = await new Promise<string>((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result as string);
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                });
                extractedSamples.push({ className, imageDataUrl: dataUrl, fileName: imageFile.name.split('/').pop() || imageFile.name });
            } catch (imgError) {
                console.error(`Erro ao processar imagem ${imageFiles[i].name} da classe ${className}:`, imgError);
            }
          }
        }
      }
      setSampleImagesFromZip(extractedSamples);

      if (config.fineTune) {
        setConfig(prev => ({ ...prev, numClasses: detectedNumClasses }));
      } else {
        if (config.numClasses === DEFAULT_CONFIG.numClasses || config.numClasses === 0 || config.numClasses === 2) {
           setConfig(prev => ({ ...prev, numClasses: detectedNumClasses }));
        }
      }

    } catch (error) {
      console.error("Erro ao processar arquivo ZIP:", error);
      alert("Ocorreu um erro ao processar o arquivo ZIP. Verifique se é um arquivo ZIP válido e tente novamente.");
      setNumClassesInData(DEFAULT_CONFIG.numClasses); 
      setClassNamesInData(Array.from({ length: DEFAULT_CONFIG.numClasses }, (_, i) => `Classe Padrão ${String.fromCharCode(65 + i)}`));
    } finally {
      setIsProcessingZip(false);
    }
    setCurrentSection(AppSection.DATA_CONFIG);
  };
  
  const saveConfigToJson = () => {
    const effectiveNumClasses = config.fineTune && zipFile ? numClassesInData : config.numClasses;
    const configToSave: ConfigDataEntry[] = [
        { parameter: "Modelo", value: config.modelName },
        { parameter: "Fine-Tuning Completo", value: config.fineTune ? "Sim" : "Não" },
        { parameter: "Número de Classes Efetivo", value: effectiveNumClasses.toString() },
        { parameter: "Épocas", value: config.epochs.toString() },
        { parameter: "Taxa de Aprendizagem", value: config.learningRate.toString() },
        { parameter: "Tamanho de Lote", value: config.batchSize.toString() },
        { parameter: "Divisão Treino", value: config.trainSplit.toString() },
        { parameter: "Divisão Validação", value: config.validSplit.toString() },
        { parameter: "Estratégia de Validação", value: config.validationStrategy },
        { parameter: "Regularização L2", value: config.l2Lambda.toString() },
        { parameter: "Paciência Early Stopping", value: config.patience.toString() },
        { parameter: "Usar Perda Ponderada", value: config.useWeightedLoss ? "Sim" : "Não" },
        { parameter: "Apresentar Score de Incerteza", value: config.simulatedUncertainty ? "Sim" : "Não" },
        { parameter: "Otimizador", value: config.optimizerName },
        { parameter: "Agendador LR", value: config.lrSchedulerName },
        { parameter: "Aumento de Dados", value: config.dataAugmentationMethod },
        { parameter: "Método XAI", value: config.camMethod },
    ];
    const jsonString = JSON.stringify(configToSave, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `config_${config.modelName}_execucao_${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleExportResults = () => {
    const currentClassNames = config.fineTune && zipFile ? classNamesInData : Array.from({ length: config.numClasses }, (_, i) => `Classe ${String.fromCharCode(65 + i)}`);
    exportResultsToCSV(
        trainingMetrics,
        evaluationReport,
        confusionMatrix,
        errorAnalysisData,
        clusterData,
        individualEval,
        currentClassNames,
        config.modelName,
        rocCurveData, 
        prCurveData   
    );
  };

  const handleExportTrainingMetrics = () => {
    exportTrainingMetricsToCSV(trainingMetrics, config.modelName);
  };

  const handleExportAllResultsToJson = () => {
    const effectiveNumClasses = config.fineTune && zipFile ? numClassesInData : config.numClasses;
    const effectiveClassNames = config.fineTune && zipFile && classNamesInData.length > 0 ? classNamesInData : Array.from({ length: effectiveNumClasses > 0 ? effectiveNumClasses : DEFAULT_CONFIG.numClasses }, (_, i) => `Classe ${String.fromCharCode(65 + i)}`);

    const allResults = {
        configuration: { ...config, numClasses: effectiveNumClasses, classNames: effectiveClassNames },
        trainingMetrics,
        evaluationReport,
        confusionMatrix: confusionMatrix ? { ...confusionMatrix, labels: effectiveClassNames } : null,
        rocCurveData,
        prCurveData,
        errorAnalysisData,
        clusterData: clusterData ? { ...clusterData, classNames: effectiveClassNames } : null,
        augmentedEmbeddingsData,
        individualEvaluation: individualEval,
        metadata: {
            exportedAt: new Date().toISOString(),
            zipFileName: zipFile?.name || "N/A",
            userClassificationType: userClassificationType || "Não especificado",
        }
    };

    const jsonString = JSON.stringify(allResults, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `todos_resultados_${config.modelName}_${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };


  const getResultsAsTextContext = useCallback((): string => {
    if (!areResultsAvailable) {
      return "Nenhum resultado de modelo disponível para análise. Por favor, execute o processamento primeiro ou carregue um contexto.";
    }

    let context = "## Contexto dos Resultados da Sessão de Análise de Imagens (Resumo Textual e JSON) ##\n\n";
    const effectiveClassNames = config.fineTune && zipFile && classNamesInData.length > 0 ? classNamesInData : Array.from({ length: config.numClasses > 0 ? config.numClasses : DEFAULT_CONFIG.numClasses }, (_, i) => `Classe ${String.fromCharCode(65 + i)}`);
    const effectiveNumClasses = config.fineTune && zipFile ? numClassesInData : config.numClasses;

    const summaryJson = {
      modelConfiguration: {
        modelName: config.modelName,
        fineTune: config.fineTune,
        effectiveNumClasses: effectiveNumClasses,
        effectiveClassNames: effectiveClassNames,
        epochsConfigured: config.epochs,
        actualEpochsRun: trainingMetrics?.epochs?.length || 0,
        learningRate: config.learningRate,
        batchSize: config.batchSize,
        optimizer: config.optimizerName,
        xaiMethod: config.camMethod,
        validationStrategy: config.validationStrategy,
      },
      overallPerformance: evaluationReport ? {
        accuracy: evaluationReport.accuracy != null ? evaluationReport.accuracy.toFixed(4) : "N/A",
        macroAvgF1: evaluationReport.macroAvg.f1Score != null ? evaluationReport.macroAvg.f1Score.toFixed(4) : "N/A",
        weightedAvgF1: evaluationReport.weightedAvg.f1Score != null ? evaluationReport.weightedAvg.f1Score.toFixed(4) : "N/A",
        aucpr: evaluationReport.aucpr != null ? evaluationReport.aucpr.toFixed(4) : "N/A",
      } : "N/A",
      clusteringMetrics: clusterData ? {
        hierarchicalARI: clusterData.metrics.hierarchicalARI != null ? clusterData.metrics.hierarchicalARI.toFixed(3) : "N/A",
        kmeansARI: clusterData.metrics.kmeansARI != null ? clusterData.metrics.kmeansARI.toFixed(3) : "N/A",
      } : "N/A",
      userContext: {
        zipFileName: zipFile?.name || "N/A",
        userClassificationType: userClassificationType || "Ainda não especificado pelo usuário.",
      },
      trainingSummary: trainingMetrics && trainingMetrics.epochs.length > 0 ? {
        lastEpoch: trainingMetrics.epochs[trainingMetrics.epochs.length -1],
        lastTrainLoss: trainingMetrics.trainLoss[trainingMetrics.trainLoss.length -1]?.toFixed(4) ?? "N/A",
        lastValidLoss: trainingMetrics.validLoss[trainingMetrics.validLoss.length -1]?.toFixed(4) ?? "N/A",
        lastTrainAcc: trainingMetrics.trainAcc[trainingMetrics.trainAcc.length -1]?.toFixed(4) ?? "N/A",
        lastValidAcc: trainingMetrics.validAcc[trainingMetrics.validAcc.length -1]?.toFixed(4) ?? "N/A",
      } : "Nenhuma métrica de treinamento detalhada.",
    };

    context += "### Resumo Estruturado dos Dados (JSON para referência rápida da IA):\n";
    context += "```json\n";
    context += JSON.stringify(summaryJson, null, 2);
    context += "\n```\n\n";
    context += "### Detalhes Adicionais da Sessão (Formato Textual/CSV Simulado):\n\n";
    
    context += "#### Configuração do Modelo e Execução (Detalhes):\n";
    context += `- Modelo Base: ${config.modelName}\n`;
    context += `- Fine-Tuning Completo: ${config.fineTune ? "Sim" : "Não"}\n`;
    context += `- Número de Classes Efetivo: ${effectiveNumClasses}\n`;
    context += `- Nomes das Classes: ${effectiveClassNames.join(', ')}\n`;
    context += `- Épocas Configurado: ${config.epochs}\n`;
    context += `- Taxa de Aprendizagem: ${config.learningRate}\n`;
    context += `- Tamanho do Lote: ${config.batchSize}\n`;
    context += `- Estratégia de Validação: ${config.validationStrategy}\n`;
    context += `- Otimizador: ${config.optimizerName}\n`;
    context += `- Método XAI: ${config.camMethod}\n`;
    context += `Tipo de Classificação Informado pelo Usuário: ${userClassificationType || "Ainda não especificado pelo usuário."}\n\n`;


    if (trainingMetrics && trainingMetrics.epochs.length > 0) {
      context += "#### Métricas de Treinamento (Época a Época):\n";
      context += "Epoca,Perda_Treino,Perda_Validacao,Acuracia_Treino,Acuracia_Validacao\n";
      trainingMetrics.epochs.forEach((epoch, index) => {
        context += `${epoch},${trainingMetrics.trainLoss[index]?.toFixed(4) ?? 'N/A'},${trainingMetrics.validLoss[index]?.toFixed(4) ?? 'N/A'},${trainingMetrics.trainAcc[index]?.toFixed(4) ?? 'N/A'},${trainingMetrics.validAcc[index]?.toFixed(4) ?? 'N/A'}\n`;
      });
      context += "\n";
    } else {
      context += "Nenhuma métrica de treinamento detalhada registrada.\n\n";
    }

    if (evaluationReport) {
      context += "#### Relatório de Classificação:\n";
      context += "Classe,Precisao,Sensibilidade(Recall),Especificidade,F1_Score,Suporte\n";
      effectiveClassNames.forEach(className => {
          const m = evaluationReport.classMetrics[className];
          if(m) context += `${className},${m.precision.toFixed(3)},${m.recall.toFixed(3)},${m.specificity?.toFixed(3) ?? 'N/A'},${m.f1Score.toFixed(3)},${m.support}\n`;
      });
      context += `Média Macro,${evaluationReport.macroAvg.precision.toFixed(3)},${evaluationReport.macroAvg.recall.toFixed(3)},${evaluationReport.macroAvg.specificity?.toFixed(3) ?? 'N/A'},${evaluationReport.macroAvg.f1Score.toFixed(3)},${evaluationReport.macroAvg.support}\n`;
      context += `Média Ponderada,${evaluationReport.weightedAvg.precision.toFixed(3)},${evaluationReport.weightedAvg.recall.toFixed(3)},${evaluationReport.weightedAvg.specificity?.toFixed(3) ?? 'N/A'},${evaluationReport.weightedAvg.f1Score.toFixed(3)},${evaluationReport.weightedAvg.support}\n`;
      context += `Acurácia Geral,,,,${evaluationReport.accuracy?.toFixed(4)}\n`;
      if (evaluationReport.aucpr) {
        context += `AUC-PR (Macro),,,,${evaluationReport.aucpr.toFixed(3)}\n`;
      }
      context += "\n";
    }

    if (confusionMatrix) {
      context += "#### Matriz de Confusão (Normalizada):\n";
      context += `Real\\Predito,${confusionMatrix.labels.join(',')}\n`;
      confusionMatrix.matrix.forEach((row, i) => {
          context += `${confusionMatrix.labels[i]},${row.map(cell => cell.toFixed(2)).join(',')}\n`;
      });
      context += "\n";
    }
    
    if (rocCurveData) {
      context += `#### Curva ROC (AUC: ${rocCurveData.auc.toFixed(3)}):\n`;
      context += `Tipo da Curva: ${rocCurveData.curveType}, Classe Associada: ${rocCurveData.className || 'N/A'}\n`;
      context += `FPR,TPR,Threshold\n`;
      rocCurveData.points.slice(0, 5).forEach(p => { // Limiting to a few points for brevity
          context += `${p.fpr?.toFixed(3) ?? 'N/A'},${p.tpr?.toFixed(3) ?? 'N/A'},${p.threshold?.toFixed(2) ?? 'N/A'}\n`;
      });
      if (rocCurveData.points.length > 5) context += `...(mais ${rocCurveData.points.length - 5} pontos)\n`;
      context += "\n";
    }

    if (prCurveData) {
      context += `#### Curva Precision-Recall (AUC: ${prCurveData.auc.toFixed(3)}):\n`;
      context += `Tipo da Curva: ${prCurveData.curveType}, Classe Associada: ${prCurveData.className || 'N/A'}\n`;
      context += `Recall,Precisao,Threshold\n`;
      prCurveData.points.slice(0, 5).forEach(p => { // Limiting to a few points for brevity
          context += `${p.recall.toFixed(3)},${p.precision?.toFixed(3) ?? 'N/A'},${p.threshold?.toFixed(2) ?? 'N/A'}\n`;
      });
      if (prCurveData.points.length > 5) context += `...(mais ${prCurveData.points.length - 5} pontos)\n`;
      context += "\n";
    }
    
    if (errorAnalysisData && errorAnalysisData.length > 0) {
        context += "#### Análise de Erros (Amostra de Imagens Mal Classificadas):\n";
        errorAnalysisData.forEach((item, index) => { 
            const imageDescription = item.image.startsWith('data:') ? `(Dados da imagem em base64, nome original: ${sampleImagesFromZip.find(si => si.imageDataUrl === item.image)?.fileName || 'desconhecido'})` : `(URL/Caminho: ${item.image})`;
            context += `- Amostra de Erro ${index + 1}: Classe Real: ${item.trueLabel}, Classe Predita: ${item.predLabel} ${imageDescription}\n`;
        });
        context += "\n";
    }

    if (clusterData) {
        context += "#### Resultados da Análise de Clusterização:\n";
        context += "##### Métricas de Clusterização:\n";
        context += `- Hierárquico: ARI=${clusterData.metrics.hierarchicalARI.toFixed(3)}, NMI=${clusterData.metrics.hierarchicalNMI.toFixed(3)}\n`;
        context += `- K-Means: ARI=${clusterData.metrics.kmeansARI.toFixed(3)}, NMI=${clusterData.metrics.kmeansNMI.toFixed(3)}\n\n`;
        context += "##### Amostra de Pontos de Cluster (PCA - Hierárquico):\n";
        clusterData.hierarchical.slice(0,3).forEach((p, idx) => {
            context += `- Ponto ${idx+1}: X=${p.x.toFixed(2)}, Y=${p.y.toFixed(2)}, Cluster Atribuído=${p.cluster}, Classe Verdadeira=${p.trueLabel || 'N/A'}\n`;
        });
        if(clusterData.hierarchical.length > 3) context += `...(mais ${clusterData.hierarchical.length - 3} pontos)\n`;
        context += "\n";
    }
    
    if (augmentedEmbeddingsData && augmentedEmbeddingsData.length > 0) {
        context += "#### Visualização de Embeddings Após Aumento de Dados (PCA - Amostra):\n";
        augmentedEmbeddingsData.slice(0,3).forEach((p, idx) => {
             context += `- Ponto Aumentado ${idx+1}: X=${p.x.toFixed(2)}, Y=${p.y.toFixed(2)}, Classe Original=${p.trueLabel || 'N/A'}\n`;
        });
        if(augmentedEmbeddingsData.length > 3) context += `...(mais ${augmentedEmbeddingsData.length - 3} pontos)\n`;
        context += "\n";
    }

    if (individualEval) {
      context += "#### Resultados da Inspeção de Imagem Individual:\n";
      context += `- Imagem Inspecionada: (Pré-visualização carregada pelo usuário)\n`;
      context += `- Classe Predita: ${individualEval.predictedClass}\n`;
      context += `- Confiança: ${(individualEval.confidence * 100).toFixed(2)}%\n`;
      if (individualEval.uncertaintyScore) {
        context += `- Score de Incerteza: ${individualEval.uncertaintyScore.toFixed(3)}\n`;
      }
      if (camImage) {
        context += `- Visualização XAI (CAM): Gerada e exibida.\n`
      }
      context += "\n";
    }
    context += "## Fim do Contexto dos Resultados ##\n";
    return context;
  }, [
      config, trainingMetrics, evaluationReport, confusionMatrix, classNamesInData, zipFile, 
      rocCurveData, prCurveData, errorAnalysisData, clusterData, areResultsAvailable, 
      userClassificationType, augmentedEmbeddingsData, individualEval, camImage, sampleImagesFromZip, numClassesInData
    ]);


  const initializeChatIA = useCallback(async (forceRefresh: boolean = false) => {
    if (!process.env.API_KEY) {
      console.warn("API Key for Gemini not found. Chat IA disabled.");
      setChatMessages([{ id: 'system-no-api', role: 'system', text: "Chave da API para Gemini não configurada (process.env.API_KEY não encontrada). O Chat IA está desabilitado.", timestamp: new Date() }]);
      return null;
    }
    if (geminiChatInstance && !forceRefresh) {
        return geminiChatInstance;
    }

    setIsChatLoading(true);
    setSimulatedAgentLog([]);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      let resultsContext = getResultsAsTextContext();
      
      const systemInstruction = `Você é Marcelo Claro, um assistente especialista em IA e ciência de dados. Seu objetivo é analisar os resultados de um modelo de classificação de imagens e responder a perguntas sobre eles. Os resultados relevantes da execução atual (fornecidos como um resumo textual abrangente, que inclui um bloco JSON com os principais dados de configuração e desempenho, além de detalhes adicionais em formato de texto e tabelas simuladas) são fornecidos abaixo.
      Contexto dos Resultados:
      ${resultsContext}

      Instruções Adicionais:
      - Consulte o bloco JSON para um resumo rápido e os detalhes textuais/CSV para informações mais granulares.
      - Se o "Tipo de Classificação Informado pelo Usuário" no contexto dos resultados for "Ainda não especificado pelo usuário.", e a pergunta do usuário se beneficiaria de conhecer esse tipo (ex: "como posso melhorar meu modelo?"), sinta-se à vontade para perguntar ao usuário sobre o tipo de classificação que ele está realizando (ex: diagnóstico de melanoma, identificação de tipos de rochas, etc.) para fornecer uma análise mais precisa. Caso contrário, responda com base nas informações disponíveis.
      - Seja claro, conciso e útil. Se os resultados não estiverem disponíveis ou forem insuficientes (conforme indicado no contexto), informe o usuário que ele precisa treinar um modelo primeiro ou fornecer mais detalhes.
      - Se o usuário perguntar algo que exija pesquisa externa (ex: "quais os últimos avanços em..."), simule a ativação de "agentes de pesquisa" e, em seguida, forneça uma resposta abrangente com base no seu conhecimento e no contexto.`;

      const newChat = ai.chats.create({
        model: 'gemini-2.5-flash-preview-04-17',
        config: { systemInstruction: systemInstruction },
      });
      setGeminiChatInstance(newChat);
      
      let initialMessageText: string;
      if (!areResultsAvailable) {
        initialMessageText = "Olá! Eu sou Marcelo Claro, seu assistente de IA. Estou aqui para ajudar a analisar os resultados do seu modelo, discutir características das imagens, ou explicar conceitos de IA.\n\nNo momento, parece que nenhum resultado específico desta sessão foi carregado. Se você já realizou um processamento na aplicação, tente 'Recarregar Contexto'. Caso contrário, por favor, inicie o processamento para que eu possa analisar os dados gerados, ou podemos conversar sobre IA em geral!";
      } else {
        let contextSummary = "";
         if (trainingMetrics && trainingMetrics.epochs.length > 0) contextSummary += "métricas de treinamento, ";
         if (evaluationReport) contextSummary += "relatório de classificação, ";
         if (confusionMatrix) contextSummary += "matriz de confusão, ";
         if (rocCurveData || prCurveData) contextSummary += "curvas ROC/PR, ";
         if (clusterData) contextSummary += "clusterização, ";

         if (contextSummary.endsWith(", ")) contextSummary = contextSummary.slice(0, -2);

        initialMessageText = "Olá! Eu sou Marcelo Claro, seu assistente de IA. Os resultados da sua sessão ";
        if (contextSummary) {
            initialMessageText += `(incluindo ${contextSummary}) `;
        }
        initialMessageText += "e a configuração do modelo foram carregados em meu contexto. ";
        
        if (userClassificationType) {
            initialMessageText += `Seu foco informado é em "${userClassificationType}". `;
        } else {
            initialMessageText += "Para uma análise mais direcionada, você pode me informar o tipo de classificação que está realizando (ex: diagnóstico médico, geologia, etc.). ";
        }
        initialMessageText += "Como posso ajudar com a análise hoje?";
      }
      
      const initialSystemMessage: UserOrModelMessage = {
        id: crypto.randomUUID(), 
        role: 'model', 
        text: initialMessageText, 
        timestamp: new Date()
      };
      
      setChatMessages(prev => {
          const lastMessage = prev.length > 0 ? prev[prev.length - 1] : null;
          if (prev.length === 0 || (lastMessage && lastMessage.id.startsWith('system-')) || forceRefresh) {
              return [initialSystemMessage];
          }
          
          const alreadyHasWelcomeModelMessage = prev.some(m => {
              if ('role' in m && m.role === 'model') {
                  return m.text.startsWith("Olá! Eu sou Marcelo Claro");
              }
              return false;
          });

          if (alreadyHasWelcomeModelMessage && !forceRefresh) {
            const lastModelWelcome = prev.slice().reverse().find(m => {
                if ('role' in m && m.role === 'model') {
                    return m.text.startsWith("Olá! Eu sou Marcelo Claro");
                }
                return false;
            }) as UserOrModelMessage | undefined; 

            if (lastModelWelcome && lastModelWelcome.text === initialSystemMessage.text) {
                 return prev;
            }
          }
          return [...prev, initialSystemMessage];
      });

      return newChat;
    } catch (error) {
      console.error("Failed to initialize Gemini chat:", error);
      setChatMessages([{ id: 'system-error-init', role: 'system', text: "Falha ao inicializar o chat com IA. Verifique o console para erros.", timestamp: new Date() }]);
      return null;
    } finally {
      setIsChatLoading(false);
    }
  }, [getResultsAsTextContext, geminiChatInstance, areResultsAvailable, userClassificationType, trainingMetrics, evaluationReport, confusionMatrix, rocCurveData, prCurveData, clusterData]); // Removed errorAnalysisData as it's not directly used for greeting or system prompt structure

  const forceRefreshChatContext = useCallback(async () => {
    setGeminiChatInstance(null); 
    await initializeChatIA(true); 
  }, [initializeChatIA]);

  useEffect(() => {
    // This effect ensures chat is re-initialized with fresh results when they become available.
    if (areResultsAvailable && process.env.API_KEY) {
      console.log("INFO: Resultados disponíveis detectados. Forçando atualização do contexto do Chat IA.");
      forceRefreshChatContext();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps 
  }, [areResultsAvailable]); // Only re-run if areResultsAvailable changes. forceRefreshChatContext is stable if initializeChatIA is.

  useEffect(() => {
    // This effect handles initial chat loading when the CHAT_IA section is activated.
    if (currentSection === AppSection.CHAT_IA && process.env.API_KEY) {
      if (!geminiChatInstance) {
        // If no instance, or if results just became available and instance was nulled by other effect.
        console.log("INFO: Seção Chat IA ativa e sem instância de chat ou contexto potencialmente desatualizado. Inicializando...");
        initializeChatIA(true); // Force creation of new instance
      }
    }
  }, [currentSection, geminiChatInstance, initializeChatIA]);


  const handleSendMessageToChatIA = async () => {
    if (!chatInput.trim()) return;
    
    const userMessage: UserOrModelMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      text: chatInput,
      timestamp: new Date(),
    };
    setChatMessages(prev => [...prev, userMessage]);
    const currentChatInput = chatInput;
    setChatInput(''); 
    setIsChatLoading(true);
    setSimulatedAgentLog([]);

    const classificationKeywords = ["classificação de", "tipo de imagem é", "estou analisando", "meu dataset é sobre", "foco é em", "objetivo é", "problema de", "trabalhando com", "dataset para", "imagens de"];
    const lowerCaseInput = currentChatInput.toLowerCase();
    let potentialNewClassificationType: string | null = null;

    if (!userClassificationType) { 
        for (const keyword of classificationKeywords) {
            if (lowerCaseInput.includes(keyword)) {
                let extractedType = lowerCaseInput.substring(lowerCaseInput.indexOf(keyword) + keyword.length).trim();
                extractedType = extractedType.replace(/, e.*$/, "").replace(/\.$/, "").replace(/\?$/, "").trim();
                if (extractedType && extractedType.length > 2 && extractedType.length < 100) { 
                    potentialNewClassificationType = extractedType;
                    potentialNewClassificationType = potentialNewClassificationType
                        .split(' ')
                        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                        .join(' ');
                    break;
                }
            }
        }
    }
    
    let chatInstanceToUse = geminiChatInstance;

    if (potentialNewClassificationType && potentialNewClassificationType !== userClassificationType) {
        setUserClassificationType(potentialNewClassificationType);
        chatInstanceToUse = await initializeChatIA(true); 
    } else if (!chatInstanceToUse) {
        chatInstanceToUse = await initializeChatIA(true);
    }
    
    if (!chatInstanceToUse) {
       const errorMessage: UserOrModelMessage = {
          id: crypto.randomUUID(),
          role: 'model',
          text: "Desculpe, não consigo processar sua mensagem. O chat não foi inicializado corretamente. Verifique se a API Key está configurada.",
          timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
      setIsChatLoading(false);
      return;
    }

    const researchKeywords = ["artigos", "pesquisa", "avanços", "multidisciplinar", "literatura", "estudos recentes", "tendências em"];
    const needsResearch = researchKeywords.some(kw => currentChatInput.toLowerCase().includes(kw));

    if (needsResearch && userClassificationType) {
        const agentLogUpdates: string[] = [];
        agentLogUpdates.push(`[${new Date().toLocaleTimeString()}] INFO: Consulta do usuário sugere pesquisa aprofundada.`);
        agentLogUpdates.push(`[${new Date().toLocaleTimeString()}] AGENT_SYSTEM: Ativando Agente de Pesquisa Especializado em "${userClassificationType}".`);
        setSimulatedAgentLog(prev => [...prev, ...agentLogUpdates]);
        
        await new Promise(resolve => setTimeout(resolve, 800)); 
        agentLogUpdates.length = 0; 
        agentLogUpdates.push(`[${new Date().toLocaleTimeString()}] AGENT_WEB_QUERY: Buscando artigos e dados sobre "IA para ${userClassificationType}" e "${currentChatInput.substring(0,30)}...".`);
        setSimulatedAgentLog(prev => [...prev, ...agentLogUpdates]);

        await new Promise(resolve => setTimeout(resolve, 1200));
        agentLogUpdates.length = 0;
        agentLogUpdates.push(`[${new Date().toLocaleTimeString()}] AGENT_ANALYSIS: Processando e sintetizando informações de múltiplas fontes...`);
        setSimulatedAgentLog(prev => [...prev, ...agentLogUpdates]);
        
        await new Promise(resolve => setTimeout(resolve, 1000));
        agentLogUpdates.length = 0;
        agentLogUpdates.push(`[${new Date().toLocaleTimeString()}] AGENT_SYSTEM: Síntese concluída. Preparando resposta...`);
        setSimulatedAgentLog(prev => [...prev, ...agentLogUpdates]);
    }

    try {
      const response = await chatInstanceToUse.sendMessage({ message: currentChatInput });
      const modelResponse: UserOrModelMessage = {
        id: crypto.randomUUID(),
        role: 'model',
        text: response.text,
        timestamp: new Date(),
      };
      setChatMessages(prev => [...prev, modelResponse]);
    } catch (error) {
      console.error("Error sending message to Gemini:", error);
      const errorMessage: UserOrModelMessage = {
        id: crypto.randomUUID(),
        role: 'model',
        text: "Desculpe, ocorreu um erro ao tentar obter uma resposta da IA. Verifique o console para detalhes.",
        timestamp: new Date(),
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsChatLoading(false);
    }
  };


  const startTraining = useCallback(() => {
    if (!zipFile) {
      alert("Por favor, carregue um arquivo ZIP primeiro.");
      return;
    }

    const currentEffectiveNumClasses = config.fineTune ? numClassesInData : config.numClasses;
    const currentClassNames = config.fineTune && classNamesInData.length > 0 ? classNamesInData : Array.from({ length: config.numClasses > 0 ? config.numClasses: DEFAULT_CONFIG.numClasses }, (_, i) => `Classe ${String.fromCharCode(65 + i)}`);

    if (currentEffectiveNumClasses <=0) {
        alert("Número de classes inválido. Verifique os dados carregados ou a configuração.");
        return;
    }
    
    const effectiveConfig = { ...config, numClasses: currentEffectiveNumClasses };

    if (effectiveConfig.fineTune && effectiveConfig.numClasses !== numClassesInData) {
      alert(`Aviso: 'Fine-Tuning Completo' está ativo. O número de classes (${numClassesInData}) e os nomes das classes foram determinados pelo arquivo ZIP. A configuração de 'Número de Classes' no painel foi ajustada.`);
    }
    
    setIsTraining(true);
    setIsTrainingComplete(false);
    setAreResultsAvailable(false); // Results are not available at start of training
    setGeminiChatInstance(null); // Clear chat instance as context will change
    setChatMessages([]);
    setSimulatedAgentLog([]);
    
    const newLog: string[] = ["INFO: Iniciando processo de treinamento do modelo..."];
    newLog.push("INFO: Parâmetros de Configuração Aplicados:");
    (Object.keys(effectiveConfig) as Array<keyof SidebarConfig>).forEach(key => {
        let prettyKey = key.replace(/([A-Z])/g, ' $1'); 
        prettyKey = prettyKey.charAt(0).toUpperCase() + prettyKey.slice(1); 
        
        const configKey = key as keyof SidebarConfig;
        const value: SidebarConfig[typeof configKey] = effectiveConfig[configKey];
        let displayValue: string;

        if (typeof value === 'boolean') {
            displayValue = value ? 'Sim' : 'Não';
        } else if (value === null || typeof value === 'undefined') {
            displayValue = 'Não definido';
        } else {
            displayValue = String(value);
        }
        
        if (SIDEBAR_EXPLANATIONS.hasOwnProperty(key)) {
             const explanationTitle = SIDEBAR_EXPLANATIONS[key as keyof typeof SIDEBAR_EXPLANATIONS].split('\n')[0].replace('O que é: ', '').replace('Como afeta: ', '').split('.')[0];
             if (explanationTitle.length < 40 && !explanationTitle.toLowerCase().includes('o que é')) { 
                 prettyKey = explanationTitle;
             }
        }
        newLog.push(`INFO:  - ${prettyKey}: ${displayValue}`);
    });
    newLog.push("INFO: ---");
    setTrainingLog(newLog);

    setTrainingStatus({ currentEpoch: 0, totalEpochs: effectiveConfig.epochs, message: "INFO: Inicializando ambiente de treinamento..." });
    
    const initialMetrics: TrainingMetrics = { epochs: [], trainLoss: [], validLoss: [], trainAcc: [], validAcc: [] };
    setTrainingMetrics(initialMetrics);
    setEvaluationReport(null); setConfusionMatrix(null); setErrorAnalysisData(null); setClusterData(null);
    setAugmentedEmbeddingsData(null); setRocCurveData(null); setPrCurveData(null);
    setCurrentSection(AppSection.TRAINING);

    let epoch = 0;
    const epochMetrics = { ...initialMetrics }; 
    let bestValidLoss = Infinity;
    let epochsNoImprove = 0;

    const interval = setInterval(() => {
      epoch++;
      const currentEpochLog = [...trainingLog]; 

      const newTrainLoss = 1 / Math.log10(epoch + 1) + Math.random() * 0.2;
      const newValidLoss = 1 / Math.log10(epoch + 1) + 0.1 + Math.random() * 0.2;
      const newTrainAcc = Math.min(0.95, 0.5 + Math.log(epoch) * 0.1 + Math.random() * 0.1);
      const newValidAcc = Math.min(0.90, 0.45 + Math.log(epoch) * 0.1 + Math.random() * 0.1);

      epochMetrics.epochs.push(epoch);
      epochMetrics.trainLoss.push(newTrainLoss);
      epochMetrics.validLoss.push(newValidLoss);
      epochMetrics.trainAcc.push(newTrainAcc);
      epochMetrics.validAcc.push(newValidAcc);
      
      setTrainingMetrics({ ...epochMetrics });
      setTrainingStatus({ currentEpoch: epoch, totalEpochs: effectiveConfig.epochs, message: `INFO: Época ${epoch}/${effectiveConfig.epochs} em processamento...` });
      currentEpochLog.push(`DEBUG: Época ${epoch}: Perda Treino: ${newTrainLoss.toFixed(4)}, Acc Treino: ${newTrainAcc.toFixed(4)}, Perda Val: ${newValidLoss.toFixed(4)}, Acc Val: ${newValidAcc.toFixed(4)}`);

      if (newValidLoss < bestValidLoss) {
        bestValidLoss = newValidLoss;
        epochsNoImprove = 0;
        currentEpochLog.push(`INFO: Época ${epoch}: Nova melhor perda de validação: ${bestValidLoss.toFixed(4)}.`);
      } else {
        epochsNoImprove++;
        currentEpochLog.push(`INFO: Época ${epoch}: Perda de validação (${newValidLoss.toFixed(4)}) não melhorou. Melhor: ${bestValidLoss.toFixed(4)}. Sem melhora por ${epochsNoImprove} épocas.`);
      }
      setTrainingLog(currentEpochLog); 

      const finishTraining = () => { // Removed async as initializeChatIA is not awaited here
        clearInterval(interval);
        setIsTraining(false);
        setIsTrainingComplete(true);
        
        const generatedReport = generateMockClassificationReport(currentClassNames);
        const generatedCM = generateMockConfusionMatrix(currentClassNames);
        const generatedErrorAnalysis = generateMockErrorAnalysis(currentClassNames, sampleImagesFromZip);
        const generatedClusterData = generateMockClusterData(currentEffectiveNumClasses > 0 ? currentEffectiveNumClasses : 2, currentClassNames);
        const generatedAugmentedEmbeddings = generateMockAugmentedEmbeddings(50, 3, currentClassNames);
        const generatedRocData = generateMockROCCurveData(currentClassNames);
        const generatedPrData = generateMockPRCurveData(currentClassNames);

        setEvaluationReport(generatedReport); setConfusionMatrix(generatedCM); setErrorAnalysisData(generatedErrorAnalysis);
        setClusterData(generatedClusterData); setAugmentedEmbeddingsData(generatedAugmentedEmbeddings);
        setRocCurveData(generatedRocData); setPrCurveData(generatedPrData);
        
        setAreResultsAvailable(true); // This will trigger the useEffect to refresh chat context
        setCurrentSection(AppSection.EVALUATION);
        // The useEffect watching areResultsAvailable will handle chat initialization/refresh
      };

      if (epochsNoImprove >= effectiveConfig.patience) {
        const earlyStopMessage = `INFO: Processamento interrompido por Parada Antecipada na época ${epoch}. Paciência (${effectiveConfig.patience}) atingida.`;
        setTrainingStatus({ currentEpoch: epoch, totalEpochs: effectiveConfig.epochs, message: earlyStopMessage });
        setTrainingLog(prevLog => [...prevLog, "INFO: ---", earlyStopMessage, "INFO: Gerando resultados finais..."]);
        finishTraining();
        return; 
      }

      if (epoch >= effectiveConfig.epochs) { 
        const completionMessage = "INFO: Processamento completo (todas as épocas concluídas)!";
        setTrainingStatus({ currentEpoch: epoch, totalEpochs: effectiveConfig.epochs, message: completionMessage });
        setTrainingLog(prevLog => [...prevLog, "INFO: ---", completionMessage, "INFO: Gerando resultados finais..."]);
        finishTraining();
      }
    }, 700); 
  }, [zipFile, config, numClassesInData, classNamesInData, trainingLog, sampleImagesFromZip]); // Removed initializeChatIA, getResultsAsTextContext from deps

  const handleImageForEvalUpload = (file: File) => {
    setUploadedImageForEval(file);
    const reader = new FileReader();
    const currentClassNames = config.fineTune && zipFile && classNamesInData.length > 0 ? classNamesInData : Array.from({ length: config.numClasses > 0 ? config.numClasses : DEFAULT_CONFIG.numClasses }, (_, i) => `Classe ${String.fromCharCode(65 + i)}`);
    
    reader.onloadend = async () => {
        const predClass = currentClassNames.length > 0 ? currentClassNames[Math.floor(Math.random() * currentClassNames.length)] : "Classe Indefinida";
        let uncertainty: number | undefined = undefined;
        if (config.simulatedUncertainty) {
            uncertainty = generateMockUncertaintyScore();
        }

      setIndividualEval({
        imageSrc: reader.result as string,
        predictedClass: predClass,
        confidence: Math.random() * 0.5 + 0.5,
        uncertaintyScore: uncertainty, 
      });
      try {
        const camDataUrl = await generateMockCAMImage(reader.result as string, config.camMethod);
        setCamImage(camDataUrl);
      } catch (error) {
        console.error("Erro ao gerar imagem CAM:", error);
        setCamImage(null); 
      }
    };
    reader.readAsDataURL(file);
    setCurrentSection(AppSection.INSPECTOR);
  };

  useEffect(() => {
    // Effect for general config changes, if any specific action needed.
  }, [config.numClasses, config.fineTune, zipFile]);


  const renderSection = () => {
    const effectiveNumClasses = config.fineTune && zipFile ? numClassesInData : config.numClasses;
    const effectiveClassNames = config.fineTune && zipFile && classNamesInData.length > 0 ? classNamesInData : Array.from({ length: effectiveNumClasses > 0 ? effectiveNumClasses : DEFAULT_CONFIG.numClasses }, (_, i) => `Classe ${String.fromCharCode(65 + i)}`);

    switch (currentSection) {
      case AppSection.DATA_CONFIG:
        return <DataUpload 
                  onFileUpload={handleZipUpload} 
                  numClasses={numClassesInData > 0 ? numClassesInData : config.numClasses} 
                  classNames={classNamesInData.length > 0 ? classNamesInData : Array.from({ length: config.numClasses > 0 ? config.numClasses : DEFAULT_CONFIG.numClasses }, (_, i) => `Classe Padrão ${String.fromCharCode(65 + i)}`)}
                  sampleImages={sampleImagesFromZip}
                  isProcessingZip={isProcessingZip}
                  zipFileLoaded={zipFile !== null}
               />;
      case AppSection.TRAINING:
        return trainingStatus ? (
            <TrainingMonitor 
                status={trainingStatus} 
                metrics={trainingMetrics} 
                isTrainingComplete={isTrainingComplete}
                onExportMetrics={handleExportTrainingMetrics}
                trainingLog={trainingLog} 
            />
        ) : <p className="text-center text-gray-400 py-10">Processamento não iniciado. Configure e inicie na barra lateral.</p>;
      case AppSection.EVALUATION:
        return <EvaluationResults 
                  report={evaluationReport} 
                  confusionMatrix={confusionMatrix} 
                  errorAnalysis={errorAnalysisData} 
                  classNames={effectiveClassNames}
                  rocCurveData={rocCurveData}
                  prCurveData={prCurveData} 
                  isTrainingComplete={isTrainingComplete} 
                />;
      case AppSection.CLUSTERING:
        return clusterData || augmentedEmbeddingsData ? (
            <ClusteringAnalysis 
                data={clusterData} 
                augmentedEmbeddings={augmentedEmbeddingsData}
                classNames={effectiveClassNames} 
            />
        ) : <p className="text-center text-gray-400 py-10">Análise de clusterização não disponível. Conclua o processamento primeiro.</p>;
      case AppSection.INSPECTOR:
        return <ImageInspector 
                  evaluation={individualEval} 
                  camImage={camImage} 
                  onImageUpload={handleImageForEvalUpload} 
                />;
      case AppSection.CHAT_IA:
        return <ChatIA
                  messages={chatMessages}
                  inputValue={chatInput}
                  onInputChange={(e) => setChatInput(e.target.value)}
                  onSendMessage={handleSendMessageToChatIA}
                  isLoading={isChatLoading}
                  onRetryInit={forceRefreshChatContext} 
                  isReady={!!process.env.API_KEY}
                  resultsAvailable={areResultsAvailable}
                  simulatedAgentLog={simulatedAgentLog}
               />;
      case AppSection.EXPLANATIONS:
        return <ExplanationHub topics={ALL_EXPLANATIONS} />;
      default:
        return <p className="text-center text-gray-400 py-10">Selecione uma seção no menu de navegação.</p>;
    }
  };

  return (
    <div className="flex h-screen font-sans">
      <Sidebar 
        config={config} 
        onConfigChange={handleConfigChange} 
        onStartTraining={startTraining} 
        onSaveConfig={saveConfigToJson} 
        isTraining={isTraining}
        zipFileLoaded={zipFile !== null} 
      />
      
      <main className="flex-1 p-6 overflow-y-auto bg-gray-850 flex flex-col">
        <header className="mb-6">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
                <div>
                    <h1 className="text-3xl lg:text-4xl font-bold text-blue-400">Geomaker AI - Analisador de Imagens DL</h1>
                    <p className="text-gray-400 text-sm lg:text-base">Plataforma Interativa para Análise de Classificação e Clusterização de Imagens.</p>
                </div>
            </div>
            {areResultsAvailable && (
                <div className="flex space-x-2">
                    <button
                        onClick={handleExportResults}
                        title="Baixar todos os resultados e métricas em um arquivo CSV."
                        className="px-4 py-2 border border-blue-500 text-blue-400 rounded-md shadow-sm text-sm font-medium hover:bg-blue-500 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-blue-600 transition-colors"
                    >
                        <i className="fas fa-file-csv mr-2"></i> Baixar (CSV)
                    </button>
                    <button
                        onClick={handleExportAllResultsToJson}
                        title="Baixar todos os resultados e configuração em um arquivo JSON."
                        className="px-4 py-2 border border-purple-500 text-purple-400 rounded-md shadow-sm text-sm font-medium hover:bg-purple-500 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-purple-600 transition-colors"
                    >
                        <i className="fas fa-file-code mr-2"></i> Baixar (JSON)
                    </button>
                </div>
            )}
          </div>
        </header>

        <nav className="mb-6">
          <ul className="flex space-x-1 sm:space-x-2 border-b-2 border-gray-700 overflow-x-auto pb-px">
            {Object.values(AppSection).map((section) => (
              <li key={section} className="flex-shrink-0">
                <button
                  onClick={() => setCurrentSection(section)}
                  className={`px-3 py-2 sm:px-4 font-medium rounded-t-lg transition-colors duration-150 text-xs sm:text-sm
                    ${currentSection === section ? 'bg-blue-600 text-gray-900' : 'text-gray-300 hover:bg-gray-700 hover:text-blue-400'}`}
                >
                  {section}
                </button>
              </li>
            ))}
          </ul>
        </nav>

        <div className="flex-grow bg-gray-850 p-4 sm:p-6 rounded-lg shadow-xl min-h-[50vh]">
          {renderSection()}
        </div>
       
        <footer className="mt-auto pt-8 text-center text-sm text-gray-500 pb-6"> 
            <p className="text-xs">Atenção: O pipeline principal de treinamento de modelos e geração de resultados é representativo e opera no frontend. O Chat com IA (Marcelo Claro) utiliza a API Gemini para respostas reais. Para pesquisa e diagnóstico com o pipeline principal, seria necessária uma implementação com backend dedicado. Veja "Aprender Conceitos &gt; Arquitetura da Aplicação" para mais detalhes.</p>
        </footer>
      </main>
    </div>
  );
};

export default App;
