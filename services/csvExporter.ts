
import type {
  TrainingMetrics,
  ClassificationReportData,
  ConfusionMatrixData,
  ClusterVisualizationData,
  IndividualEvaluationData,
  ErrorAnalysisItem,
  ROCCurveData,
  ROCPoint
} from '../types';

const escapeCSVField = (field: any): string => {
  if (field === null || typeof field === 'undefined') {
    return '';
  }
  const stringField = String(field);
  if (stringField.includes(',') || stringField.includes('"') || stringField.includes('\n')) {
    return `"${stringField.replace(/"/g, '""')}"`;
  }
  return stringField;
};

const formatMetricsToCSV = (metrics: TrainingMetrics | null): string => {
  if (!metrics || metrics.epochs.length === 0) return 'Seção Monitor de Treinamento Vazia ou Sem Dados\n';
  let csvString = 'Seção: Monitor de Treinamento\n';
  csvString += 'Epoca,Perda_Treino,Perda_Validacao,Acuracia_Treino,Acuracia_Validacao\n';
  metrics.epochs.forEach((epoch, index) => {
    csvString += `${epoch},${escapeCSVField(metrics.trainLoss[index]?.toFixed(4))},${escapeCSVField(metrics.validLoss[index]?.toFixed(4))},${escapeCSVField(metrics.trainAcc[index]?.toFixed(4))},${escapeCSVField(metrics.validAcc[index]?.toFixed(4))}\n`;
  });
  return csvString + '\n';
};

const formatEvaluationReportToCSV = (report: ClassificationReportData | null, classNames: string[]): string => {
  if (!report) return 'Seção Relatório de Avaliação Vazia\n';
  let csvString = 'Seção: Relatório de Classificação\n';
  csvString += 'Classe,Precisao,Sensibilidade(Recall),Especificidade,F1_Score,Suporte\n';
  
  classNames.forEach(className => {
    const classMetric = report.classMetrics[className];
    if (classMetric) {
      csvString += `${escapeCSVField(className)},${escapeCSVField(classMetric.precision?.toFixed(4))},${escapeCSVField(classMetric.recall?.toFixed(4))},${escapeCSVField(classMetric.specificity?.toFixed(4))},${escapeCSVField(classMetric.f1Score?.toFixed(4))},${escapeCSVField(classMetric.support)}\n`;
    }
  });
  csvString += `Média Macro,${escapeCSVField(report.macroAvg?.precision?.toFixed(4))},${escapeCSVField(report.macroAvg?.recall?.toFixed(4))},${escapeCSVField(report.macroAvg?.specificity?.toFixed(4))},${escapeCSVField(report.macroAvg?.f1Score?.toFixed(4))},${escapeCSVField(report.macroAvg?.support)}\n`;
  csvString += `Média Ponderada,${escapeCSVField(report.weightedAvg?.precision?.toFixed(4))},${escapeCSVField(report.weightedAvg?.recall?.toFixed(4))},${escapeCSVField(report.weightedAvg?.specificity?.toFixed(4))},${escapeCSVField(report.weightedAvg?.f1Score?.toFixed(4))},${escapeCSVField(report.weightedAvg?.support)}\n`;
  csvString += `Acurácia Geral,,,,${escapeCSVField(report.accuracy?.toFixed(4))}\n`;
  csvString += `AUC-PR (Macro),,,,${escapeCSVField(report.aucpr?.toFixed(4))}\n`;
  return csvString + '\n';
};

const formatConfusionMatrixToCSV = (matrixData: ConfusionMatrixData | null): string => {
  if (!matrixData) return 'Seção Matriz de Confusão Vazia\n';
  let csvString = 'Seção: Matriz de Confusão\n';
  csvString += 'Real\\Predito,' + matrixData.labels.map(label => escapeCSVField(label)).join(',') + '\n';
  matrixData.matrix.forEach((row, i) => {
    csvString += escapeCSVField(matrixData.labels[i]) + ',' + row.map(cell => escapeCSVField(cell?.toFixed(4))).join(',') + '\n';
  });
  return csvString + '\n';
};

const formatErrorAnalysisToCSV = (errorAnalysis: ErrorAnalysisItem[] | null): string => {
    if (!errorAnalysis || errorAnalysis.length === 0) return 'Seção Análise de Erros Vazia\n';
    let csvString = 'Seção: Análise de Erros (Amostra)\n';
    csvString += 'Imagem_Placeholder_URL,Classe_Real,Classe_Predita\n';
    errorAnalysis.forEach(item => {
        csvString += `${escapeCSVField(item.image)},${escapeCSVField(item.trueLabel)},${escapeCSVField(item.predLabel)}\n`;
    });
    return csvString + '\n';
};


const formatClusterDataToCSV = (clusterData: ClusterVisualizationData | null): string => {
  if (!clusterData) return 'Seção Análise de Clusterização Vazia\n';
  let csvString = 'Seção: Métricas de Clusterização\n';
  csvString += 'Metodo,Metrica,Valor\n';
  csvString += `Hierárquico,ARI,${escapeCSVField(clusterData.metrics.hierarchicalARI?.toFixed(4))}\n`;
  csvString += `Hierárquico,NMI,${escapeCSVField(clusterData.metrics.hierarchicalNMI?.toFixed(4))}\n`;
  csvString += `K-Means,ARI,${escapeCSVField(clusterData.metrics.kmeansARI?.toFixed(4))}\n`;
  csvString += `K-Means,NMI,${escapeCSVField(clusterData.metrics.kmeansNMI?.toFixed(4))}\n`;
  // Add point data if necessary in future
  return csvString + '\n';
};

const formatImageInspectorToCSV = (evalData: IndividualEvaluationData | null): string => {
  if (!evalData) return 'Seção Inspetor de Imagem Vazia (Nenhuma imagem avaliada)\n';
  let csvString = 'Seção: Inspetor de Imagem\n';
  csvString += 'Metrica,Valor\n';
  csvString += `Classe_Predita,${escapeCSVField(evalData.predictedClass)}\n`;
  csvString += `Confianca,${escapeCSVField(evalData.confidence?.toFixed(4))}\n`;
  if (evalData.uncertaintyScore !== undefined) {
    csvString += `Score_Incerteza,${escapeCSVField(evalData.uncertaintyScore?.toFixed(4))}\n`;
  }
  return csvString + '\n';
};

const formatCurveDataToCSV = (curveData: ROCCurveData | null, curveName: string): string => {
    if (!curveData || curveData.points.length === 0) return `Seção ${curveName} Vazia\n`;
    let csvString = `Seção: ${curveName} (AUC: ${curveData.auc.toFixed(4)})\n`;
    if (curveData.curveType === 'ROC') {
        csvString += 'FPR,TPR,Threshold\n';
        curveData.points.forEach(p => {
            csvString += `${escapeCSVField(p.fpr?.toFixed(4))},${escapeCSVField(p.tpr?.toFixed(4))},${escapeCSVField(p.threshold?.toFixed(2))}\n`;
        });
    } else { // PR Curve
        csvString += 'Recall,Precisao,Threshold\n';
        curveData.points.forEach(p => {
            csvString += `${escapeCSVField(p.recall?.toFixed(4))},${escapeCSVField(p.precision?.toFixed(4))},${escapeCSVField(p.threshold?.toFixed(2))}\n`;
        });
    }
    return csvString + '\n';
};


export const exportTrainingMetricsToCSV = (
  trainingMetrics: TrainingMetrics | null,
  modelName: string
): void => {
  if (!trainingMetrics || trainingMetrics.epochs.length === 0) {
    alert("Nenhuma métrica de treinamento para exportar.");
    return;
  }
  let csvString = `Métricas de Treinamento para Modelo: ${modelName}\n`;
  csvString += formatMetricsToCSV(trainingMetrics);

  const blob = new Blob([`\uFEFF${csvString}`], { type: 'text/csv;charset=utf-8;' }); // Added BOM for Excel
  const link = document.createElement('a');
  if (link.download !== undefined) {
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `metricas_treinamento_${modelName}_${new Date().toISOString().slice(0,10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
};


export const exportResultsToCSV = (
  trainingMetrics: TrainingMetrics | null,
  evaluationReport: ClassificationReportData | null,
  confusionMatrix: ConfusionMatrixData | null,
  errorAnalysis: ErrorAnalysisItem[] | null,
  clusterData: ClusterVisualizationData | null,
  individualEval: IndividualEvaluationData | null,
  classNames: string[],
  modelName: string,
  rocCurveData: ROCCurveData | null,
  prCurveData: ROCCurveData | null
): void => {
  let fullCSVString = `Resultados Consolidados para Modelo: ${modelName}\n\n`;
  
  fullCSVString += formatMetricsToCSV(trainingMetrics);
  fullCSVString += formatEvaluationReportToCSV(evaluationReport, classNames);
  fullCSVString += formatConfusionMatrixToCSV(confusionMatrix);
  fullCSVString += formatCurveDataToCSV(rocCurveData, "Curva ROC");
  fullCSVString += formatCurveDataToCSV(prCurveData, "Curva Precision-Recall");
  fullCSVString += formatErrorAnalysisToCSV(errorAnalysis);
  fullCSVString += formatClusterDataToCSV(clusterData);
  fullCSVString += formatImageInspectorToCSV(individualEval);

  const blob = new Blob([`\uFEFF${fullCSVString}`], { type: 'text/csv;charset=utf-8;' }); // Added BOM for Excel
  const link = document.createElement('a');
  if (link.download !== undefined) {
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `resultados_consolidados_${modelName}_${new Date().toISOString().slice(0,10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
};
