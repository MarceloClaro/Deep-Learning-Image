
import { AppSection } from './constants';

export interface SidebarConfig {
  numClasses: number;
  modelName: 'ResNet18' | 'ResNet50' | 'DenseNet121' | 'VisionTransformer (ViT)' | 'EfficientNetB0';
  fineTune: boolean;
  epochs: number;
  learningRate: number;
  batchSize: number;
  trainSplit: number;
  validSplit: number;
  useWeightedLoss: boolean;
  l2Lambda: number;
  patience: number;
  optimizerName: 'Adam' | 'AdamW' | 'Ranger';
  lrSchedulerName: 'Nenhum' | 'Recozimento por Cosseno' | 'Política de Um Ciclo';
  dataAugmentationMethod: 'Padrão' | 'Mixup' | 'Cutmix';
  camMethod: 'Grad-CAM' | 'Grad-CAM++' | 'Score-CAM' | 'LayerCAM' | 'LIME' | 'SHAP';
  validationStrategy: 'Hold-out (Treino/Validação/Teste)' | 'K-Fold Cross-Validation';
  simulatedUncertainty: boolean;
}

export interface TrainingStatus {
  currentEpoch: number;
  totalEpochs: number;
  message: string;
}

export interface TrainingMetrics {
  epochs: number[];
  trainLoss: number[];
  validLoss: number[];
  trainAcc: number[];
  validAcc: number[];
}

export interface ClassificationReportRow {
  precision: number;
  recall: number; // Same as Sensitivity
  f1Score: number;
  support: number;
  specificity?: number; // Added
}

export interface ClassificationReportData {
  accuracy: number;
  macroAvg: ClassificationReportRow;
  weightedAvg: ClassificationReportRow;
  classMetrics: {
    [className: string]: ClassificationReportRow;
  };
  aucpr?: number; // Area Under Precision-Recall Curve (overall or macro-averaged)
}


export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
}

export interface ErrorAnalysisItem {
  image: string; // URL or base64
  trueLabel: string;
  predLabel: string;
}

export interface ClusterPoint {
  x: number; // PCA1 or t-SNE1
  y: number; // PCA2 or t-SNE2
  cluster: number; // Assigned cluster ID by an algorithm, or true class index
  trueLabel?: string; // Optional: for visualizing true classes or original label for augmented data
  id?: string; // Optional unique ID for the point
}

export interface ClusterVisualizationData {
  hierarchical: ClusterPoint[];
  kmeans: ClusterPoint[];
  trueClasses?: ClusterPoint[]; // For ground truth visualization
  metrics: {
    hierarchicalARI: number;
    hierarchicalNMI: number;
    kmeansARI: number;
    kmeansNMI: number;
  };
}

export interface IndividualEvaluationData {
  imageSrc: string; // base64
  predictedClass: string;
  confidence: number;
  uncertaintyScore?: number; // Added for simulated uncertainty
}

export interface ExplanationTopic {
  id: string;
  title: string;
  summary: string; // Pre-defined summary
  details?: string; // Could be fetched from Gemini or pre-defined
  keywords: string[]; // For Gemini prompt enhancement
  promptSuggestion?: string; // Specific prompt for Gemini
}

export interface ConfigDataEntry {
  parameter: string;
  value: string;
}

export interface SampleImage {
  className: string;
  imageDataUrl: string;
  fileName?: string;
}

export interface ROCPoint { // Also usable for Precision-Recall curves
  fpr?: number; // False Positive Rate (for ROC)
  recall: number; // Same as TPR for ROC, main x-axis for PR
  tpr?: number; // True Positive Rate (for ROC)
  precision?: number; // Precision (for PR)
  threshold?: number;
}

export interface ROCCurveData { // Can also be PRCurveData
  points: ROCPoint[];
  auc: number; // AUC-ROC or AUC-PR
  className?: string;
  curveType: 'ROC' | 'PR';
}

export interface UserOrModelMessage {
  id: string;
  role: 'user' | 'model' | 'system'; // 'system' for critical errors or initial prompts
  text: string;
  timestamp: Date;
}

export interface SystemLogMessage {
    id: string;
    type: 'agent_log' | 'system_info';
    text: string;
    timestamp: Date;
}

export type ChatMessage = UserOrModelMessage | SystemLogMessage;


export { AppSection }; // Re-export for convenience
