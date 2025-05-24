
import React from 'react';
import type { SidebarConfig } from '../types';
import { MODEL_OPTIONS, LEARNING_RATE_OPTIONS, BATCH_SIZE_OPTIONS, OPTIMIZER_OPTIONS, LR_SCHEDULER_OPTIONS, DATA_AUGMENTATION_OPTIONS, CAM_METHOD_OPTIONS, VALIDATION_STRATEGY_OPTIONS, SIDEBAR_EXPLANATIONS } from '../constants';
import { InfoTooltip } from './InfoTooltip';

interface SidebarProps {
  config: SidebarConfig;
  onConfigChange: (newConfig: SidebarConfig) => void;
  onStartTraining: () => void;
  onSaveConfig: () => void;
  isTraining: boolean;
  zipFileLoaded: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ config, onConfigChange, onStartTraining, onSaveConfig, isTraining, zipFileLoaded }) => {
  const handleChange = (key: keyof SidebarConfig, value: any) => {
    onConfigChange({ ...config, [key]: value });
  };

  const isNumClassesDisabled = config.fineTune && zipFileLoaded;

  return (
    <aside className="w-96 bg-gray-800 p-6 space-y-3 overflow-y-auto border-r border-gray-700 shadow-lg flex flex-col">
      <div className="flex-grow space-y-3"> {/* Wrapper for scrollable content */}
        <div className="flex items-center justify-center mb-3">
          <img 
              src="https://github.com/MarceloClaro/CLASSIFICACAO-DE-ROCHAS/blob/main/logo.png?raw=true" 
              alt="Geomaker Logo" 
              className="h-16 w-16" // Adjusted size
          />
        </div>
        <h2 className="text-xl font-semibold text-blue-400 border-b border-gray-700 pb-2">Configuração de Treinamento</h2>

        <div>
          <label htmlFor="numClasses" className="block text-sm font-medium text-gray-300">
            Número de Classes
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.numClasses} />
          </label>
          <input
            type="number"
            id="numClasses"
            value={config.numClasses}
            onChange={(e) => handleChange('numClasses', parseInt(e.target.value))}
            min="1"
            className={`mt-1 block w-full p-2 bg-gray-700 border border-gray-600 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-200 ${isNumClassesDisabled ? 'opacity-70 cursor-not-allowed' : ''}`}
            disabled={isNumClassesDisabled}
          />
          {isNumClassesDisabled && (
            <p className="text-xs text-blue-300 mt-1">{SIDEBAR_EXPLANATIONS.numClassesEffectiveNote}</p>
          )}
        </div>
        
        <div>
          <label htmlFor="modelName" className="block text-sm font-medium text-gray-300">
            Modelo Pré-treinado (SOTA)
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.modelName} />
          </label>
          <select
            id="modelName"
            value={config.modelName}
            onChange={(e) => handleChange('modelName', e.target.value as SidebarConfig['modelName'])}
            className="mt-1 block w-full p-2 bg-gray-700 border border-gray-600 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-200"
          >
            {MODEL_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
          </select>
        </div>

        <div className="flex items-center">
          <input
            id="fineTune"
            type="checkbox"
            checked={config.fineTune}
            onChange={(e) => handleChange('fineTune', e.target.checked)}
            className="h-4 w-4 text-blue-600 border-gray-600 rounded focus:ring-blue-500 bg-gray-700"
          />
          <label htmlFor="fineTune" className="ml-2 block text-sm text-gray-300">
            Fine-Tuning Completo
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.fineTune} />
          </label>
        </div>

        <div>
          <label htmlFor="epochs" className="block text-sm font-medium text-gray-300">
            Épocas: {config.epochs}
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.epochs} />
          </label>
          <input
            type="range"
            id="epochs"
            min="1"
            max="100"
            value={config.epochs}
            onChange={(e) => handleChange('epochs', parseInt(e.target.value))}
            className="mt-1 block w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>
        
        <div>
          <label htmlFor="validationStrategy" className="block text-sm font-medium text-gray-300">
            Estratégia de Validação
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.validationStrategy} />
          </label>
          <select
            id="validationStrategy"
            value={config.validationStrategy}
            onChange={(e) => handleChange('validationStrategy', e.target.value as SidebarConfig['validationStrategy'])}
            className="mt-1 block w-full p-2 bg-gray-700 border border-gray-600 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-200"
          >
            {VALIDATION_STRATEGY_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
          </select>
        </div>

        <div>
          <label htmlFor="learningRate" className="block text-sm font-medium text-gray-300">
            Taxa de Aprendizagem
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.learningRate} />
          </label>
          <select
            id="learningRate"
            value={config.learningRate}
            onChange={(e) => handleChange('learningRate', parseFloat(e.target.value))}
            className="mt-1 block w-full p-2 bg-gray-700 border border-gray-600 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-200"
          >
            {LEARNING_RATE_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
          </select>
        </div>

        <div>
          <label htmlFor="batchSize" className="block text-sm font-medium text-gray-300">
            Tamanho do Lote (Batch Size)
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.batchSize} />
          </label>
          <select
            id="batchSize"
            value={config.batchSize}
            onChange={(e) => handleChange('batchSize', parseInt(e.target.value))}
            className="mt-1 block w-full p-2 bg-gray-700 border border-gray-600 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-200"
          >
            {BATCH_SIZE_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
          </select>
        </div>

        <div>
          <label htmlFor="trainSplit" className="block text-sm font-medium text-gray-300">
            Divisão de Treinamento: {config.trainSplit.toFixed(2)}
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.trainSplit} />
          </label>
          <input
            type="range"
            id="trainSplit"
            min="0.5"
            max="0.9"
            step="0.05"
            value={config.trainSplit}
            onChange={(e) => handleChange('trainSplit', parseFloat(e.target.value))}
            className="mt-1 block w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>
        
        <div>
          <label htmlFor="validSplit" className="block text-sm font-medium text-gray-300">
            Divisão de Validação: {config.validSplit.toFixed(2)}
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.validSplit} />
          </label>
          <input
            type="range"
            id="validSplit"
            min="0.05"
            max="0.4"
            step="0.05"
            value={config.validSplit}
            onChange={(e) => handleChange('validSplit', parseFloat(e.target.value))}
            className="mt-1 block w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>
        
        <div className="flex items-center">
          <input
            id="useWeightedLoss"
            type="checkbox"
            checked={config.useWeightedLoss}
            onChange={(e) => handleChange('useWeightedLoss', e.target.checked)}
            className="h-4 w-4 text-blue-600 border-gray-600 rounded focus:ring-blue-500 bg-gray-700"
          />
          <label htmlFor="useWeightedLoss" className="ml-2 block text-sm text-gray-300">
            Usar Perda Ponderada
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.useWeightedLoss} />
          </label>
        </div>
        
        <div className="flex items-center">
          <input
            id="simulatedUncertainty"
            type="checkbox"
            checked={config.simulatedUncertainty}
            onChange={(e) => handleChange('simulatedUncertainty', e.target.checked)}
            className="h-4 w-4 text-blue-600 border-gray-600 rounded focus:ring-blue-500 bg-gray-700"
          />
          <label htmlFor="simulatedUncertainty" className="ml-2 block text-sm text-gray-300">
            Apresentar Score de Incerteza
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.simulatedUncertainty} />
          </label>
        </div>

        <div>
          <label htmlFor="l2Lambda" className="block text-sm font-medium text-gray-300">
            Regularização L2 (λ): {config.l2Lambda.toFixed(2)}
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.l2Lambda} />
          </label>
          <input
            type="range"
            id="l2Lambda"
            min="0.0"
            max="0.1"
            step="0.01"
            value={config.l2Lambda}
            onChange={(e) => handleChange('l2Lambda', parseFloat(e.target.value))}
            className="mt-1 block w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>

        <div>
          <label htmlFor="patience" className="block text-sm font-medium text-gray-300">
            Paciência (Early Stopping): {config.patience}
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.patience} />
          </label>
          <input
            type="range"
            id="patience"
            min="1"
            max="10"
            value={config.patience}
            onChange={(e) => handleChange('patience', parseInt(e.target.value))}
            className="mt-1 block w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>
        
        <h3 className="text-lg font-semibold text-blue-400 border-t border-gray-700 pt-3 mt-3">Opções Avançadas</h3>
        
        <div>
          <label htmlFor="optimizerName" className="block text-sm font-medium text-gray-300">
            Otimizador
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.optimizerName} />
          </label>
          <select
            id="optimizerName"
            value={config.optimizerName}
            onChange={(e) => handleChange('optimizerName', e.target.value as SidebarConfig['optimizerName'])}
            className="mt-1 block w-full p-2 bg-gray-700 border border-gray-600 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-200"
          >
            {OPTIMIZER_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
          </select>
        </div>

        <div>
          <label htmlFor="lrSchedulerName" className="block text-sm font-medium text-gray-300">
            Agendador de LR
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.lrSchedulerName} />
          </label>
          <select
            id="lrSchedulerName"
            value={config.lrSchedulerName}
            onChange={(e) => handleChange('lrSchedulerName', e.target.value as SidebarConfig['lrSchedulerName'])}
            className="mt-1 block w-full p-2 bg-gray-700 border border-gray-600 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-200"
          >
            {LR_SCHEDULER_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
          </select>
        </div>

        <div>
          <label htmlFor="dataAugmentationMethod" className="block text-sm font-medium text-gray-300">
            Aumento de Dados
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.dataAugmentationMethod} />
          </label>
          <select
            id="dataAugmentationMethod"
            value={config.dataAugmentationMethod}
            onChange={(e) => handleChange('dataAugmentationMethod', e.target.value as SidebarConfig['dataAugmentationMethod'])}
            className="mt-1 block w-full p-2 bg-gray-700 border border-gray-600 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-200"
          >
            {DATA_AUGMENTATION_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
          </select>
        </div>

        <div>
          <label htmlFor="camMethod" className="block text-sm font-medium text-gray-300">
            Método XAI (Visualização)
            <InfoTooltip text={SIDEBAR_EXPLANATIONS.camMethod} />
          </label>
          <select
            id="camMethod"
            value={config.camMethod}
            onChange={(e) => handleChange('camMethod', e.target.value as SidebarConfig['camMethod'])}
            className="mt-1 block w-full p-2 bg-gray-700 border border-gray-600 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-200"
          >
            {CAM_METHOD_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
          </select>
        </div>
      </div> {/* End of scrollable content wrapper */}
      
      <div className="pt-4 space-y-3 border-t border-gray-700 mt-auto"> {/* Buttons section */}
        <button
            onClick={onStartTraining}
            disabled={isTraining || !zipFileLoaded}
            title={!zipFileLoaded ? "Carregue um arquivo ZIP primeiro" : ""}
            className="w-full flex items-center justify-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
            <i className="fas fa-play mr-2"></i> Iniciar Processamento
        </button>
        <button
            onClick={onSaveConfig}
            className="w-full flex items-center justify-center px-4 py-2 border border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-300 bg-gray-700 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-blue-500"
        >
            <i className="fas fa-save mr-2"></i> Salvar Configuração
        </button>
      </div>

      <div className="mt-auto border-t border-gray-700 pt-3 text-xs text-gray-400 text-center space-y-1 pb-2">
        <p>Produzido por: Projeto Geomaker + IA - Professor: Marcelo Claro</p>
        <p>
          Contato: marceloclaro@gmail.com | DOI: <a href="https://doi.org/10.5281/zenodo.13910277" target="_blank" rel="noopener noreferrer" className="hover:text-blue-400 underline">10.5281/zenodo.13910277</a>
        </p>
      </div>
    </aside>
  );
};
