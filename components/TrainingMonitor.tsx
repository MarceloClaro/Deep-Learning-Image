
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { TrainingStatus, TrainingMetrics } from '../types';

interface TrainingMonitorProps {
  status: TrainingStatus;
  metrics: TrainingMetrics | null;
  isTrainingComplete: boolean; 
  onExportMetrics: () => void; 
  trainingLog: string[]; 
}

export const TrainingMonitor: React.FC<TrainingMonitorProps> = ({ status, metrics, isTrainingComplete, onExportMetrics, trainingLog }) => {
  const progressPercent = status.totalEpochs > 0 ? (status.currentEpoch / status.totalEpochs) * 100 : 0;

  const chartData = metrics ? metrics.epochs.map((epoch, index) => ({
    epoch,
    trainLoss: metrics.trainLoss[index],
    validLoss: metrics.validLoss[index],
    trainAcc: metrics.trainAcc[index],
    validAcc: metrics.validAcc[index],
  })) : [];

  const trainingInProgress = status.message !== "Processamento completo!" && !status.message.includes("Parada Antecipada") && !status.message.includes("concluídas");

  return (
    <div className="space-y-8"> 
      <h2 className="text-3xl font-bold text-blue-400 mb-6 text-center">Monitor de Processamento</h2>
      
      {/* Status and Progress Bar Card */}
      <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
        <div className="flex justify-between items-center mb-4">
            <p className="text-xl text-gray-200">{status.message}</p>
            {isTrainingComplete && (
              <button
                onClick={onExportMetrics}
                title="Baixar apenas as métricas de treinamento epoch-a-epoch em CSV."
                className="px-3 py-1.5 border border-green-500 text-green-400 rounded-md shadow-sm text-xs font-medium hover:bg-green-500 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-green-600 transition-colors"
              >
                <i className="fas fa-chart-line mr-1"></i> Baixar Métricas de Treinamento (CSV)
              </button>
            )}
        </div>
        {status.totalEpochs > 0 && (
          <div className="w-full bg-gray-700 rounded-full h-6 shadow-inner">
            <div
              className="bg-blue-500 h-6 rounded-full text-xs font-medium text-blue-100 text-center p-1 leading-none transition-all duration-500 ease-out"
              style={{ width: `${progressPercent}%` }}
            >
              {status.currentEpoch}/{status.totalEpochs} ({Math.round(progressPercent)}%)
            </div>
          </div>
        )}
      </div>

      {/* Training Log Card */}
      {trainingLog && trainingLog.length > 0 && (
        <details className="bg-gray-800 p-6 rounded-lg shadow-xl group" open>
          <summary className="text-xl font-semibold text-blue-300 cursor-pointer hover:text-blue-200 transition-colors list-none">
            <div className="flex items-center justify-between">
              <span>Log de Processamento</span>
              <i className="fas fa-chevron-down transform transition-transform duration-200 group-open:rotate-180"></i>
            </div>
          </summary>
          <pre className="mt-4 text-xs text-gray-300 bg-gray-750 p-4 rounded-md overflow-x-auto max-h-80 whitespace-pre-line">
            {trainingLog.join('\n')}
          </pre>
        </details>
      )}

      {/* Charts Card */}
      {metrics && metrics.epochs.length > 0 && (
        <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6"> 
              <div>
                <h3 className="text-xl font-semibold text-blue-300 mb-3 text-center">Perda por Época {trainingInProgress ? "(Atualizando...)" : ""}</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="epoch" stroke="#9CA3AF" name="Época" />
                    <YAxis stroke="#9CA3AF" domain={['auto', 'auto']} />
                    <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '0.375rem' }} labelStyle={{ color: '#E5E7EB' }} itemStyle={{ color: '#D1D5DB' }}/>
                    <Legend wrapperStyle={{ color: '#9CA3AF', paddingTop: '10px' }} />
                    <Line type="monotone" dataKey="trainLoss" name="Perda Treino" stroke="#4299E1" strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
                    <Line type="monotone" dataKey="validLoss" name="Perda Validação" stroke="#F56565" strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-blue-300 mb-3 text-center">Acurácia por Época {trainingInProgress ? "(Atualizando...)" : ""}</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="epoch" stroke="#9CA3AF" name="Época"/>
                    <YAxis stroke="#9CA3AF" domain={[0, 1]}/>
                    <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '0.375rem' }} labelStyle={{ color: '#E5E7EB' }} itemStyle={{ color: '#D1D5DB' }} />
                    <Legend wrapperStyle={{ color: '#9CA3AF', paddingTop: '10px' }} />
                    <Line type="monotone" dataKey="trainAcc" name="Acurácia Treino" stroke="#63B3ED" strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
                    <Line type="monotone" dataKey="validAcc" name="Acurácia Validação" stroke="#D53F8C" strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
        </div>
      )}
      {(!metrics || metrics.epochs.length === 0) && isTrainingComplete && (
         <p className="text-gray-400 text-center py-10">Nenhuma métrica de processamento foi registrada.</p>
      )}
      {(!metrics || metrics.epochs.length === 0) && !isTrainingComplete && !trainingInProgress && (
         <p className="text-gray-400 text-center py-10">Aguardando início do processamento para exibir métricas...</p>
      )}
    </div>
  );
};
