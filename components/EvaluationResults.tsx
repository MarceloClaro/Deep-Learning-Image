
import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell, LineChart, Line, AreaChart, Area } from 'recharts';
import type { ClassificationReportData, ConfusionMatrixData, ErrorAnalysisItem, ROCCurveData, ROCPoint } from '../types';

interface EvaluationResultsProps {
  report: ClassificationReportData | null;
  confusionMatrix: ConfusionMatrixData | null;
  errorAnalysis: ErrorAnalysisItem[] | null;
  classNames: string[];
  rocCurveData: ROCCurveData | null;
  prCurveData: ROCCurveData | null; // Precision-Recall Curve data
  isTrainingComplete: boolean;
}

const BLUE_HEATMAP_COLORS = [
  '#EFF6FF', '#DBEAFE', '#BFDBFE', '#93C5FD', '#60A5FA', 
  '#3B82F6', '#2563EB', '#1D4ED8', '#1E40AF', '#1E3A8A'
];

const getCellStyling = (value: number, maxValue: number): { backgroundColor: string; color: string } => {
  if (maxValue === 0) {
    return { backgroundColor: BLUE_HEATMAP_COLORS[0], color: '#1A202C' };
  }
  const ratio = value / maxValue;
  const colorIndex = Math.min(Math.floor(ratio * (BLUE_HEATMAP_COLORS.length -1)), BLUE_HEATMAP_COLORS.length - 1);
  const backgroundColor = BLUE_HEATMAP_COLORS[colorIndex];
  const textColor = colorIndex >= BLUE_HEATMAP_COLORS.length / 2 ? '#F3F4F6' : '#1A202C';
  return { backgroundColor, color: textColor };
};

const CustomCurveTooltip = ({ active, payload, label, curveType }: any) => {
  if (active && payload && payload.length) {
    const point = payload[0].payload;
    let xLabel = curveType === 'ROC' ? 'FPR' : 'Recall';
    let yLabel = curveType === 'ROC' ? 'TPR' : 'Precisão';
    let xValue = curveType === 'ROC' ? point.fpr : point.recall;
    let yValue = curveType === 'ROC' ? point.tpr : point.precision;

    return (
      <div className="bg-gray-700 p-2 border border-gray-600 rounded shadow-lg text-sm">
        <p className="label text-gray-200">{`${xLabel} : ${xValue?.toFixed(3) ?? 'N/A'}`}</p>
        <p className="intro text-gray-200">{`${yLabel} : ${yValue?.toFixed(3) ?? 'N/A'}`}</p>
        {point.threshold !== undefined && <p className="text-gray-400">{`Threshold : ${point.threshold.toFixed(2)}`}</p>}
      </div>
    );
  }
  return null;
};

const renderCurveChart = (curveData: ROCCurveData, titlePrefix: string) => {
  if (!curveData) return null;

  const isROC = curveData.curveType === 'ROC';
  const xKey = isROC ? 'fpr' : 'recall';
  const yKey = isROC ? 'tpr' : 'precision';
  const xLabel = isROC ? 'FPR (Taxa de Falsos Positivos)' : 'Recall (Sensibilidade)';
  const yLabel = isROC ? 'TPR (Taxa de Verdadeiros Positivos)' : 'Precisão';
  const lineColor = isROC ? '#60A5FA' : '#34D399'; // Blue for ROC, Green for PR
  const randomLineName = isROC ? 'Aleatório (ROC)' : 'Baseline (PR)';

  return (
    <div className="bg-gray-750 p-4 rounded-md shadow-inner">
      <ResponsiveContainer width="100%" height={350}>
        <AreaChart data={curveData.points} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis 
            type="number" 
            dataKey={xKey} 
            name={xLabel}
            stroke="#9CA3AF" 
            domain={[0, 1]}
            label={{ value: xLabel, position: 'insideBottomRight', offset: -15, fill: '#9CA3AF', fontSize: 12 }}
            tickFormatter={(tick) => tick.toFixed(1)}
            ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
          />
          <YAxis 
            type="number"
            dataKey={yKey}
            name={yLabel} 
            stroke="#9CA3AF" 
            domain={[0, 1]}
            label={{ value: yLabel, angle: -90, position: 'insideLeft', fill: '#9CA3AF', fontSize: 12, dx: -5 }}
            tickFormatter={(tick) => tick.toFixed(1)}
            ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
          />
          <Tooltip content={<CustomCurveTooltip curveType={curveData.curveType} />} />
          <Legend 
            verticalAlign="top" 
            height={36} 
            wrapperStyle={{color: '#D1D5DB'}} 
            payload={[{ value: `AUC: ${curveData.auc.toFixed(3)}`, type: 'line', id: 'ID01', color: lineColor }]}
          />
          <Area type="monotone" dataKey={yKey as string} stroke={lineColor} fill={lineColor} fillOpacity={0.2} strokeWidth={2} dot={false} name={`${titlePrefix} Curve`} />
          {isROC && <Line type="linear" dataKey={xKey as string} stroke="#F87171" strokeDasharray="5 5" dot={false} name={randomLineName} strokeWidth={1.5} />}
          {!isROC && curveData.auc !== undefined && ( 
             <Line type="linear" dataKey={() => curveData.auc} stroke="#A1A1AA" strokeDasharray="5 5" dot={false} name="Baseline (Pode variar com desbalanceamento)" strokeWidth={1.5} />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};


export const EvaluationResults: React.FC<EvaluationResultsProps> = ({ report, confusionMatrix, errorAnalysis, classNames, rocCurveData, prCurveData, isTrainingComplete }) => {
  if (!isTrainingComplete || !report || !confusionMatrix ) {
    return <p className="text-gray-400 text-center py-10">Resultados da avaliação ainda não disponíveis. Conclua o processamento primeiro.</p>;
  }
  
  const reportArray = classNames.map(className => {
    const metrics = report.classMetrics[className];
    if (!metrics) { 
        console.warn(`Métricas não encontradas para a classe: ${className}`);
        return { name: className, precision: 0, recall: 0, f1Score: 0, support: 0, specificity: 0 };
    }
    return { name: className, ...metrics };
  }).concat([
    { name: 'Média Macro', ...report.macroAvg },
    { name: 'Média Ponderada', ...report.weightedAvg }
  ]);
  
  let maxValCM = 0;
  if (confusionMatrix) {
    confusionMatrix.matrix.forEach(row => row.forEach(val => {
      if (val > maxValCM) maxValCM = val;
    }));
  }

  return (
    <div className="space-y-8"> 
      <h2 className="text-3xl font-bold text-blue-400 mb-6 text-center">Resultados da Avaliação do Modelo</h2>

      {/* Classification Report */}
      <div className="bg-gray-800 p-6 rounded-lg shadow-xl mb-8">
        <h3 className="text-xl font-semibold text-blue-300 mb-4">Relatório de Classificação</h3>
        <div className="overflow-x-auto bg-gray-750 p-3 rounded-md shadow-inner">
          <table className="min-w-full divide-y divide-gray-600">
            <thead className="bg-gray-700">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Classe</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Precisão</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Sensibilidade (Recall)</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Especificidade</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">F1-Score</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Suporte</th>
              </tr>
            </thead>
            <tbody className="bg-gray-750 divide-y divide-gray-600">
              {reportArray.map((row) => (
                <tr key={row.name} className="hover:bg-gray-700 transition-colors">
                  <td className="px-4 py-2 whitespace-nowrap text-sm font-medium text-gray-200">{row.name}</td>
                  <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">{row.precision?.toFixed(3) ?? 'N/A'}</td>
                  <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">{row.recall?.toFixed(3) ?? 'N/A'}</td>
                  <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">{row.specificity?.toFixed(3) ?? 'N/A'}</td>
                  <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">{row.f1Score?.toFixed(3) ?? 'N/A'}</td>
                  <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">{row.support ?? 'N/A'}</td>
                </tr>
              ))}
            </tbody>
            <tfoot className="bg-gray-700">
                <tr>
                    <td colSpan={3} className="px-4 py-2 text-right text-sm font-bold text-gray-200">Acurácia Geral:</td>
                    <td className="px-4 py-2 text-sm font-bold text-gray-200">{report.accuracy?.toFixed(3) ?? 'N/A'}</td>
                    <td colSpan={2} className="px-4 py-2 text-right text-sm font-bold text-gray-200">AUC-PR (Macro): {report.aucpr?.toFixed(3) ?? 'N/A'}</td>
                </tr>
            </tfoot>
          </table>
        </div>
      </div>

      {/* Confusion Matrix */}
      <div className="bg-gray-800 p-6 rounded-lg shadow-xl mb-8">
        <h3 className="text-xl font-semibold text-blue-300 mb-4">Matriz de Confusão (Normalizada)</h3>
         <div className="bg-gray-750 p-4 rounded-md shadow-inner overflow-x-auto">
            <table className="min-w-full border-collapse border border-gray-600">
                <thead>
                    <tr>
                        <th className="border border-gray-600 px-2 py-1 bg-gray-700 text-gray-300 sticky left-0 z-10">Real<span className="text-xs font-thin"> \ Predito</span></th>
                        {confusionMatrix.labels.map(label => (
                            <th key={label} className="border border-gray-600 px-2 py-1 bg-gray-700 text-gray-300 text-xs break-all">{label}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {confusionMatrix.matrix.map((row, i) => (
                        <tr key={i}>
                            <td className="border border-gray-600 px-2 py-1 font-semibold bg-gray-700 text-gray-300 text-xs break-all sticky left-0 z-10">{confusionMatrix.labels[i]}</td>
                            {row.map((cell, j) => {
                                const styling = getCellStyling(cell, maxValCM);
                                return (
                                    <td 
                                        key={j} 
                                        className="border border-gray-600 px-2 py-1 text-center text-sm font-mono"
                                        style={{ backgroundColor: styling.backgroundColor, color: styling.color }}
                                    >
                                        {cell?.toFixed(2) ?? '0.00'}
                                    </td>
                                );
                            })}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
        <p className="text-xs text-gray-500 mt-2 text-center">Células com fundo azul mais escuro indicam valores mais altos. A escala de cores é relativa ao valor máximo na matriz.</p>
      </div>
      
      {/* ROC and PR Curves Section */}
      <div className="bg-gray-800 p-6 rounded-lg shadow-xl mb-8">
        <h3 className="text-xl font-semibold text-blue-300 mb-1 text-center">Curvas de Avaliação</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start mt-4">
            {/* ROC Curve */}
            {rocCurveData && classNames.length >=1 && (
                <div>
                    <p className="text-md text-blue-200 mb-2 text-center">
                        {rocCurveData.className || (classNames.length === 2 ? "Curva ROC Binária" : "Curva ROC Média")}
                    </p>
                    {renderCurveChart(rocCurveData, "ROC")}
                </div>
            )}

            {/* Precision-Recall Curve */}
            {prCurveData && classNames.length >=1 && (
                 <div>
                    <p className="text-md text-green-300 mb-2 text-center">
                        {prCurveData.className || (classNames.length === 2 ? "Curva PR Binária" : "Curva PR Média")}
                    </p>
                    {renderCurveChart(prCurveData, "PR")}
                </div>
            )}
        </div>
         {(!rocCurveData && !prCurveData) && <p className="text-gray-400 text-center">Nenhuma curva de avaliação disponível.</p>}
      </div>


      {/* Error Analysis */}
      {errorAnalysis && errorAnalysis.length > 0 && (
        <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
          <h3 className="text-xl font-semibold text-blue-300 mb-4">Análise de Erros (Amostra de Imagens Mal Classificadas do Dataset)</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-2 gap-4">
            {errorAnalysis.map((item, index) => (
              <div key={index} className="bg-gray-750 p-3 rounded-md shadow-inner text-center flex flex-col items-center">
                <img 
                    src={item.image} 
                    alt={`Mal classificada ${index + 1} - Real: ${item.trueLabel}, Predito: ${item.predLabel}`} 
                    className="w-48 h-48 object-cover rounded-md mb-2 bg-gray-600"
                    onError={(e) => { (e.target as HTMLImageElement).src = `https://via.placeholder.com/200x200.png/1F2937/9CA3AF?text=Erro+Img`; }}
                />
                <p className="text-xs text-gray-400 mt-1">Real: <span className="font-semibold text-green-400">{item.trueLabel}</span></p>
                <p className="text-xs text-gray-400">Predito: <span className="font-semibold text-red-400">{item.predLabel}</span></p>
              </div>
            ))}
          </div>
        </div>
      )}
      {errorAnalysis && errorAnalysis.length === 0 && isTrainingComplete && (
        <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
            <h3 className="text-xl font-semibold text-blue-300 mb-4">Análise de Erros</h3>
            <p className="text-gray-400 mt-4">Nenhuma imagem mal classificada na amostra para análise, ou não há amostras de imagem suficientes/classes para gerar erros.</p>
        </div>
      )}
    </div>
  );
};
