
import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import type { ClusterVisualizationData, ClusterPoint } from '../types';

interface ClusteringAnalysisProps {
  data: ClusterVisualizationData | null; // Can be null if only augmented embeddings are available
  augmentedEmbeddings: ClusterPoint[] | null;
  classNames: string[]; // Used for labeling true classes and augmented embeddings
}

const DISTINCT_COLORS = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#06B6D4', '#D97706', '#EC4899', '#0EA5E9', '#6366F1'];

const renderCustomTooltip = (props: any) => {
  const { active, payload } = props;
  if (active && payload && payload.length) {
    const data = payload[0].payload as ClusterPoint;
    const color = DISTINCT_COLORS[data.cluster % DISTINCT_COLORS.length];
    return (
      <div className="bg-gray-700 p-3 border border-gray-600 rounded shadow-lg text-sm">
        <p className="text-gray-200">{`PCA Componente 1 (X): ${data.x.toFixed(3)}`}</p>
        <p className="text-gray-200">{`PCA Componente 2 (Y): ${data.y.toFixed(3)}`}</p>
        <p className="text-gray-200" style={{ color }}>
          {data.trueLabel ? `Classe Original: ${data.trueLabel}` : `Cluster Atribuído: ${data.cluster}`}
        </p>
        {!data.trueLabel && <p className="text-gray-400 text-xs">ID do Ponto: {data.id}</p>}
      </div>
    );
  }
  return null;
};

interface ClusterPlotProps {
  title: string;
  points: ClusterPoint[];
  /** Type of plot: 'algorithm' for K-Means/Hierarchical, 'true_classes' for ground truth or augmented embeddings. */
  plotType: 'algorithm' | 'true_classes';
  /** Only provide for plotType 'true_classes' or 'augmented_embeddings' to name legends correctly. */
  legendClassNames?: string[]; 
}

const ClusterPlot: React.FC<ClusterPlotProps> = ({ title, points, plotType, legendClassNames }) => {
  if (!points || points.length === 0) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg shadow-xl mb-8">
        <h3 className="text-xl font-semibold text-blue-300 mb-4 text-center">{title}</h3>
        <div className="flex items-center justify-center h-80 bg-gray-750 rounded-md">
          <p className="text-gray-400">Não há dados para exibir neste gráfico.</p>
        </div>
      </div>
    );
  }

  const dataByCluster = points.reduce((acc, point) => {
    const key = point.cluster; // This 'cluster' field is the true class index for 'true_classes' type
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(point);
    return acc;
  }, {} as Record<number, ClusterPoint[]>);

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-xl mb-8">
      <h3 className="text-xl font-semibold text-blue-300 mb-4 text-center">{title}</h3>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis 
            type="number" 
            dataKey="x" 
            name="PCA Comp. 1" 
            stroke="#9CA3AF" 
            label={{ value: "PCA Componente 1", position: 'insideBottomRight', offset: -15, fill: '#9CA3AF', fontSize: 12 }}
            domain={['auto', 'auto']}
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            name="PCA Comp. 2" 
            stroke="#9CA3AF" 
            label={{ value: "PCA Componente 2", angle: -90, position: 'insideLeft', fill: '#9CA3AF', fontSize: 12, dx: -5 }}
            domain={['auto', 'auto']}
          />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} content={renderCustomTooltip} />
          <Legend wrapperStyle={{ color: '#9CA3AF', paddingTop: '10px' }} />
          
          {Object.entries(dataByCluster).map(([clusterIdStr, clusterPoints]) => {
            const clusterId = parseInt(clusterIdStr, 10);
            let legendName: string;

            if (plotType === 'true_classes' && legendClassNames && legendClassNames[clusterId]) {
              legendName = legendClassNames[clusterId];
            } else if (plotType === 'true_classes') {
              legendName = `Classe ${clusterId}`; // Fallback for true classes if names are missing
            } else { // plotType === 'algorithm'
              legendName = `Cluster ${clusterId}`;
            }
            
            return (
              <Scatter
                key={clusterId}
                name={legendName}
                data={clusterPoints}
                fill={DISTINCT_COLORS[clusterId % DISTINCT_COLORS.length]}
                fillOpacity={0.7}
              />
            );
          })}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};


export const ClusteringAnalysis: React.FC<ClusteringAnalysisProps> = ({ data, augmentedEmbeddings, classNames }) => {
  if (!data && !augmentedEmbeddings) {
    return <p className="text-gray-400 text-center py-10">Dados de clusterização não disponíveis. Conclua o processamento primeiro.</p>;
  }

  return (
    <div className="space-y-8">
      <h2 className="text-3xl font-bold text-blue-400 mb-6 text-center">Análise de Clusterização (com PCA)</h2>

      {data && (
        <>
          {/* Metrics Table Card */}
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl mb-8">
            <h3 className="text-xl font-semibold text-blue-300 mb-4">Métricas de Avaliação de Clusterização</h3>
            <div className="overflow-x-auto bg-gray-750 p-3 rounded-md shadow-inner">
              <table className="min-w-full divide-y divide-gray-600">
                <thead className="bg-gray-700">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Método</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Adjusted Rand Index (ARI)</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Normalized Mutual Information (NMI)</th>
                  </tr>
                </thead>
                <tbody className="bg-gray-750 divide-y divide-gray-600">
                  <tr className="hover:bg-gray-700 transition-colors">
                    <td className="px-4 py-2 whitespace-nowrap text-sm font-medium text-gray-200">Hierárquico</td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">{data.metrics.hierarchicalARI.toFixed(3)}</td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">{data.metrics.hierarchicalNMI.toFixed(3)}</td>
                  </tr>
                  <tr className="hover:bg-gray-700 transition-colors">
                    <td className="px-4 py-2 whitespace-nowrap text-sm font-medium text-gray-200">K-Means</td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">{data.metrics.kmeansARI.toFixed(3)}</td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">{data.metrics.kmeansNMI.toFixed(3)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-xs text-gray-500 mt-2 text-center">ARI e NMI mais próximos de 1 indicam melhor concordância com as classes verdadeiras.</p>
          </div>
          
          <ClusterPlot title="Clusterização Hierárquica (PCA 2D)" points={data.hierarchical} plotType="algorithm" />
          <ClusterPlot title="Clusterização K-Means (PCA 2D)" points={data.kmeans} plotType="algorithm" />
          {data.trueClasses && data.trueClasses.length > 0 && (
            <ClusterPlot 
                title="Classes Verdadeiras (PCA 2D)" 
                points={data.trueClasses} 
                plotType="true_classes"
                legendClassNames={classNames}
            />
          )}
        </>
      )}

      {augmentedEmbeddings && augmentedEmbeddings.length > 0 && (
         <ClusterPlot 
            title="Visualização de Embeddings Após Aumento de Dados (PCA 2D)" 
            points={augmentedEmbeddings} 
            plotType="true_classes" // Using 'true_classes' type because we color by original class
            legendClassNames={classNames} // Use original class names for the legend
        />
      )}
       {(!data && (!augmentedEmbeddings || augmentedEmbeddings.length === 0)) && (
         <p className="text-gray-400 text-center py-10">Nenhum dado disponível para a análise de clusterização ou visualização de embeddings.</p>
      )}

    </div>
  );
};
