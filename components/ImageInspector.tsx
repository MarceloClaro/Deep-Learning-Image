
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import type { IndividualEvaluationData } from '../types';

interface ImageInspectorProps {
  evaluation: IndividualEvaluationData | null;
  camImage: string | null; // base64 CAM overlay image
  onImageUpload: (file: File) => void;
}

export const ImageInspector: React.FC<ImageInspectorProps> = ({ evaluation, camImage, onImageUpload }) => {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
      onImageUpload(file);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onImageUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.gif'] },
    multiple: false,
  });

  return (
    <div className="space-y-8"> 
      <h2 className="text-3xl font-bold text-blue-400 mb-6 text-center">Inspetor de Imagem & Visualização XAI</h2>

      <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
        <h3 className="text-xl font-semibold text-blue-300 mb-4">Carregar Imagem para Inspeção</h3>
        <div
          {...getRootProps()}
          className={`p-8 border-2 border-dashed rounded-md cursor-pointer transition-colors
            ${isDragActive ? 'border-blue-500 bg-gray-700' : 'border-gray-600 hover:border-blue-600 bg-gray-750'}`}
        >
          <input {...getInputProps()} />
          <div className="text-center">
            <i className="fas fa-image text-5xl text-gray-500 mb-3"></i>
            {isDragActive ? (
              <p className="text-lg text-blue-400">Solte a imagem aqui...</p>
            ) : (
              <p className="text-lg text-gray-400">Arraste e solte uma imagem aqui, ou clique para selecionar</p>
            )}
            <p className="text-sm text-gray-500 mt-1">Formatos suportados: JPG, PNG, BMP, GIF.</p>
          </div>
        </div>
      </div>

      {evaluation && preview && (
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
          {/* Original Image and Prediction Card */}
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
            <h3 className="text-xl font-semibold text-blue-300 mb-4">Imagem Carregada & Predição</h3>
            <div className="bg-gray-750 p-4 rounded-md shadow-inner">
                <img src={evaluation.imageSrc} alt="Carregada para avaliação" className="w-full max-w-md mx-auto rounded-md shadow-lg mb-3" />
                <div className="text-center text-gray-200 space-y-1">
                <p>Classe Predita: <span className="font-semibold text-blue-400">{evaluation.predictedClass}</span></p>
                <p>Confiança: <span className="font-semibold text-blue-400">{(evaluation.confidence * 100).toFixed(2)}%</span></p>
                {evaluation.uncertaintyScore !== undefined && (
                    <p>Score de Incerteza: <span className={`font-semibold ${evaluation.uncertaintyScore > 0.15 ? 'text-yellow-400' : 'text-green-400'}`}>{evaluation.uncertaintyScore.toFixed(3)}</span></p>
                )}
                </div>
            </div>
          </div>

          {/* CAM Visualization Card */}
          {camImage && (
            <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
              <h3 className="text-xl font-semibold text-blue-300 mb-4">Visualização XAI</h3>
               <div className="bg-gray-750 p-4 rounded-md shadow-inner">
                <img src={camImage} alt="Visualização XAI" className="w-full max-w-md mx-auto rounded-md shadow-lg" />
                <p className="text-xs text-gray-500 mt-2 text-center">Esta é uma sobreposição XAI. Áreas coloridas indicam regiões de importância para a predição, com variações baseadas no método XAI selecionado na configuração. (Nota: A geração real de XAI requer um backend com o modelo treinado).</p>
              </div>
            </div>
          )}
           {!camImage && (
            <div className="bg-gray-800 p-6 rounded-lg shadow-xl flex items-center justify-center h-full">
                <p className="text-gray-400">Visualização XAI indisponível ou carregando...</p>
            </div>
           )}
        </div>
      )}
    </div>
  );
};
