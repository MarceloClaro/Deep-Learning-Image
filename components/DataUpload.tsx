
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import type { SampleImage } from '../types';
import { InfoTooltip } from './InfoTooltip';
import { SIDEBAR_EXPLANATIONS } from '../constants';


interface DataUploadProps {
  onFileUpload: (file: File) => void;
  numClasses: number;
  classNames: string[];
  sampleImages: SampleImage[];
  isProcessingZip: boolean;
  zipFileLoaded: boolean;
}

export const DataUpload: React.FC<DataUploadProps> = ({ onFileUpload, numClasses, classNames, sampleImages, isProcessingZip, zipFileLoaded }) => {
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadedFileSize, setUploadedFileSize] = useState<string | null>(null);
  // State for dummy DICOM/Segmentation options
  const [supportsDicom, setSupportsDicom] = useState(false);
  const [segmentationMethod, setSegmentationMethod] = useState('Nenhuma');


  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setUploadedFileName(file.name);
      setUploadedFileSize((file.size / 1024 / 1024).toFixed(2) + " MB");
      onFileUpload(file);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/zip': ['.zip'] },
    multiple: false,
    disabled: isProcessingZip,
  });

  return (
    <div className="space-y-6"> 
      <h2 className="text-3xl font-bold text-blue-400 mb-6 text-center">Carregar Conjunto de Dados (ZIP) & Opções de Processamento</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Dropzone Card */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-xl h-full flex flex-col">
          <h3 className="text-xl font-semibold text-blue-300 mb-4">Selecione o Arquivo ZIP</h3>
          <div
            {...getRootProps()}
            className={`flex-grow flex flex-col justify-center items-center p-8 border-2 border-dashed rounded-md cursor-pointer transition-colors
              ${isProcessingZip ? 'opacity-50 cursor-default' : (isDragActive ? 'border-blue-500 bg-gray-700' : 'border-gray-600 hover:border-blue-600 bg-gray-750')}`}
          >
            <input {...getInputProps()} />
            <div className="text-center">
                {isProcessingZip ? (
                    <>
                        <i className="fas fa-spinner fa-spin text-5xl text-blue-500 mb-3"></i>
                        <p className="text-lg text-blue-400">Processando arquivo ZIP...</p>
                        <p className="text-sm text-gray-500 mt-1">Aguarde, isso pode levar alguns instantes.</p>
                    </>
                ) : (
                    <>
                        <i className="fas fa-cloud-upload-alt text-5xl text-gray-500 mb-3"></i>
                        {isDragActive ? (
                            <p className="text-lg text-blue-400">Solte o arquivo ZIP aqui...</p>
                        ) : (
                            <p className="text-lg text-gray-400">Arraste e solte um arquivo ZIP aqui, ou clique para selecionar</p>
                        )}
                        <p className="text-sm text-gray-500 mt-1">Deve ser um arquivo ZIP contendo pastas de imagens (uma por classe).</p>
                    </>
                )}
            </div>
          </div>
        </div>

        {/* Advanced Data Options Card (Simulated) */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
            <h3 className="text-xl font-semibold text-blue-300 mb-4">Opções Avançadas de Dados (Demonstrativo)</h3>
            <div className="space-y-4">
                 <div className="flex items-center">
                    <input
                        id="dicomSupport"
                        type="checkbox"
                        checked={supportsDicom}
                        onChange={(e) => setSupportsDicom(e.target.checked)}
                        className="h-4 w-4 text-blue-600 border-gray-600 rounded focus:ring-blue-500 bg-gray-700"
                        disabled // Non-functional
                    />
                    <label htmlFor="dicomSupport" className="ml-2 block text-sm text-gray-300 opacity-70">
                        Processar Metadados DICOM
                        <InfoTooltip text={SIDEBAR_EXPLANATIONS.dicomSupport} />
                    </label>
                </div>
                <div>
                    <label htmlFor="segmentationMethod" className="block text-sm font-medium text-gray-300 opacity-70">
                        Método de Segmentação de Lesão (Pré-processamento)
                        <InfoTooltip text={SIDEBAR_EXPLANATIONS.segmentationMethod} />
                    </label>
                    <select
                        id="segmentationMethod"
                        value={segmentationMethod}
                        onChange={(e) => setSegmentationMethod(e.target.value)}
                        className="mt-1 block w-full p-2 bg-gray-700 border border-gray-600 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-200 opacity-70 cursor-not-allowed"
                        disabled // Non-functional
                    >
                        <option>Nenhuma</option>
                        <option>Limiarização Automática</option>
                        <option>U-Net (Simulado)</option>
                        <option>Watershed (Simulado)</option>
                    </select>
                </div>
                <p className="text-xs text-gray-500">Nota: Estas opções são demonstrativas e não estão funcionalmente implementadas no frontend. Em um sistema completo, elas controlariam o pré-processamento real dos dados.</p>
            </div>
        </div>
      </div>


      {zipFileLoaded && !isProcessingZip && (
        <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
          <h3 className="text-xl font-semibold text-blue-300 mb-3">Informações do Arquivo Carregado</h3>
          <div className="bg-gray-750 p-4 rounded-md shadow-inner">
            <p className="text-gray-200"><i className="fas fa-file-archive mr-2 text-blue-400"></i>Nome: {uploadedFileName}</p>
            <p className="text-gray-200"><i className="fas fa-weight-hanging mr-2 text-blue-400"></i>Tamanho: {uploadedFileSize}</p>
            {numClasses > 0 ? (
              <div className="mt-3 border-t border-gray-600 pt-3">
                <h4 className="text-md font-medium text-blue-300">Dados Detectados:</h4>
                <p className="text-gray-300">Número de Classes: {numClasses}</p>
                <p className="text-gray-300">Nomes das Classes: {classNames.join(', ')}</p>
              </div>
            ) : (
               <p className="text-yellow-400 mt-3 border-t border-gray-600 pt-3">Nenhuma classe válida detectada no ZIP ou ZIP ainda não processado completamente.</p>
            )}
          </div>
        </div>
      )}

      {!isProcessingZip && sampleImages.length > 0 && (
        <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
            <h3 className="text-xl font-semibold text-blue-300 mb-4">Amostras de Imagens (Extraídas do ZIP)</h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {sampleImages.map((sample, index) => (
                <div key={index} className="bg-gray-750 p-2 rounded shadow-inner">
                  <img 
                    src={sample.imageDataUrl}
                    alt={`Amostra ${sample.className} - ${sample.fileName || index + 1}`} 
                    className="w-full h-32 object-cover rounded mb-1"
                  />
                  <p className="text-center text-xs text-gray-300 truncate" title={sample.className}>Classe: {sample.className}</p>
                  {sample.fileName && <p className="text-center text-xs text-gray-500 truncate" title={sample.fileName}>{sample.fileName}</p>}
                </div>
              ))}
            </div>
        </div>
      )}
       {!isProcessingZip && zipFileLoaded && sampleImages.length === 0 && numClasses > 0 && (
         <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
            <h3 className="text-xl font-semibold text-blue-300 mb-3">Amostras de Imagens</h3>
            <p className="text-gray-400">Nenhuma imagem de amostra pôde ser extraída ou exibida para as classes detectadas. Verifique o conteúdo das pastas de classe no arquivo ZIP.</p>
        </div>
      )}
    </div>
  );
};
