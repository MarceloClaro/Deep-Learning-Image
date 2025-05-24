
import React, { useRef, useEffect, useState } from 'react';
import type { ChatMessage, UserOrModelMessage, SystemLogMessage } from '../types';

interface ChatIAProps {
  messages: ChatMessage[];
  inputValue: string;
  onInputChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onSendMessage: () => void;
  isLoading: boolean;
  onRetryInit: () => Promise<any>;
  isReady: boolean; // True if API key exists
  resultsAvailable: boolean;
  simulatedAgentLog: string[];
}

export const ChatIA: React.FC<ChatIAProps> = ({ 
    messages, 
    inputValue, 
    onInputChange, 
    onSendMessage, 
    isLoading, 
    onRetryInit, 
    isReady, 
    resultsAvailable,
    simulatedAgentLog 
}) => {
  const messagesEndRef = useRef<null | HTMLDivElement>(null);
  const agentLogEndRef = useRef<null | HTMLDivElement>(null);
  
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null);
  const [speechSynthesisInstance, setSpeechSynthesisInstance] = useState<SpeechSynthesis | null>(null);
  const currentUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      setSpeechSynthesisInstance(window.speechSynthesis);
    }
    // Cleanup function to cancel any ongoing speech when the component unmounts
    return () => {
      if (speechSynthesisInstance && currentUtteranceRef.current) {
        speechSynthesisInstance.cancel();
      }
    };
  }, [speechSynthesisInstance]);


  const scrollToBottomMessages = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
   const scrollToBottomAgentLog = () => {
    agentLogEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottomMessages, [messages]);
  useEffect(scrollToBottomAgentLog, [simulatedAgentLog]);


  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !isLoading) {
      onSendMessage();
    }
  };

  const handleVocalize = (text: string, messageId: string) => {
    if (!speechSynthesisInstance) return;

    if (speechSynthesisInstance.speaking) {
      speechSynthesisInstance.cancel(); // Stop current speech if any
      // If the canceled speech was the one we just clicked, it will be set to null by onend, 
      // so we don't re-trigger. If it was another, we proceed.
      if (speakingMessageId === messageId) {
        setSpeakingMessageId(null);
        currentUtteranceRef.current = null;
        return;
      }
    }
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'pt-BR';
    utterance.rate = 1;
    utterance.pitch = 1;

    utterance.onstart = () => {
      setSpeakingMessageId(messageId);
    };
    utterance.onend = () => {
      setSpeakingMessageId(null);
      currentUtteranceRef.current = null;
    };
    utterance.onerror = (event) => {
      console.error("Speech synthesis error:", event);
      setSpeakingMessageId(null);
      currentUtteranceRef.current = null;
    };
    
    currentUtteranceRef.current = utterance;
    speechSynthesisInstance.speak(utterance);
  };

  const handleStopVocalization = () => {
    if (speechSynthesisInstance) {
      speechSynthesisInstance.cancel();
      setSpeakingMessageId(null);
      currentUtteranceRef.current = null;
    }
  };


  return (
    <div className="flex flex-col h-full bg-gray-850" style={{maxHeight: 'calc(100vh - 180px)' /* Adjust based on header/footer */}}>
      <header className="p-4 border-b border-gray-700 sticky top-0 bg-gray-850 z-10">
        <h2 className="text-2xl font-bold text-blue-400 text-center">
          <i className="fas fa-comments mr-2"></i>Chat com IA - Marcelo Claro
        </h2>
        <p className="text-sm text-gray-400 text-center">
          Pergunte sobre os resultados do seu modelo, CSVs, ou conceitos de IA.
        </p>
      </header>

      {!isReady && !messages.some(m => (m as UserOrModelMessage).role === 'system' && (m.id === 'system-no-api' || m.id === 'system-error-init')) && (
        <div className="p-4 text-center">
             <p className="text-yellow-400 bg-yellow-900 bg-opacity-30 p-3 rounded-md border border-yellow-700"><i className="fas fa-exclamation-triangle mr-2"></i>A API Gemini não está configurada (API_KEY não encontrada no ambiente de execução). O chat não funcionará.</p>
        </div>
      )}
       
      {isReady && !resultsAvailable && !messages.some(m => (m as UserOrModelMessage).text.toLowerCase().includes("nenhum resultado de modelo")) && (
        <div className="p-4 text-center">
             <p className="text-orange-400 bg-orange-900 bg-opacity-30 p-3 rounded-md border border-orange-700"><i className="fas fa-info-circle mr-2"></i>Os resultados do modelo ainda não estão disponíveis. Execute o treinamento para uma análise detalhada ou faça perguntas gerais sobre IA.</p>
              <button 
                onClick={onRetryInit} 
                className="mt-3 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm disabled:opacity-50"
                disabled={isLoading}
                title="Tentar recarregar o contexto dos resultados (se o treinamento foi concluído)"
              >
                {isLoading ? <><i className="fas fa-spinner fa-spin mr-2"></i>Carregando...</> : <><i className="fas fa-sync-alt mr-2"></i>Recarregar Contexto</>}
              </button>
        </div>
      )}

      <div className="flex-grow overflow-y-auto p-4 space-y-4">
        {messages.map((msg) => {
            if ('role' in msg) { // UserOrModelMessage
                 const typedMsg = msg as UserOrModelMessage;
                 return (
                    <div key={typedMsg.id} className={`flex ${typedMsg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-xl lg:max-w-2xl px-4 py-3 rounded-xl shadow-lg relative group ${
                            typedMsg.role === 'user' ? 'bg-blue-600 text-white' : 
                            (typedMsg.role === 'system' ? 'bg-yellow-700 bg-opacity-80 text-black text-xs italic p-2' : 'bg-gray-700 text-gray-200')
                        }`}
                        >
                        <p className="whitespace-pre-wrap text-sm leading-relaxed">{typedMsg.text}</p>
                        <div className="flex justify-end items-center mt-1">
                             {typedMsg.role === 'model' && speechSynthesisInstance && (
                                <button
                                    onClick={() => speakingMessageId === typedMsg.id ? handleStopVocalization() : handleVocalize(typedMsg.text, typedMsg.id)}
                                    className={`p-1 rounded-full hover:bg-gray-600 transition-colors text-xs 
                                        ${speakingMessageId === typedMsg.id ? 'text-red-400' : 'text-blue-300'}
                                        ${speakingMessageId && speakingMessageId !== typedMsg.id ? 'opacity-50 cursor-not-allowed' : ''}
                                        absolute -top-2 -right-2 bg-gray-750 border border-gray-600 opacity-0 group-hover:opacity-100 focus:opacity-100
                                    `}
                                    title={speakingMessageId === typedMsg.id ? "Parar Vocalização" : "Vocalizar Resposta"}
                                    aria-label={speakingMessageId === typedMsg.id ? "Parar Vocalização" : "Vocalizar Resposta"}
                                    disabled={speakingMessageId !== null && speakingMessageId !== typedMsg.id}
                                >
                                    {speakingMessageId === typedMsg.id ? <i className="fas fa-stop-circle"></i> : <i className="fas fa-volume-up"></i>}
                                </button>
                            )}
                            <p className={`text-xs ${
                                typedMsg.role === 'user' ? 'text-blue-200' : 
                                (typedMsg.role === 'system' ? 'text-gray-800' :'text-gray-400')
                                }`}>
                                {typedMsg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </p>
                        </div>
                        </div>
                    </div>
                );
            }
            return null; 
        })}
        <div ref={messagesEndRef} />
      </div>

      {simulatedAgentLog && simulatedAgentLog.length > 0 && (
        <div className="mt-2 mb-2 px-4 max-h-32 overflow-y-auto">
          <h4 className="text-sm font-semibold text-purple-400 mb-1 sticky top-0 bg-gray-850 py-1">Log dos Agentes (Simulado):</h4>
          <pre className="text-xs text-purple-300 bg-gray-750 p-2 rounded-md whitespace-pre-wrap">
            {simulatedAgentLog.join('\n')}
            <div ref={agentLogEndRef} />
          </pre>
        </div>
      )}

      {isLoading && (
        <div className="p-4 text-center text-sm text-gray-400">
          <i className="fas fa-spinner fa-spin mr-2 text-lg"></i>Marcelo Claro está pensando...
        </div>
      )}

      <div className="p-4 border-t border-gray-700 bg-gray-850 sticky bottom-0">
        <div className="flex items-center space-x-3">
          <input
            type="text"
            value={inputValue}
            onChange={onInputChange}
            onKeyPress={handleKeyPress}
            placeholder={isReady ? "Digite sua pergunta..." : "Chat indisponível..."}
            className="flex-grow p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-200 placeholder-gray-500 disabled:opacity-70"
            disabled={isLoading || !isReady}
            aria-label="Campo de entrada para mensagem de chat"
          />
          <button
            onClick={onSendMessage}
            disabled={isLoading || !inputValue.trim() || !isReady}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            aria-label="Enviar mensagem"
          >
            <i className="fas fa-paper-plane"></i>
          </button>
        </div>
      </div>
    </div>
  );
};
