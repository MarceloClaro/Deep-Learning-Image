
import React, { useState, useRef, useEffect } from 'react';
import type { ExplanationTopic } from '../types';
import { GoogleGenAI } from "@google/genai";

interface ExplanationHubProps {
  topics: ExplanationTopic[];
}

const API_KEY = process.env.API_KEY;
let geminiAiInstance: GoogleGenAI | null = null;
if (API_KEY) {
  geminiAiInstance = new GoogleGenAI({ apiKey: API_KEY });
}

interface MiniChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
}

interface ExplanationCardProps {
  topic: ExplanationTopic;
  aiInstance: GoogleGenAI | null;
}

const ExplanationCard: React.FC<ExplanationCardProps> = ({ topic, aiInstance }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isInputVisible, setIsInputVisible] = useState(false);
  const [userQuestion, setUserQuestion] = useState('');
  const [aiChatHistory, setAiChatHistory] = useState<MiniChatMessage[]>([]);
  const [isLoadingAi, setIsLoadingAi] = useState(false);
  const chatEndRef = useRef<null | HTMLDivElement>(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [aiChatHistory]);

  const handleToggleInput = () => {
    setIsInputVisible(!isInputVisible);
    if (!isInputVisible && aiChatHistory.length === 0 && topic.summary) {
       // Optional: If opening input for the first time and no chat, add summary as initial model message.
       // setAiChatHistory([{ id: `summary-${topic.id}`, role: 'model', text: `Resumo inicial:\n${topic.summary}` }]);
    }
  };
  
  const handleSendTopicQuestion = async () => {
    if (!userQuestion.trim() || !aiInstance) {
      if (!aiInstance) {
        setAiChatHistory(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: "API Gemini não configurada. Não é possível buscar explicação da IA." }]);
      }
      return;
    }

    const newUserMessage: MiniChatMessage = { id: crypto.randomUUID(), role: 'user', text: userQuestion };
    setAiChatHistory(prev => [...prev, newUserMessage]);
    const currentQuestion = userQuestion;
    setUserQuestion('');
    setIsLoadingAi(true);

    try {
      const model = 'gemini-2.5-flash-preview-04-17';
      const prompt = `Contexto do Tópico:
Título: ${topic.title}
Resumo Original (para sua referência): ${topic.summary}
Palavras-chave: ${topic.keywords.join(', ')}

Com base no contexto do tópico "${topic.title}" acima, por favor, responda à seguinte pergunta específica do usuário de forma clara e concisa:
"${currentQuestion}"

Se a pergunta for muito genérica ou se desviar significativamente do tópico, gentilmente relembre o usuário sobre o tópico em questão ou peça para que reformule a pergunta de forma mais específica ao tópico.`;
      
      const result = await aiInstance.models.generateContent({
        model: model,
        contents: prompt,
      });
      
      const responseText = result.text;
      setAiChatHistory(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: responseText }]);

    } catch (error) {
      console.error("Erro ao buscar explicação da IA para o tópico:", error);
      setAiChatHistory(prev => [...prev, { id: crypto.randomUUID(), role: 'model', text: "Desculpe, não foi possível buscar uma explicação da IA no momento." }]);
    } finally {
      setIsLoadingAi(false);
    }
  };

  return (
    <div className="bg-gray-800 p-5 rounded-lg shadow-xl mb-6">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full text-left flex justify-between items-center py-2 text-xl font-semibold text-blue-300 hover:text-blue-200 transition-colors"
        aria-expanded={isOpen}
        aria-controls={`explanation-${topic.id}`}
      >
        {topic.title}
        <i className={`fas fa-chevron-down transform transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}></i>
      </button>
      {isOpen && (
        <div id={`explanation-${topic.id}`} className="mt-3 space-y-3 text-gray-300 text-sm">
          <p className="leading-relaxed whitespace-pre-line">{topic.summary}</p>
          {topic.details && <p className="border-t border-gray-700 pt-3 mt-3 leading-relaxed whitespace-pre-line">{topic.details}</p>}
          
          <button 
            onClick={handleToggleInput}
            disabled={!aiInstance}
            className="mt-3 px-4 py-2 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded-md shadow-md disabled:opacity-60 flex items-center transition-colors"
          >
            <i className={`fas ${isInputVisible ? 'fa-times-circle' : 'fa-comments'} mr-2`}></i>
            {isInputVisible ? 'Fechar Chat do Tópico' : 'Perguntar à IA sobre este Tópico'}
          </button>
          
          {isInputVisible && (
            <div className="mt-4 p-4 bg-gray-750 border border-gray-600 rounded-md shadow-inner">
              <div className="max-h-60 overflow-y-auto space-y-2 mb-3 pr-2">
                {aiChatHistory.map(msg => (
                  <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-md px-3 py-2 rounded-lg shadow ${msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-600 text-gray-200'}`}>
                      <p className="whitespace-pre-wrap text-xs">{msg.text}</p>
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>
              {isLoadingAi && (
                <div className="text-center text-xs text-gray-400 my-2">
                  <i className="fas fa-spinner fa-spin mr-1"></i>A IA está pensando...
                </div>
              )}
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={userQuestion}
                  onChange={(e) => setUserQuestion(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && !isLoadingAi && handleSendTopicQuestion()}
                  placeholder="Sua pergunta sobre este tópico..."
                  className="flex-grow p-2 text-xs bg-gray-600 border border-gray-500 rounded-md focus:ring-1 focus:ring-blue-500 focus:border-blue-500 text-gray-200 placeholder-gray-400"
                  disabled={isLoadingAi}
                />
                <button
                  onClick={handleSendTopicQuestion}
                  disabled={isLoadingAi || !userQuestion.trim()}
                  className="px-3 py-2 text-xs bg-green-600 hover:bg-green-700 text-white rounded-md shadow-md disabled:opacity-60"
                >
                  <i className="fas fa-paper-plane"></i> Enviar
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export const ExplanationHub: React.FC<ExplanationHubProps> = ({ topics }) => {
  return (
    <div className="space-y-8"> 
      <h2 className="text-3xl font-bold text-blue-400 mb-4 text-center">Aprenda Conceitos Chave</h2>
      <p className="text-gray-400 mb-6 text-center max-w-2xl mx-auto">Expanda os tópicos abaixo para uma breve explicação. Você também pode iniciar um chat com a IA para tirar dúvidas específicas sobre cada conceito.</p>
      {!geminiAiInstance && <p className="text-yellow-400 text-sm bg-yellow-900 bg-opacity-50 p-3 rounded-md text-center mb-6 border border-yellow-700"><i className="fas fa-exclamation-triangle mr-2"></i>Chave da API Gemini não configurada. Chat sobre tópicos com IA está desabilitado.</p>}
      <div className="space-y-4">
        {topics.map(topic => (
          <ExplanationCard key={topic.id} topic={topic} aiInstance={geminiAiInstance} />
        ))}
      </div>
    </div>
  );
};
