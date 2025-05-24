
import React, { useState } from 'react';

interface InfoTooltipProps {
  text: string;
  className?: string;
}

export const InfoTooltip: React.FC<InfoTooltipProps> = ({ text, className }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div className={`relative inline-block ml-1 ${className}`}>
      <i
        className="fas fa-info-circle text-blue-500 cursor-pointer hover:text-blue-300 text-xs"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onClick={(e) => {e.stopPropagation(); setShowTooltip(!showTooltip);}} // Stop propagation to prevent label click if nested
        role="button"
        aria-label="Mais informações"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowTooltip(!showTooltip);}}
      ></i>
      {showTooltip && (
        <div 
          className="absolute z-20 w-72 p-3 text-xs leading-normal text-left text-white break-words transform -translate-x-1/2 bg-gray-700 rounded-lg shadow-xl left-1/2 bottom-full mb-2 tooltip-text whitespace-pre-line"
          role="tooltip"
        >
          {text}
        </div>
      )}
    </div>
  );
};