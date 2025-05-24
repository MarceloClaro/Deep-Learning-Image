
import type { TrainingMetrics, ClassificationReportData, ClassificationReportRow, ConfusionMatrixData, ClusterVisualizationData, ErrorAnalysisItem, ROCCurveData, ROCPoint, SampleImage, ClusterPoint, SidebarConfig } from '../types';

export const generateMockMetrics = (epochs: number): TrainingMetrics => {
  const epochNumbers = Array.from({ length: epochs }, (_, i) => i + 1);
  return {
    epochs: epochNumbers,
    trainLoss: epochNumbers.map(e => 1 / Math.log10(e + 1) + Math.random() * 0.2),
    validLoss: epochNumbers.map(e => 1 / Math.log10(e + 1) + 0.1 + Math.random() * 0.2),
    trainAcc: epochNumbers.map(e => Math.min(0.95, 0.5 + Math.log(e) * 0.1 + Math.random() * 0.1)),
    validAcc: epochNumbers.map(e => Math.min(0.90, 0.45 + Math.log(e) * 0.1 + Math.random() * 0.1)),
  };
};

const mockReportRow = (): ClassificationReportRow => ({
  precision: Math.random() * 0.3 + 0.65, // 0.65 - 0.95
  recall: Math.random() * 0.3 + 0.65, // Sensitivity
  f1Score: Math.random() * 0.3 + 0.65,
  support: Math.floor(Math.random() * 100) + 50,
  specificity: Math.random() * 0.3 + 0.68, // Specificity typically also high
});

export const generateMockClassificationReport = (classNames: string[]): ClassificationReportData => {
  const classMetrics: { [className: string]: ClassificationReportRow } = {};
  classNames.forEach(name => {
    classMetrics[name] = mockReportRow();
  });

  return {
    accuracy: Math.random() * 0.2 + 0.75, // 0.75 - 0.95
    macroAvg: mockReportRow(),
    weightedAvg: mockReportRow(),
    classMetrics: classMetrics,
    aucpr: Math.random() * 0.25 + 0.70, // Mock AUC-PR
  };
};

export const generateMockConfusionMatrix = (classNames: string[]): ConfusionMatrixData => {
  const size = classNames.length;
  const matrix = Array(size).fill(null).map(() => Array(size).fill(0));
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      if (i === j) { // Diagonal
        matrix[i][j] = Math.random() * 0.3 + 0.6; // 0.6 - 0.9 (normalized)
      } else {
        matrix[i][j] = Math.random() * 0.1; // 0 - 0.1 (normalized off-diagonal)
      }
    }
  }
  // Ensure rows sum to roughly 1 for normalized view (optional, depends on interpretation)
  // For now, raw counts or mock proportions are fine.
  return { labels: classNames, matrix };
};

export const generateMockErrorAnalysis = (
  classNames: string[],
  sampleImages: SampleImage[] = []
): ErrorAnalysisItem[] => {
  if (classNames.length === 0) return [];
  
  const numErrorsToDisplay = Math.min(4, sampleImages.length, classNames.length > 1 ? 4 : 0);
  
  if (numErrorsToDisplay === 0 && classNames.length > 1 && sampleImages.length === 0) {
    return Array.from({ length: Math.min(4, classNames.length > 1 ? 4 : 0) }).map((_, i) => {
      const trueIdx = Math.floor(Math.random() * classNames.length);
      let predIdx = Math.floor(Math.random() * classNames.length);
      while (predIdx === trueIdx && classNames.length > 1) {
        predIdx = Math.floor(Math.random() * classNames.length);
      }
      if (classNames.length === 1) predIdx = trueIdx; // Cannot misclassify if only one class
      return {
        image: `https://via.placeholder.com/200x200.png/1F2937/9CA3AF?text=Erro+Img+${i+1}`,
        trueLabel: classNames[trueIdx],
        predLabel: classNames[predIdx],
      };
    });
  }
  
  if (numErrorsToDisplay === 0) return [];

  const errorItems: ErrorAnalysisItem[] = [];
  const usedImageIndices = new Set<number>();

  for (let i = 0; i < numErrorsToDisplay; i++) {
    let imageIndex: number;
    do {
      imageIndex = Math.floor(Math.random() * sampleImages.length);
    } while (usedImageIndices.has(imageIndex) && usedImageIndices.size < sampleImages.length);
    
    if (usedImageIndices.size >= sampleImages.length && i < numErrorsToDisplay) break;
    usedImageIndices.add(imageIndex);
    
    const selectedImage = sampleImages[imageIndex];
    const actualClassIndex = classNames.indexOf(selectedImage.className);
    const trueIdx = actualClassIndex !== -1 ? actualClassIndex : Math.floor(Math.random() * classNames.length);

    let predIdx = Math.floor(Math.random() * classNames.length);
    while (predIdx === trueIdx && classNames.length > 1) { 
      predIdx = Math.floor(Math.random() * classNames.length);
    }
    if (classNames.length === 1) predIdx = trueIdx;

    errorItems.push({
      image: selectedImage.imageDataUrl, 
      trueLabel: classNames[trueIdx],
      predLabel: classNames[predIdx],
    });
  }
  return errorItems;
};

export const generateMockClusterData = (numClustersAlgorithm: number, classNames: string[]): ClusterVisualizationData => {
  const numPoints = 100;
  const numTrueClasses = classNames.length;

  const generatePoints = (isForTrueClasses = false) => Array.from({ length: numPoints }).map((_, i) => {
    const trueClassIdx = i % numTrueClasses;
    const pointClusterId = isForTrueClasses ? trueClassIdx : Math.floor(Math.random() * numClustersAlgorithm);
    
    const base_x = (trueClassIdx % Math.ceil(Math.sqrt(numTrueClasses))) * 10;
    const base_y = Math.floor(trueClassIdx / Math.ceil(Math.sqrt(numTrueClasses))) * 10;

    return {
      id: `p-${i}`,
      x: base_x + (Math.random() * 8 - 4),
      y: base_y + (Math.random() * 8 - 4),
      cluster: pointClusterId,
      trueLabel: classNames[trueClassIdx],
    };
  });
  
  const truePoints = generatePoints(true);

  return {
    hierarchical: generatePoints(),
    kmeans: generatePoints(),
    trueClasses: truePoints,
    metrics: {
      hierarchicalARI: Math.random() * 0.5 + 0.3, 
      hierarchicalNMI: Math.random() * 0.5 + 0.3,
      kmeansARI: Math.random() * 0.5 + 0.4,       
      kmeansNMI: Math.random() * 0.5 + 0.4,
    },
  };
};

export const generateMockAugmentedEmbeddings = (numOriginalPoints: number, numAugmentationsPerPoint: number, classNames: string[]): ClusterPoint[] => {
  const augmentedPoints: ClusterPoint[] = [];
  const numTrueClasses = classNames.length;
  let pointIdCounter = 0;

  for (let i = 0; i < numOriginalPoints; i++) {
    const originalClassIdx = i % numTrueClasses;
    const originalClassName = classNames[originalClassIdx];

    const base_x = (originalClassIdx % Math.ceil(Math.sqrt(numTrueClasses))) * 15;
    const base_y = Math.floor(originalClassIdx / Math.ceil(Math.sqrt(numTrueClasses))) * 15;
    
    const originalPoint_x = base_x + (Math.random() * 5 - 2.5);
    const originalPoint_y = base_y + (Math.random() * 5 - 2.5);

    for (let j = 0; j < numAugmentationsPerPoint; j++) {
      augmentedPoints.push({
        id: `aug-p-${pointIdCounter++}`,
        x: originalPoint_x + (Math.random() * 4 - 2),
        y: originalPoint_y + (Math.random() * 4 - 2),
        cluster: originalClassIdx,
        trueLabel: originalClassName,
      });
    }
  }
  return augmentedPoints;
};

export const generateMockCAMImage = (originalImageSrc: string, camMethod?: SidebarConfig['camMethod']): Promise<string> => {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const img = new Image();
  
  return new Promise<string>((resolve) => {
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      if (ctx) {
        ctx.drawImage(img, 0, 0);
        
        let opacityBase = 0.6;
        let blurAmount = '1px';
        let primaryColor = 'rgba(255, 87, 51, OPACITY)'; // Bright orange-red
        let secondaryColor = 'rgba(255, 199, 0, OPACITY)'; // Yellowish

        if (camMethod?.includes('LIME')) {
            opacityBase = 0.5;
            blurAmount = '0px'; // LIME often shows sharper regions
            primaryColor = 'rgba(50, 205, 50, OPACITY)'; // Lime green
            secondaryColor = 'rgba(34, 139, 34, OPACITY)';
        } else if (camMethod?.includes('SHAP')) {
            opacityBase = 0.7;
            primaryColor = 'rgba(0, 100, 255, OPACITY)'; // Blue for positive SHAP
            secondaryColor = 'rgba(255, 0, 100, OPACITY)'; // Red for negative SHAP (simplified here)
        } else if (camMethod?.includes('Grad-CAM++')) {
            opacityBase = 0.65;
        }


        ctx.filter = `blur(${blurAmount})`;

        const grad1 = ctx.createRadialGradient(
            img.width * (0.4 + Math.random()*0.2), img.height * (0.4 + Math.random()*0.2), img.width * 0.05,
            img.width * (0.4 + Math.random()*0.2), img.height * (0.4 + Math.random()*0.2), img.width * (0.25 + Math.random()*0.15)
        );
        grad1.addColorStop(0, primaryColor.replace('OPACITY', opacityBase.toString()));
        grad1.addColorStop(0.5, secondaryColor.replace('OPACITY', (opacityBase*0.66).toString()));
        grad1.addColorStop(1, secondaryColor.replace('OPACITY', '0'));
        ctx.fillStyle = grad1;
        ctx.fillRect(0, 0, img.width, img.height);
        
        if (!camMethod?.includes('LIME')) { // LIME might be more focused, less secondary
            const grad2 = ctx.createRadialGradient(
                img.width * (0.65 + Math.random()*0.2), img.height * (0.6 + Math.random()*0.2), img.width * 0.02,
                img.width * (0.65 + Math.random()*0.2), img.height * (0.6 + Math.random()*0.2), img.width * (0.1 + Math.random()*0.1)
            );
            grad2.addColorStop(0, primaryColor.replace('OPACITY', (opacityBase*0.8).toString()));
            grad2.addColorStop(0.7, secondaryColor.replace('OPACITY', (opacityBase*0.33).toString()));
            grad2.addColorStop(1, secondaryColor.replace('OPACITY', '0'));
            ctx.fillStyle = grad2;
            ctx.fillRect(0,0, img.width, img.height);
        }
        ctx.filter = 'none';
      }
      resolve(canvas.toDataURL());
    };
    img.onerror = () => {
      console.error("Failed to load image for CAM generation:", originalImageSrc);
      resolve(`https://via.placeholder.com/${300}x${200}.png/1F2937/9CA3AF?text=CAM+Load+Error`);
    };
    img.src = originalImageSrc;
  });
};

export const generateMockROCCurveData = (classNames: string[]): ROCCurveData => {
  const points: ROCPoint[] = [];
  const numPoints = 15;
  let auc = 0;

  points.push({ fpr: 0, tpr: 0, recall: 0, threshold: 1 }); // recall is TPR for ROC

  let lastFPR = 0;
  let lastTPR = 0;

  for (let i = 1; i < numPoints; i++) {
    const fprIncrement = Math.random() * (1 / (numPoints -1));
    let tprIncrement = fprIncrement + (Math.random() * 0.2 - 0.05);
    
    let newFPR = Math.min(1, lastFPR + fprIncrement);
    let newTPR = Math.min(1, lastTPR + tprIncrement);

    if (i < numPoints / 2) {
        newTPR = Math.min(1, lastTPR + tprIncrement + Math.random() * 0.1);
    } else {
        newFPR = Math.min(1, lastFPR + fprIncrement + Math.random() * 0.05);
    }
    
    newTPR = Math.max(newTPR, newFPR * (0.7 + Math.random()*0.3) ); 
    newTPR = Math.min(1, newTPR);
    newFPR = Math.min(1, newFPR);

    if (newFPR < lastFPR) newFPR = lastFPR + 0.001;
    if (newTPR < lastTPR) newTPR = lastTPR;

    newFPR = Math.min(1, newFPR);
    newTPR = Math.min(1, newTPR);
    
    points.push({ 
        fpr: parseFloat(newFPR.toFixed(4)), 
        tpr: parseFloat(newTPR.toFixed(4)), 
        recall: parseFloat(newTPR.toFixed(4)), // recall is TPR for ROC
        threshold: parseFloat((1 - (i / numPoints)).toFixed(2)) 
    });
    
    auc += (newFPR - lastFPR) * (lastTPR + newTPR) / 2;

    lastFPR = newFPR;
    lastTPR = newTPR;
  }

  if (lastFPR < 1 || lastTPR < 1) {
     auc += (1 - lastFPR) * (lastTPR + 1) / 2;
     points.push({ fpr: 1, tpr: 1, recall: 1, threshold: 0 });
  }
  
  points.sort((a, b) => {
    if (a.fpr! !== b.fpr!) {
      return a.fpr! - b.fpr!;
    }
    return a.tpr! - b.tpr!;
  });

  const uniqueFprPoints: ROCPoint[] = [];
  const fprMap = new Map<number, {tprs: number[], thresholds: (number | undefined)[]}>();
  points.forEach(p => {
      if (!fprMap.has(p.fpr!)) {
          fprMap.set(p.fpr!, {tprs: [], thresholds: []});
      }
      fprMap.get(p.fpr!)!.tprs.push(p.tpr!);
      fprMap.get(p.fpr!)!.thresholds.push(p.threshold);
  });

  Array.from(fprMap.keys()).sort((a,b) => a-b).forEach(fprValue => {
      const data = fprMap.get(fprValue)!;
      const avgTpr = data.tprs.reduce((sum, val) => sum + val, 0) / data.tprs.length;
      const threshold = data.thresholds.length > 0 ? data.thresholds[0] : undefined; 
      uniqueFprPoints.push({fpr: fprValue, tpr: avgTpr, recall: avgTpr, threshold: threshold});
  });
  
  let finalAuc = 0;
  if (uniqueFprPoints.length > 1) {
    for(let i = 0; i < uniqueFprPoints.length - 1; i++) {
        const p1 = uniqueFprPoints[i];
        const p2 = uniqueFprPoints[i+1];
        finalAuc += (p2.fpr! - p1.fpr!) * (p1.tpr! + p2.tpr!) / 2;
    }
  }

  return {
    points: uniqueFprPoints,
    auc: parseFloat(Math.min(0.99, Math.max(0.65, finalAuc)).toFixed(3)),
    className: classNames.length === 2 ? 'Curva ROC Binária' : (classNames.length === 1 ? `Classe: ${classNames[0]}` : 'Curva ROC Média (Multiclasse)'),
    curveType: 'ROC',
  };
};

export const generateMockPRCurveData = (classNames: string[]): ROCCurveData => {
    const points: ROCPoint[] = [];
    const numPoints = 15;
    let aucpr = 0;

    // Start with high precision at low recall (often)
    points.push({ recall: 0, precision: Math.random() * 0.2 + 0.8, threshold: 1 }); 

    let lastRecall = 0;
    let lastPrecision = points[0].precision!;

    for (let i = 1; i < numPoints; i++) {
        const recallIncrement = Math.random() * (1 / (numPoints -1)) + 0.01; // Recall generally increases
        let precisionDecrement = Math.random() * 0.1; // Precision generally decreases
        
        let newRecall = Math.min(1, lastRecall + recallIncrement);
        let newPrecision = Math.max(0, lastPrecision - precisionDecrement);

        // Make it a bit more realistic (less jagged straight lines)
        if (Math.random() < 0.3) { // Occasionally, precision might not drop much or even slightly increase locally
            newPrecision = Math.max(0, lastPrecision - precisionDecrement * 0.5 + Math.random() * 0.05);
        }
        newPrecision = Math.min(1, newPrecision);

        // Ensure points are ordered by Recall
        if (newRecall < lastRecall) newRecall = lastRecall + 0.001;
        newRecall = Math.min(1, newRecall);
        
        points.push({ 
            recall: parseFloat(newRecall.toFixed(4)), 
            precision: parseFloat(newPrecision.toFixed(4)), 
            threshold: parseFloat((1 - (i / numPoints)).toFixed(2)) 
        });
        
        // Trapezoidal rule for AUC: (x2-x1) * (y1+y2)/2
        aucpr += (newRecall - lastRecall) * (lastPrecision + newPrecision) / 2;

        lastRecall = newRecall;
        lastPrecision = newPrecision;
    }

    // Ensure last point is (1, P_at_R=1)
    if (lastRecall < 1) {
        const finalPrecision = Math.max(0, lastPrecision - Math.random()*0.2); // Precision at Recall=1
        aucpr += (1 - lastRecall) * (lastPrecision + finalPrecision) / 2;
        points.push({ recall: 1, precision: finalPrecision, threshold: 0 });
    }
    
    points.sort((a, b) => {
        if (a.recall !== b.recall) {
        return a.recall - b.recall;
        }
        return b.precision! - a.precision!; // Higher precision first for same recall
    });

    // Ensure unique Recall points for line chart, averaging precision or taking max
    const uniqueRecallPoints: ROCPoint[] = [];
    const recallMap = new Map<number, {precisions: number[], thresholds: (number | undefined)[]}>();
    points.forEach(p => {
        if (!recallMap.has(p.recall)) {
            recallMap.set(p.recall, {precisions: [], thresholds: []});
        }
        recallMap.get(p.recall)!.precisions.push(p.precision!);
        recallMap.get(p.recall)!.thresholds.push(p.threshold);
    });

    Array.from(recallMap.keys()).sort((a,b) => a-b).forEach(recallValue => {
        const data = recallMap.get(recallValue)!;
        // For PR curves, often take the max precision for a given recall (if multiple thresholds yield same recall)
        const maxPrecision = Math.max(...data.precisions); 
        const threshold = data.thresholds.length > 0 ? data.thresholds[data.precisions.indexOf(maxPrecision)] : undefined; 
        uniqueRecallPoints.push({recall: recallValue, precision: maxPrecision, threshold: threshold});
    });

    let finalAucPr = 0;
    if (uniqueRecallPoints.length > 1) {
        for(let i = 0; i < uniqueRecallPoints.length - 1; i++) {
            const p1 = uniqueRecallPoints[i];
            const p2 = uniqueRecallPoints[i+1];
            finalAucPr += (p2.recall - p1.recall) * (p1.precision! + p2.precision!) / 2;
        }
    }
    
    return {
        points: uniqueRecallPoints,
        auc: parseFloat(Math.min(0.99, Math.max(0.60, finalAucPr)).toFixed(3)), // Clamp AUC-PR
        className: classNames.length === 2 ? 'Curva PR Binária' : (classNames.length === 1 ? `Classe: ${classNames[0]}`: 'Curva PR Média (Multiclasse)'),
        curveType: 'PR',
    };
};

export const generateMockUncertaintyScore = (): number => {
    return parseFloat((Math.random() * 0.3 + 0.05).toFixed(3)); // Range 0.05 - 0.35
};
