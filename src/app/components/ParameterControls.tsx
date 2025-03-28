import { JSX } from 'react';
import { Parameters } from './utils';

interface ParameterControlsProps {
  params: Parameters;
  setParams: React.Dispatch<React.SetStateAction<Parameters>>;
}

export default function ParameterControls({ params, setParams }: ParameterControlsProps): JSX.Element {
  const handleAdjust = (param: string, value: number) => {
    // Update parent's state instead of a local one
    setParams(prev => ({
      ...prev,
      [param]: Math.max(0, prev[param] + value),  // Prevent negative values
    }));
  };

  const formatValue = (value: number): string => {
    if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
    if (value <= 0.001) return `${(value * 1000).toFixed(0)}m`;
    return value.toFixed(3);
  };

  return (
    <div className="space-y-4 mb-4">
      {Object.entries(params).map(([param, value]) => (
        <div key={param} className="flex items-center gap-2">
          <label className="block w-24">{param}:</label>
          <div className="flex items-center gap-1">
            <button
              onClick={() => handleAdjust(param, -getStep(value))}
              className="px-2 py-1 bg-gray-200 rounded hover:bg-gray-300"
              aria-label={`Decrease ${param}`}
            >
              -
            </button>
            <span className="w-20 text-center">{formatValue(value)}</span>
            <button
              onClick={() => handleAdjust(param, getStep(value))}
              className="px-2 py-1 bg-gray-200 rounded hover:bg-gray-300"
              aria-label={`Increase ${param}`}
            >
              +
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

// Helper function to determine step size based on current value
const getStep = (value: number): number => {
  if (value >= 1) return 1;
  if (value >= 0.1) return 0.1;
  if (value >= 0.01) return 0.01;
  return 0.001;
};
