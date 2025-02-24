'use client';
import { useState } from 'react';
import ParameterControls from '@/app/components/ParameterControls';
import SignalVisualization from './components/SignalVisualization';
import { Parameters } from './components/utils';


export default function SignalPage() {
  // Initialize parameters state
  const [params, setParams] = useState<Parameters>({
    sinus_period: 0.005,
    amplitude: 1.0,
    phase: 0.0,
    duration: 20e-3,    
    impulse_period: 0.002
  });

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl mb-4 font-bold text-gray-800">
        Signal Visualization Dashboard
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Parameter Controls */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-4">Signal Parameters</h2>
          <ParameterControls params={params} setParams={setParams} />
        </div>

        {/* Visualization */}
        <div className="lg:col-span-2">
          <SignalVisualization
            endpoint="sample_sinus"
            params={params}
            title="Sampled Sinusoidal Signal"
          />
        </div>
      </div>
    </div>
  );
}