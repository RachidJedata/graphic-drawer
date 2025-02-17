'use client'

import { useState } from "react";
import SignalVisualization from "../components/SignalVisualization";

interface Parameters {
    [key: string]: number;
}

const getStep = (value: number): number => {
    if (value >= 1) return 1;
    if (value >= 0.1) return 0.1;
    if (value >= 0.01) return 0.01;
    return 0.001;
};

const ParameterControls = ({
    params,
    onParamsChange
}: {
    params: Parameters;
    onParamsChange: (newParams: Parameters) => void;
}) => {
    const handleAdjust = (param: string, value: number) => {
        const updated = {
            ...params,
            [param]: Math.max(getStep(params[param]), params[param] + value)
        };
        onParamsChange(updated);
    };

    const formatValue = (value: number): string => {
        if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
        if (value <= 0.001) return `${(value * 1000).toFixed(0)}ms`;
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
                        >
                            -
                        </button>
                        <span className="w-20 text-center">
                            {formatValue(value)}
                        </span>
                        <button
                            onClick={() => handleAdjust(param, getStep(value))}
                            className="px-2 py-1 bg-gray-200 rounded hover:bg-gray-300"
                        >
                            +
                        </button>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default function Home() {
    // const [params, setParams] = useState<Parameters>({ period: 0.005, amplitude: 2.5 });
    const [params, setParams] = useState<Parameters>({ period:0.005,amplitude:4 });


    return (
        <main className="min-h-screen p-8">
            <div className="w-full flex">
                <div className="parameter-box">

                </div>
                <ParameterControls
                    params={params}
                    onParamsChange={setParams}
                />
            </div>
            <SignalVisualization
                endpoint="sinus"
                params={params}
                title="Sinusoidal Wave"
            />
        </main>
    );
}