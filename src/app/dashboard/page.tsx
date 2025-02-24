'use client'

import { useState } from "react";
import SignalVisualization from "../components/SignalVisualization";
import { Parameters } from "../components/utils";
import ParameterControls from "../components/ParameterControls";

export default function Home() {
    const [params, setParams] = useState<Parameters>({ period: 0.005, amplitude: 2.5, duration: 10e-3 });
    // const [params, setParams] = useState<Parameters>({ Te:0.0025 });


    return (
        <main className="min-h-screen p-8">
            <div className="w-full flex">
                <div className="parameter-box">

                </div>
                <ParameterControls
                    params={params}
                    setParams={setParams}
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