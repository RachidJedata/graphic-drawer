from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- Add this import
import numpy as np

app = FastAPI()

# Add CORS middleware  <-- This is the critical part
# More secure local development configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js default
        "http://127.0.0.1:3000"
    ],
    allow_methods=["GET"],
)

def rect(x: np.ndarray) -> np.ndarray:
    """Vectorized rectangular function"""
    return np.where(np.abs(x) <= 0.5, 1, 0).astype(int)

@app.get("/rect")
async def get_rect(x: float):
    try:
        if x == 0:
            raise ValueError("x cannot be zero")
            
        # Signal parameters
        fs = 250e3  # Sampling frequency (250 kHz)
        T = 1e-3    # Time duration (1 ms)
        
        # Time array (-5ms to 5ms with dt=4μs)
        t = np.arange(-5e-3, 5e-3, 1/fs)
        
        # Frequency array (for potential FFT usage)
        n = len(t)
        freq = np.fft.fftshift(np.fft.fftfreq(n, d=1/fs))
        
        # Generate the rectangular waveform
        scaled_time = t / (x * T)
        y = rect(scaled_time).tolist()

        return {
            "parameters": {
                "input_x": x,
                "sampling_frequency": fs,
                "duration": T,
                "num_samples": len(y)
            },
            "time": t.tolist(),
            "frequency": freq.tolist(),
            "signal": y
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    
def generate_sinusoidal(period: float, amplitude: float = 1.0, phase: float = 0.0) -> dict:
    """Generate sinusoidal signal with specified parameters"""
    if period <= 0:
        raise ValueError("Period must be positive")
        
    fs = 250e3  # Sampling frequency (250 kHz)
    duration = 10e-3  # Total duration (10 ms)
    
    # Time array from -5ms to 5ms
    t = np.arange(-duration/2, duration/2, 1/fs)
    
    # Frequency calculation
    frequency = 1 / period
    
    # Generate signal
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    # Frequency domain
    n = len(t)
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1/fs))
    
    return {
        "time": t.tolist(),
        "frequency": freq.tolist(),
        "signal": signal.tolist(),
        "parameters": {
            "period": period,
            "frequency": frequency,
            "amplitude": amplitude,
            "phase": phase,
            "sampling_rate": fs,
            "duration": duration,
            "num_samples": n
        }
    }

@app.get("/sinus")
async def get_sinus(
    period: float,
    amplitude: float = 1.0,
    phase: float = 0.0
):
    try:
        data = generate_sinusoidal(period, amplitude, phase)
        return data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")
    