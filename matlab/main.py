from fastapi import FastAPI, Query,HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- Add this import
import numpy as np
import math


app = FastAPI()

# Add CORS middleware  <-- This is the critical part
# More secure local development configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    
def generate_sinusoidal(period: float, amplitude: float = 1.0, phase: float = 0.0,duration: float = 10e-3) -> dict:
    """Generate sinusoidal signal with specified parameters"""
    if period <= 0:
        raise ValueError("Period must be positive")
        
    fs = 250e3  # Sampling frequency (250 kHz)    
    
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
    phase: float = 0.0,
    duration:float = 10e-3
):
    try:
        data = generate_sinusoidal(period, amplitude, phase,duration)
        return data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")
    
@app.get("/impulse")
async def get_impulse(
    Te: float = 0.001,
    duration: float = 1.0
):
    # Create a time array centered around 0
    time = np.arange(-duration/2, duration/2, Te).tolist()
    
    # Create an impulse signal: 1 at t=0 (middle index), 0 elsewhere
    signal = [0] * len(time)
    if signal:
        middle_index = len(time) // 2  # Find the middle index
        signal[middle_index] = 1  # Set the impulse at the center

    parameters = {"Te": Te, "duration": duration}

    return {
        "time": time,
        "signal": signal,
        "parameters": parameters
    }

@app.get("/dirac_comb")
async def get_dirac_comb(
    Te: float = Query(0.001, gt=0, description="Sampling interval"),
    duration: float = Query(1.0, gt=0, description="Total time duration"),
    period: float = Query(0.1, gt=0, description="Spacing between impulses")
):
    # Generate time array centered around 0
    time_array = np.arange(-duration/2, duration/2, Te)
    time = time_array.tolist()
    
    # Initialize signal with zeros
    signal = [0] * len(time)
    
    # Calculate impulse positions
    n_min = math.ceil((-duration/2) / period)
    n_max = math.floor((duration/2 - 1e-9) / period)  # Avoid boundary inclusion
    t_impulses = [n * period for n in range(n_min, n_max + 1)]
    
    # Set impulses at calculated positions
    for t in t_impulses:
        index = int(np.round((t + duration/2) / Te))
        if 0 <= index < len(signal):
            signal[index] = 1

    parameters = {
        "Te": Te,
        "duration": duration,
        "period": period,
        "impulse_count": len(t_impulses)
    }

    return {
        "time": time,
        "signal": signal,
        "parameters": parameters
    }


@app.get("/sample_sinus")
async def get_sample_sinus(
    sinus_period: float = 0.005,
    amplitude: float = 1.0,
    phase: float = 0.0,
    duration: float = 20e-3,  # Increased duration for better visibility
    Te: float = 1/250e3,
    impulse_period: float = 0.002  # Adjusted for better demonstration
):
    """
    Generate a sampled sinusoidal signal using Dirac comb sampling
    """
    if any(param <= 0 for param in [sinus_period, Te, impulse_period, duration]):
        raise HTTPException(status_code=400, detail="All parameters must be positive")

    # Create centered time array
    t = np.arange(-duration/2, duration/2, Te)
    
    # Generate sinusoid
    frequency = 1 / sinus_period
    sinus_signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    # Generate Dirac comb (peigne de Dirac)
    comb_signal = np.zeros_like(t)
    n_min = math.ceil((-duration/2) / impulse_period)
    n_max = math.floor((duration/2 - 1e-9) / impulse_period)
    
    # Find exact impulse positions
    for n in range(n_min, n_max + 1):
        t_imp = n * impulse_period
        # Direct index calculation for perfect alignment
        index = int(round((t_imp - t[0]) / Te))
        if 0 <= index < len(comb_signal):
            comb_signal[index] = 1
    
    # Sample the sinusoid by multiplying with the comb
    sampled_signal = sinus_signal * comb_signal

    parameters = {
        "sinus_frequency": frequency,
        "sampling_frequency": 1/impulse_period,
        "nyquist_frequency": 1/(2*impulse_period),
        "num_impulses": comb_signal.sum()
    }

    return {
        "time": t.tolist(),        
        "signal": sampled_signal.tolist(),        
        "parameters": parameters
    }
