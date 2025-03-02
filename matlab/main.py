from fastapi import FastAPI, Query,HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore # <-- Add this import
import numpy as np # type: ignore
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

# --------------------------
# Reusable Helper Functions
# --------------------------
def generate_time_array(duration: float, Te: float) -> np.ndarray:
    """Generate centered time array [-duration/2, duration/2) with step Te"""
    return np.arange(-duration/2, duration/2, Te)

def generate_frequency_axis(t: np.ndarray, Te: float) -> np.ndarray:
    """Generate centered frequency axis for FFT"""
    n = len(t)
    return np.fft.fftshift(np.fft.fftfreq(n, d=Te))

def generate_comb_signal(duration: float, period: float, Te: float) -> dict:
    """
    Generates a Dirac comb signal with impulses at specified intervals.
    
    Parameters:
        duration (float): Total time window (centered around 0)
        period (float): Spacing between impulses (must be > 0)
        Te (float): Time resolution/sampling interval (must be > 0)
    
    Returns:
        dict: {"time": list_of_timestamps, "signal": list_of_0s_and_1s}
    """
    # Generate time axis from -duration/2 to duration/2 (centered)
    time_array = np.arange(-duration/2, duration/2, Te)
    time = time_array.tolist()
    
    # Initialize signal with zeros (NumPy array for performance)
    signal = np.zeros_like(time_array, dtype=int)
    
    if len(time_array) == 0:  # Handle empty time array edge case
        return {"time": time, "signal": signal.tolist()}
    
    # Calculate first/last impulse indices within the time range
    t_start, t_end = -duration/2, duration/2 - 1e-9  # Boundary adjustment
    n_min = math.ceil(t_start / period)
    n_max = math.floor(t_end / period)
    
    # Place impulses at calculated positions
    for n in range(n_min, n_max + 1):
        t_impulse = n * period
        index = int(np.round((t_impulse - time_array[0]) / Te))
        if 0 <= index < len(signal):
            signal[index] = 1
    
    return {"time": time, "signal": signal.tolist()}
             
def validate_positive(**params):
    """Validate parameters are positive"""
    for name, value in params.items():
        if value <= 0:
            raise HTTPException(400, f"{name} must be positive")

def rect(x: np.ndarray) -> np.ndarray:
    """Vectorized rectangular function"""
    return np.where(np.abs(x) <= 0.5, 1, 0).astype(int)


# --------------------------
# Signal Generation Endpoints
# --------------------------

@app.get("/rect")
async def get_rect(x: float,duration:float=10e-3):
    try:
        if x == 0:
            raise ValueError("x cannot be zero")
            
        # Signal parameters
        fs, T_pulse = 250e3, 1e-3
       
        
        # Time array 
        t = generate_time_array(duration, 1/fs)        
        scaled_time = t / (x * T_pulse)                

        return {
            "time": t.tolist(),            
            "signal": rect(scaled_time).tolist(),
            "parameters": {
                "largeur": x,
                "sampling_frequency": fs,
                "pulse_width": T_pulse
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    

@app.get("/sinus")
async def get_sinus(
    period: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
    duration:float = 10e-3
):
    Te=1/250e3
    validate_positive(period=period, duration=duration, Te=Te)
    t = generate_time_array(duration, Te)
    freq = 1/period
    signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return {
        "time": t.tolist(),
        "signal": signal.tolist(),
        "parameters": {
            "frequency": freq,
            "amplitude": amplitude,
            "phase": phase,
            "duration": duration,
            "sampling_interval": Te
        }
    }

    
@app.get("/impulse")
async def get_impulse(
    Te: float = 0.001,
    duration: float = 1.0
):
    validate_positive(Te=Te, duration=duration)
    t = generate_time_array(duration, Te)
    
    signal = np.zeros(len(t), dtype=int)
    if len(signal) > 0:
        signal[len(signal)//2] = 1
    
    return {
        "time": t.tolist(),
        "signal": signal.tolist(),
        "parameters": {"Te": Te, "duration": duration}
    }

@app.get("/dirac_comb")
async def get_dirac_comb(
    Te: float = Query(0.001, gt=0, description="Sampling interval"),
    duration: float = Query(1.0, gt=0, description="Total time duration"),
    t_impulse: float = Query(0.1, gt=0, description="Spacing between impulses")
):
        
    signalTime = generate_comb_signal(duration=duration,period=t_impulse,Te=Te)

    parameters = {
        "Te": Te,
        "duration": duration,
        "t_impulse": t_impulse,
        "impulse_count": len(signalTime["signal"])
    }

    return {
        "time": signalTime["time"],
        "signal": signalTime["signal"],
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
    # Validate inputs
    validate_positive(
        sinus_period=sinus_period,
        Te=Te,
        impulse_period=impulse_period,
        duration=duration
    )

      # Generate comb signal with corrected parameter name
    comb_data = generate_comb_signal(
        duration=duration,
        period=impulse_period,  # Key fix: using 'period' parameter
        Te=Te
    )    
    # Create centered time array
    t = comb_data["time"]
    comb_signal= comb_data["signal"]
    # Generate sinusoid
    frequency = 1 / sinus_period
    sinus_signal = amplitude * np.sin(2 * np.pi * frequency * np.array(t) + phase)
       
    # Explicit elementwise multiplication using list comprehension
    sampled_signal = [c * s for c, s in zip(comb_signal, sinus_signal.tolist())]

    parameters = {
        "sinus_frequency": frequency,
        "sampling_frequency": 1/impulse_period,
        "nyquist_frequency": 1/(2*impulse_period),        
    }

    return {
        "time": t,        
        "signal": sampled_signal,   
        "parameters": parameters
    }