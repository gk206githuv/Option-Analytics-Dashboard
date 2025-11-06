import numpy as np
from scipy.stats import norm
from typing import Literal

"""
Black-Scholes-Merton model implementation for European option
pricing and calculation of the Greeks.
"""

# Type hint for option
OptionType = Literal["call", "put"]

def _calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Internal helper function to calculate d1 and d2."""
    if T <= 0 or sigma <= 0:
        # Handle division by zero or log(0) issues at expiry
        if S > K:
            return (np.inf, np.inf)  # Deep in the money
        else:
            return (-np.inf, -np.inf) # Out of the money
            
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType = "call") -> float:
    """
    Calculates the price of a European option using the BSM formula.
    """
    if T <= 0:
        # Return intrinsic value for expired option
        if option_type == "call":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)
    
    if option_type == "call":
        price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "put":
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")
        
    return price

def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType = "call") -> float:
    """Calculates the Delta of an option."""
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1, _ = _calculate_d1_d2(S, K, T, r, sigma)
    
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculates the Gamma of an option."""
    if T <= 0 or sigma <= 0:
        return 0.0
        
    d1, _ = _calculate_d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculates the Vega of an option."""
    if T <= 0 or sigma <= 0:
        return 0.0
        
    d1, _ = _calculate_d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType = "call") -> float:
    """Calculates the Theta of an option (per day)."""
    if T <= 0:
        return 0.0

    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)
    
    term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    
    if option_type == "call":
        term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
        return (term1 - term2) / 365.0  # Annualized to daily
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        return (term1 + term2) / 365.0  # Annualized to daily

def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType = "call") -> float:
    """Calculates the Rho of an option (per 1% rate change)."""
    if T <= 0:
        return 0.0

    _, d2 = _calculate_d1_d2(S, K, T, r, sigma)
    
    if option_type == "call":
        return (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100.0  # Per 1% change
    else:
        return (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100.0 # Per 1% change