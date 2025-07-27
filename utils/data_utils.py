#!/usr/bin/env python3
"""
Data utility functions
"""

import numpy as np


def prepare_hnn_data(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> tuple:
    """Prepare data for HNN training with correct physics"""
    q = x.reshape(-1, 1)
    p = v.reshape(-1, 1)
    
    from config import SYSTEMS
    params = SYSTEMS['damped_oscillator']['parameters']
    k = params['k']
    m = params['m'] 
    c = params['c']
    
    dq_dt = v.reshape(-1, 1)
    dp_dt = -(k/m) * q - (c/m) * p
    
    print(f"🔧 HNN Data: Using physics-based derivatives (k={k}, m={m}, c={c})")
    print(f"   - dq_dt = velocity (direct)")
    print(f"   - dp_dt = -(k/m)*q - (c/m)*p = -k*q (since c=0)")
    
    return q, p, dq_dt, dp_dt 