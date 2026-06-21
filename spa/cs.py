import cs1
assert( '__version__' in cs1.__dict__ and cs1.__version__ > '0.1.1')

# Compatibility patch for older cs1 using deprecated np.complex
import numpy as _np
if not hasattr(_np, 'complex'):
    _np.complex = complex

from cs1.cs import *
from cs1.basis.common import *
from cs1.basis.adaptive import *


def Simulate_ECG(duration=2.0, heart_rate=60, sampling_rate=512):
    """Generate a realistic simulated ECG signal for compressed sensing demo.

    Parameters
    ----------
    duration : float
        Signal duration in seconds.
    heart_rate : float
        Heart rate in beats per minute (bpm).
    sampling_rate : int
        Sampling rate in Hz.
    """
    n_pts_cycle = int(sampling_rate * 60 / heart_rate)  # points per cardiac cycle
    t_cycle = np.linspace(0, 1, n_pts_cycle)

    # P-Q-R-S-T cardiac cycle template
    ecg_cycle = (
        0.15 * np.exp(-((t_cycle - 0.15) / 0.03) ** 2)    # P wave
        - 0.1  * np.exp(-((t_cycle - 0.35) / 0.015) ** 2)  # Q wave
        + 1.0  * np.exp(-((t_cycle - 0.38) / 0.02) ** 2)   # R wave
        - 0.35 * np.exp(-((t_cycle - 0.42) / 0.02) ** 2)   # S wave
        + 0.25 * np.exp(-((t_cycle - 0.62) / 0.04) ** 2)   # T wave
    )
    ecg_cycle = ecg_cycle / np.max(np.abs(ecg_cycle))

    # Replicate to fill the requested duration
    n_cycles = max(1, int(heart_rate * duration / 60))
    y = np.tile(ecg_cycle, n_cycles)
    total_pts = int(duration * sampling_rate)
    y = y[:total_pts]
    t = np.linspace(0, duration, len(y))

    plt.figure(figsize=(12, 4))
    plt.plot(t, y)
    plt.title(f'Simulated ECG Signal (HR={heart_rate} bpm, {duration}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (normalized)')
    plt.tight_layout()
    plt.show()
    return y


def Analyze_Sparsity(x, PSIs, topk=10):
    """Analyze signal sparsity across multiple transform bases.

    Parameters
    ----------
    x : ndarray
        1D signal.
    PSIs : dict
        Dictionary mapping base name to transform matrix.
    topk : int
        Number of top coefficients to show.

    Returns
    -------
    results : dict
        Sparsity ratio and top coefficients for each basis.
    """
    from IPython.display import display, HTML
    results = {}
    rows = []
    for name, psi in PSIs.items():
        coeffs = np.abs(psi @ x)
        n = len(coeffs)
        k = topk
        top_indices = np.argsort(coeffs)[-k:][::-1]
        top_coeffs = coeffs[top_indices]
        sum_top = np.sum(top_coeffs)
        sum_all = np.sum(coeffs)
        ratio = round(sum_top / sum_all, 3)
        results[name] = {'ratio': ratio, 'indices': top_indices, 'coeffs': top_coeffs}
        rows.append(f'<tr><td>{name}</td><td>{ratio}</td><td>{k}/{n}</td>'
                    f'<td>{list(np.round(top_coeffs, 2))}</td></tr>')

    html = (f'<table><tr><th>Basis</th><th>Top-{topk}/Total energy</th>'
            f'<th>K/N</th><th>Top-{topk} coeffs</th></tr>'
            + ''.join(rows) + '</table>')
    display(HTML(html))
    return results