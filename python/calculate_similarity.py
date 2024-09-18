import numpy as np
from scipy.fft import fft, ifft

from tic import tic


@tic
def fast_dft(query_descriptor, candidate_descriptor):
    """
    Perform FFT-based cross-correlation between query and candidate descriptors.

    Args:
        query_descriptor (numpy.ndarray): Query descriptor of shape (N, M).
        candidate_descriptor (numpy.ndarray): Candidate descriptor of shape (N, M).

    Returns:
        tuple:
            correlation_map (numpy.ndarray): Correlation map of shape (N,).
            max_correlation (float): Maximum correlation value.
    """
    # Perform FFT along the theta axis for both descriptors
    query_fft = fft(query_descriptor, axis=1)
    candidate_fft = fft(candidate_descriptor, axis=1)

    # Compute the cross-correlation in the frequency domain
    correlation_map_2d = ifft(query_fft * np.conj(candidate_fft), axis=1)

    # Sum the correlation map along the second axis to get a 1D correlation map
    correlation_map = np.sum(correlation_map_2d.real, axis=1)

    # Find the maximum correlation value
    max_correlation = np.max(correlation_map)

    return correlation_map, max_correlation
