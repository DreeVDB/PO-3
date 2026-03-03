import numpy as np

def sort_roots_canonical(roots: np.ndarray) -> np.ndarray:
    """Sorteer wortels op (reel deel, imaginair deel) voor consistente volgorde."""
    roots = np.asarray(roots, dtype=np.complex128).reshape(-1)
    order = np.lexsort((np.imag(roots), np.real(roots)))
    return roots[order]

### NN train op 10 floats, NR werkt op complexe tensors => encoderen en decoderen

def encode_roots_ri(roots: np.ndarray) -> np.ndarray:
    """Codeer 5 complexe wortels naar 10 floats: [re0..re4, im0..im4]."""
    roots = sort_roots_canonical(roots)
    out = np.zeros(10, dtype=np.float64)
    out[:5] = np.real(roots)
    out[5:] = np.imag(roots)
    return out


def decode_roots_ri(vec10: np.ndarray) -> np.ndarray:
    """Decodeer 10 floats [re0..re4, im0..im4] naar 5 complexe wortels."""
    v = np.asarray(vec10, dtype=np.float64).reshape(-1)
    if v.size != 10:
        raise ValueError("Expected 10 values for root representation.")
    return v[:5] + 1j * v[5:]