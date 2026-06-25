'''
Device selection for the (PyTorch-based) models in spa.

Priority: CUDA -> MPS (Apple Silicon) -> CPU.

Usage
-----
    from spa.device import resolve_device, resolve_device_str
    device = resolve_device()          # torch.device, auto best accelerator
    device = resolve_device(cuda)      # honour a user flag/string, see below

The ``prefer`` / ``cuda`` argument accepted everywhere:
    None / True / 'auto' / 'cuda' -> best available (cuda, else mps, else cpu)
    False / 'cpu'                 -> force CPU
    'mps'                         -> MPS if available, else cuda, else cpu
    'cuda:1', torch.device(...)   -> used as given
'''


def _mps_available():
    '''True only on a working Apple-Silicon MPS backend.'''
    try:
        import torch
        backend = getattr(torch.backends, 'mps', None)
        return bool(backend) and torch.backends.mps.is_available()
    except Exception:
        return False


def resolve_device(prefer=None):
    '''
    Return a ``torch.device`` following the priority CUDA -> MPS -> CPU.

    See the module docstring for the accepted values of ``prefer``.
    '''
    import torch

    # explicit CPU
    if prefer is False or prefer == 'cpu':
        return torch.device('cpu')

    # already a device object
    if isinstance(prefer, torch.device):
        return prefer

    # explicit MPS request (fall through to the priority chain if unavailable)
    if prefer == 'mps' and _mps_available():
        return torch.device('mps')

    # explicit custom string such as 'cuda:1'
    if isinstance(prefer, str) and prefer not in ('auto', 'cuda', 'mps'):
        return torch.device(prefer)

    # auto / True / 'cuda' / 'mps'(unavailable): CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    if _mps_available():
        return torch.device('mps')
    return torch.device('cpu')


def resolve_device_str(prefer=None):
    '''Same priority as :func:`resolve_device` but returns the string form.

    Useful for libraries (e.g. ctgan) whose ``cuda`` argument accepts a device
    string ('cuda' / 'mps' / 'cpu').
    '''
    return str(resolve_device(prefer))
