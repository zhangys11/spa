import importlib.metadata
try:
    __version__ = importlib.metadata.version('spa-ds')
except importlib.metadata.PackageNotFoundError:
    __version__ = '2.0.2'