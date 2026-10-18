import importlib
import sys

print(f"Python {sys.version_info.major}.{sys.version_info.minor}")

for name in ["numpy", "pandas", "matplotlib", "seaborn", "sklearn"]:
    try:
        module = importlib.import_module(name)
        print(f"{name:<12} OK (versi {module.__version__})")
    except ImportError:
        print(f"{name:<12} BELUM TERPASANG")
