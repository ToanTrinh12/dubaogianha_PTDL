import sys
print("Python:", sys.version)
try:
    import fastapi, pandas, numpy, sklearn
    print("fastapi:", fastapi.__version__)
    print("pandas:", pandas.__version__)
    print("numpy:", numpy.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("Dependencies: OK")
except Exception as e:
    print("Dependency error:", repr(e))
    raise
