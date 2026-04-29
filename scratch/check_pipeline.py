import sys, logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
sys.path.insert(0, ".")
from src.data.pipeline import build_tensors

t = build_tensors()

print()
print("=== PIPELINE OK ===")
print(f"X_train : {t['X_train'].shape}")
print(f"X_val   : {t['X_val'].shape}")
print(f"X_test  : {t['X_test'].shape}")
print(f"Features: {len(t['feature_cols'])} columnas")
print(f"Target  : {t['target_col']}")
print(f"Fechas test: {t['test_dates'][0].date()} -> {t['test_dates'][-1].date()}")
