import pandas as pd

df = pd.read_csv('data/processed/dataset_orinoco_features.csv', parse_dates=['fecha']).set_index('fecha').sort_index()
n = len(df)

n_train = int(n * 0.60)
n_val   = int(n * 0.20)
n_test  = n - n_train - n_val

train_start = df.index[0]
train_end   = df.index[n_train - 1]
val_start   = df.index[n_train]
val_end     = df.index[n_train + n_val - 1]
test_start  = df.index[n_train + n_val]
test_end    = df.index[-1]

print(f"Total filas  : {n}")
print(f"Train  (60%) : {train_start.date()} -> {train_end.date()} | {n_train} dias")
print(f"Val    (20%) : {val_start.date()} -> {val_end.date()} | {n_val} dias")
print(f"Test   (20%) : {test_start.date()} -> {test_end.date()} | {n_test} dias")
print()
print(f'train_end: "{train_end.date()}"')
print(f'val_end:   "{val_end.date()}"')
