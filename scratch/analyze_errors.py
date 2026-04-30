import pandas as pd
import numpy as np

meses = ['', 'Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']

# Cargar ambos experimentos
df_mse   = pd.read_csv('results/experiments/lstm_residual_lb90/predictions_test.csv', parse_dates=['fecha'])
df_huber = pd.read_csv('results/experiments/lstm_residual_lb90_huber/predictions_test.csv', parse_dates=['fecha'])

for df, label in [(df_mse, 'lb90_MSE'), (df_huber, 'lb90_Huber')]:
    df['error']     = df['y_pred'] - df['y_true']
    df['abs_error'] = df['error'].abs()
    df['month']     = df['fecha'].dt.month

print(f"{'Mes':<5} {'MSE MAE':>10} {'Huber MAE':>12} {'Delta MAE':>12} {'MSE bias':>10} {'Huber bias':>12}")
print("-" * 65)
for m in range(1, 13):
    g_mse   = df_mse[df_mse['month'] == m]
    g_huber = df_huber[df_huber['month'] == m]
    mae_mse   = g_mse['abs_error'].mean() * 100
    mae_huber = g_huber['abs_error'].mean() * 100
    bias_mse   = g_mse['error'].mean() * 100
    bias_huber = g_huber['error'].mean() * 100
    delta = mae_huber - mae_mse
    arrow = "v" if delta < -1 else ("^" if delta > 1 else "=")
    print(f"{meses[m]:<5} {mae_mse:>9.1f}cm {mae_huber:>10.1f}cm  {arrow}{abs(delta):>7.1f}cm {bias_mse:>9.1f}cm {bias_huber:>10.1f}cm")

print()
print("GLOBAL:")
mae_mse_g   = df_mse['abs_error'].mean() * 100
mae_huber_g = df_huber['abs_error'].mean() * 100
print(f"  lb90_MSE  : MAE={mae_mse_g:.1f}cm")
print(f"  lb90_Huber: MAE={mae_huber_g:.1f}cm  delta={mae_huber_g-mae_mse_g:+.1f}cm")
