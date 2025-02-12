import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('UFC Fight Statistics (July 2016 - Nov 2024).csv')
df.columns = df.columns.str.strip()

df_reg = df[['Total Strike Landed F1R1', 'Total Strike Landed F2R1']].dropna().copy()
df_reg = df_reg.sort_values(by='Total Strike Landed F1R1')

X = df_reg["Total Strike Landed F1R1"]
y = df_reg["Total Strike Landed F2R1"]

n = len(X)
sum_x = np.sum(X)
sum_y = np.sum(y)
sum_xy = np.sum(X * y)
sum_x2 = np.sum(X ** 2)

m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
b = (sum_y - m * sum_x) / n

y_pred = m * X + b

plt.figure(figsize=(12, 6))
sns.lineplot(x=X, y=y, marker="o", color="blue", label="Actual Values")
sns.lineplot(x=X, y=y_pred, color="red", label="Regression Line")

plt.xlabel("Total Strike Landed F1R1")
plt.ylabel("Total Strike Landed F2R1")
plt.title("Linear Regression: F1R1 vs. F2R1 Strikes Landed")
plt.legend()
plt.grid(True)
plt.show()
print(f"Linear Regression Equation: y = {m:.4f}x + {b:.4f}")
