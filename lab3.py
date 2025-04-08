import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, t
x = np.array([5, 2, 12, 9, 15, 6, 25, 16])  
y = np.array([64, 87, 50, 71, 44, 56, 42, 60])  
print("It seems like the insurance premium might decrease with more driving experience.")
SSxx = np.sum((x - np.mean(x))**2)
SSyy = np.sum((y - np.mean(y))**2)
SSxy = np.sum((x - np.mean(x)) * (y - np.mean(y)))
print(f"SSxx = {SSxx:.2f}")
print(f"SSyy = {SSyy:.2f}")
print(f"SSxy = {SSxy:.2f}")
b = SSxy / SSxx
a = np.mean(y) - b * np.mean(x)
print(f"Regression line: y = {a:.2f} + {b:.2f}x")
print(f"a = {a:.2f} → Expected premium when experience is 0 years")
print(f"b = {b:.2f} → Change in premium per additional year of experience")
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, a + b * x, color='red', label='Line of Best Fit')
plt.xlabel('Driving Experience (years)')
plt.ylabel('Monthly Auto Insurance Premium')
plt.title('Driving Experience vs Insurance Premium')
plt.legend()
plt.show()
r = SSxy / np.sqrt(SSxx * SSyy)
r_squared = r**2
print(f"Correlation coefficient: r = {r:.2f}")
print(f"Coefficient of determination: r² = {r_squared:.2f}")
predicted_premium = a + b * 10
print(f"Predicted premium for 10 years of experience: {predicted_premium:.2f}")
y_pred = a + b * x
errors = y - y_pred
std_dev = np.sqrt(np.sum(errors**2) / (len(x) - 2))
print(f"Standard deviation of errors = {std_dev:.2f}")
n = len(x)
alpha = 0.10
t_critical = t.ppf(1 - alpha/2, n - 2)
SE_b = std_dev / np.sqrt(SSxx)
CI_lower = b - t_critical * SE_b
CI_upper = b + t_critical * SE_b
print(f"90% confidence interval for b: ({CI_lower:.2f}, {CI_upper:.2f})")
alpha = 0.05
t_statistic = b / SE_b
p_value = t.sf(abs(t_statistic), n - 2)
if p_value < alpha and b < 0:
    print(f"At 5% significance, B is negative. p-value = {p_value:.3f}")
else:
    print(f"At 5% significance, B is not negative. p-value = {p_value:.3f}")
if abs(r) > 0:
    print("Correlation coefficient is significantly different from zero.")
else:
    print("Correlation coefficient is not significantly different from zero.")
