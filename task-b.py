# ============================================================
# import libraries
# ============================================================
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# O'zingizning binary (0/1) ma'lumotingizni tuzing (masalan: mijoz xarid qildimi yoki yo'qmi).

# ============================================================
# data
# ============================================================
# x (saytdagi vaqt, daqiqa): [0.5, 1.2, 2.0, 3.5, 4.2, 5.0]
# y (xarid holati): [0, 0, 0, 1, 1, 1]
z = sp.symbols("z")
x = np.array([0.5, 1.2, 2.0, 3.5, 4.2, 5.0])
y = np.array([0, 0, 0, 1, 0, 1])


# ============================================================
# 1) Sigmoid va uning hosilasini SymPy bilan tekshiring ("ayirma = 0" usuli)
# ============================================================
sigmoid = 1 / (1 + sp.exp(-z))
d_sigmoid = sp.diff(sigmoid, z)
print(f"d-sigmoid -> {sp.simplify(d_sigmoid)}")  # d-sigmoid -> 1/(4*cosh(z/2)**2)
print(sp.simplify(d_sigmoid - sigmoid * (1 - sigmoid)))  # 0


# ============================================================
# 2) Logistik regressiyani gradient descent bilan o'qiting
# ============================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


w, b, lr = 0.0, 0.0, 0.1

for step in range(2000):
    p = sigmoid(w * x + b)
    dw = np.mean((p - y) * x)
    db = np.mean(p - y)
    w -= lr * dw
    b -= lr * db


print(f"step 2000: w -> {w:.3f},  b -> {b:.3f}")
# step 2000: w -> 2.729,  b -> -7.167


# ============================================================
# 3) ROC egri chizig'ini chizing va AUC ni np.trapezoid bilan hisoblang
# ============================================================
pred = sigmoid(w * x + b)
tpr, fpr = [0.0], [0.0]
for t in np.sort(np.unique(pred))[::-1]:
    prediction = (pred >= t).astype(int)
    tpr.append(np.sum((prediction == 1) & (y == 1)) / np.sum(y == 1))
    fpr.append(np.sum((prediction == 1) & (y == 0)) / np.sum(y == 0))

auc = np.trapezoid(tpr, fpr)
print(f"AUC -> {auc:.3f}")
# AUC -> 0.875

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC egri chizig'i (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Tasodifiy taxmin (0.5)")
plt.fill_between(fpr, tpr, alpha=0.2, color="blue", label="AUC maydoni")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.title("ROC Egri Chizig'i")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")

plt.legend()

plt.grid(True)
plt.show()


# ============================================================
# 4) Natijani sklearn.metrics.roc_auc_score bilan solishtiring
# ============================================================
# hali bu mavzu o'tilmagan
