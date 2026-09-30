# Foundations of Machine Learning: Linear & Logistic Regression from Scratch

This repository explores the mathematical underpinnings and from-scratch implementations of fundamental machine learning algorithms: **Linear Regression** and **Logistic Regression**. Rather than relying on black-box estimators, the models are built using analytical differentiation, iterative optimization (Gradient Descent), and manual metric evaluations.

---

## 📌 Project Overview

### 1. Task A: Linear Regression (Analytical Solution vs. Gradient Descent)
* **Dataset:** Number of rooms ($x$) vs. Monthly power consumption ($y$).
* **MSE Cost Function:**
  $$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left(w x^{(i)} + b - y^{(i)}\right)^2$$
* **Symbolic Derivation (SymPy):**
  * Evaluated partial derivatives analytically: $\frac{\partial J}{\partial w}$ and $\frac{\partial J}{\partial b}$.
  * Solved the system of equations algebraically using `sp.solve` to determine the exact global minimum: **$w = 2.7$, $b = -0.1$**.
* **Gradient Descent Optimization:**
  * Implemented vectorized parameter updates using pure NumPy.
  * Experimented with multiple learning rates ($\alpha = 0.02, 0.03, 0.04$) to analyze convergence rates toward the analytical minimum.
* **Loss Landscape Visualization:**
  * Plotted the parameter space using a 2D contour map (`matplotlib.pyplot.contour`) to visualize convergence paths toward the global minimum.

---

### 2. Task B: Logistic Regression & From-Scratch ROC-AUC
* **Dataset:** Time spent on site in minutes ($x$) vs. Binary purchase decision ($y \in \{0, 1\}$).
* **Sigmoid Derivative Identity:**
  * Analytically verified the algebraic property using SymPy:
    $$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$
* **Model Training:**
  * Optimized parameters ($w \approx 2.73, b \approx -7.17$) over 2,000 iterations using batch Gradient Descent on the Binary Cross-Entropy loss gradient.
* **ROC Curve & AUC (From Scratch):**
  * Calculated True Positive Rate (TPR) and False Positive Rate (FPR) across all unique decision thresholds manually:
    $$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
  * Evaluated the Area Under the Curve (AUC) using numerical integration via the trapezoidal rule (`np.trapezoid`): **$\text{AUC} \approx 0.88$**.

---

## 📊 Visualizations

| Loss Landscape (Contour Map) | ROC Curve (AUC = 0.88) |
| :---: | :---: |
| ![Contour Map](contour.png) | ![ROC Curve](roc_curve.png) |

---

## 🛠 Tech Stack
* **Python 3**
* **NumPy** — Vectorized linear algebra, numerical array operations, and trapezoidal integration.
* **SymPy** — Symbolic mathematics, partial derivatives, and algebraic equation solving.
* **Matplotlib** — Contour visualizations and ROC performance curves.

---

## 🚀 Getting Started

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/ml-engineer04/mini-project-2month.git](https://github.com/ml-engineer04/mini-project-2month.git)
cd mini-project-2month
pip install numpy sympy matplotlib
