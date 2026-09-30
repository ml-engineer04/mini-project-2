# Foundations of Machine Learning: Linear & Logistic Regression from Scratch

Ushbu repozitoriyda klassik mashina o‘rganishi algoritmlari — **Chiziqli regressiya (Linear Regression)** va **Logistik regressiya (Logistic Regression)** tayyor yuqori darajadagi kutubxonalardan foydalanmasdan, matematik formulalar va noldan yozilgan algoritmlar orqali tahlil qilingan.

---

## 📌 Loyiha tarkibi

### 1. Task A: Chiziqli regressiya (Analitik yechim vs Gradient Descent)
* **Ma'lumotlar:** Xonalar soni ($x$) va oylik elektr energiyasi iste'moli ($y$).
* **MSE xatolik funksiyasi:** 
  $$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left(w x^{(i)} + b - y^{(i)}\right)^2$$
* **SymPy orqali analitik tahlil:**
  * Qisman hosilalar hisoblandi: $\frac{\partial J}{\partial w}$ va $\frac{\partial J}{\partial b}$.
  * `sp.solve` orqali optimal parametrlar topildi: **$w = 2.7$, $b = -0.1$**.
* **Gradient Descent optimizatsiyasi:**
  * Algoritm noldan NumPy orqali yozildi.
  * Turli learning rate ($\alpha = 0.02, 0.03, 0.04$) qiymatlari konvergensiya tezligiga ta'siri bo‘yicha o‘zaro solishtirildi.
* **Loss Landscape:**
  * Xatolik sathi `matplotlib` kontur xaritasi (Contour map) orqali chizildi va global minimum nuqtasi belgilandi.

---

### 2. Task B: Logistik regressiya va From-Scratch ROC-AUC
* **Ma'lumotlar:** Saytda o‘tkazilgan vaqt ($x$) va xarid holati ($y \in \{0, 1\}$).
* **Sigmoid xossasini tekshirish:**
  * Sigmoid hosilasi SymPy orqali tekshirilib, quyidagi ayniyat analitik isbotlandi:
    $$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$
* **Logistik regressiya trenirovkasi:**
  * Binary Cross-Entropy gradientlari orqali model parametrlari ($w \approx 2.73, b \approx -7.17$) 2000 iteratsiyada optimallashtirildi.
* **ROC Egri chizig'i va AUC (From Scratch):**
  * Har xil ehtimollik bo'sag'alari (thresholds) bo'yicha **TPR** va **FPR** nuqtalari qo'lda hisoblandi:
    $$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
  * Egri chiziq ostidagi maydon (AUC) trapetsiya integrallash usuli (`np.trapezoid`) orqali hisoblandi: **$\text{AUC} \approx 0.88$**.

---

## 📊 Natijalar va Vizualizatsiya

| Contour Map (Loss Landscape) | ROC Curve (AUC = 0.88) |
| :---: | :---: |
| ![Contour Map](contour.png) | ![ROC Curve](roc_curve.png) |

---

## 🛠 Ishlatilgan texnologiyalar
* **Python 3**
* **NumPy** — vektorlashgan hisob-kitoblar va trapetsiya qoidasi integratsiyasi uchun.
* **SymPy** — analitik differensiallash, qisman hosilalar va algebraik tenglamalar sistemasini yechish uchun.
* **Matplotlib** — kontur xaritasi va ROC egri chizig'ini vizualizatsiya qilish uchun.

---

## 🚀 Ishga tushirish

Repozitoriyani yuklab oling va kerakli kutubxonalarni o'rnating:

```bash
git clone [https://github.com/ml-engineer04/](https://github.com/ml-engineer04/)<repo-nomi>.git
cd <repo-nomi>
pip install numpy sympy matplotlib
