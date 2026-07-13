# Bank Marketing Predictor & Decision Tree Chronograph

An interactive decision tree classifier dashboard for predicting bank term deposit subscriptions from client demographics and behavioral metrics. Built with scikit-learn, HTML5, CSS3, and native SVG.

The interface is styled in a **Bright Vintage Car Console theme** (resembling classic sports cars with a bright parchment/ivory dashboard face, ash-wood panels, burnished brass bezel trims, and a mechanical analog speedometer).

---

## 📈 Model Performance Metrics

Tuned via 5-fold cross-validation (`GridSearchCV`) targeting the F1-Score:

- **Accuracy**: `90.0%`
- **ROC-AUC**: `81.0%`
- **Precision**: `63.5%`
- **Recall**: `26.8%`
- **F1-Score**: `0.38`

### Top Feature Importances
1. **Number of Employees** (`nr.employed`): `57.8%` (High social/economic indicator split)
2. **Days Since Last Contact** (`pdays`): `11.6%` (A shorter window strongly correlates with success)
3. **Euribor 3-Month Interest Rate** (`euribor3m`): `7.0%`

---

## 📂 Repository Contents

- `train_model.py`: Preprocesses the dataset (preventing target leakage by dropping `duration`), fits and tunes the decision tree model, and exports model metadata (splits, nodes, thresholds, evaluation stats) to JSON.
- `eda_modeling.ipynb`: Explanatory Data Analysis and model training notebook.
- `index.html`: Clean, professional dashboard layout.
- `style.css`: Vintage dashboard styling sheet.
- `app.js`: Dynamic script updating the analog needle and tracing tree logic.
- `bank-additional/` and `bank/`: Source datasets.

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn
```

### 2. Retrain/Generate Model Data (Optional)
```bash
python train_model.py
```

### 3. Launch the Local Web Dashboard
```bash
python -m http.server 8000
```
Open `http://localhost:8000` in your web browser.
