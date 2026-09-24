# Credit Analytics: Default Prediction & Portfolio Risk

Classification models for predicting borrower default probability, plus a Monte Carlo portfolio simulation to evaluate lending strategies under credit risk.

## Overview

1. **Data simulation** — two synthetic borrower datasets are generated from explanatory variables (age, monthly income, and a third feature), with default probabilities following:
   - **Dataset 1**: a linear (logistic) decision boundary
   - **Dataset 2**: a nonlinear decision boundary (age-threshold effects)
2. **Classification models**
   - Logistic regression (linear decision boundary)
   - Kernel SVM (RBF kernel, nonlinear decision boundary) with probability outputs
   - Models are compared via cross-entropy loss, confusion matrices, and ROC/AUC curves on both datasets, in-sample and out-of-sample
3. **Portfolio risk simulation** — using the fitted default probabilities, three lending strategies are compared via Monte Carlo scenario simulation:
   - **Scenario I**: lend to everyone at a flat interest rate (naive baseline)
   - **Scenario II**: lend only to borrowers with predicted default probability ≥ 0.95 (logistic regression), at a lower rate
   - **Scenario III**: same selection rule using the SVM model's predicted probabilities

   Each strategy's expected profit and Value-at-Risk (VaR) are estimated from simulated repayment/default outcomes.

## Contents

```
notebooks/
  credit_analytics.ipynb   # dataset generation, model fitting, evaluation, and portfolio simulation
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/credit_analytics.ipynb
```

## Key finding

The naive "lend to everyone" strategy outperforms the logistic-regression-filtered strategy in expected profit, since the linear decision boundary cannot separate good from bad borrowers on the nonlinear dataset. The SVM-filtered strategy, which captures the nonlinear boundary, achieves both higher expected profit and a better VaR profile than either alternative.

## Authors

_Add group member names here._
