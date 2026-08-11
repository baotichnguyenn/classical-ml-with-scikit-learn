# Classical ML Problems — Four Supervised & Unsupervised Baselines, Built From Scratch

Four end-to-end notebooks covering the four canonical problem shapes in classical machine learning:
**regression**, **binary classification**, **time-series forecasting**, and **clustering**.

Each one goes from raw CSV to a tuned model, and each one is measured against an explicit dumb
baseline. Where the model beat the baseline, the margin is reported. Where it barely did — or
where the data turned out to have almost no signal — that is reported too.

> **What this repo is.** This is early work: my first full pass at the classical ML stack, written
> during Year 1 of my Applied AI & Analytics diploma (DAAA), late 2023 – early 2024. It is not a
> production system and it is not novel research. It is the foundation layer — the point where
> I stopped treating `model.fit()` as the interesting part and started treating **validation
> protocol, baselines, and leakage** as the interesting part.
>
> **Why it's still up.** The [Post-Hoc Audit](#post-hoc-audit--what-id-do-differently-now) section
> near the bottom is a line-by-line critique of my own methodology, written well after the fact.
> The delta between the notebooks and that audit is the actual point of this repository. If you
> only read one section, read that one.

**Stack:** Python 3.9 / 3.12 · pandas · NumPy · scikit-learn · imbalanced-learn · statsmodels · Matplotlib · Seaborn

---

## Results at a Glance

| Notebook | Problem | Data | Baseline | Best model | Result | Δ vs baseline |
|---|---|---|---|---|---|---|
| [Hospital Cost](#1-regression--hospital-cost-prediction) | Regression | 1,338 patients × 7 cols | `DummyRegressor(mean)` — RMSE \$11,142 | Gradient Boosting (tuned) | **RMSE \$4,352**, MAPE 31.7% | **−61% RMSE** |
| [Water Quality](#2-binary-classification--water-potability) | Binary classification | 3,276 samples × 9 features, 61/39 imbalance | `DummyClassifier(uniform)` — 51.3% acc | ExtraTrees (tuned) | **63.6% acc**, macro-F1 0.613 | **+12.3 pp acc** |
| [Stock Prices](#3-time-series--multi-asset-price-forecasting) † | Time-series forecasting | 1,257 trading days × 3 tickers | Mean forecast — MAPE 17–31% | SARIMA (grid-searched) | **MAPE 4.4% – 8.1%** | **−13 to −24 pp MAPE** |
| [Student Segmentation](#4-clustering--student-segmentation) | Clustering | 1,000 students × 5 features | — (unsupervised) | K-Means, k=8 | **Silhouette 0.215** — weak separation, reported as such | n/a |

† See [Attribution](#attribution) — the time-series notebook is a classmate's work, included here for repo completeness.

---

## 1. Regression — Hospital Cost Prediction

**Problem.** Predict per-patient hospitalisation cost in USD from demographics and health markers
(age, gender, BMI, smoking status, region). Heavy right tail: mean \$13,270, max \$63,770.

**Approach.**
- EDA: correlation heatmap over label-encoded features, per-feature distributions, box plots for tail structure.
- Cleaning: z-score screen at `|z| > 2.5` on `Age`, `BMI`, `Cost` → 63 rows removed (4.7%), 1,275 retained.
- Encoding: binary map for `Gender`/`Smoker`; one-hot for `Region` with `drop_first=True` — **explicitly to avoid the dummy-variable trap**, not by default.
- `StandardScaler` fit on train only, applied to test.
- Screened **8 model families**, then `GridSearchCV` (`cv=3`, neg-MSE) on the top 3 by RMSE.

**Model screen (test set, before tuning):**

| Model | RMSE | MAPE |
|---|---|---|
| **Gradient Boosting** | **4,435.90** | **30.55%** |
| Random Forest | 4,741.52 | 39.66% |
| KNN | 5,200.32 | 39.79% |
| Lasso | 5,795.41 | 39.57% |
| Linear Regression | 5,795.49 | 39.57% |
| Ridge | 5,796.45 | 39.60% |
| Decision Tree | 6,168.69 | 39.57% |
| ElasticNet | 6,881.50 | 63.76% |

**After tuning:** Gradient Boosting `{learning_rate: 0.1, max_depth: 3, n_estimators: 50}` →
**RMSE 4,351.55 / MAPE 31.74%**. Tuned Random Forest landed at 4,405.87; tuned KNN at 5,295.84.

**Feature importance (tuned GBR):** `Smoker` **0.649**, `Age` 0.181, `BMI` 0.165 — all three region
dummies and `Gender` together contribute **< 0.5%**. The model is, in effect, a smoker premium plus
an age–BMI surface. That interpretability check is what makes the result trustworthy rather than
just low-error.

**What it taught me.** A shallow ensemble (depth 3, 50 trees) beat a deep one. The tuning gain over
default GBR was ~2% RMSE — the win came from *model family selection*, not hyperparameters. That
ratio has held true on nearly everything I've built since.

📓 [`[Regression] Hospital Cost .ipynb`](%5BRegression%5D%20Hospital%20Cost%20.ipynb)

---

## 2. Binary Classification — Water Potability

**Problem.** Classify a water sample as potable (1) or not (0) from 9 chemical and physical
properties. The hard case: **~24% of `Sulfate` and 15% of `ph` are missing**, and the classes are
imbalanced 61/39.

**The finding that shaped the whole notebook.** Every feature's Pearson correlation with the target
falls in **[−0.030, +0.007]**. The strongest single linear relationship in the dataset is
`Organic_carbon` at r = −0.030. I wrote that up as a finding rather than quietly moving on, and
argued explicitly *against* dropping features on correlation grounds — a near-zero r rules out a
linear relationship, not a relationship.

**Approach.**
- `IterativeImputer` (MICE-style, round-robin regression on the other features) for the three columns with missing values — chosen over mean/median because the missingness is 5–24%, large enough that central-tendency fill would compress variance.
- Stratified 75/25 split, `random_state=42`.
- **SMOTE oversampling placed *inside* an `imblearn.Pipeline`**, so resampling happens within each CV fold and never touches the validation fold. This is the single most important design choice in the notebook — SMOTE applied before cross-validation is one of the most common ways to silently inflate a classification score.
- **12 classifiers** benchmarked under 10-fold CV across 5 metrics (accuracy, balanced accuracy, recall, F1, ROC-AUC), with **train and test scores retained** so overfit is visible per model.
- **Learning curves plotted for all 12** (training score vs CV score vs training-set size) — this is how ExtraTrees and Random Forest were identified as the two models still generalising as data grew, rather than picking on a single number.

**Result.**

| Model | Accuracy | Macro-F1 | Recall (class 1) | Precision (class 1) |
|---|---|---|---|---|
| `DummyClassifier(uniform)` | 0.501 | 0.496 | 0.525 | 0.395 |
| ExtraTrees (default) | 0.637 | 0.601 | 0.431 | 0.545 |
| **ExtraTrees (tuned)** | **0.636** | **0.613** | **0.503** | 0.537 |

`RandomizedSearchCV` on F1 → `{n_estimators: 500, min_samples_leaf: 5, max_features: 4}`.

**The honest read, which is in the notebook verbatim:** tuning moved accuracy by −0.1 pp. It did
buy a real **+7.2 pp on minority-class recall** (0.431 → 0.503) at a small precision cost — the
right trade for a potability screen, where a missed unsafe sample costs more than a false alarm —
but the headline number did not move. I wrote *"rough hyperparameter tuning made little improvement"*
rather than reporting the one metric that happened to tick up.

Training accuracy of **1.000** against 0.64 test accuracy is flagged in the notebook as the memorisation
signature it is.

**Feature importance:** `pH`, `Hardness`, `Sulfate` dominate — a different ranking than the
correlation matrix suggested, which is exactly the point about non-linear structure.

📓 [`[Binary Classification] Water Quality.ipynb`](%5BBinary%20Classification%5D%20Water%20Quality.ipynb)

---

## 3. Time Series — Multi-Asset Price Forecasting

**Problem.** Forecast 60-day-ahead prices for AAPL, AMZN (USD) and DBS `D05.SI` (SGD) from
1,257 trading days, 2018-10-01 → 2023-09-28.

**Approach.**
- **Augmented Dickey-Fuller** on all three series: p = 0.765 / 0.459 / 0.656 → fail to reject the unit root. After first differencing, p < 0.001 across the board → `d = 1` fixed on evidence, not convention.
- **ACF/PACF** read for order selection: slow ACF decay + PACF cutting off after lag 1 → AR(1)-dominant structure.
- **Expanding-window walk-forward validation** (`TimeSeriesSplit`, `test_size=80`, `n_splits=3`). Train on the past, forecast forward, roll, repeat. No random shuffling, no lookahead — the model never sees a future bar.
- Every candidate scored on **RMSE, MAPE *and* AIC** together — out-of-sample error alongside an in-sample complexity penalty.
- **Grid search:** ARIMA over 25 (p,q) combinations per ticker; SARIMA over 144 (p,d,q)(P,D,Q,7) combinations per ticker — roughly **3.5 hours of compute per ticker**.

**Result (walk-forward, averaged over folds):**

| Ticker | Mean-forecast baseline MAPE | Best ARIMA | Best SARIMA | SARIMA MAPE |
|---|---|---|---|---|
| AAPL | 31.04% | (4,1,1) — 7.44% | (4,1,4)(1,1,1,7) | **7.14%** |
| AMZN | 24.74% | (3,1,3) — 8.29% | (2,1,3)(1,1,1,7) | **8.09%** |
| D05.SI | 17.30% | — ‡ | (2,1,4)(1,1,1,7) | **4.37%** |

‡ The DBS ARIMA sweep was interrupted mid-run; its row in the notebook's summary table is not
reproducible from the saved cell outputs. Noted rather than quoted.

**The result worth actually reading.** In the AMZN sweep, **`ARIMA(0,1,0)` — a pure random walk —
ranked 4th of 25 configurations at MAPE 8.34%, against the best model's 8.29%.** A relative gap of
0.6%. The entire 25-model grid spans a MAPE band narrower than one percentage point, and the same
compression shows up in the AAPL sweep (7.44% to ~7.47%).

Translated: on price *levels*, none of these models found information beyond "tomorrow ≈ today."
That is precisely what a weak-form-efficient market predicts, and it is the most useful thing in the
notebook. A grid search that finds a 0.6% edge over a random walk has not found an edge — it has
found the noise floor, and the honest move is to say so instead of shipping "SARIMA beats baseline
by 24 percentage points" as a headline.

📓 [`[Time Series Analysis] Stock Prices.ipynb`](%5BTime%20Series%20Analysis%5D%20Stock%20Prices.ipynb)

---

## 4. Clustering — Student Segmentation

**Problem.** Segment 1,000 students by age and English/Math/Science scores to identify cohorts
needing targeted support. No labels, no ground truth.

**Approach.**
- Linear interpolation for 29 missing English and 33 missing Math scores.
- `StandardScaler` on all numeric features — mandatory before any Euclidean-distance method.
- **A feature ablation I'd still defend.** `Gender` had 7 categories (Female, Male, Non-binary, Genderqueer, Genderfluid, Bigender, Polygender). One-hot encoding it and re-running the full k sweep showed *higher inertia and lower silhouette at every k* versus dropping it. I dropped it — six sparse binary dimensions were pushing k-means into the curse of dimensionality, where every point drifts equidistant and cluster structure dissolves. Testing that empirically instead of assuming it either way is the habit worth having.
- k swept 2 → 10, evaluated by **both** elbow (inertia) and silhouette.
- **Four algorithms compared at k=8**, spanning three different notions of "cluster":

| Algorithm | Silhouette @ k=8 | Notion of a cluster |
|---|---|---|
| **K-Means** | **0.215** | Distance to centroid |
| Spectral | 0.212 | Eigenstructure of the affinity graph |
| Agglomerative | 0.142 | Hierarchical merge distance |
| DBSCAN (defaults) | −0.359 | Density reachability |

**The honest read.** The elbow plot has **no elbow** — inertia falls smoothly with k, and the
notebook says so instead of inventing one. The winning silhouette of 0.215 is weak; on a −1 to +1
scale, that means substantially overlapping clusters. The correct conclusion is that **this dataset
has no strong natural cluster structure**, and k=8 is a defensible operational choice for delivering
differentiated support, not a discovered truth about students.

📓 [`[Clustering] Student Performance.ipynb`](%5BClustering%5D%20Student%20Performance.ipynb)

---

## The Habits This Repo Was Built to Establish

Four different problem types, one set of non-negotiables:

1. **Every model is measured against a dumb baseline.** `DummyRegressor`, `DummyClassifier`, and a mean-forecast benchmark. A number with no baseline next to it is not a result.
2. **Resampling and imputation belong inside the pipeline.** SMOTE inside `imblearn.Pipeline` so it runs per-fold, never across the validation boundary.
3. **Validation protocol matches data structure.** Stratified splits for imbalanced classification; expanding-window walk-forward for time series. Never `train_test_split(shuffle=True)` on a time index.
4. **Metrics chosen for the problem, not the leaderboard.** Balanced accuracy and per-class recall under imbalance; RMSE *and* MAPE together for a heavy-tailed cost target; AIC alongside out-of-sample error for time-series order selection; silhouette for label-free clustering.
5. **Interpretability as a correctness check.** Feature importances read back against domain intuition every time — `Smoker` at 0.649 in the cost model, `pH`/`Hardness`/`Sulfate` in the water model.
6. **Negative results reported as results.** Near-zero correlations, no elbow, tuning that bought nothing, a random walk matching a 144-config grid search. All four notebooks contain at least one.

---

## Post-Hoc Audit — What I'd Do Differently Now

Written later, against my own notebooks. Not a disclaimer — the specific gap between these two
lists is the growth I want a reviewer to see.

**Validation and leakage**

- **`IterativeImputer` is fit on the full dataframe before the train/test split** in the water-quality notebook, then again inside the pipeline. The first fit leaks test-set distribution into the imputed training values. The pipeline version is correct; the pre-fit should be deleted outright.
- **Outlier removal in the regression notebook happens before splitting.** Z-scores are computed over all 1,338 rows, then 63 are dropped, then the split occurs — the cleaning step has already seen the test set. Worse, the dummy baseline is computed on the *uncleaned* frame, so the headline "−61% RMSE" compares models fit on different populations. The real gain is smaller and I no longer quote it without that caveat.
- **Trimming the tail removes the problem.** That `|z| > 2.5` screen deletes exactly the high-cost smokers the model exists to predict. Winsorising, or modelling `log(cost)`, preserves the tail instead of legislating it away.
- **Model selection and final evaluation share a test set.** Screening 8 families on the test set, then tuning the top 3 and reporting test RMSE, makes the reported error mildly optimistic. Nested CV, or a proper train/validation/test three-way split.

**Search and thresholds**

- **`RandomizedSearchCV(n_iter=60)` over a 27-point grid** — sklearn raised the warning and I kept going. That is `GridSearchCV` wearing a costume. And `cv=2` is far too few folds for a stable F1 estimate.
- **The 0.5 decision threshold was never tuned.** Under a 61/39 imbalance with an asymmetric cost (a missed unsafe sample beats a false alarm), the threshold should be selected on a precision–recall curve. SMOTE addresses the training distribution; it does not choose an operating point.
- **DBSCAN was run at default `eps`/`min_samples`.** A silhouette of −0.36 is a hyperparameter failure, not evidence against density clustering. A k-distance plot to pick `eps` is a ten-line fix, and comparing a swept K-Means against an unswept DBSCAN is not a fair comparison.
- **144-config SARIMA grids at ~3.5 h per ticker** to land within noise of a random walk. Coarse-to-fine search, `auto_arima` for a starting region, or an early-stopping rule on the AIC surface.

**Time-series specifics**

- **Forecasting price levels flatters every model** on a near-unit-root series, and the mean-forecast baseline is a strawman for a trending one. The correct benchmark is a **naive random-walk (last-value) forecast**, and the correct target is **returns**, not levels. `ARIMA(0,1,0)` placing 4th of 25 already showed this from inside the notebook.
- **The 7-day seasonality is an artifact.** Reindexing to a full calendar and linearly interpolating weekends *manufactures* the weekly period that `seasonal_decompose` then recovers. Markets trade 5 days. The notebook half-catches this and proceeds anyway; the fix is a business-day frequency (`freq='B'`) and `m=5`, or dropping seasonal decomposition entirely for daily equity data.

**Reproducibility and hygiene**

- **The clustering notebook's final profile table reports 2 clusters (49.4% / 50.6%), inconsistent with the chosen k=8**, and the eight written cluster personas are not derived from it. That section needs a re-run with a fixed seed and profiles computed from the actual k=8 centroids, inverse-transformed to original score units.
- **Unseeded estimators produce run-to-run drift.** The water-quality model reports 0.636 accuracy in the comparison cell and 0.62 in the final cell — same model, different refit. Every estimator gets a `random_state` now.
- **No `requirements.txt`, two Python versions across four notebooks (3.9.12 and 3.12.0), datasets not committed.** Nothing here is reproducible by a stranger, which for a public repo is the most consequential item on this list.
- **6 MB of base64 plot images committed inside the `.ipynb` files**, and filenames with spaces and brackets. `nbstripout` on a pre-commit hook, and kebab-case filenames.

---

## Repository Structure

```
classical-ml-problems/
├── [Regression] Hospital Cost .ipynb          # GBR / RF / KNN + GridSearchCV, 8-model screen
├── [Binary Classification] Water Quality.ipynb # 12-model screen, SMOTE-in-pipeline, learning curves
├── [Time Series Analysis] Stock Prices.ipynb   # ADF, ACF/PACF, ARIMA + SARIMA walk-forward grids
└── [Clustering] Student Performance.ipynb      # K-Means / Spectral / Agglomerative / DBSCAN
```

**Datasets are not included in this repository.** The notebooks expect:

| Notebook | Expected path |
|---|---|
| Regression | `CA1-Dataset/CA1-Regression-Dataset.csv` |
| Classification | `../Datasets/waterquality.csv` |
| Time series | `./CA2 Datasets/CA2-Stock-Price-Data.csv` |
| Clustering | `./CA2 Datasets/Student_Performance_dataset.csv` |

## Running

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn imbalanced-learn statsmodels matplotlib seaborn tqdm openpyxl
jupyter lab
```

Notebooks were originally run on Python 3.9.12 (regression, time series, clustering) and Python
3.12.0 (classification). Nothing is version-pinned — see the audit above.

⚠️ The SARIMA grid searches in the time-series notebook take **~3.5 hours per ticker**. Reduce the
parameter ranges before running.

## Attribution

- **Regression, Binary Classification, Clustering** — Nguyen Tich Bao
- **Time Series Analysis** — Shaun Kwo Rui Yu, included here for completeness of the four-problem set

Coursework produced for an Applied AI & Machine Learning module. Datasets were provided by the
module and are not redistributed here.
