# Research Enhancement Plan for Corrosion Inhibitors ML Project

## Current State Diagnosis

The project predicts inhibition efficiency (IE) from 6 features (C#, Mw, HLB, EO, Conc, pH) using RF and SVR on ~300 samples across three corrosive media. The best general model achieves **Test R^2 = 0.417** (RF). Key problems identified:

- pH acts as a proxy for medium, dominating feature importance (SHAP = 18.4)
- Medium-specific models underperform (R^2 < 0.2) due to small per-medium sample sizes
- Leave-one-medium-out shows negative R^2 -- no cross-medium generalization
- Only 2 model types explored (RF, SVR)
- No data augmentation despite small dataset
- No polynomial/interaction features despite Paper 4 using the *exact same molecular features* (C#, Mw, HLB, EO) with polynomial terms to reach R^2 = 0.80

---

## PATHWAY A: Data Augmentation via KDE-based Virtual Sample Generation

**Source:** Paper 3 (Herowati et al.) -- improved R^2 from 0.05 to 0.99 on a 54-sample pyrimidine dataset using this technique.

**What to do:**

- Implement Kernel Density Estimation (KDE) to generate ~500-1000 virtual training samples
- Apply VSG **only to the training set** after the train/val/test split (critical to prevent data leakage)
- Use `scipy.stats.gaussian_kde` or `sklearn.neighbors.KernelDensity`
- Tune the bandwidth parameter carefully (Paper 3 did not discuss this -- an opportunity to be more rigorous)
- Validate that synthetic samples are chemically plausible (descriptor ranges stay within physically meaningful bounds)
- Consider medium-aware VSG: generate synthetic samples per medium to balance the dataset (HCl: 160, NaCl: 74, CPS: 61 -- CPS is underrepresented)

**Expected impact:** Potentially dramatic improvement in R^2, especially for medium-specific models that currently fail due to tiny sample sizes.

**Caution:** Paper 3's R^2 = 0.99 results are suspiciously perfect. Report results honestly with proper caveats about synthetic data limitations. Always evaluate on the **original** held-out test set.

---

## PATHWAY B: Massively Expanded Model Comparison

**Source:** Paper 3 tested 23 models (14 linear + 9 nonlinear); Paper 1 used SVR, XGBoost, GPR; Paper 4 used 8 regularized linear models.

**Currently missing models to add:**

- **Gradient Boosting / XGBoost / LightGBM** -- consistently top performers in tabular data
- **Gaussian Process Regression (GPR)** -- provides natural uncertainty quantification (Paper 1)
- **ElasticNet, Lasso, Ridge** -- regularized linear models suited for small datasets (Paper 4)
- **AdaBoost, Bagging, Extra Trees** -- ensemble variants
- **KNN Regressor** -- non-parametric baseline
- **Partial Least Squares (PLS)** -- good for interpretability + feature selection (Paper 2)
- **Bayesian Ridge / ARD Regression** -- automatic relevance determination

**Implementation:** Use a systematic benchmark table comparing all models with consistent metrics (R^2, RMSE, MAE, MSE) on the same train/val/test splits, both with and without VSG augmentation. This creates a powerful results table for publication.

---

## PATHWAY C: Polynomial Feature Engineering and Interaction Terms

**Source:** Paper 4 (Tale Masoule et al.) used the **exact same molecular features** (C#, Mw, HLB, EO) and achieved R^2 = 0.80 with only 59 samples by applying 2nd-degree polynomial transformations.

**What to do:**

- Generate all 2nd-degree polynomial features: HLB^2, EO^2, Conc^2, pH^2, HLB x EO, HLB x Conc, Conc x pH, C# x EO, etc.
- Use `sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=False)`
- Aggressively prune: retain only top 4-5 polynomial features by composite importance score
- Key interactions to investigate based on domain knowledge:
  - **Conc x pH** -- concentration effects likely differ by pH/medium
  - **HLB x EO** -- both relate to surfactant hydrophilicity
  - **C# x Conc** -- carbon chain length may interact with dosage
  - **Conc^2** -- diminishing returns on concentration (known phenomenon)

**Expected impact:** Capturing nonlinear structure-property relationships without requiring complex models, directly applicable since Paper 4 demonstrated this with the same feature space.

---

## PATHWAY D: Ensemble Feature Selection (5-Algorithm Composite)

**Source:** Paper 4 used 5 algorithms aggregated into composite scores; Paper 1 compared 5 IVS methods; Paper 2 used PLS VIP.

**Current gap:** The project only uses basic importance (linear coefficients, RF importance, permutation importance) without systematic selection.

**What to do:**

1. **Pearson Correlation** -- linear filter
2. **Mutual Information** -- nonlinear dependency
3. **F-test p-values** -- statistical significance
4. **Recursive Feature Elimination (RFE)** -- iterative model-based
5. **PLS VIP scores** -- projection-based importance

Aggregate all five into a composite ranking per feature. Apply this to:

- Raw features (6 original)
- Polynomial-expanded features (after Pathway C)
- Per-medium subsets (to understand medium-specific drivers)

Additionally, perform **collinearity filtering** (Pearson > 0.7 threshold, per Paper 4) before training -- currently C#, Mw, HLB, EO may be intercorrelated.

---

## PATHWAY E: Sobol Global Sensitivity Analysis

**Source:** Paper 1 (Jayaweera et al.) -- central finding was that Sobol analysis revealed models trivialize inhibitor sensitivity.

**Why this matters for your project:** SHAP shows pH dominates (18.4 vs Conc 10.5), but you need to verify this isn't masking the real structure-activity relationships. Sobol's 1st-order and total indices provide a mathematically rigorous decomposition.

**What to do:**

- Use `SALib` library for Sobol sensitivity analysis
- Compute 1st-order indices (individual variable contribution) and total indices (including interactions)
- Compare Sobol results vs SHAP results -- discrepancies reveal interaction effects
- Run Sobol **with and without pH** to see how other features redistribute
- Run Sobol per medium to understand medium-specific drivers

**Expected insight:** Whether pH is truly important for IE prediction, or whether it's just a medium identifier whose effect could be better captured by one-hot encoding of medium.

---

## PATHWAY F: Rethinking the Medium/pH Encoding

**Source:** Paper 1's finding that linearly correlated variables dominate and trivialize the variables of interest.

**Core problem:** pH = {0.5, 4.9, 12.5} is a near-perfect proxy for medium = {HCl, NaCl, CPS}. The model uses pH to distinguish mediums rather than learning actual pH-IE chemistry.

**Options to test:**

1. **Replace pH with one-hot encoded medium** (3 binary columns: is_HCl, is_NaCl, is_CPS) -- makes the medium relationship explicit, freeing the model to learn from other features
2. **Drop pH entirely** and use only molecular/concentration features -- tests if structure-activity relationships alone are predictive
3. **Use medium as a grouping variable** in a mixed-effects model (`statsmodels.MixedLM`) -- models medium-specific intercepts/slopes while sharing structure-activity coefficients
4. **Interaction with medium:** Create medium x feature interactions (e.g., Conc_in_HCl, Conc_in_NaCl, Conc_in_CPS) to capture medium-specific concentration effects

Compare all four approaches to the current pH-as-numeric baseline. This directly addresses the project's biggest weakness.

---

## PATHWAY G: Improved Cross-Validation and Evaluation

**Source:** Paper 3 used 5-fold CV; Paper 4 used 5-fold + separate validation holdout; Paper 1 used temporal 4-fold.

**Current weakness:** Only 3-fold CV with 18 iterations of RandomizedSearchCV. With ~200 training samples, 3-fold means ~133 training samples per fold -- tight.

**Improvements:**

- Increase to **5-fold or 10-fold CV** for more robust estimates
- Use **stratified K-fold by medium** to ensure each fold has proportional representation of HCl/NaCl/CPS
- Implement **nested CV** (outer loop for evaluation, inner loop for hyperparameter tuning) to get unbiased performance estimates
- Increase RandomizedSearchCV iterations from 18 to 50-100 (or switch to `BayesSearchCV` from `scikit-optimize`)
- Report **mean +/- std** across CV folds, not just single-split metrics

---

## PATHWAY H: Uncertainty Quantification

**Source:** Paper 1 used GPR for probabilistic predictions; Paper 4 acknowledged prediction bias.

**What to add:**

- **Gaussian Process Regression** -- natural prediction intervals
- **Quantile Regression Forests** -- prediction intervals from RF
- **Conformal Prediction** -- distribution-free prediction intervals with coverage guarantees (using `MAPIE` library)
- **Bootstrap prediction intervals** -- resample-based confidence bands

**Why it matters:** For practical inhibitor optimization, knowing that IE = 65% +/- 12% is far more useful than just IE = 65%. Papers in the field are starting to demand this.

---

## PATHWAY I: Outlier Detection and Data Quality

**Source:** Paper 4 used 4 outlier detection algorithms simultaneously; Paper 1 used LOF over Z-score.

**What to do:**

- Apply **4 algorithms in parallel**: Isolation Forest, Local Outlier Factor, One-Class SVM, Elliptic Envelope
- Normalize anomaly scores across algorithms for consensus
- Visualize outliers on PCA-projected scatter plot (Paper 4's approach -- very effective figure)
- **Do not automatically remove** -- analyze whether outliers are measurement errors or legitimate chemical behavior
- This analysis may reveal why certain mediums (CPS) have higher prediction error

---

## PATHWAY J: Multilayer / Hierarchical Modeling

**Source:** Paper 4 (Tale Masoule et al.) -- used a structure-property-performance hierarchical framework with the same type of molecular features.

**How to adapt:**

- **Layer 1 (Molecular):** C#, Mw, HLB, EO (surfactant structure)
- **Layer 2 (Operational):** Conc, pH/medium (test conditions)
- **Layer 3 (Output):** IE

Build separate sub-models:

1. Molecular features --> intermediate predictions (e.g., predicted surface coverage, predicted adsorption strength)
2. Intermediate predictions + operational conditions --> IE

Even without measuring intermediate properties, you could use an **autoencoder or latent representation** to learn an implicit intermediate layer. Alternatively, use the molecular features to predict medium-specific behavior and combine.

---

## PATHWAY K: Publication-Quality Presentation Enhancements

**Source:** Best practices from all four papers.

**Figures to add:**

1. **Methodology flowchart** (Paper 1, Fig. 1; Paper 4, Fig. 10) -- full pipeline from data to optimization
2. **PCA outlier visualization** with multi-algorithm overlay and scaled anomaly circles (Paper 4, Fig. 8)
3. **Sobol sensitivity bar charts** comparing 1st-order vs total indices per feature (Paper 1, Fig. 6)
4. **Model comparison heatmap** -- all models x all metrics x with/without VSG (inspired by Paper 3's Tables 2-3)
5. **Domain of applicability plot** -- Williams plot showing standardized residuals vs leverage (Paper 2)
6. **Before/after VSG comparison** -- dual bar charts showing R^2 and RMSE improvement (Paper 3, Fig. 2)
7. **Polynomial feature scatter plots** with R^2 annotations (Paper 4, Fig. 14) -- e.g., Conc^2 vs IE, HLB x EO vs IE
8. **Medium-stratified predicted vs actual** with different colors/markers per medium and separate regression lines
9. **Learning curves** -- performance vs training set size to diagnose whether more data would help
10. **Feature correlation heatmap with embedded coefficients** (Paper 4, Fig. 9)

**Tables to add:**

- Comprehensive model comparison table (23 models, before/after augmentation)
- Feature selection composite scores from 5 algorithms
- Per-medium performance breakdown for all models
- Comparison with prior literature results

---

## PATHWAY L: External Validation and Generalization

**Source:** Paper 2 designed 5 new molecules; Paper 3 validated on 3 external datasets.

**Options:**

1. If new lab data can be collected, use it as a completely independent test set
2. Cross-validate with the pyrimidine datasets from Papers 2/3 (different compound class but same ML framework)
3. Perform **domain of applicability analysis** -- for each test point, calculate its distance to the training distribution and flag extrapolation risks
4. Use the model to **predict optimal formulations** and verify experimentally (strongest form of external validation for a journal paper)

---

## Recommended Priority Ordering

Based on expected impact vs. implementation effort:

| Priority | Pathway | Impact | Effort | Justification |
|----------|---------|--------|--------|---------------|
| 1 | C (Polynomial Features) | High | Low | Paper 4 proved this works with same features; easy to implement |
| 2 | F (Medium/pH Encoding) | High | Low | Directly addresses the #1 weakness; simple code changes |
| 3 | B (Expanded Models) | High | Medium | XGBoost/GBM likely to outperform current RF; systematic comparison |
| 4 | A (KDE Data Augmentation) | Very High | Medium | Paper 3 showed dramatic improvement; needs careful implementation |
| 5 | D (Ensemble Feature Selection) | Medium | Medium | Better feature understanding; improves interpretability |
| 6 | E (Sobol Analysis) | Medium | Low | Adds rigor to sensitivity claims; uses SALib |
| 7 | G (Better CV) | Medium | Low | More credible results with stratified 5-fold + nested CV |
| 8 | I (Outlier Detection) | Medium | Low | May explain CPS prediction errors |
| 9 | H (Uncertainty Quantification) | High | Medium | Novel contribution for the field; practical value |
| 10 | K (Presentation) | High | Medium | Makes the paper publishable at higher-tier venues |
| 11 | J (Hierarchical Model) | High | High | Novel framework; mirrors Paper 4's successful approach |
| 12 | L (External Validation) | Very High | High | Requires lab work but strongest possible evidence |

---

## Concrete Implementation Sequence

**Phase 1 -- Quick Wins (1-2 days):**

- Implement polynomial features (Pathway C)
- Test one-hot medium encoding vs pH (Pathway F)
- Add XGBoost/GBM/ElasticNet to model arsenal (Pathway B, partial)
- Upgrade to 5-fold stratified CV (Pathway G)

**Phase 2 -- Core Enhancements (3-5 days):**

- Implement KDE-based VSG augmentation (Pathway A)
- Run full 23-model benchmark with/without VSG (Pathway B, complete)
- Ensemble feature selection pipeline (Pathway D)
- Sobol sensitivity analysis (Pathway E)
- Multi-algorithm outlier detection (Pathway I)

**Phase 3 -- Advanced Methods (5-7 days):**

- Uncertainty quantification with GPR + conformal prediction (Pathway H)
- Hierarchical/multilayer model (Pathway J)
- Domain of applicability analysis (Pathway L, partial)

**Phase 4 -- Publication Polish (2-3 days):**

- Generate all enhanced figures (Pathway K)
- Write comprehensive comparison tables
- Methodology flowchart
- Results narrative with statistical rigor

---

## Paper References

- **Paper 1:** Jayaweera et al. -- "Assessing the Feasibility of Using a Data-Driven Corrosion Rate Model for Optimizing Dosages of Corrosion Inhibitors" (npj Materials Degradation, 2024)
- **Paper 2:** Alamri & Alhazmi -- "Development of data driven machine learning models for the prediction and design of pyrimidine corrosion inhibitors" (J. Saudi Chemical Society, 2022)
- **Paper 3:** Herowati et al. -- "Machine learning for pyrimidine corrosion inhibitor small dataset" (Theoretical Chemistry Accounts, 2024)
- **Paper 4:** Tale Masoule et al. -- "Predicting air-entraining in cement paste from the molecular attributes of nonionic surfactants with a multilayer method" (J. American Ceramic Society, 2025)
