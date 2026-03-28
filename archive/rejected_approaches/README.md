# Rejected Approaches

Methods that were implemented and tested but did not improve results. Kept here for reference when writing the paper -- documenting what was tried and why it was discarded strengthens the methodology section.

## Pathway C: Polynomial Feature Engineering

**Files:** `preprocess_polynomial.py`, `train_polynomial.py`, `train_extended_poly.py`

**What:** Generated 2nd-degree polynomial and interaction terms (HLB×pH, Conc², C#×EO, etc.) from the original 6 features. PFI selected top 15 polynomial terms. Tested alone, with VSG, and combined with RDKit descriptors.

**Results:**
| Configuration | Best Model | Test R² |
|---|---|---|
| Polynomial (6 + 15 poly) | RF | 0.379 |
| Polynomial + VSG | GB | 0.409 |
| Extended + Poly (6 + 20 RDKit + 15 poly) | GB | 0.406 |

**Why rejected:** All configurations performed worse than the baseline (0.417). Tree-based models already capture nonlinear interactions internally, so explicit polynomial terms added redundancy and overfitting risk without new information.

---

## VSG with 500 Samples (original)

**Files:** `train_with_vsg.py`

**What:** KDE-based Virtual Sample Generation with 500 synthetic samples on the original 6 features.

**Results:**
| Model | Val R² | Test R² |
|---|---|---|
| RF | 0.644 | 0.402 |
| SVR | 0.543 | 0.346 |

**Why rejected:** Too many synthetic samples (500 vs 202 real) overwhelmed the real data with noise. Reducing to 200 samples improved results (RF 0.433, GB 0.458).

---

## Extended with 40 RDKit Descriptors

**Files:** `preprocess_extended.py`, `train_extended.py`, `train_extended_vsg.py`

**What:** Top 40 RDKit descriptors (46 total features) instead of top 20 or top 10.

**Results:**
| Configuration | Best Model | Test R² |
|---|---|---|
| Extended 40 (no VSG) | RF | 0.488 |
| Extended 40 + VSG (500 samples) | RF | 0.374 |

**Why rejected:** 46 features with 202 samples gave a poor sample-to-feature ratio. Top 10 RDKit (0.499) and top 10 + VSG (0.525) both outperformed it. The extra 30 descriptors added noise, not signal.
