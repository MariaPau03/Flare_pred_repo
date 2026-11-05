# FlarePred: Forecasting Hidradenitis Suppurativa Severity

Submission for the Almirall HS Innovation Challenge (Hackathon Xartec Salut, 2026).
The goal is to predict each patient's **mean IHS4 score for 1 Jan to 30 Jun 2026** (`avg_ihs4_next6`), using only data recorded up to **31 Dec 2025**.

## Repository

| Path | Content |
|---|---|
| `submission/almirall_FlarePred.ipynb` | Full pipeline (Google Colab): EDA, features, models, analyses, answer files |
| `submission/answers_teamName_*.csv` | Final predictions for exercises 1, 2, 3a (images) and 3b (images + RNAseq) |
| `submission/Write-up.pdf` | Written report |
| `hs_public_dataset.zip` | Challenge data (`train`, `test_1`, `test_2`, evaluation templates) |

**To run:** open the notebook in Colab, point `zip_path` at `hs_public_dataset.zip` in your Drive, and run all cells. Answer files are written to `submission/`.

## Approach and results

All metrics are MAE on IHS4 points from 5-fold cross-validation on the 21,000 train patients, unless stated otherwise.

**Exercise 1: next-6-month IHS4 for every patient.**
One row per patient built from visits, treatments, surgeries and demographics (IHS4 history, visit behaviour, biologic status, antibiotics, surgeries), plus time features: exponentially weighted IHS4, 24-month slope, volatility and flare dynamics.
An XGBoost model with MAE loss reaches **1.467**, against 2.70 for the best simple baseline (mean by Hurley stage).
The time features help in 5 of 5 folds.
test_2 differs from train (adversarial AUC 0.68), mainly because fewer visits are scored there, so its expected MAE is about 1.60.

**Exercise 2: patients on a biologic.**
- **2a:** A model trained only on biologic patients, with features from before and after the biologic started, reaches MAE **1.201** (the exercise 1 model gets 1.225).
- **2b:** A super-responder is a patient whose IHS4 falls below 25% of their pre-biologic level; 12% of labelled patients qualify. The classifier reaches ROC-AUC 0.74 before treatment and 0.80 after the first 6 months. Super-responders are more often on adalimumab, on their first biologic line, earlier in their disease, and quick to respond.

**Extra analyses.**
- **Confounding by indication:** the raw biologic effect is −0.37 IHS4 points and becomes −4.2 after adjusting, mainly for Hurley stage.
- **Variables that did not help:** practitioner ID, missing-data flags and seasonality.

**Exercise 3: image and RNAseq embeddings.**
The embeddings are used to correct the residual of the exercise 1 model, with ridge regression trained on out-of-fold residuals.

| Patients | Exercise 1 MAE | Recipe | Gain in MAE points [95% CI] |
|---|---|---|---|
| All 2,800 image patients | 1.529 | 3a, gated images | +0.032 [+0.015, +0.047] |
| 661 recent biologic starters with an image | 2.028 | 3a, gated images | **+0.134** [+0.075, +0.198] |
| 600 RNAseq patients | 2.094 | 3b, RNAseq | **+0.393** [+0.293, +0.489] |

## Why the embeddings are gated to recent biologic starters

**Long-term patients (≥ 9 months on their biologic) don't need biomarkers.**
The clinical record already holds years of disease history: recent IHS4 scores, how often they visit and how they respond to treatment.
The tabular model predicts them well. For patients on a biologic for more than 2 years, MAE is 0.95.

**Recent starters (< 9 months) have a thin record.**
Their visit scores do not yet show whether the drug works, so they are the hardest group to predict: tabular MAE is about 1.8 at 3 to 9 months and 2.2 under 3 months.
This is the gap that biological embeddings can fill.

**RNAseq is a pre-treatment baseline screen.**
Every RNAseq sample was collected on the day the patient started their current biologic line.
The transcriptomic embedding therefore captures how likely the patient is to respond to that drug before any clinical change can be measured.
The signal lies on a single latent axis that flags deep responders, whom the tabular model over-predicts by about 4 IHS4 points.
On these patients, RNAseq improves MAE by about **+0.38 to +0.39 points**.
The gain is largest under 3 months (+0.52) and disappears after about 9 months.

**Images are a noisy proxy of the same biology.**
In cross-modal tests, image embeddings recovered about 30% of the RNAseq response axis (R² 0.29).
Applied only to recent starters, the image correction gains **+0.134 MAE points**.

**Applying the embeddings to everyone leads to overfitting.**
On image-only patients, who are mostly long-term, a global image correction makes MAE worse by **−0.024 points** (95% CI excludes 0).
Their history already contains the flare severity the image shows, so the 256-dimensional embedding only adds noise.
The image embedding also strongly encodes the camera (`device_id` AUC 0.91).
However, device and capture date alone predict nothing, and centring the embeddings by device or batch leaves the results unchanged, so the gains are not a technical artefact.

**Conclusion.**
The gated approach applies the image and RNAseq corrections only to recent biologic starters.
It extracts the biological signal for the patients who need it most and leaves the rest of the cohort on the tabular prediction.

## Limitations

- Cross-validation uses random folds only, with no temporal validation.
- The response proxy (next-6-month IHS4 ÷ pre-biologic IHS4) is rough.
- test_2 has no targets, so its error can only be estimated.
- Few image-only patients are recent starters (about 28), so the image gain cannot be validated outside the RNAseq group.

## License

MIT, see `LICENSE`.
