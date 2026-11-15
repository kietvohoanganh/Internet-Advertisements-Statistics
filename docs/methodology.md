# Methodology

[Project overview](../README.md) · [Verified model comparison](../results/model_comparison.csv) · [Statistical summary](../results/statistical_summary.csv)

## Data and statistical analysis

The source labels are retained. Unknown geometry is preserved as `NA` for descriptive statistics and inference. Descriptive tables use the original rows; inference uses distinct labeled records to reduce repeated counting. Each test records its observed sample sizes. Deduplication does not establish independence between images from the same website.

For height, width and aspect ratio, Welch's test evaluates equality of class means while allowing unequal variances. Individual 95% mean-difference intervals are unadjusted for multiplicity; Cohen's d describes standardized effect size. Wilcoxon rank-sum is a distribution/rank sensitivity analysis, not an equivalent test of means. Q-Q plots and Shapiro-Wilk checks diagnose distribution shape without mechanically choosing a test.

Binary associations use Pearson chi-square only when all expected cells are at least five; otherwise Fisher's exact test applies. Degenerate tables are explicitly marked not estimable. Phi describes association strength. Benjamini-Hochberg adjustment covers four primary comparisons and, separately, three exploratory geometry-missingness comparisons.

Geometry Logistic Regression includes observed dimensions, `local` and nonconstant missingness indicators. Continuous-feature odds ratios correspond to one training standard deviation after imputation; the binary OR compares 1 with 0. Unstable Wald intervals are flagged and omitted from the published interpretation. Raw estimates remain in the generated tables. Statistical significance does not establish causation, practical importance or predictive usefulness. A separate ANOVA would duplicate the two-class mean comparison and is not added.

## Grouped modeling protocol

Identical full predictor vectors define groups; labels never enter the group key. A group stays entirely within train or test and within every inner/outer fold, even when its labels conflict. Training uses distinct labeled records. The historical split remains fixed and is development-exposed.

The benchmark evaluates geometry Logistic Regression, LASSO, Elastic Net, Random Forest, similarity SVM and pattern correction on matched outer observations. Median/mode imputation, standardization, frequent-size vocabularies and geometric bins are learned only inside the relevant fitting fold. Public dictionary names define fixed token groups. No full-sample statistical test selects predictive variables or thresholds.

The release uses five outer and three inner folds with seed 2026. Inner OOF predictions select hyperparameters and thresholds. Accuracy and precision-constrained F2 are separate experiments. Random Forest's six configurations cover tree count, maximum depth, minimum leaf size, feature subsampling and class weighting. Python tree construction is seeded; probability reductions run sequentially for repeatable scores.

Sparse penalized fits begin near the fold-specific null-model penalty and follow a dense warm-start path. Only the declared tuning penalties are scored. Requested solutions must be finite and complete, with a successful solver status. Incomplete paths are rejected, and convergence warnings remain visible.

## Selection, correction and uncertainty

The primary rule maximizes advertisement F2 among candidates meeting 90% precision on outer training-CV predictions. If no candidate qualifies, the fallback is recorded. Accuracy-only selection does not apply that constraint. This finite-sample condition is not a precision guarantee for new observations.

The SVM's training false-positive budget is supplied by Elastic Net. At inference, the selected SVM uses its stored similarity reference and margin threshold. Its sigmoid-transformed margin is a ranking score, not a calibrated probability.

Pattern correction can replace an Elastic Net decision only with sufficient pure training-pattern support for the SVM alternative and no blocking evidence for Elastic Net. Its support is chosen inside training. Paired group-bootstrap comparisons and outer-fold improvements determine whether it is eligible for deployment. It remains optional, and no ROC-AUC/AP is assigned to its final class decisions using the underlying Elastic Net probability.

F2 and precision intervals resample entire predictor-profile groups while holding fold predictions fixed. They exclude full retraining variation and domain shift. Outer CV results also inform candidate selection, so the selected model's estimate is not independent external validation. The selected SVM's historical-test performance does not improve on the retained accuracy-oriented legacy classifier; the README displays both contexts explicitly.

Training-only IQR filtering and clipping are sensitivity variants with the same evaluation observations. Neither is automatically deployed. Extreme image dimensions may be meaningful banner characteristics rather than errors. Random Forest importance permutes predeclared feature families within outer validation groups; it is diagnostic and never selects features.

## Published schema

`model_comparison.csv` contains the six primary-objective CV rows, the selected historical-test row, and the current legacy historical evaluation when that stage ran. Protocol labels separate these rows; intervals apply only to the corresponding CV estimates. Blank ranking scores for correction are intentional.

`statistical_summary.csv` is a tidy table with row kinds for dataset counts, primary tests, Logistic Regression effects, missingness tests and provenance. A phi estimate has no fabricated CI. Logistic OR intervals are Wald intervals; Welch rows contain mean-difference intervals. Provenance rows bind the configuration, analysis sources, software environment and four published figure hashes.
