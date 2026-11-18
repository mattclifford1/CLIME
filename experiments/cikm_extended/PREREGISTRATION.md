# Predictions for the explanations run

Written 2026-11-18, before `run_explanations.py` was first run, and not edited afterwards. The outcome of each is assessed in the paper's results section.

## What the run adds

The earlier runs score how well a surrogate's predictions agree with the black box. This run scores the explanation itself, the surrogate's coefficient vector, in three ways:

- **Ground truth.** For the black boxes with a gradient (logistic regression, RBF SVM, MLP), the cosine between the coefficients and the gradient of the black box's log-odds at the query point.
- **Invariance.** Whether the coefficients change when the black box's threshold is moved, which shifts its log-odds by a constant and leaves that gradient unchanged.
- **Stability.** Whether the coefficients change when the neighbourhood is drawn again.

Six surrogates share each neighbourhood: LIME, CLIME, Logit-LIME, Logit-CLIME, and the soft-label logistic surrogate with and without CLIME's class weights. Brier score and KL divergence are also reported class balanced, and as skill over a constant surrogate.

## Predictions

1. **CLIME does not move the explanation towards the truth.** On the SVM and the MLP, CLIME's cosine with the log-odds gradient at the query point is not above LIME's: the paired difference, CLIME minus LIME, is below +0.02 in all three settings. Reason: LIME's slope is the kernel-weighted mean of the probability gradient, which is concentrated at the decision boundary; CLIME's class weights move that mean further across the boundary, away from the query point.
2. **The logit surrogate is closest to the truth where the truth exists.** Logit-LIME's cosine with the gradient at the query point is above LIME's on the SVM and the MLP in all three settings. On logistic regression all six surrogates are above 0.99.
3. **Logit-LIME does not change when the black box's threshold moves; LIME does.** Between the plain and the re-thresholded black box at 200:20, Logit-LIME's coefficients have median cosine above 0.999 and median norm ratio within 1% of 1. LIME's coefficients grow: median norm ratio above 1.5 at query points whose neighbourhood was one-sided before the move. LIME's direction moves less than its scale: median cosine above 0.9.
4. **CLIME changes more than LIME when the threshold moves.** Its class weights follow the black box's classes, which the move changes: CLIME's median cosine between the plain and the re-thresholded black box is below LIME's.
5. **Each weighting wins its own loss.** By kernel-weighted Brier score LIME is better than CLIME; by class-balanced kernel-weighted Brier score CLIME is better than LIME. The sign of the difference flips in all three settings, on the test set and on the local sample. So neither score can say which weighting gives the better explanation.
6. **LIME's Brier skill collapses under imbalance.** LIME's skill over a constant surrogate, 1 − Brier/Brier of the constant, is lower at 200:20 than at 110:110, where its plain Brier score is better (lower).
7. **CLIME is less stable.** At 200:20, the cosine between the coefficients from two draws of the neighbourhood is lower for CLIME than for LIME, because CLIME's fit leans on the few samples of the rare class.
8. **Rebalanced training changes LIME's explanation more than it changes the model's gradient.** At 200:20 on the SVM and the MLP, the cosine between the plain and the class-weighted model's log-odds gradients at a query point is above the cosine between LIME's coefficients for the two models.

## What would change the conclusion

- If prediction 1 fails (CLIME is closer to the gradient by more than 0.02), CLIME improves the explanation and not only the score, and the paper's claim that it is mostly a threshold move is too strong.
- If prediction 3 fails for Logit-LIME, the cause is the clipping of saturated probabilities, and the share of such neighbourhoods is reported.
- Predictions 4, 7 and 8 are the uncertain ones.

## Setup

`run_explanations.py`: every dataset of the main run at its own ratio, 110:110 and 200:20; six model families; five seeds; the black box plain, class weighted, SMOTE trained, re-thresholded on validation data and placed by deltas (CP Minimax); the 20 line points and 10 test points per class of the main run. Differences are paired within (dataset, seed) blocks and averaged dataset first, as in the main run.
