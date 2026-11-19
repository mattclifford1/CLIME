# Predictions for the image run

Written 2026-11-19, while the plain networks of `run_images.py` were training and before any explanation was fitted, and not edited afterwards. One image's logits under every mask at 5000:50 had been seen while timing the code: all of them were on the majority side. The outcome of each prediction is assessed in the paper.

## What the run is

CIFAR-10, one class against the rest, as in the companion notes (`imbalanced-explanations`), with a ResNet-20 trained from scratch at 4000:4000, 5000:500 and 5000:50; ten classes, two seeds, 15 test images of each class per network.

- **Explained in patches.** The image is cut into a 4×4 grid of 16 patches. A mask keeps some patches and replaces the others by their mean colour. The surrogate's features are the 16 on/off indicators.
- **Every mask is evaluated.** The network's logit is stored for all 65,536 masks of each query image, so each surrogate's fit on all of them (its *exact fit*, the limit of infinitely many samples) is known, and nothing below depends on a held-out sample.
- **Six surrogates**, as in the tabular run: LIME, CLIME, Logit-LIME, Logit-CLIME and the soft-label logistic surrogate with each weighting. Masks are drawn with each patch on with probability 0.5 and weighted by `lime_image`'s kernel (exponential, width 0.25, on the cosine distance to the full image); 1,000 masks unless said otherwise, which is `lime_image`'s default.
- **Truth.** (a) The leave-one-out effect of each patch, logit of the full image minus logit with that patch hidden. (b) The exact Shapley values of the logit over the 16 patches. A threshold move adds a constant to the logit and changes neither.
- **Threshold moves** of the plain network: the prior shift log(ratio), the threshold maximising balanced accuracy on validation logits, and deltas (Clopper–Pearson minimax) on training logits.

## Checks that must hold exactly

- Logit-LIME's coefficients on a fixed set of masks are unchanged by any threshold move (cosine 1, norm ratio 1). A failure is a bug.
- Where every mask is in one black-box class, CLIME's coefficients equal LIME's.

## Predictions

1. **Neighbourhoods are one-sided under imbalance.** For the plain network at 5000:50, under 1% of the kernel-weighted masks are in the rarer black-box class at 80% or more of majority query images and at 50% or more of minority query images. The share over all query images is at least 20 points higher at 5000:50 than at 4000:4000.
2. **The published score saturates.** At 5000:50 LIME's fidelity over all masks is 0.95 or more on average and its balanced fidelity 0.65 or less. CLIME's fidelity minus LIME's is below +0.01.
3. **CLIME is not closer to the truth.** CLIME's cosine with the leave-one-out effects minus LIME's is below +0.02 at every ratio, paired over (class, seed) blocks.
4. **The logit surrogate is closer to the truth.** Logit-LIME's cosine with the leave-one-out effects, and with the Shapley values, is above LIME's at 5000:500 and at 5000:50.
5. **A threshold move changes LIME's explanation.** Under the prior shift at 5000:50, LIME's coefficients have a median norm ratio above 1.5 and a different top patch at 10% or more of query images. CLIME changes more than LIME: its median cosine between before and after is lower.
6. **The logit surrogate needs fewer masks.** At 1,000 masks and 5000:50, the cosine between the fit and the surrogate's own exact fit is higher for Logit-LIME than for LIME, and that gap is larger at 5000:50 than at 4000:4000. Reason: LIME's target there is a probability near zero, close to an exponential of the logit and far from linear in the patches.
7. **The logit surrogate is more stable; CLIME is not.** At 5000:50, between two draws of 1,000 masks, Logit-LIME's cosine is above LIME's and CLIME's is not above LIME's.
8. **Rebalanced training changes the truth.** At 5000:50 the leave-one-out effects of the class-weighted network and of the network trained with the logit-adjusted loss each have a median cosine below 0.9 with the plain network's on the same image. (Stage two of the run.)
9. **A threshold move works better on pretrained features.** At 5000:50 the prior shift leaves a smaller share of neighbourhoods one-sided for a fine-tuned ImageNet ResNet-18 than for the ResNet-20 trained from scratch. (Stage three.)

## What would change the conclusion

- If prediction 4 fails, the logit surrogate's advantage in the tabular run does not carry to a network explained in patches, and the paper says so.
- If prediction 6 fails, the tabular result on samples rests on the black box being linear in its inputs.
- If prediction 1 fails, CLIME has more to do on images than expected and predictions 2 and 3 are the ones to read.
- Predictions 5 (the sizes), 6 and 8 are the uncertain ones.

## Stages

1. Plain networks, all three ratios: predictions 1 to 7.
2. Class-weighted and logit-adjusted networks at 5000:500 and 5000:50: prediction 8.
3. Fine-tuned ResNet-18, at a reduced scale set by its cost: prediction 9.
