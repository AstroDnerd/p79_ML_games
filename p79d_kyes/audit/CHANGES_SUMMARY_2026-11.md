# p79d paper: what changed, 2026-11-01 to 2026-11-20 (by section)

High-level guide; details and numbers in `AUDIT_LOG.md`. Nothing here is committed by Claude.

**Whole paper**
- Every number now comes from the final 42-run models (five simulation-grouped folds per model, six models).
- Observations use a model trained on all runs, with the five fold models quoted as a range.
- Figures went from 15 to 13 by merging 9+10 and 14+15.

**Abstract, Introduction, Conclusion**
- Rewritten to match the results below:
  - offsets of 1.0–1.3 (¹³CO) and 1.6 (two lines on MWISP), and where the two-line excess comes from;
  - χᵥ treated as a run-level quantity;
  - dust at 0.3–0.4 times the CO value, no longer called a "lower limit".

**§2 Data**
- Final dataset of 42 runs, with three re-extracted.
- New statements: all runs share one forcing seed; the fold assignment and the spin-up snapshots are described.

**§3 Methods**
- New trainings:
  - a second seed of the main model;
  - an all-runs model whose uncertainty calibration comes from the pooled held-out residuals of the five folds;
  - mask-augmented ¹³CO and two-line models, as a test.
- Corrected wording on temperature scaling and MC dropout.

**§4.1 Mach number**
- Final scores: R² = 0.980, median error 7.4%.
- The second-seed training (7.5%) showed the per-run outliers are partly training scatter, so the "fold assignment" explanation was corrected.
- Training without mask augmentation (7.7%) showed it costs nothing.
- Leaving the spin-up frames out of the evaluation changes no score by more than 0.003.

**§4.2 Compressibility (χᵥ)**
- New within-run statistics.
- New baseline: nearest-neighbour regression on true Mach number and forcing (R² = 0.64), against which the network's 0.75 is compared.
- The second seed shows χᵥ depends on the training (R² 0.75 → 0.63).

**§4.3–4.5 Density head, column density alone, calibration**
- Per-fold coverage.
- Uncertainty split into its two parts with MC dropout (the epistemic part is small).
- New resolution test: the error goes from 7.4% to 7.6% at a 2.5× lower resolution.

**§4.6 Synthetic CO**
- New comparison with the plain linewidth estimate, also with the thermal width removed.
- The linewidth's excess at Mach 1–3 traced to ¹³CO opacity broadening, using 10.8k images.
- On held-out synthetic maps the network reads 1.15× the linewidth estimate.
- This week: the second-seed and mask-augmented CO models fall within training scatter. Figure 9 is now one Mach-error figure (by Mach bin, and against extinction); the χᵥ values per Mach bin moved into Table 2.

**§4.7 Non-LTE**
- Rerun with the final models, each predicting snapshots from runs it never saw. Table replaced.
- The two-line model's Mach number is robust, but its χᵥ is not.

**§5.1–5.2 Nearby clouds**
- Rerun with the all-runs models; Table 4 gives fold ranges.
- Sources added for the Serpens and Orion A cubes.
- Figure 11 rebuilt from the data with readable tile labels.

**§5.3 MWISP**
- Final ratios: 1.17 (¹³CO) and 1.56 (two lines).
- A new control on synthetic maps passed through the real-data processing path reproduces the ¹³CO ratio but not the two-line one.

**§5.4 Dust**
- "Lower limit" claim removed.
- New control: the column-density model applied to ¹³CO integrated intensity.
- This week: switched to the all-runs model (0.4 times the CO value; the fold models give 0.3). Per-region values updated; Figures 14 and 15 merged.

**§5.5 Offset**
- The ¹³CO offset is reproduced on synthetic maps (1.15). The processing-path control (0.98) rules out the upsampling.
- The two-line excess traced to the real ¹²CO linewidth map lying outside the training range (¹³CO only 35% detected). These values are now called extrapolations.
- This week: the mask-augmented CO models run on the real data give the same two-line excess (1.58) and a ¹³CO offset within the scatter between trainings, so partial detection alone is not the cause.

**§6 Discussion**
- Limitations added: single trainings, the shared forcing seed, scale dependence.
- Mask-augmentation and dust caveats updated.

**Bibliography**
- Eight entries verified or added; wrong citation keys repointed.

**Still open**
- Paper TODOs: package citations, the Herschel map releases, UAT IDs, and the decision on keeping the dust subsection.
- Thesis Ch. 3 refresh.
