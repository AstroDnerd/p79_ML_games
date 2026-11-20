# p79d audit log (auditor: Fable)

Batches 1-6 were read-only. On 2026-11-10 the user asked for the fixes to be made; what was changed and queued is in the
last section, "Changes made on 2026-11-10". Nothing was committed or pushed.

Severity: **A** = likely wrong, or a claim the paper's own numbers do not support (fix before submission);
**B** = a referee will probably raise it / needs a check or a caveat; **C** = minor, editorial, leftover.
"Verify" = I could not confirm from the paper alone; scheduled for a later batch.

## Batch status

| Batch | Scope | Status |
|---|---|---|
| 1 | Paper (`p79d_paper/.../*.tex`): internal math, physics, numbers, logic, ApJ content | **done 2026-11-09** |
| 2 | Code vs paper (net9010 pipeline, RADMC/LTE/LVG, obs prediction, numbers scripts, json configs) | **done 2026-11-09** |
| 3 | Opus session transcript + memory vs what the paper/code actually do (advisor doc not read directly) | **done 2026-11-09** |
| 4 | Thesis Ch. 3 (this project) vs paper and code (s6-s9 read in full; s1-s5 spot-checked) | **done 2026-11-09** |
| 5 | Thesis Ch. 2, Ch. 4, intro; cross-chapter consistency (all sections now read; see 5b and 5c) | **done 2026-11-09** |
| 6a | Paper commit `499534a` (final observation numbers) and code commits `be9820c`, `ae39518` | **done 2026-11-10** |
| 6b | Literature claims and bibliography metadata against the sources | **done 2026-11-10** |

State of the paper when audited: working tree with uncommitted edits (Nov 6-9); `main.pdf` is from Nov 3 and older than the tex.
All 57 numbers in Section 5 and 11 elsewhere are still `\prov` (pre-final models), so Section 5 numbers must be re-audited once final.

---

## Batch 1: paper

### A. Physics / logic problems

**P-A1. The "periodic box vs real cloud" explanation of the 1.1-1.5x offset is contradicted by Table 4.**
(abstract; 5_Observations §5.3, §5.5; 7_Conclusion item 6)
The offset is attributed to box-trained models meeting real clouds. But the projected-map model is equally box-trained and shows
almost no offset: Perseus 10.6/10.4 = 1.02, Ophiuchus 7.2/6.4 = 1.12, Serpens 12.5/13.9 = 0.90; only Orion A is high (1.35).
The offset is a property of the CO-trained models (1.14-1.33) and MWISP (1.33-1.48). So the cause is more likely something specific
to the RADMC training set (LTE, uniform abundance, fixed 10 K, fixed 4.6 pc, beam/tile differences) than periodicity.
The abstract and conclusion state "predictions ... are 1.1-1.5 times higher" without saying this holds only for the CO models.
The tiles also differ between models (Table 4 note), which the comparison does not control for.

**P-A2. "Below Ms ~ 3 the thermal width and the channel width dominate the line" is numerically wrong.** (4_Results §4.6; repeated in abstract/conclusion as "Ms <~ 3")
At Ms = 3, sigma_1D = 3 x 0.188/sqrt(3) = 0.33 km/s; 13CO thermal sigma = 0.053 km/s adds 1.3% in quadrature; a 0.066 km/s channel
adds less. Even at Ms = 1 (sigma_1D = 0.109) thermal adds ~11%. So thermal + channel cannot produce the quoted 20% linewidth error
at Ms = 1-3. §5.2 of the same paper says thermal broadening "matters only for Ms <~ 1", and §4.6 itself puts the limit at Ms < 0.5.
Three sections give three thresholds (0.5, 1, 3). The real cause of the 20% (single-sightline sqrt(3) sigma_LOS vs 3D rms, 3-sigma
masking clipping the wings, opacity broadening, volume- vs mass-weighting) should be identified. Verify in batch 2 whether
Ms_obs subtracts the thermal width in quadrature; if it does not, that is an easy fix that would improve the baseline further.

**P-A3. LVG paragraph contradicts itself on mechanism.** (4_Results §4.7; 6_Discussion "LTE radiative transfer")
(a) It says subthermal excitation "rescales the emission without much changing its velocity structure" and that the 13CO model is
insensitive "because the input transform removes the absolute intensity". Two sentences later the 12CO failure is explained by "a change in
the shape of the normalised intensity map". A pure rescale cannot change the normalised map; these are two different pictures.
(b) Arithmetic: W drops by 0.32-0.77 while "peak brightness" changes < 10% and linewidth <= 13%. Since W ~ T_peak x width this only
works if "peak brightness" is the map maximum, i.e. the contrast between diffuse and dense gas changed (not a rescale). Define the quantity.
(c) Text says intensities fall "by factors of 0.5-0.8"; Table 5 has 0.32, 0.45, 0.47 for 12CO.
(d) W13 ratio is 0.70 at n = 102 and 0.70 at n = 1000, 0.52 at n = 200: no density trend for the optically thin line, where the
subthermal effect should be strongest at low n. May be real (different snapshots, dense gas dominates emission) but needs a sentence,
and a code check (CMB background, LVG length scale) in batch 2.

**P-A4. Using the projected-map model for observed chi_v is a non sequitur.** (4_Results §4.6 last paragraph; §5.3)
§4.6: chi_v is recovered worse from CO because intensity-weighted moments of a saturating line differ from density-weighted
projections; "We therefore base the observational chi_v estimates on the projected-map model". The observed inputs are real 13CO
maps, which have exactly that problem, and the projected-map model was never trained on it. The lower R2 of the CO models is the
honest estimate of what CO maps contain. No cross-domain test is reported (projected-map model applied to synthetic 13CO maps,
chi_v R2). Also untested: whether partial detection/masking biases chi_v upward (§4.4 says the network reads chi_v from "larger voids,
sharper fronts"; masked regions look like voids). The observed chi_v = 0.34-0.53 all sit above the solenoidal value, which is what
such a bias would produce.

**P-A5. Degeneracy argument is applied asymmetrically and in a regime the paper says is invalid.** (4_Results §4.4, Fig. 4 caption; Conclusion item 2)
(a) The strongest anti-correlation quoted (r = -0.71) is at Ms < 1 and is attributed to the b*Ms trade-off of Eq. 1. §4.3 says Eq. 1
(and b) is meaningful only for supersonic gas, and §4.2 says chi_v is unconstrained (R2 = 0.00) there.
(b) The positive correlation at Ms > 3 is dismissed as a between-run artifact of one fold. The negative correlation at low Ms can be the
same kind of artifact: xi_0_mach5.5 is over-predicted in Ms by 1.6x and under-predicted in chi_v (0.97 -> 0.20), which alone
produces a strong negative r. A within-run (run-median-subtracted) correlation would separate the two; not reported.
(c) Bins are mismatched: column-density model quoted at "Ms < 1" and "1-3"; full-moment model at "0.5-1" and "1-3".
(d) The argument links b to chi_v ("it must underestimate b, and with it chi_v") without showing the b-chi_v relation.

**P-A6. Dust "lower limits" are not established as lower limits, and one of them is implausible.** (§5.4, abstract, Discussion §6.1, Limitations 5, Conclusion 8)
(a) Chamaeleon III "Ms > 11.8" is the highest of all ten regions, for a quiescent cloud whose CO linewidths (order 1-2 km/s FWHM)
imply a box-scale Ms of a few to ~8 at 10 K. A lower limit above the likely true value means the estimate is not a lower limit.
The background subtraction (first percentile, "up to a factor of 2 in Musca and Chamaeleon") is the probable cause.
(b) The control ("the column-density model on 13CO integrated intensity gives ratio 1.2, so the shortfall comes from the dust")
does not isolate the dust: 13CO W is not column density (saturates at high column, truncated/mask-filled at low column), so its
contrast differs from the true column for reasons unrelated to line-of-sight dilution.
(c) Unconsidered alternatives: the simulations are hydrodynamic, and magnetic fields narrow the density PDF at fixed Ms
(sigma_s^2 = ln(1 + b^2 Ms^2 beta/(beta+1))), so a hydro-trained column model reads a real magnetised cloud low; and Ms_obs from a
tile spectrum is inflated by systematic motions and multiple components. Either would give dust/CO < 1 without any dilution.
(d) "13 matched tiles of Perseus, Ophiuchus and Orion A" but Table 4 lists 6 + 4 + 4 = 14 for the projected-map model.
(e) The author's own red notes (§5.4 top, §6.1 end) say the explanation is uncertain and the section may be dropped, yet the abstract,
intro, limitations and conclusion 8 present it as a result. Decide and make consistent.

**P-A7. MWISP headline numbers describe 22 clouds, not 55.** (abstract, Conclusion 6 vs §5.4)
rho = 0.9 and the 1.33-1.48 ratio are for the 22 clouds with tiles <= 10 pc; the other 33 are excluded because the network saturates.
Tile size = 128 px x (50"/3.1) = 2064", i.e. 10 pc at 1 kpc, so "<= 10 pc" is essentially "d <= 1 kpc", which is largely the set
assigned 500 pc by fiat (V_LSR > -10 km/s). The statement "the distance enters only the physical size of a tile" is true for the inputs
but the distance decides which clouds enter the statistics.

**P-A8. Effective ceiling of the network is ~14-15, not 20, and the observed clouds sit near it.** (§5.3, §5.4 vs abstract "0.02 to 20")
Taurus tiles (spectra 13.0, 24.2) give 13.8 and 15.6 and are said to "saturate near the training maximum"; MWISP "saturates" above
13.5. Table 1 shows why: only two runs (mach160, mach320) have any snapshots above ~16, and those are partly cut at 20.
Perseus, Serpens and Orion A (Ms_obs 9-14, predictions 11-13) are therefore close to the saturation level; Serpens is under-predicted
by the projected model (12.5 vs 13.9). The paper should state the effective range and check whether saturation affects the nearby clouds.

### B. Referee-level concerns / needs a check

**P-B1. Transient snapshots are in the training and test sets.** (§2.1, Table 1)
Ms ranges include "the initial growth of the turbulence from rest": every run spans a factor ~5-8 in Ms (e.g. mach320: 8.4-47; mach1: 0.02-0.18).
The low end of each run is not developed turbulence (near-uniform density, large-scale forcing pattern). The quoted lower bound
Ms = 0.02 exists only as a transient of one run. A referee will ask how many snapshots precede ~2 turnover times and whether results
change without them. Also relevant to "within each forcing class chi_v follows ln Ms (|r| = 0.85-0.93)". Verify counts in batch 2.

**P-B2. Run 2 has fewer snapshots than the projected set despite a looser cut.** (§2.4 vs §2.5)
Projected: 35 337 images = 11 779 snapshots with Ms <= 20. Run 2: "10 937 snapshots with Ms <= 22". A looser cut with ~840 fewer
snapshots means some runs or snapshots are missing from run 2 (the count is close to the three re-extracted runs mach10/20/70, 1100 snapshots).
Table 3 does not give N per row, so it is unclear whether the CO rows are 42-run results. Also the 22 vs 20 cut is unexplained. Verify in batch 2.

**P-B3. LTE and uniform abundance underlie the "98% detected at n = 100" claim.** (§2.4, Discussion "Mask augmentation")
At n = 100 cm^-3 (A_V ~ 1.5) 13CO is strongly subthermal and, in real clouds, largely photodissociated. The conclusion that partial
coverage of real tiles "comes from cloud edges ... not from low mean density" is derived from LTE + constant X(13CO) and is not
independent evidence. The LVG check gives W13 ratio 0.5-0.7, which would already lower that 98%.

**P-B4. LVG check is small and outside the regime where the network is claimed to help.**
Five snapshots, Ms = 3.4-6.9, no retraining. The paper's claimed advantage over the linewidth is at Ms <~ 3 and for 12CO; the 12CO
model fails the check and no snapshot below Ms = 3.4 is tested.

**P-B5. "Run 1 saturates over most of the map" depends on Ms.** (§2.4)
With N(13CO) = 1.4e16 cm^-2 at 10 K, integral tau dv ~ 2 km/s. For Ms = 1 (FWHM ~ 0.26 km/s) tau ~ 8; for Ms = 10 (FWHM ~ 2.6 km/s)
tau < 1. The statement holds for low-Mach runs only. Also 7.59 K is J(10 K) without subtracting the CMB term (0.87 K); fine for RADMC
output, but a real spectrum would saturate at 6.7 K. Check in batch 2 whether a background is included.

**P-B6. b_lin: 14% median error vs an 18% systematic bias for two thirds of the suite.** (§4.3)
For xi >= 0.5 (28 of 42 runs) the true 0.28/0.28/0.33 are predicted 0.23/0.23/0.28, i.e. -18%; xi = 0 is -25%. A median absolute
fractional error of 14% at Ms >= 2 is hard to reconcile with that unless scatter is odd or the sets differ. Verify in batch 2.

**P-B7. kNN reference argument.** (§4.2 last paragraph; Discussion §6.2; Conclusion 3)
The in-sample kNN (k = 50, R2 = 0.97) mostly averages neighbours from the same run, so it shows chi_v is nearly constant along a run,
not that it is "a smooth function of Ms and xi". "The network ... does better than exact knowledge of both would allow" is a
contradiction as worded (the paper then says the reference is not an upper bound). The conclusion "much of the remaining error
comes from sparse sampling" rests on this.

**P-B8. Fold design.** (§2.5, §4.1, §4.5)
"Each fold contains about one simulation of each xi": with 7/7/16/6/6 runs and five folds it is 1-2 for four classes and ~3 for xi = 0.5.
The fold that over-predicts holds "the three runs near <Ms> = 6.5 with xi >= 0.5" (mach50, mach45, mach43), so all near-identical
flows at that Mach number were removed from training together; the 20-26% error and q_0.9 = 5.7 may be an artifact of the fold
assignment. Note also q is computed on the *calibration* fold, so q = 5.7 means that model is badly off on its calibration runs,
not (only) on the test runs; the text conflates the two. Verify fold lists in batch 2.

**P-B9. Conformal guarantee.** (§3.4, §4.5)
Exchangeability holds at the level of simulations (8 per fold), not images, so the finite-sample guarantee is that of n ~ 8, not n ~ 7000.
The paper half-says this ("exchangeable only approximately"); a referee may want it stated. Calibration improved four folds only
from 0.04-0.10 to 0.04-0.08, so "calibrated uncertainties" (abstract, intro) is generous; coverage at Ms < 0.5 is 55%.

**P-B10. Resolution channel is advertised, then shown to be unnecessary.**
Intro P5: "a resolution-aware input channel lets one network serve observations of different angular resolution". §4.5 and §6.5: setting
the channel to 1 is as good or better. Reword the intro so it credits the degradation augmentation.

**P-B11. No resolution/convergence statement for the labels.** sigma_s, b and chi_v at 256^3 are compared with Federrath et al. values
(b ~ 1/3, 1) obtained at higher resolution; §2.5 says shocks are under-resolved above Ms = 20, Limitations says "marginally resolved"
at Ms >~ 10. One sentence on how the labels depend on resolution (or a single 512^3 check) would pre-empt this.

**P-B12. Taurus uses the model the paper declares unreliable.** §4.7: "we use LTE-trained models only with 13CO in the input";
§5.1/Table 4: Taurus is predicted with the run-2 12CO-only model, on 14 pc tiles outside the domain. It adds nothing and invites criticism.

**P-B13. Literature details to verify against the sources** (LLM-drafted summaries are a common place for errors):
Peek & Burkhart 2019 ("density slices", "98% even after Fourier amplitudes are removed"); Schmaltz 2025 (32x32 sub-fields, six
simulations, Ms = 2.2-11, errors 0.5 and ~2); Hu 2025 (Ms ~ 0.25-1, two HI clouds); Zhang 2026; "Both report errors several times
larger on a withheld simulation"; Shimajiri 2014 velocity resolution (paper says 0.5 km/s); Padoan 2003 "Ms ~ 14" for Perseus;
Arce 2010 "5.4-7.6", "T_ex = 10-15 K". Also the data sources recorded in the Opus memory (Taurus 13CO Panopoulou 2014, Ophiuchus
White 2015, Serpens 2021A&A...646A.170G) differ from those in the paper (COMPLETE for Ophiuchus/Serpens, CfA 12CO for Taurus). Batch 3.

**P-B14. 13CO abundance differs between run 1 (1e-6) and run 2 (1.5e-6)** with no reason given; Limitations says "one temperature
and abundance". Orion A is run at T = 20 K through models trained only at 10 K (opacity and brightness scale differ).

### C. Minor / editorial

- **P-C1.** Leftover author notes that must go: `\red{(from my understanding of FT of v)}` (§4.2), `\red{Maybe we can skip this one?}` (Fig. 7 caption),
  `\red{Talk about the flattening ...}` (§5.4), `\red{These results were ok ...}` (§5.4 dust), `\red{Provided my juistification ...}` (§6.1),
  `\red{(This part reads more like an introduction ...)}` (§6.4). 13 `\TODO`s remain (forcing band k = 2, citations, figures, table check).
- **P-C2.** Table 2 caption TODO mentions "net4003-net9009 entries" but net4003 is not in the table; and Discussion "Logarithmic Mach target"
  TODO asks for "R2 ~ 0.49 -> 0.97 if confirmed in the logs". Both are placeholders for numbers from the earlier nets.
- **P-C3.** Discussion §6.2: "5-16 [runs] per value" of xi; Table 1 gives 6-16 (7, 7, 16, 6, 6).
- **P-C4.** Fig. 3 caption: n = 100 cm^-3 "(A_V ~ 1.3)"; text says mean A_V = 1.5 at n = 100. Presumably the median; say so.
- **P-C5.** MWISP channel width 0.16 km/s (§2.4) vs 0.166 km/s (§5.4).
- **P-C6.** Linewidth-estimator precision: "~20-30%" in the intro (with a citation TODO) vs "15-30%" in §5.5.
- **P-C7.** Intro P3: Ms = sigma_v / c_s(T_ex): the sound speed depends on the kinetic temperature (T_ex is a proxy only when thermalised),
  and the 1D linewidth needs sqrt(3) to match the 3D definition given two paragraphs earlier.
- **P-C8.** "Compressive modes at low Mach number radiate away as sound waves" (§2.3, §6.2): in a periodic box nothing radiates away; and the
  suite's own subsonic xi = 0 runs keep chi_v = 0.84-0.97. Reword (compressive motions persist as sound waves under compressive driving and are
  not generated under solenoidal driving).
- **P-C9.** "above three times the noise in a run" (§2.4, §5.2): "run" = contiguous channels; the minimum length is never given.
- **P-C10.** §5.3: Orion A tiles "1-35% above" Ms_obs while the Table 4 medians give 12.7/9.4 = +35%, the top of that range. Recheck when final.
- **P-C11.** Abstract says Mach numbers "0.02 to 20", intro "0.02-47"; consistent with the cut but worth one phrase.
- **P-C12.** Title names only "Synthetic 13CO Observations"; the paper's main results are on projected maps, plus 12CO, real clouds and dust.
- **P-C13.** §4.2 "A question we ask is if the run-level error a limitation ..." (grammar). §4.3 heading has body text on the same line.
- **P-C14.** Results header comment cites `kfold_net9010.py`, which is not in `p79d_kyes/` (only `analyze_kfold_net9010.py`, `orchestrate_net9010.py`).
- **P-C15.** "1-2% per free-fall time" is cited to Krumholz 2005/2007 as an observation; fine, but Krumholz & McKee 2005 is the theory paper.
- **P-C16.** Acknowledgement grant AAG-1616026 is a 2016 award; confirm it is still the right one. Keyword UAT IDs flagged TODO.
- **P-C17.** `figures/fig_xi_confusion.png` is unused (leftover from the xi-classification version).

### Checked and found consistent (no action)

Table 1 totals (42 runs; 12 376 snapshots; 30 runs with 1 <= <Ms> <= 20; 32 runs reaching Ms >= 1); Eq. 3 values (1, 0.82, 0.33, 0.05, 0);
strain-rate identity; Eq. 6 as the inverse of Eq. 1; chi = 1/3 for a random field; A_V = 15 for n = 1000 over 4.6 pc and 1.5-45 for run 2;
J(10 K) = 7.59 K; c_s = 0.188 km/s; beam = 3.1 pixels; network dimensions and 1.23e7 parameters; conformal quantile formula; Eq. 13 and the
15-30% range; Table 3 chi_v ranges vs text; q_0.9 -> rescale factors; epistemic variance share (~11%); Taurus tile size (14 pc);
CO cloud-median ratios 1.1-1.3; the Brunt & Federrath projection argument; all \cite keys exist in bibliography.bib.


---

## Batch 2: code vs paper

Method: read `networks_nbisht/net9010.py`, `ViT_setup_test_net9010.py`, `extract_sim_data.py`, `extract_sim_co.py`,
`extract_sim_radmc.py` (moment/IO parts), `get_Observation_Data.py` (moment/mask parts), `predict_observations_net9010.py`,
`predict_dust_net9010.py` (logic), `paper_numbers_net9010.py`, `analyze_kfold_net9010.py`, `compare_lvg_lte.py`, `robustness_net9010.py`
(header), all json configs, `models/orch_final.json` and the final training logs. I re-ran `paper_numbers_net9010.py --no_linewidth`
(read-only, output to my scratchpad) and a small script of my own on the saved `net9010_final_*_test.npz` files and the dataset labels.
Not read in detail: `make_paper_figures.py`, `orchestrate_net9010.py`, `process_mwips_*.py`, tile-selection code, plotting code.

### Updates to batch-1 items

- **P-A5 (degeneracy) -> downgraded to B, with a better number available.** Removing each run's mean residual, the
  anti-correlation survives and is stronger for the column-density model: within-run r = -0.85 (Ms 0.5-1), -0.76 (1-3), -0.15 (3-7);
  full-moment model -0.23, -0.23, -0.02. So the trade-off is real at Ms 0.5-3. Below Ms = 0.5 it is a between-run artifact
  (raw -0.54 / -0.28, within-run -0.20 / +0.30). The paper should quote the within-run values, use the same bins for both models
  (script prints 0.5-1: -0.84 vs -0.37; <1: -0.71 vs -0.47), and drop the Ms < 0.5 range from the claim.
- **P-A6 (dust) -> mechanism found, see C-A3.**
- **P-B2 resolved as a stale sentence.** All six final models have 35 337 test images from 42 runs, Ms max 18.95. The sentence "Run 2 covers
  10 937 snapshots with Ms <= 22" (2_Sim_and_Data) is from before the three runs were re-extracted; the config still has `ms_max_frame` 22
  but training cuts at 20. Also the dataset has 12 374 frames; Table 1 sums to 12 376.
- **P-B6 (b_lin) reproduced as printed** (median error 0.140; class medians 0.275->0.226 etc.). Not an error, but the systematic
  -18% / -25% offsets are the main error and the text should lead with them.
- **P-B8 (folds) confirmed.** `assign_sim_folds` deals runs round-robin by Ms within each xi, and the suite has ~5 runs at each Mach level,
  so fold 4 = {xi_0_mach65, xi_0.5_mach50, xi_0.75_mach45, xi_1_mach43, + mach1, mach10, mach12, mach320}: four of the five runs at <Ms> ~ 6.5.
  With fold 4 held out (and fold 0 used for calibration) the training set has one run between <Ms> = 5.5 and 9.5. The 20-26% error is a
  fold-design effect. Folds 0 and 1 have 9 runs, the others 8. The model's q_0.9 = 5.65 and its raw coverage error 0.348 are both real.
- **P-B1 (transients) quantified.** 8.1% of frames (997 of 12 374) are at t < 2 crossing times; runs last ~20-25 crossing times, except
  xi_1_mach80 (38 frames, 2.8 crossing times, 71% transient) and xi_0_mach100 (117 frames). Early chi_v differs from the late value
  (xi = 0.25: 0.63-0.80 early vs 0.30-0.46 late; xi = 1: 0.07 vs 0.19). The script's "worst run at Ms 0.5-1" is a transient:
  xi_0.25_mach16 with true chi_v = 0.80 predicted 0.08. So the paper's account of that bin ("rests on a single run, xi_0_mach10")
  is incomplete.
- **Checked OK:** chi_v from ln Ms alone (leave-run-out kNN) gives R2 = -0.54 at Ms >= 1, so the network's 0.75 is not a by-product of Ms.

### A. Code contradicts the paper, or changes a conclusion

**C-A1. chi_v is effectively a per-run constant, which is what the paper criticises about xi.** (`paper_numbers` output; 6_Discussion §6.2)
At Ms >= 1 the within-run variance of chi_v is 3.1% of the total (a perfect run-mean predictor would score R2 = 0.969); the median
within-run correlation between predicted and true chi_v is 0.17; the per-run regression slope is 0.76; replacing every prediction by
its run median raises R2 from 0.753 to 0.838. None of these printed numbers is in the paper. The Discussion says xi is a poor target
because "a suite of 42 runs provides only 42 independent labels" while chi "var[ies] from snapshot to snapshot". By the paper's own
numbers the network identifies the run's chi_v level (about 32 independent labels) and does not follow its variation.

**C-A2. The uncertainties are not combined as the abstract and §3.4/§4.5 say.**
- Conformal scores and intervals use the sigma of one deterministic pass (aleatoric head only): `compute_conformal_scores`,
  `build_conformal_intervals`, `predict`. MC dropout is never used for the intervals.
- The "raw Gaussian intervals, built from the total variance of Equation 12" (§4.5) are built from the aleatoric sigma only
  (`paper_numbers` 4.4, `plot_coverage_calibration`).
- MC dropout is a separate diagnostic on 1500 images per fold (`robustness_net9010.py`). `mc_forward` switches on every `nn.Dropout`
  (attention, projection, MLP, positional and residual), not only "the residual-branch dropout layers" (§3.3, §3.4).
- The "learnable temperature (Guo 2017)" is an ordinary parameter trained jointly with everything else by AdamW (it ends at
  log T ~ -0.02, i.e. T ~ 0.98). It is a redundant bias on the log-variance, not post-hoc temperature scaling.
- The per-fold "after calibration" coverage error (0.04-0.08) is averaged over 3 levels (80/90/95) while the "before" error
  (0.04-0.10) is over 10 levels (0.5-0.95); §3.4 says both use 0.5-0.95. Conformal made folds 0 and 1 slightly worse (0.043->0.055,
  0.063->0.080). Per-fold 90% coverage ranges 0.85-0.97 for Ms and 0.69-0.94 for chi_v.

**C-A3. Dust background subtraction manufactures contrast; this is the likely origin of Chamaeleon III "Ms > 11.8".**
`predict_dust_net9010.py`: `with_bg(t, bg) = max(t - bg, 0.05*bg)` with bg = 1st percentile of the map, then the model sees
ln(Sigma/mean). A pixel at the background becomes 0.05 bg while one at 2 bg becomes bg: a factor 2 in column turns into a factor 20.
For low-contrast maps (Musca, Chamaeleon) most of the tile is near the floor, so sigma(ln N) and hence the predicted Ms inflate. The
"lower limit" quoted is this larger, background-subtracted value. Also: Aquila is placed at 260 pc while Serpens (same complex)
uses 436 pc; Chamaeleon beams are "check the header" assumptions.

**C-A4. LVG table: pre-final models, evaluated in-sample, and it contradicts the 98% detection claim.** (`compare_lvg_lte.py`, `~/plots/lvg_check`, `radmc_co_lvg.json`)
- Table 5 is the Oct 28 output for tags `co2_13/co2_12/co2_both` (pre-final), `foldall` models. It is not marked `\prov`. Final models
  (`final_co13` etc.) exist and the check has not been re-run with them.
- `foldall` trains on every run except calibration fold 0, so the LTE versions of all five test frames were in its training set.
  The LTE predictions are in-sample; the out-of-fold models were not used.
- Four of the five frames (mach65, mach50, mach45, mach43) are fold-4 runs, three of them the runs the paper singles out as badly predicted.
- At n = 102 the 13CO detected fraction is 0.68 (LTE) and 0.60 (LVG) for xi_0_mach65, against "Even at n = 100 the FCRAO-like setup
  detects 98% of the 13CO pixels" (§2.4, Fig. 3). The 98% is one solenoidal-mix run; compressive runs have large undetected voids even in LTE.
  This weakens "the partial coverage of real tiles comes from cloud edges ... not from low mean density".
- "Peak brightness" in §4.7 is the map maximum of the noise-free cube (`rt/Tpeak`), confirming P-A3(b).
- No CMB/background term in the RADMC setup (`radmc3d.inp` has none; `read_image` converts I_nu directly). For LVG this removes
  radiative excitation by the 2.7 K field, and observed spectra are background-subtracted. State it or test it.
- A directory `bad_lvg_lte_bug` exists in the dataset folder; batch 3 will check from the transcript that Table 5 post-dates the fix.

**C-A5. Seven final trainings were stopped by a wall-clock budget, not by the stated criterion.** (`models/logs/*final*`)
§3.4: "training stops after 15 epochs without improvement". `--time_budget_h` stopped final_proj fold 4 at epoch 67 (its best epoch
was 66, i.e. still improving), final_mom0 fold 1 (epoch 75), final_coboth folds 3 and 4, and the `foldall` models for mom0, co13 and co12
(the ones applied to observations). The other final_proj folds stopped early at 21-45 epochs. Not necessarily harmful, but it is
unreported, and the worst fold is one of the truncated ones.

### B. Mismatches a referee could find, or that affect interpretation

**C-B1. Mask augmentation: only the projected full-moment model has it.** `orch_final.json`: `--mask_aug_prob 0.5` only for `final_proj`.
`final_mom0` and all four CO groups are trained without it. So (a) the column-density vs full-moment comparison differs in augmentation
as well as input; (b) §3.4's "In the two-line mode both lines share the field and the 12CO map loses 0.3q" describes code that the final
two-line model never ran; (c) the column-density model applied to partly detected 13CO integrated-intensity tiles (the dust "control",
ratio 1.2) has never seen a mask.

**C-B2. Linewidth baseline (`paper_numbers` lines 226-252).** Ms_obs = sqrt(3) x sigma of the masked moments, compared with the
volume-weighted 3D Ms. No thermal deconvolution (13CO sigma_th = 0.053 km/s), no correction for the 3-sigma clip. That explains errors
below Ms ~ 1, not at 1-3 (see P-A2). A fairer baseline subtracts sigma_th^2 and compares with `Ms_los_mw`-based truth as well.

**C-B3. Synthetic noise is white per pixel; real noise is beam-correlated.** `_reobserve_file` convolves the cube with the beam, then adds
independent Gaussian noise of 0.2 K to every 128^2 pixel (3.1 pixels per beam). Real COMPLETE maps have 0.19 K per 23" pixel with a
46" beam, and the pipeline then upsamples them by 1.55 (bilinear), which correlates the noise further. The noise texture and the
effective sensitivity per beam differ between training and real tiles; relevant to detection fractions and to the CO-model offset (P-A1).

**C-B4. Effective suite is 40 runs above the cut.** xi_0.5_mach160 contributes 15 images and xi_0.5_mach320 6 images with Ms <= 20, all
spin-up frames. They count toward "42 runs" and "32 runs that reach Ms >= 1". On them the column-density model predicts 0.07x and 0.06x
the true Ms and the full-moment model 0.41x for mach320; §4.4 says the column-density model's largest error apart from the subsonic runs
is a factor 1.8. Also §4.2 "chi_v = 0.18-0.26 for every run with xi >= 0.5": the script prints [0.201, 0.383] for xi = 0.5 because of
these two runs.

**C-B5. Resolution semantics.** In training, r is a bilinear down-and-up resampling factor (no anti-alias filter, `_degrade_resolution`).
On real data r = clip(native_beam_pix / beam_pix, 0.4, 1): a Gaussian beam of 2 pixels is declared as r = 0.5. These are different
degradations. Since §4.5 shows the channel is ignored it does not change results, but the description in §5.2 should not imply equivalence.

**C-B6. RADMC details not stated in the paper.** No microturbulence; line profile function is purely thermal, on a 256^3 grid with
piecewise-constant velocities and no Doppler-catching option, so at high Mach number cell-to-cell velocity jumps exceed the local
line width (0.05 km/s). The brightness is a Rayleigh-Jeans temperature. o/p-H2 = 0. Run 1 used a 45" beam (`radmc_13co.json`), the
paper says 46". Worth two sentences in §2.4.

**C-B7. Observation pipeline details.** The W floor for filling masked pixels is the 1st percentile over the whole cloud map for
real tiles (`floors`) but per image in training (`obs_to_training_units`, `mask_augment`). MWISP cubes are pre-masked at 2 sigma by
DBSCAN and then masked again at 3 sigma; all 57 clouds are assigned 10 K; 18 clouds use the 500 pc placeholder (so the "22 clouds
with tiles <= 10 pc" are mostly these, as suspected in P-A7).

**C-B8. One training seed.** `set_seed(8675309)` for every fold and model; no seed-to-seed scatter is available, so differences such
as 7.4% vs 7.8% (mask augmentation), or 0.75 vs 0.74 (chi_v, full vs column), cannot be called significant or not.

### C. Minor

- **C-C1.** `Attention.forward` applies dropout to the attention logits before the softmax (`self.attn_drop(logits).softmax()`), not to the
  attention weights. Works, but is not standard and not what "dropout within the attention" usually means.
- **C-C2.** Epistemic sigma uses `mu.std(0)` (N-1) vs Eq. 12 (1/N). Negligible.
- **C-C3.** `obs_clouds.json` notes still say Taurus tile "~39 pc" (paper: 14 pc), and list references (Panopoulou 2014, White 2015,
  Gong 2021) as "not the data used", which matches the paper but not the memory note of 2026-10-12. Batch 3.
- **C-C4.** 4_Results header cites `kfold_net9010.py`; the scripts are `ViT_setup_test_net9010.py`, `analyze_kfold_net9010.py`, `paper_numbers_net9010.py`.
- **C-C5.** `moment_maps` normalises by N only (assumes rho0 = 1); Eq. 4 has rho0. Fine for this suite.
- **C-C6.** `predict_dust` default `--co_tiles_dir` is `observations_net9010_projmask` and default `--tag mom0`: pre-final names.
  Confirm the final dust run used the final tiles and `final_mom0`.

### Verified against the code (no action)

Eq. 5 (Ms), Helmholtz decomposition and chi_v / chi_w, theta2 / omega2, b and b_lin, block-summed moments (Eqs. 4-6 of §2.2), periodic
shifts shared between datasets, input transform (Eq. 8), architecture and parameter count, loss (Eq. 10, weights 1/2/3 at Ms 4 and 10),
AdamW / LR schedule / patience, augmentation probabilities and ranges, split logic (test fold, next fold for calibration, last 15% of
frames with a 5-frame gap for validation), conformal quantile, A_V conversion (N_H2 / 9.4e20), sound speed, run-2 abundances and
density sampling, FCRAO/MWISP re-observation parameters, contiguous-channel masking, kinematic-distance formula, tile-spectrum Ms_obs,
and every number of Table 3 and §4.1-4.6 that `paper_numbers_net9010.py --no_linewidth` prints (all reproduce).


---

## Batch 3: Opus session (transcript `4ddc5c70...jsonl`, 2026-09-23 to 2026-11-09) and memory notes

Method: extracted the 56 user messages and 339 assistant text messages (tool output excluded); read all user messages (truncated),
the compaction summaries of Nov 4 and Nov 6, and in full the assistant reports of Sep 26, Oct 1, 5, 19, 21, 23, 28, 29, 30, Nov 2, 4, 6, 9.
Earlier reports were keyword-searched only. The advisor Claude Doc was not opened; I relied on what the session reported writing into it.

### Open items from earlier batches, now settled

- **LVG bug: Table 5 post-dates the fix.** On Oct 29 Opus found the first LVG run was bit-identical to LTE (`prepare_populations` truncated
  `radmc3d.inp`), fixed it, moved the bad output to `bad_lvg_lte_bug/`, and reran; `~/plots/lvg_check/lvg_lte_summary.txt` (Oct 28 22:07 local)
  is the corrected run. C-A4's other points (pre-final `co2_*` models, `foldall`, in-sample) stand.
- **Data sources vs the 2026-10-12 memory note.** The note lists the papers the user used as references, not the data used. Opus checked them
  on Oct 5 and recorded that Orkisz 2017 is Orion B, White 2015 is J=3-2, Gong 2021 is one filament, and Panopoulou 2014 uses the FCRAO 13CO
  Taurus map (no public cube found, so CfA 12CO stayed). The paper cites none of these as data. No inconsistency; the memory note's wording
  ("data sources") is misleading and could be reworded.
- **13 vs 14 matched dust tiles (P-A6d):** one Ophiuchus tile has no dust coverage (Opus' Oct 28 table has 6 + 3 + 4). Say so in the paper.

### A. Things said or found in the session that the paper does not reflect

**S-A1. The dust "control" fails in Orion A, and the paper leaves that out.** (Oct 28 report; 5_Observations §5.4)
The column-density model on the 13CO integrated intensity gives 5.4, 4.4, 2.0, 5.0 on the four Orion A tiles against Ms_obs = 13.7, 8.8,
9.4, 9.5, and the dust gives 4.8, 3.4, 2.4, 2.5: in Orion A the column-density model is low on CO as well as on dust, so there the
shortfall is the model or the resolution (Opus: "the NRO beam puts the resolution setting at its 0.4 lower limit"), not line-of-sight
dilution. The paper quotes the control only for Perseus and Ophiuchus ("median ratio 1.2 ... so the shortfall comes from the dust maps
rather than from the model") while the headline "median 0.5 (0.2-0.8) on 13 matched tiles" includes the four Orion A tiles.
Opus also listed a second cause the paper dropped (13CO is absent from diffuse gas, so its maps have more contrast than dust), and
noted that the dust-only Ms grows with tile size (0.8-4.5 pc), so the ten "lower limits" mix scale with turbulence.

**S-A2. Orion A is outside the trained resolution range for the projected-map model.** (Oct 28 report; `predict_observations` line 175; thesis Table "Observed clouds")
NRO pixels are 7.5" with a 25.8" beam, i.e. 3.44 pixels per beam. The code never downsamples (`up >= 1`), so r = clip(1/3.44, 0.4, 1) = 0.4,
with the true value 0.29 below the training minimum. Table 4's caption and §5.2 say the projected-map model sees every map at
"two pixels (r = 0.5)". That is true only for the three FCRAO clouds (Taurus is r = 0.89). Orion A is also the one cloud with 0.5 km/s
channels (2-channel masking) and T = 20 K, and it is the cloud where the projected model is 35% high. Likewise the CO models see
Orion A at r = 0.90 without upsampling, not "upsampled to their training beam (r = 1)".

**S-A3. The 35% Orion A excess is the `foldall` model; the five fold models are close to Ms_obs.** (thesis s8 "Current results"; Oct 21 report)
Thesis: fold models give 9.7-11.3 in Orion A "all below the final model's 12.7" (Ms_obs 9.4, i.e. ratios 1.03-1.20); 6.4-10.5 in Ophiuchus
(ratios 1.0-1.6); 10.6-12.3 in Perseus. Opus noted on Oct 21 that "an earlier final model sat about 20% off its own folds". So
independently trained models of the same kind differ by 20-60% on the same real tiles, which is as large as the 1.1-1.5 offset the
paper interprets. Table 4 reports a single model per column with no spread. (To be redone anyway when the \prov numbers are replaced.)

**S-A4. The observed chi_v depends on mask augmentation and on the detected fraction.** (thesis s8 "Compressive fraction")
"The projected model without augmentation had given 0.33-0.37 for every cloud"; with mask augmentation 0.34-0.53, "the highest values
in Ophiuchus, which has the most partly detected tiles"; the CO models give 0.29-0.71. The paper reports only 0.50/0.53/0.34/0.35 and
reads them as lying between natural and compressive driving. This confirms the concern of P-A4: the cloud-to-cloud differences in
chi_v track masking, not turbulence. The thesis says so ("not yet robust"); the paper should not present cloud values.

**S-A5. The simulated-MWISP control does not exercise the code path where the real-data bug was.** (Oct 29 reports)
The "median ratio 1.05 on simulated MWISP re-observations" is used in the paper (§5.4) and thesis to rule out the survey setup as the
cause of the offset. Those synthetic maps are at the simulation's own pixel scale and are fed directly. Real maps go through
`regrid()`: nearest-neighbour velocity fill, then bilinear upsampling of the moment maps by 1.5-1.9, which is where the v = 0 bug
was (fixed Oct 29) and which the projected-map model (no offset) never uses. A control that block-averages a synthetic cube to
1.7-2 pixels per beam, adds noise, and runs the full real-data path has not been done. This is the cheapest test of P-A1.

**S-A6. LVG would have cost 2.5x LTE, not 50x.** (Oct 30 report; thesis s7 "How large is the LTE bias?")
With populations solved once per frame and 15-level molecule files, an LVG frame takes ~25 min against ~10 min for LTE. The thesis says
this; the paper says only that the solver "as first configured" cost >50x. A referee who learns the 2.5x figure will ask why run 2 is
LTE. Either state it and argue from Table 5, or render run 2 in LVG.

### B. Process points worth knowing

- **S-B1. Results swing between retrainings by more than the effects discussed.** Column-density model: 23.5% (41 runs, Oct) -> 16.1% (42 runs, Nov),
  with the Ms 3-7 bin going from 69% to ~10% and "three times worse" becoming "two times". chi_v R2 by bin with/without mask augmentation:
  -0.50 -> 0.63 (Ms < 0.5), 0.39 -> 0.09 (0.5-1). Run-1 13CO chi_v at Ms 7-20: -0.18 -> 0.79 after a masking change, and 0.60 -> -0.26 at Ms 1-3.
  kNN reference 0.78 -> 0.72 -> 0.64. With one seed (C-B8) none of these can be attributed to the stated cause.
- **S-B2. Opus recommended on Oct 30 that "the Ms headline should go, since the linewidth matches it for 13CO"** and called the real-cloud
  excess "a network bias". The abstract still leads with R2 = 0.98 and the title still names 13CO. This is the user's and advisor's call;
  noted because the session's own assessment is more cautious than the paper's framing.
- **S-B3. Earlier networks (net9001-net9009) were trained on the old data products** with column density transposed relative to the
  velocity maps for two of three lines of sight, variance instead of dispersion in channel 2, and half of each run duplicated
  (Oct 1 report). Table 2 of the paper labels their scores only as "an earlier, smaller dataset with image-level splits".
- **S-B4. Decisions that were made and are reflected correctly:** raw chi_v, no Mach cut on the chi loss, density head (Oct 28);
  leakage evidence in the thesis only; no dust radiative transfer, with the reason given; dust as lower limits (user, Oct 28);
  Serpens from the COMPLETE cube; tiles <= 10 pc for MWISP.
- **S-B5. Standing thesis fixes Opus listed but has not made** (left for the user's manual pass): `s6_results.tex:210`, the s8
  "observational floor" paragraph, s9 summary item 7. See batch 4.

---

## Batch 4: thesis Chapter 3 (`~/thesis/6669b1f35be7cdc2643f33d9/chapter3`, last commit 2026-11-02)

State: the chapter describes the pre-final models (41 runs, 32 313 images). §s9 says so. Every number therefore differs from the paper;
I list only what is wrong or contradictory beyond that. s6-s9 were read in full; s1-s5 were spot-checked (no mention of lost data or
of net9005-8, consistent with the memory note).

### A. Errors and contradictions

**T-A1. Wrong 13CO thermal width.** `s7_radmc.tex`, linewidth subsection, conclusion 2: "the thermal width (0.19 km/s at 10 K for 13CO)".
0.19 km/s is the isothermal sound speed for mu = 2.33. The 13CO thermal dispersion at 10 K is 0.053 km/s (the paper's §5.2 has 0.05).
This is probably where the paper's "below Ms ~ 3 the thermal width dominates" (P-A2) comes from.

**T-A2. Larson relation misquoted.** `s7_radmc.tex` "Physical scaling": "sigma_1D ~ 0.7 (L/pc)^{1/2} km/s (Larson 1981)". Larson (1981) gives
sigma = 1.10 L^0.38 (three-dimensional). The 0.7 L^0.5 form is Solomon et al. (1987) (with radius) / Heyer & Brunt (2004). The derived
"L ~ 0.023 Ms^2 pc" follows from the quoted form, so only the citation and the 1D/3D statement need fixing; the paper's Fig. 13
caption also cites Larson (1981), so check which form that figure plots.

**T-A3. chi_v "varies from snapshot to snapshot, so there are thousands of distinct labels"** (`s6_results.tex:27`) against, in the same file
(line 210), "the network identifies runs, not snapshots", and against the final numbers (within-run variance 3%; C-A1).
Line 210 itself still has the "0.5 in logit" and "close to the ceiling" claims that Opus retracted on Nov 4 (true spread ~0.15 in logit;
a run-mean predictor would score 0.97).

**T-A4. Dust control overstated.** `s8_obs.tex` dust paragraph: "The same column-density model applied to the 13CO integrated intensity of
these tiles gives values close to the CO spectrum, so the model is not the problem". False for the four Orion A tiles (S-A1), whose
dust ratios (0.26-0.39) are in the same table.

**T-A5. "Even at n = 100 the mocks are almost fully detected"** (s7 "Run 2 is still almost fully detected", s9 summary item 6) against, in the
LVG subsection of the same chapter, a detected fraction of 0.68 (LTE) and 0.60 (LVG) at n = 102 for the compressive frame (C-A4).

**T-A6. "The observational floor"** (`s8_obs.tex`): "The agreement on fully detected tiles is therefore as close as the observations can
currently test." Contradicted by the linewidth subsection and by s9 ("the network is responsible"). Summary item 7 mixes the first
comparison (1-7%, plain projected model, old processing) with the current one. Both were flagged by Opus on Nov 2 and are unchanged.

**T-A7. Causal stories attached to single retrainings.** "Augmentation acts as a regulariser" (s6 mask augmentation); "the noise spikes ...
had been inflating the dispersion maps ... the density and velocity structure carries the compressive signal again" (s7, run-1 v2);
"We have not isolated the cause" of the 69% column-density error at Ms 3-7, which disappeared in the final retraining. Given S-B1
these should be stated as changes between two trainings, not as effects.

### B. Thesis vs paper: places where they disagree in substance (not just stale numbers)

- **T-B1. Cause of the real-cloud excess.** Thesis: "Either the tile spectrum underestimates the three-dimensional Mach number of real clouds, or
  the real maps contain structure ... We have not separated the two" (s8 MWISP), and s9 lists magnetic field, gravity, temperature and
  foreground as candidates. Paper: "we attribute [it] to the difference between periodic boxes and real clouds". The thesis is the
  defensible version.
- **T-B2. MWISP sample.** Thesis abstract and summary say 22 clouds; paper abstract and conclusion say 55 (P-A7). Thesis also notes that
  Perseus-arm streaming motions can make kinematic distances too large by ~1 kpc; the paper does not.
- **T-B3. LVG text.** Thesis: "30-48% less 13CO emission and up to 68% less 12CO"; paper: "factors of 0.5-0.8" (P-A3c). Thesis states the
  models are `--fold all` and gives the 2.5x cost; paper gives neither.
- **T-B4. Run 1 saturation.** Thesis: "the line saturates at the 7.59 K ceiling in almost every image" (i.e. the map maximum reaches it);
  paper: "saturates ... over most of the map" (P-B5). The thesis wording is the supportable one.
- **T-B5. Training stop.** Thesis mentions the time budget; paper does not (C-A5).
- **T-B6. Run-1 beam** 45" / 3.04 px in the thesis and config; 46" / 3.1 px in the paper for the same data.
- **T-B7. Arce et al. regions** "0.6-2 pc" (thesis, radius) vs "1-4 pc" (paper); consistent only if the paper means diameter. Say which.
- **T-B8. Ophiuchus noise** is 0.37 K (thesis table), nearly twice the 0.2 K of the synthetic FCRAO setup; the paper mentions only Perseus (0.19 K).
- **T-B9. Stale counts the paper inherited:** "32 811 images from 10 937 frames" for run 2 (thesis s7) is the source of the paper's stale sentence (P-B2).

### C. Minor

- Abstract and summary numbers (0.983, 7.8%, 23.5%, 6.8-9.5%, chi_v 0.6-0.7) are pre-final; "41 runs ... 5-16 per class" (s6:24).
- Open `\tc{}` notes: 15 in the chapter (Orion/Taurus noise, Ortiz-Leon citation, HGBS citations, uncertainty-decomposition figure,
  literature chi values, acknowledgements).
- s8 Processing step 2 says the CO training beam is 3.04 px; run 2 (`fcrao500`) is 3.1 px.
- s9 item 9 lists "a circular calibration check" among the bugs fixed; nothing else in the chapter describes it.


---

## Batch 5: thesis introduction, Chapter 2 (XGBoost / Paper I), Chapter 4 (3D forecasting), cross-chapter consistency

Scope actually read: `main_thesis_file.tex` structure; `introduction/intro.tex` (first part); Ch. 2 abstract, `s3_sims`, `s4_methods`,
`s7_conclusion` in full, `s5_results`/`s6_discussion` by keyword; Ch. 4 abstract, `s3_sims`, `s4_data` (first half), `s9_discussion` in full,
`s6`-`s8` by keyword. **Not read:** Ch. 2 `s1`, `s2_history`, most of `s5`, `s6`; Ch. 4 `s1`, `s2_history`, `s5_models`, `s7_evaluation`,
`s8_results` (8.7k words), `s9b_hybrids`, `s10_ongoing`; `background_from_paper3.tex`. Numbers inside those sections are unaudited.

### Cross-chapter inconsistencies

**X-1. Sound speed for the same "4.6 pc, 1000 cm^-3, 10 K (Collins 2012)" scaling.** Ch. 3 and Paper II: c_s = 0.188 km/s (mu = 2.33, 10 K).
Ch. 4 Table "Physical scaling": c_s = 0.21 km/s, code time L/c_s = 21.75 Myr. The Ch. 4 value is what 4 pi G = 1620 with rho_0 = 4.1e-21
and L = 4.6 pc requires (c_s = 0.207 km/s), which corresponds to 12.1 K for mu = 2.33, not 10 K. The scaling set is over-determined;
either state T ~ 12 K (or the mu used) in Ch. 4, or note in Ch. 3 that its 10 K value differs from the gravity-fixed one. Also
rho_0 = 4.1e-21 g/cm^3 for "1000 cm^-3" implies 2.45 m_H per particle, neither 2.33 (per particle) nor 2.8 (per H2); say which n is meant.

**X-2. Free-fall time vs the forecast horizon in Ch. 2 / Paper I.** With rho_0 = 4.1e-21, t_ff = 1.04 Myr (Ch. 4 Table gives "~1.1 Myr",
Ch. 2 "time scale 1.1 Myr"). Ch. 2 says 0.25 t_ff ~ 0.31 Myr and "121 snapshots spanning 1.0 t_ff, each ~0.01 Myr", which need
t_ff = 1.2-1.24 Myr. 0.25 x 1.04 = 0.26 Myr. A 15-20% inconsistency that is in the abstract of the published chapter. Check the code time
of the last snapshot against t_ff.

**X-3. Evaluation standards differ between chapters.** Ch. 3 / Paper II argue at length that correlated samples must not be split at random
and hold out whole simulations. Ch. 2's headline R2 = 0.9988 ("Global performance") is a random 20% split over (particle, snapshot)
rows of one simulation, where the same particle at adjacent snapshots is on both sides. Ch. 2 does have temporal, core-wise and
other-simulation tests, and its thesis additions report that linear advection x + v dT beats the model on the unseen simulation
and that the model's gain within the training simulation is 20-50%. Consequently "R2 > 0.99" mostly measures advection. Paper II's
introduction cites Paper I as forecasting trajectories "with R2 > 0.99" without that qualification; a reader of the thesis sees both.
Ch. 4 uses three seeds per model; Ch. 3 uses one (C-B8).

**X-4. Same forcing module, described three ways.** Ch. 4 (from the parameter file): parabolic profile peaking at k = 2, vanishing at k = 0 and 4,
autocorrelation of one turnover time, weight 0.5. Ch. 3 / Paper II: "parabolic spectral profile peaking at k = 2 \TODO{confirm the forcing band}".
Ch. 2: "wavenumbers between 1 and 2" (a different suite, Mac Low 1999 driving). The Paper II TODO can be closed from the mach_grid parameter
files exactly as Ch. 4 did (the config already stores `driving_profile` and `driving_autocorrelation`).

### Chapter 2

- **B2-1. "Systematically conservative false-positive rate (37% precision)"** (abstract) contradicts §s6 "systematically over-predicting collapse".
  37% precision means 63% of predicted core particles are false positives; that is the opposite of conservative.
- **B2-2. Over-interpretation of SHAP.** Conclusion item 2: SHAP "proved that the model independently re-derived fundamental non-linear MHD
  collapse criteria", with strong B "driving high core-formation probabilities". The test set is snapshots 80-90 (late), where
  future core particles are already dense, and B rises with density by flux freezing. High |B| is a tracer of compressed gas, not a
  cause of collapse (physically the field opposes it), and a "threshold" at log10(rho/rho_0) ~ 0.8 is a classification by current
  density. The same chapter's conclusion item 6 says the outcome is not a point-wise quantity. Tone down to "the classifier keys on".
- **B2-3. "A_3 ... corresponds to a 3-sigma confidence interval"** (§s4 metrics). Three times the mean absolute error is not a 3-sigma interval.
  The thesis addition already notes that sigma_CV is used for two different quantities.
- **B2-4. Pseudo-cores "a few hundred AU in size"**: the finest cell is 4.6 pc / 2048 = 463 AU, so they are at the grid scale. Say so.
- **C2-1.** Future work proposes applying the gas-parcel model to Gaia/APOGEE stellar kinematics; stars are not tracers of the gas after formation. Speculative; flag as such.
- Verified: box mass 5900 Msun, B_0 = 13 muG for beta_0 = 0.2, 2048^3 effective resolution, core fraction 5.6%, w_pos = 16.87,
  the XGBoost leaf-weight and gain formulae, SHAP additivity and base-value discussion.

### Chapter 4

- **B4-1. Virial parameter is not stated and spans a factor of 64.** With 4 pi G = 1620 fixed and Ms = 1-8, alpha_vir = 5 Ms^2 / (3 G rho_0 L^2 / c_s^2)
  = Ms^2 / 77, i.e. 0.013 at Ms = 1 to 0.83 at Ms = 8 (Ch. 2 used alpha_vir = 1 at Ms = 9). The low-Mach runs are strongly bound and
  collapse globally; "typical cloud conditions" (s3 units) applies only to the top of the range. This matters for the star-formation
  conclusions and for the "check the virial state lies in the training range" step of the Enzo coupling. State alpha_vir per Mach number.
- **B4-2. Jeans resolution of the "truth".** The collapse runs are on a static 512^3 grid. The Jeans length of the mean density is L/6.4, so
  the Truelove condition (4 zones) fails above rho ~ 400 rho_0, while the pseudo-cores reach 1e5 rho_0 "only a few zones across".
  Fragmentation and merging of clumps at those densities are resolution-dependent, and the star-matching metric uses these clumps as
  ground truth. The chapter says the resolution "does not follow" the collapse but never mentions artificial fragmentation. Add a caveat.
- **B4-3. Independence of the simulations.** The four realisations per Mach number are dumps of one driven box at 3, 5, 7, 9 turnover times,
  and the `\tc{}` at s3 asks whether all eight boxes share `DrivenFlowSeed = 20150419`. If they do, the forcing sequence is identical in
  units of the turnover time, so boxes of different Mach number have the same large-scale pattern at the same dump. That would weaken
  both "held-out simulations" and "generalize to Mach numbers not seen in training" (abstract; s8: a U-Net trained without any Ms = 4
  run scores +0.55 on them, higher than the reference model's +0.51 that was trained with them). Resolve the `\tc{}` before making either claim.
- **C4-1.** Abstract says "generalize to Mach numbers not seen in training"; s4 split paragraph says the test set measures "new realizations
  at known Mach numbers". Both are true of different experiments; the abstract should say which.
- Verified: t_ff = 0.0478 code units from 4 pi G = 1620; zone sizes 1850 AU / 0.036 pc / 0.072 pc, and 0.29 / 0.58 pc for 16^3 / 8^3;
  conservative downsampling formulae; the description of the two dataset errors and the flux-convergence test; the discussion's
  treatment of baselines (persistence, coarse Enzo) is careful and self-critical.
- 24 open `\tc{}` notes in the chapter (driving amplitudes for seven boxes, seed, core-hours, abstract numbers, funding).

### Thesis front matter and structure

- `introduction/intro.tex` still opens with the template ("The introduction to your thesis goes here", `\citep{Example2019}`, "Part I/II/III")
  followed by "STUFF FROM THE PROSPECTUS THAT WAS CUT". `Example2019` is the one undefined citation in the build.
- Chapters 3-5 have placeholder titles ("Chapter 3 Title: Mach Predictor", "Short Chapter 3 Title") and each carries
  "This thesis chapter originally appeared in the literature as", which is not true of Ch. 3-5 yet. Chapter 5 is the empty template.
- Intro details to recheck when it is written: "more than 150 chemical species" in GMCs (the count of interstellar molecules is
  now about twice that); Gaia described as providing "a 6D position-position-velocity cube".


---

## Batch 5b: second pass over the sections skipped in batch 5

Now read in full: Ch. 2 `s1_intro`, `s2_history`, `s5_results`, `s6_discussion`; Ch. 4 `s7_evaluation`, `s8_results`, `s9b_hybrids`, `s10_ongoing`.
**Still unread:** Ch. 4 `s1_intro`, `s2_history`, `s5_models` (5.4k words of architecture description), `s6_training` (keyword search only);
`introduction/background_from_paper3.tex`.

### Chapter 2 (additional)

**X-3 strengthened -> A. The classifier is tested on the same particles it was trained on.** (`s5_results` setup paragraph; Table "Training and test sets")
The label ("ends in a core at t_ff") is a constant per particle. Training uses all particles at snapshots 20-79, testing the same
particles at snapshots 80-90, with position among the inputs. A particle at snapshot 80 has nearly the feature vector it had at
snapshot 79, with the same label. This is the constant-label leakage that Ch. 3 identifies for xi ("can be learned by recognising the
simulation"), here at the level of particles. The chapter's own side experiment shows a tree ensemble memorises the density field
of one run from coordinates (R2 = 0.96). AUC = 0.975, average precision 0.784 and the SHAP results all come from this split. The
regression has a split by core (57 held-out cores); the classifier has none, and was not tested on u502/u503. A particle- or
core-held-out classifier test (or a test on another simulation) is needed before the AUC can be quoted as generalisation.
The 10-fold "repeated k-fold" of the grid search is likewise over rows of the same particles, so "strictly disjoint from the
training data, mimicking the evaluation of an unknown physical state" (s5 first paragraph) is not accurate.

**X-2 confirmed with the chapter's own numbers.** `s5_results` advection subsection: DeltaT = 30 snapshots = 0.0112 code time units. With the
code time unit L/c_s = 22-24 Myr (c_s = 0.19-0.21 km/s) this is 0.25-0.27 Myr, and one snapshot is 0.008-0.009 Myr, not 0.31 Myr and
0.01 Myr. 0.31 Myr would need c_s = 0.16 km/s. The value appears in the abstract, s3, s6 and the conclusion.

- **B2-2 revised.** The thesis additions already say what I flagged ("SHAP ... do[es] not show that the magnetic field promotes collapse";
  the ablation changes AUC by 0.003). The problem is now internal: the published text in the same chapter still says the model
  "re-derives classical gravoturbulent collapse criteria", "successfully extracts fundamental MHD collapse dynamics" and (conclusion 2)
  that SHAP "proved" it. Add a pointer from those passages to the thesis additions, or reword the conclusion.
- **B2-5.** "prestellar core formation ... affecting only a small fraction (<~ 1%) of turbulent gas" (s6 subgrid) vs 5.6% of tracer particles
  in this simulation.
- **B2-6.** `s1_intro`: "tens of parsecs (~1e19 m)". Ten parsecs is 3e17 m; 1e19 m is 320 pc. ("0.1 pc ~ 3e15 m" is right.)
- **B2-7. Series numbering.** `s6` says a large suite of collapse simulations "will be the focus of the second paper in this series". The paper
  titled "ML in Astrophysical Turbulence II" is the Mach-number predictor (Ch. 3), and the collapse suite is Ch. 4.
- "Conservative" is used for the classifier's tendency to flag gas as collapsing (s5, s6) and for its false-positive rate (abstract);
  the first is defensible, the second is not (B2-1 stands).
- Checked: the unwrapping rule and its half-box assumption; the gamma / number-of-trees explanation; snapshot ranges of the temporal
  test (0.167-0.66 t_ff train, targets to t_ff); recall 93% / precision 37% / FPR 0.093 are mutually consistent with a 5.6% base rate.

### Chapter 4 (additional)

**B4-4. Abstract and discussion say the forecasts "merge most neighbouring cores"; the sensitivity analysis says otherwise.** (`s8` star matching)
"Why is the best F1 only 0.28?" concludes that "neighbouring cores merge into fewer, larger clumps". The sensitivity paragraph two
bullets later finds that defining stars as density peaks raises the forecast stars only from 333 to 418 (truth ~1400) and states
"The forecasts contain genuinely fewer cores, not merely merged ones." The abstract, Discussion finding 5 ("Regions, not cores ...
they merge most neighbouring cores") and §9 selection text keep the merging explanation. One of the two has to go.

**B4-5. "About twice as well as a density recipe or a sink criterion" holds only in 0.29 pc cells.** (abstract; Discussion finding 6; Tables c2f16/c2f8)
0.29 pc, one interval: learned 0.36-0.44 vs recipe 0.22 and sink 0.18 (about 2x). 0.58 pc: learned 0.44-0.58 vs recipe 0.43 and sink 0.25,
i.e. 1.0-1.35x the recipe. The abstract names both cell sizes.

**B4-6. "Original-resolution simulation" is 128^3, not 512^3.** Abstract: the star-matching metric "compares the dense clumps of a forecast
with those of the original-resolution simulation". §7: "True stars. We identify them in the original 128^3 data", which is the 512^3 run
block-averaged by 4. Say 128^3 in the abstract. (Related to B4-2: the 1000 <rho> threshold is chosen because the Jeans length there
is below one 128^3 zone, so the "stars" are defined at the resolution limit.)

- **B4-3 addendum.** The Mach hold-out result (s8 ablations: +0.550 and +0.535 with the Mach number held out, against +0.511 and +0.497
  for the reference trained with it) compares four runs with one and a 28-run with a 24-run training set, as the text says. With the
  open seed question it supports "Mach number need not be seen" only weakly; s8 itself still has `\tc{Hold-out results.}` in the
  Mach paragraph while the abstract states the conclusion.
- **C4-2. Gate 3 as planned vs as passed.** Plan (s10 step 4): "closer to the truth than Enzo at N at every dump". Result (s9b): mean below
  Enzo N at every dump, individual simulations at 88% of dumps at 64^3, declared "passes". State the relaxed criterion.
- **C4-3. Stale status.** s10 "Status (29 October)": "online runs with seeds 2 and 3 of every H2 model (submitted)"; s9b already reports
  three seeds for the online runs.
- **C4-4.** Table "Skill per test simulation": the CSNO-UNet-GRU row (+0.12 to +0.14) is the single-step head, unlabelled, in a chapter whose
  conclusions rest on the substepped one.
- **C4-5.** "With 4000 instead of 1000 training samples, the U-Net's star F1 triples (0.05 to 0.15)": 0.05 is seed 1, the three-seed table has 0.06.
- Checked and consistent: Jeans length 0.023 pc at 1000 <rho>; mass thresholds 0.06 / 0.6 / 6 Msun; FSS and CSI-F1 relations;
  precision 0.74-0.79, recall 0.17 and F1 0.28 for ~300 forecast vs ~1400 true stars; "Jeans length of the mean density is 0.16 of
  the box" and the 0.4 <rho> statement; cell-crossing times 0.1-0.3 t_ff at Ms >= 4; H2 cost ratios (2.4 / 1.5 / 1.06 x Enzo N;
  22% / 10% / 4% of Enzo 2N); numbers quoted in the Discussion against Tables skill, stars, coarse64, coarse16 and the hybrid text.
  The chapter reports its negative results (PINN, DeepONet, H1, solver beats forecasters) plainly, uses three seeds, and discloses
  its own loss-scale slip in H2.


---

## Batch 5c: the last unread sections (after the limit reset, 2026-11-09 evening)

Read in full: Ch. 4 `s1_intro`, `s2_history`, `s5_models`, `s6_training`; `introduction/background_from_paper3.tex`.
With this, every `.tex` section of Chapters 2-4 and the introduction files has been read at least once. Figures were not inspected,
and literature claims were not checked against the sources.

### Chapter 4

- **C4-6. Counts that disagree between sections.** Models: "nine models" (abstract, background file) vs "all eight models" (`s6` protocol and
  campaign 1; the DeepONet was added later). Ablations: "twenty single-seed variants" (`s6` campaigns) vs "Twenty-four" (`s8`, `s10`).
- **C4-7. Stale status notes.** `s6` campaign 4: stand-alone CSNO with the substepped head "(\tc{training})", while `s8` reports it
  "trained to completion". `s5` PINN paragraph ends "\tc{Results.}" though `s8` has them. `s1`: "The coarse-to-fine model of
  Section [ongoing]" now lives in the results and evaluation sections.
- **C4-8. Introduction vs the chapter's own numbers.** "Only a small fraction of the gas, typically 1-2% per free-fall time, ends up in stars"
  opens a chapter whose simulations give epsilon_ff = 0.2-0.8 (`s8` whole-box paragraph, which explains why). A forward pointer would help.
- **C4-9. Claims about prior work to verify against the papers:** StarFind "F1 > 0.995", "recall 0.88 but precision of about 0.26",
  StarNet "~0.3 dex at 5.5 times lower cost", and "StarFind and StarNet have no baselines" (a strong statement about someone else's work).
- **C4-10. Momentum.** `s4_data` says the conservative downsampling makes the coarse fields "carry the same mass and momentum"; `s5` (CSNO head)
  adds a gate so total momentum can change because "the stored data do not conserve it exactly". Say in `s4` what is and is not conserved
  between snapshots (mass yes; momentum only approximately) so the two statements do not read as a contradiction.
- Checked and consistent: sample counts (32 x 67 x 24 = 51 456; 24 training runs = 38 592; 268 test states); linear model 72 parameters;
  XGBoost 350 rounds x 8 outputs = 2800 trees; FNO 28.3M parameters for width 32, 12 modes, four layers; PINN residuals and their
  constants (4 pi G = 1620, t_ff = 0.0478, tau = 33 x 0.01 t_ff); c_s Delta t ~ 0.07 pc ~ one 64^3 cell per unit Mach number; ConvGRU
  equations; the substep/CFL argument for the conservative head. The history section states the weaknesses of the early tests itself
  (small test sets sharing snapshots with training, loss terms that carried no gradient, DWA weights that did nothing).

### `introduction/background_from_paper3.tex`

This file is the introduction of the Paper III draft. It is not `\input` anywhere in the thesis at present (0 references in
`intro.tex` and `main_thesis_file.tex`), and it ends with "This paper provides ..." and `\ref{sec:...}` labels that do not exist
in the thesis. If it is to become thesis background it needs rewording. Content points:

- **I-1. "the size of a proto-star - a few hundred kilometers (~1e5 m)".** A protostar is a few solar radii, ~1e9 m; the quoted dynamic range
  of 1e12 changes accordingly (1e17 m / 1e9 m = 1e8; even to the first-core scale it is ~1e5-1e6).
- **I-2. "GMCs in the molecular ring (~3 kpc)".** The molecular ring is usually placed at ~4-5 kpc; "3 kpc" is the name of a different feature.
- **I-3. Survey list.** "telescopes like ALMA and IRAM 30-m in the form of surveys like ATLAS-3D CO ..., The Gaia Andromeda Photometric Survey"
  as sources of GMC structure: ATLAS-3D CO is a survey of early-type galaxies, and the Gaia Andromeda survey is optical photometry,
  not a molecular-cloud survey. The same sentence is in `intro.tex`.
- **I-4. "~97% of all naturally occurring elements"** produced by stars and mergers: state what is counted (elements by number of species?) or drop.
- **I-5. Numbers to verify:** NewHorizon "0.3 dex below"; SILCC "20-40% of the mass" in H2; Hirashima "~75% reduction"; Ono "10x / 1000x";
  Seifried polarisation "<= 10 degrees".
- **I-6. Series numbering is consistent here** (Paper I XGBoost, Paper II Mach predictor, Paper III this benchmark), which makes the Ch. 2
  sentence about "the second paper in this series" (B2-7) the odd one out. The bib key for Paper I is
  `bisht2026mlastrophysicalturbulencei` here and `Bisht2026` in Paper II; make sure both resolve to the same entry.


---

## Batch 6a: the final observation run (paper `499534a`, code `be9820c` / `ae39518`, outputs in `~/plots/observations_net9010_*final*`)

What changed: all `\prov` numbers in the abstract, Section 5 and the conclusion were replaced with values from the final models;
figures regenerated; `submit_obs_final.sh` runs every final tag, MWISP with the three run-2 models, and dust with `final_mom0`
matched to the `final_proj` tiles (this resolves C-C6). The thesis has uncommitted Ch. 4 edits that I did not look at.

**Transcription check: every new number I tested matches the output files.** Table 4 (both model columns and Ms_obs), the tile-level
statements of §5.3 (3%, -1% to +19%, -26% to +52%, the 39% and 45% tiles within 10%, CO fully detected -10% to +50% with median +18%,
cloud medians 1.1-1.3, the 2.3x Ophiuchus tile, Taurus 13.6 / 13.7 vs 13.0 / 24.2), chi_v medians 0.61 / 0.73 / 0.53 / 0.27, MWISP
(12.2, 8.0-15.6, 1.57, rho = 0.90; 13CO 1.28, 0.89; 76% above 13.5), dust (0.3, 0.1-0.5, control 1.0, all ten lower limits).

The problems are in what the numbers mean, and most earlier findings are sharpened rather than resolved:

**N-1. Every observational number is from the single `foldall` model; the five fold models often disagree with it by more than the effects discussed.**
(Table 4, §5.3, §5.4; S-A3 confirmed with final models)
- chi_v. Paper: Serpens 0.53. The five fold models give 0.18-0.38 on every Serpens tile (cloud median ~0.30). Perseus: paper 0.61, folds
  0.30-0.61 per tile; Ophiuchus: paper 0.73, folds 0.36-0.66. The statement "Perseus and Ophiuchus lie near or above the ~0.67 of
  purely compressive driving" is true of one model out of six. Between the pre-final and final `foldall` models the same clouds moved
  from 0.50 / 0.53 / 0.34 / 0.35 to 0.61 / 0.73 / 0.53 / 0.27.
- Ms, projected model: cloud medians of the fold models are 10.6-14.2 in Perseus (ratio to Ms_obs 1.02-1.37), 7.9-9.4 in Ophiuchus
  (1.23-1.47), 10.4-14.4 in Serpens (0.75-1.04), 10.4-12.6 in Orion A (1.10-1.34). One fully detected Orion A tile is predicted at
  10.2, 6.2, 5.2, 11.9, 5.1 by the five folds (Ms_obs 9.4): a factor 2.3 spread on clean data, at the clipped r = 0.4 (S-A2).
- MWISP two-line: `foldall` ratio 1.57; fold models 1.32-1.53. 13CO: 1.28; folds 1.16-1.32.
- Dust control: with `foldall` the column-density model on Perseus 13CO intensity reads 1.5-1.6x Ms_obs on four of six tiles; the
  fold mean reads ~1.0.
The outputs already contain all six models per tile. Table 4 and the text should give the fold range (or a pooled median with its
spread), and the chi_v paragraph should not quote single-model values.

**N-2. The projected-map model now also reads high, on partly detected tiles.** (updates P-A1)
Final `foldall` cloud ratios: Perseus 1.23, Ophiuchus 1.47, Serpens 0.94, Orion A 1.12 (pre-final: 1.02, 1.12, 0.90, 1.35). On fully
detected tiles it is within 3% (Perseus, Ophiuchus) and -1% to +19% (Orion A); the excess is on tiles with 63-80% coverage
(+25% to +52%). The CO models are high on fully detected tiles too (median +18%). So there are two effects, not one: a
partial-coverage bias that mask augmentation has not removed (the Discussion still says it "removes most of the high bias"), and a
CO-model offset on clean tiles. The abstract's single "1.1-1.6 times higher ... attributed to periodic boxes" covers neither accurately.
§5.3 gives the CO cloud medians (1.1-1.3) but not the projected-model ones (0.94-1.47).

**N-3. The dust control now fails in two of the four clouds.** (updates S-A1 / P-A6)
Column-density model on the 13CO integrated intensity, fold mean / Ms_obs per tile: Perseus 0.35-1.13, Ophiuchus 0.57-1.33 (median
of the ten = 1.0, as quoted); Orion A 0.22-0.33; Serpens 0.35-0.55; Taurus 0.24-0.37. In Orion A and Serpens the model returns
about 0.3 on the CO map itself, the same factor as the dust (median 0.3). The paper still quotes only Perseus and Ophiuchus and
concludes "the shortfall comes from the dust maps rather than from the model". The dust result also moved from 0.5 to 0.3 between
the pre-final and final column-density models, and every "lower limit" fell by 20-45% (Perseus 4.6 -> 2.9, L1688 3.9 -> 2.1),
which is again model-to-model scatter (S-B1).

**N-4. Chamaeleon and Musca limits still come from the background subtraction.** (C-A3 unchanged)
Raw / background-subtracted: Cha I 2.9 / 4.2, Cha II 3.5 / 7.2, Cha III 4.5 / 9.5, Musca 1.5 / 2.3. The quoted limits are the
subtracted ones; Cha III "Ms > 9.5" remains the highest of the ten.

**N-5. MWISP: the excluded tiles now correlate better than the included ones.**
Two-line model, tiles > 10 pc (33 clouds): Spearman 0.93, ratio 1.28, predictions up to 16-18. Tiles <= 10 pc (22 clouds): Spearman
0.90, ratio 1.57. The reason given for excluding the large tiles ("the network saturates") is weaker than before (the final model's
ceiling moved up), and the abstract still presents "55 clouds" with rho = 0.9 (P-A7 unchanged).

**N-6. chi_v text: improved, but see N-1.** The new sentence that partly detected tiles carry the highest chi_v, "so part of this
excess may come from the cloud edges rather than from the flow", addresses S-A4 and should stay. With N-1 the remaining sentence
comparing clouds with the driving limits should go.

**Still open from earlier batches in this commit:** the red note in §5.4 (MWISP) and at the top of §5.4 dust; Table 4 caption
"r = 0.5" for all clouds (the output shows r = 0.40 for Orion A and 0.89 for Taurus with `final_proj`, 0.90 for Orion A with the CO
models); "13 matched tiles" without saying one Ophiuchus tile lacks dust coverage; Fig. 9 caption still has its TODO; Taurus still
uses the 12CO-only model (five folds: 10.2-13.5 and 13.3-15.9).


---

## Batch 6b: literature claims against the sources

Method: three sub-agents with web access checked the claims I listed against arXiv / ADS / publisher pages; I compared their
findings with the wording in the paper and thesis. Quotes for Wells & Norman 2021, Guo et al. 2017 and Angelopoulos & Bates 2021
were read from the PDFs; the others came through a page-summarising fetch, so numbers are reliable but exact wording should be
rechecked before it is quoted. Verdicts: OK = source supports the text; FIX = the text must change; SOFTEN = defensible but overstated.

### Part 1: machine-learning prior work and methods references

| ID | Where | Claim in our text | Verdict | What the source says |
|---|---|---|---|---|
| L-1 | Paper §6.4; thesis s9 | Peek & Burkhart 2019: CNN separates sub- from super-Alfvenic turbulence in density slices with > 98% accuracy even with Fourier amplitudes removed | OK | 128x128 slices of 512^3 density cubes. 97.94% on normal images, 99.50% on fixed-Fourier-power images; ">98%" is the abstract's wording. |
| L-2 | Paper §6.4; thesis s9 | Schmaltz et al. 2025: 32x32 sub-fields, six solenoidal isothermal MHD runs, Ms 2.2-11, median error ~0.5 without noise, ~2 at noise = mean signal | OK with two caveats | Ms = 2.17-11.02, M_A = 0.43-0.69. ~0.5 holds at 90 degrees inclination ("below 0.75 across all inclination angles"); 2-2.25 at 100% noise. The paper never states its train/test split; that the main tests use training simulations is an inference from its Appendix B. |
| L-3 | Paper §6.4 ("consistent with intensity maps giving the smallest errors"); thesis s9 ("intensity maps give the smallest errors") | SOFTEN | "while their median errors are similar, the maximum and upper quartile error values tend to be lowest for intensity maps". Median errors are not smallest for intensity. This sentence carries part of our argument that fixed b lets column density determine Ms. |
| L-4 | Paper §6.4; thesis s9 | Hu 2025: Ms, M_A and field orientation from HI channel maps, Ms ~ 0.25-1, two HI clouds | OK | Conditional ResNet; also predicts total field strength (five outputs). Two clouds in Monoceros R2 (thesis has this). |
| L-5 | Paper §6.4: "Both report errors several times larger on a simulation withheld from training than on the training simulations"; thesis s9 same | FIX | Hu 2025 withholds one simulation and says only that the distributions are "broader"; no factor is given. Zhang & Hu withhold a "patch", not a simulation, and their abstract gives sigma = 0.01-0.02 (seen) vs 0.1-1.8 (unseen), i.e. 10-100x. Quote each paper's own numbers. |
| L-6 | Thesis s9: "All four suites are solenoidally driven" | SOFTEN | True of the training suites. Zhang & Hu also test a "fully compressively driven" multiphase run in an appendix (no error numbers). |
| L-7 | Thesis Ch. 4 s1 | StarFind: F1 > 0.995; recall 0.88, precision ~0.26; five input fields; (10 comoving kpc)^3, 64^3 | OK | Wells & Norman 2021 (arXiv:2011.01358): "<P> >= 0.258 and <R> >= 0.875" (lower bounds). |
| L-8 | Thesis Ch. 4 s1: StarNet "reproduces the enriched volume fraction to ~0.3 dex at 5.5 times lower cost" | FIX | 0.3 dex holds for the low-metallicity thresholds only ("at high metallicity, StarNet predicts a much lower enriched fraction"). "5.5 times" is not stated by the authors: it is 5600 vs 1025 node-hours for runs with nine vs five AMR levels. |
| L-9 | Thesis Ch. 4 s1: "StarFind and StarNet have no baselines" | SOFTEN | They compare with the resolved Phoenix simulations as ground truth; there is no alternative-method or simple-recipe baseline. Say that. |
| L-10 | Thesis Ch. 4 s9 | McGreivy & Hakim 2024: 79% compared against a weak baseline | OK | 60 of 76 papers. |
| L-11 | Thesis background file | Hirashima 2025 ~75% reduction; Ono 2025 10x CPU / 1000x GPU; Chadayammuri 2023 | OK | Hirashima: "computation cost", one isolated dwarf galaxy. "Ono et al." is Ono & Sugimura (two authors). |
| L-12 | Paper §3.3; thesis s5, s2b | "learnable temperature T that rescales the predicted variance (Guo et al. 2017)" | FIX | Guo et al.: "All methods are post-processing steps ... Each method requires a hold-out validation set"; "T is optimized with respect to NLL on the validation set"; classification logits only. Our T is trained jointly and ends at ~0.98 (C-A2). Drop the citation or describe it as an extra learnable bias on the log-variance. |
| L-13 | Paper §3.4 | Conformal quantile ceil((n+1)(1-alpha))/n, Angelopoulos & Bates | OK | Exact formula in their Section 1. Vovk et al. 2005 not checked. |


### Part 2: observational data and distances

Sources the agent could not open (values below marked "not opened" rest on citing papers or memory): Larson 1981 and Solomon 1987
full text, Bohlin 1978, Mizuno 2001, Andre 2010. I also read the FITS headers of the Orion, Serpens and Perseus cubes in
`~/projects/p79d_dataset/observations`.

| ID | Where | Claim in our text | Verdict | What the source says |
|---|---|---|---|---|
| L-14 | Paper §5.1 | "Perseus, Ophiuchus and Serpens: the FCRAO 14 m maps of the COMPLETE survey (Ridge et al. 2006), 46'' beam on 23'' pixels and 0.066 km/s channels" | FIX (citation) | Ridge et al. 2006 covers Perseus and Ophiuchus only: "Observations of the third cloud, Serpens, are less advanced and hence are not included here"; the Serpens cubes are a COMPLETE website release with no published parameter table. Our Serpens cube header has 23'' pixels and 66.4 m/s channels, so the description is right; the citation is not. Cite the COMPLETE data page for Serpens. |
| L-15 | Paper §2.4 "close to the 0.19 K measured in the COMPLETE Perseus cube"; thesis table (0.19, 0.37 K) | OK | Ridge 2006 Table 6 gives mean 13CO rms 0.17 K (Perseus) and 0.33 K (Ophiuchus). Ours are measured on the cubes; say "measured" and/or quote the published values. |
| L-16 | Paper §5.1: Orion A "Nobeyama 45 m map of Shimajiri et al. (2014), with a 25.8'' beam and 0.5 km/s channels"; thesis "7.5'' pixels" | FIX (citation or wording) | Shimajiri et al. 2014 Table 1: effective beam 25.8'', velocity resolution 0.3 km/s, grid 10''. Our cube (`orion/Fig1.fits.gz`, TELESCOP NRO45M) has 7.5'' pixels and 500 m/s channels. The numbers describe the cube we used, but they are not the parameters of the cited paper: say where the cube came from and that it is a regridded product. |
| L-17 | Paper §5.1 | Taurus CfA 1.2 m: 8.4' beam, 0.65 km/s channels (Dame et al. 2001) | OK | "beamwidth of 8'.4 +/- 0'.2"; 0.65 km/s. |
| L-18 | Paper §5.4: "The MWISP survey (Su et al. 2019) ... 50'' beam, 30'' pixels, 0.166 km/s channels, 0.25 K and 0.47 K noise" | FIX (citation) | Su et al. 2019 give ~0.16 km/s (12CO) and 0.17 km/s (13CO), rms ~0.5 K (12CO) and ~0.3 K (13CO). 0.166 km/s (13CO only; 12CO is 0.159), 0.25 K and 0.47 K are from the DR1 paper, arXiv:2512.08260 (Yang et al.), which the paper still has as a TODO citation. |
| L-19 | Paper §5.4: "the cloud catalogue of the first data release for l = 105-150 deg, |b| < 5 deg" | SOFTEN | DR1 covers l = 9.75-229.75 deg, |b| <= 5.25 deg, 2310 deg^2, 103 517 clouds (10 790 with 13CO). l = 105-150 deg is our subset; word it so. DBSCAN with a 2 sigma cut and the ScienceDB doi are confirmed. |
| L-20 | Paper §5.4 | Reid et al. 2019: R0 = 8.15 kpc, Theta0 = 236 km/s | OK | |
| L-21 | Paper §5.5, Fig. 13; thesis Table | Arce et al. 2010 Table 5 values; "1-4 pc regions" | OK | All six rows match. The size column is a radius (0.6-2 pc); the abstract's "1 to 4 pc" is the diameter. T_ex is from the 12CO peak. This settles T-B7: both statements are right, say radius or diameter. |
| L-22 | Paper §5.5 | Padoan et al. 2003: Perseus Ms ~ 14 for the whole complex | OK | sigma_v = 2.0 km/s (Perseus), 1.0 km/s (Taurus) over ~30 pc; "approximately 14 in Perseus and 7 in Taurus". |
| L-23 | Paper §2.4, Fig. 8 | Pineda et al. 2008: 13CO saturates above A_V ~ 5 | OK | "The (1-0) lines of 12CO and 13CO saturate above 4 and 5 mag". ~60% of 12CO positions sub-thermal (thesis "most"). |
| L-24 | Thesis s7 "sigma_1D ~ 0.7 (L/pc)^0.5 km/s (Larson 1981)" | FIX | Larson 1981: sigma = 1.10 L^0.38. The 0.72 R^0.5 form (1D, R a radius) is Solomon et al. 1987. Confirms T-A2. Check which relation the paper's Fig. 13 plots under the Larson citation. |
| L-25 | Code comment / paper A_V | N(H2)/A_V = 9.4e20 (Bohlin et al. 1978) | OK | Follows from N_H/E(B-V) = 5.8e21 with R_V = 3.1 and fully molecular gas (original not opened). The paper gives no citation for the A_V conversion at all; add one. |
| L-26 | Paper §5.4, conclusion: Chamaeleon III "Ms > 9.5" as a lower limit | NOT SUPPORTED | Belloche et al. 2011: C18O and 12CO linewidths in Cha III of ~0.9 and 2.6 km/s, 13CO "between those" (~1 km/s adopted); turbulent/thermal pressure ~4. That is Ms ~ 3.5-6; even the optically thick 12CO width gives ~10 as an upper bound. A lower limit of 9.5 is above the plausible value (confirms P-A6a, C-A3, N-4). |
| L-27 | Dust config and thesis | HGBS resolutions 36.3'' and 18.2'' | OK | Palmeirim 2013; Konyves 2015. |
| L-28 | Thesis s8 | Orkisz et al. 2017 is Orion B, Ms ~ 6, mostly solenoidal | OK | Solenoidal fraction 0.72 +/- 0.09 vs 2/3 at equipartition. |
| L-29 | Table 4, dust config | Distances | Two flags | Perseus 300, Ophiuchus 137, Serpens 436, Orion A 400, Taurus 140, Cha I-III 192/198/192: within 10% of current values (Zucker 2019; Ortiz-Leon 2017; Voirin 2018; no source found for Cha III = 192). **Aquila at 260 pc** is the HGBS adopted value and 40% below the VLBI distance (436 pc) that the paper uses for Serpens; **Musca at 140 pc** is defensible (Bonne et al. 2020) but other work gives ~170 pc. Both only set the quoted tile sizes. |


### Part 3: turbulence theory, radiative transfer, and the bibliography

Full texts were read through ar5iv; ADS was not used. I confirmed items L-30 to L-34 myself in `bibliography.bib`.

**Wrong papers behind citation keys (the most serious result of the literature check):**

| ID | Key | Cited for | What the bib entry actually is | What should be cited |
|---|---|---|---|---|
| L-30 | `Federrath2010b` (6 uses: intro, §2.1 x2, §2.3 x2, §6.2) | the forcing module, the projection operator (Eq. 2), zeta = 1 solenoidal, b ~ 1/3 and 1, mode conversion | Federrath et al. 2010, ApJ 713, 269: "Modeling Collapse and Accretion in Turbulent Gas Clouds: Implementation and Comparison of Sink Particles in AMR and SPH" | Federrath et al. 2010, A&A 512, A81 (arXiv:0905.1060), "Comparing the statistics of interstellar turbulence in simulations and observations: Solenoidal versus compressive turbulence forcing". Not in the bib. |
| L-31 | `Federrath2008` | b ~ 1/3 vs 1; "consistent with the expectations for the two driving limits"; b ~ 1/3 in §4.3 | Federrath, Glover, Klessen & Schmidt 2008, Physica Scripta T132, 014025 (Lagrangian tracer particles) | Federrath, Klessen & Schmidt 2008, ApJ 688, L79 (arXiv:0808.0605), which is the paper that states b ~ 1/3 (3D solenoidal) and b ~ 1 (compressive). |
| L-32 | `Padoan1998` | sigma_s^2 = ln(1 + b^2 Ms^2); column-density variance methods | Padoan, Juvela, Bally & Nordlund 1998, ApJ 504, 300 (synthetic molecular clouds, non-LTE transfer) | Padoan, Nordlund & Jones 1997, MNRAS 288, 145 (arXiv:astro-ph/9703110), which has sigma^2 = ln(1 + M^2 beta^2) with a single beta ~ 0.5. |
| L-33 | `Lazarian2006` | velocity channel analysis / velocity coordinate spectrum (intro, §6.4) | Lazarian 2006, Astron. Nachr. 327, 609: cosmic-ray propagation in the intergalactic medium | Presumably Lazarian & Pogosyan 2006, ApJ 652, 1348 (VCS). |
| L-34 | `Brunt2010` | (a) column-density variance set by b Ms "after correcting for projection"; (b) "constrains only the product b Ms"; (c) §4.3 "the quantity most directly imprinted on the column density" | Brunt, Federrath & Price 2010, MNRAS 405, L56 (reconstructing the 3D density PDF) | For the projection correction: Brunt, Federrath & Price 2010, MNRAS 403, 1507 (arXiv:1001.1046). For the Taurus application (b = 0.48, Ms = 17.6): Brunt 2010, A&A 513, A67 (arXiv:1002.1239). Neither contains the sentence about "only the product"; that is our inference and should be worded as such. |

The same keys are used in the thesis bibliography; check `your_bib_file.bib` for the same substitutions.

**Claims:**

| ID | Where | Claim | Verdict | What the source says |
|---|---|---|---|---|
| L-35 | Paper Eq. 2, Eq. 3 | Projection operator; zeta = 1 solenoidal; F_comp = (1 - zeta)^2 / (1 - 2 zeta + 3 zeta^2); natural mixture 1/3 | OK (in A&A 512, A81) | Their Eq. 6 and Eq. 9 with D = 3. |
| L-36 | Paper §2.1 "parabolic spectral profile peaking at k = 2 \TODO{confirm the forcing band}" | Resolve | Schmidt et al. 2009 (the Enzo module): weight ~ k^2 (2 k0 - k)^2, non-zero for 0 < k < 2 k0, with k0 = 2 for alpha = 2, i.e. the band 0 < k < 4 peaking at k = 2 (as thesis Ch. 4 states). Federrath et al. 2010 (FLASH) use 1 < k < 3. Autocorrelation time T = L/(2V) in both. Ours is the Schmidt form; say 0 < k < 4. |
| L-37 | Paper §4.2 | Brunt & Federrath 2014: projected line-of-sight field contains only transverse (solenoidal) modes | OK with a caveat | "the projected field ... contains only contributions from transverse structure in 3D". They apply it to momentum density rho v and state that application to velocity "is impossible except under conditions of uniform density". Our "in a medium of uniform density" is consistent. |
| L-38 | Paper §2.3 and §6.2: "supersonic flows convert solenoidal into compressive motions through shocks, while at low Mach number compressive modes radiate away as sound waves (Federrath et al. 2010)" | FIX | Neither statement is in Federrath et al. 2010 (no "sound waves", "convert"). The Mach dependence of the solenoidal ratio is in Federrath et al. 2011, PRL 107, 114504 (arXiv:1109.1760), which speaks of "vorticity generation in oblique, colliding shocks" (shocks generate solenoidal motions, the opposite direction to our sentence) and a "strong drop of solenoidal energy for low-Mach, compressively driven turbulence". Reword and re-attribute (see also P-C8). |
| L-39 | Paper §2.3, §4.2 realised chi_v | Comparison with the literature | Add | Federrath 2011 fit: compressive forcing gives E_comp/E_tot ~ 0.72 (Ms 1), 0.63 (Ms 2), 0.59 (Ms 5), ~0.58 (high Ms); solenoidal goes from ~0 to ~0.19. Our 0.66-0.68 (xi = 0, Ms >= 3) and 0.18-0.26 (xi >= 0.5) are close; the agent could not confirm the fit coefficients against the PDF. Federrath 2010 / 2013 quote inertial-range values (1/3 and 1/2; 0.43), a different quantity. The paper compares chi_v with nothing; this comparison would strengthen §2.3. |
| L-40 | Paper intro | "the bispectrum, and the Cramer statistic (Burkhart et al. 2009)" | FIX | Burkhart et al. 2009 has the bispectrum but no Cramer statistic. That is Yeremi et al. 2014 (ApJ 783, 93) / Koch et al. 2017 (from memory, not opened). |
| L-41 | Paper intro | 1-2% per free-fall time (Krumholz & McKee 2005; Krumholz & Tan 2007) as an observation | OK | 2005 is the theory (SFR_ff ~ 0.014); 2007 is the observational paper (~1%, "a few percent" in dense gas). |
| L-42 | Paper §2.3 | w = rho^{1/2} v, squared magnitude twice the kinetic energy density (cf. Kritsuk et al. 2007) | OK | Their E(k) is built on w = sqrt(rho) u. |
| L-43 | Suggested in P-A6c | Molina et al. 2012: sigma_s^2 = ln[1 + b^2 M^2 beta/(beta + 1)] | OK, with a condition | Their Eq. 20; holds for B ~ rho^{1/2}. Not in the bib. |
| L-44 | Paper §2.4, thesis s7 | Critical density ~2e3 cm^-3 for CO J = 1-0; 110.2013 GHz; X(13CO) = 1.5e-6 | OK | LAMDA: 1.9e3 (13CO) and 2.2e3 (12CO) with p-H2 at 10 K, two-level, no trapping. The rates in the current LAMDA files are from Yang et al. 2010; Schoier et al. 2005 is the database paper. |
| L-45 | C-A4, C-B6 (my earlier notes) | RADMC-3D background and Doppler catching | Updated | Manual: for LVG populations the background "default is blackbody at 2.73 K" (`lines_tbg`), so my worry that the LVG excitation lacks the CMB was unfounded for the populations. Whether a background is added to the line *images* could not be verified from the manual; our Rayleigh-Jeans ceiling of exactly 7.59 K says it is not. Doppler catching exists (`doppcatch`) and is off by default; our runs do not use it. |

**Bibliography metadata** (59 entries; 39 DOIs resolved through Crossref, 16 eprints through the arXiv API; Andre2010, Kendall2017,
Vovk2005, Dullemond2012 not verified online):

- **Correction to my own earlier note:** before the agents reported I listed four suspect entries (Padoan2011, Pogosyan2005, Ridge2006,
  Burkhart2009). Those were artefacts of my quick parser reading fields from the following entry. All four entries are internally correct.
- `Zhang2026` is now published: ApJ 998, 14, doi 10.3847/1538-4357/ae3727, title "Machine Learning Estimation of Energy Fractions in
  Magnetohydrodynamic Turbulence Modes". Entry is still `@misc`.
- `Boerner2023` pages "1--4" should be "173--176".
- `Burkhart2012` journal should be ApJ Letters (755, L19).
- `Su2019` title: "Longitudes of 25.8-49.7 deg"; add doi 10.3847/1538-4365/aaf1c8.
- Journal macros without the backslash will print literally: `Crutcher2012` and `McKee2007` (`araa`), `Krumholz2005` and `Shu1977` (`apj`), `Larson1981` (`mnras`).
- Raw Unicode / HTML that BibTeX may mangle: en dashes in pages (Brunt2010, Burkhart2009, Esquivel2010, Lazarian2000, Lazarian2006,
  Padoan1998, Pogosyan2005); "O'Shea", "Rudiger", "Falceta-Goncalves" with raw accents; `Lazarian2000` title contains `H<scp>i</scp>Power`;
  all-caps Crossref titles (Bryan2014, Burkhart2009, Burkhart2012, Esquivel2010, Padoan2011, Raskutti2016).
- Missing entries the paper needs: Federrath et al. 2010 (A&A 512, A81), Federrath, Klessen & Schmidt 2008, Padoan et al. 1997,
  Brunt et al. 2010 (MNRAS 403, 1507), Brunt 2010 (A&A 513, A67), Lazarian & Pogosyan 2006, Federrath et al. 2011 (PRL), the MWISP DR1
  paper (arXiv:2512.08260), Solomon et al. 1987 if the 0.7 R^0.5 form is used, and optionally Molina et al. 2012 and Yeremi et al. 2014.

---

## Where to start (auditor's priority list across all batches)

1. P-A1 with S-A3 and S-A5: the 1.1-1.5x offset. Report fold-model spread on real tiles, run the synthetic control through the real-data path, then decide what the paper can claim.
2. P-A6 / C-A3 / S-A1: dust section. Fix or drop the background subtraction, include Orion A in the control, or remove the section from the paper as the red note suggests.
3. C-A1 / P-A4 / S-A4: chi_v. State that the network recovers the run-level value; do not present observed cloud values as physical.
4. C-A2: describe the uncertainties as they are computed (conformal on the aleatoric sigma; MC dropout as a diagnostic; no post-hoc temperature scaling).
5. P-A2 / T-A1: thermal width and the Ms < 3 explanation; subtract sigma_th in the baseline.
6. C-A4 / P-A3 / S-A6: LVG check with final out-of-fold models, corrected text, and the 2.5x cost.
7. P-A7, P-A8, S-A2: MWISP count, effective ceiling, Orion A resolution.
8. C-B8 / S-B1: at least a second seed for the two headline models, or soften every comparison smaller than the retraining scatter.
9. Citations: L-30 to L-34 (five keys point to the wrong papers), then L-5, L-12, L-14, L-16, L-18, L-38, L-40.
10. Observations after the final run: N-1 (report fold-model ranges, not one model), N-2, N-3.
11. Thesis: T-A1 to T-A7, X-1 to X-3 (X-3 first: the Ch. 2 classifier split), B4-4 to B4-6, then the stale numbers.

---

## Changes made on 2026-11-10 (at the user's request; nothing committed or pushed)

### New finding while fixing

**N-7. All 42 runs share one forcing seed** (`DrivenFlowSeed = 20150419` in every parameter file; `sims_mach_grid.json`). The forcing
sequence is the same realisation in every run, scaled in amplitude and time and projected with a different zeta. Labels differ
between runs, so this is not label leakage, but the suite samples the large-scale flow pattern less widely than 42 runs suggest,
and "held-out simulations" share their forcing history with the training runs. Now stated in paper §2.1 and Limitations.
Also: the training split's highest Mach number is 15.8 (the validation frames hold 15.8-19.0), which is the ceiling seen on observations (P-A8).

### Code (`p79d_kyes`)

| File | Change |
|---|---|
| `networks_nbisht/net9010.py` | `seed` module switch used by the trainer. |
| `ViT_setup_test_net9010.py` | `--fold full`: train on all 42 runs; conformal quantiles from the pooled out-of-fold residuals of the five fold models (cross-conformal). `--seed`. Seed and calibration source written to the quantile json. Dry run on saved predictions: pooled q90 = 2.22 (final_proj) against 3.79 from the single calibration fold of the old `foldall` model. |
| `orchestrate_net9010.py` | Plan `full`: six `--fold full` models (all input types) and a second-seed 5-fold run of the projected full-moment model (`final_proj_s2`, `--seed 2`). Pooling skips `full` jobs. |
| `predict_observations_net9010.py` | Headline model is `foldfull` if present (else `foldall`); both excluded from the k-fold statistics; per-tile `Ms_kfold_min/max`, `chi_kfold_min/max` written. |
| `predict_dust_net9010.py` | Adopted value is the prediction on the map as observed; the background-subtracted value is kept as a labelled diagnostic (its floor manufactures contrast). `foldfull` handled. |
| `make_paper_figures.py` | Observation figures use `Ms_foldfull` when it exists. |
| `compare_lvg_lte.py` | Final run-2 models by default, evaluated out of fold (`--models oof`); output to `~/plots/lvg_check_final`. |
| `paper_numbers_net9010.py` | Linewidth baseline also with the thermal and channel widths removed in quadrature, and against sqrt(3) x the mass-weighted LOS Mach number. |
| `control_realpath_net9010.py` (new) | Synthetic cubes observed at 2 px per beam with noise, moments, then the real-data upsampling path, predicted out of fold; compares with the standard training-style image (the control of S-A5). 12-image smoke test: path/standard = 0.98. |
| `submit_checks_final.sh` (new) | One job: the control (0.2 K and 0.37 K noise), the LVG comparison, `paper_numbers` with the linewidth baseline. |

### Jobs queued

- `orch_full` (22142317): 11 highmem trainings, two at a time. Six `--fold full` models first (`final_proj` and `final_co13` started
  19:48), then `final_proj_s2` folds 0-4. Expect roughly 1.5-2 days. The orchestrator stops after 11.5 h: rerun
  `sh submit_orchestrator.sh ~/projects/p79d_dataset/models/orch_full.json` until `status` shows all done.
- `checks_final` (22142350): control, LVG, linewidth; about 1-2 h once it starts.
  Outputs: `~/plots/control_realpath*/control_realpath_final_co13.txt`, `~/plots/lvg_check_final/lvg_lte_summary.txt`,
  `models/paper_numbers_final2.txt`.

### Still to do when the jobs finish (not done, needs the outputs)

1. `sh submit_obs_final.sh` after all six full models exist (it picks up `foldfull` automatically), then `make_paper_figures.py`.
   Replace the first value in each Table 4 entry, the chi_v values, the MWISP and dust numbers; keep the fold ranges.
2. Fill the five new `\TODO`s: second-seed numbers (§3.4), corrected linewidth baseline and the cause at Ms 1-3 (§4.6), Table 5 from
   `lvg_check_final` (§4.7), the control result (§5.5), and the decision on the dust subsection (§5.4).
3. Sources for the Serpens cube (COMPLETE data page) and the Orion A cube (§5.1 TODOs).
4. Thesis Ch. 3: the full refresh to the final numbers (the chapter still describes the 41-run models), T-A3 line 210, T-A5 to T-A7.
   Thesis Ch. 2 and Ch. 4 findings (X-1 to X-3, B2-*, B4-*) are untouched; Ch. 4 has someone else's uncommitted edits.
5. Not attempted: mask augmentation for the column-density and CO models (C-B1), a non-LTE run 2 (S-A6), spin-up frames
   excluded from training (P-B1), per-Mach-bin conformal calibration.

### Paper (`p79d_paper/.../*.tex`, compiles with no undefined references)

- **Bibliography:** eight verified entries added (Federrath et al. 2010 A&A; Federrath, Klessen & Schmidt 2008; Federrath et al. 2011;
  Padoan et al. 1997; Lazarian & Pogosyan 2006; Brunt et al. 2010; Brunt 2010; Molina et al. 2012; plus the MWISP DR1 paper and
  Belloche et al. 2011) and every citation repointed (L-30 to L-34). Zhang & Hu updated to ApJ; Boerner pages; Burkhart 2012 journal;
  six journal macros.
- **Abstract / conclusion:** uncertainties described as computed; chi_v as a run-level quantity; offset attributed to the CO-trained
  models, cause stated as open, 22 MWISP clouds; dust no longer a "lower limit".
- **§1:** linewidth estimator with sqrt(3) and kinetic temperature; Brunt and Padoan citations; Cramer statistic removed; resolution
  channel and uncertainty wording.
- **§2:** forcing band (0 < k < 4, peak at 2) and shared seed; mode-conversion sentence replaced and compared with Federrath et al. 2011;
  run-1 saturation; run-2 count; RADMC details; LVG cost; white noise; contiguous-channel rule; 98% detection qualified; fold assignment
  and spin-up snapshots.
- **§3:** the learnable T is not Guo et al.'s temperature scaling; all dropout layers in MC sampling; time budget; mask augmentation
  only for the projected full-moment model; conformal on the aleatoric sigma, exchangeability between simulations; model for
  observations trained on all runs with cross-conformal quantiles.
- **§4:** fold artefact at Ms ~ 6.5; spin-up images; chi_v within-run statistics; kNN paragraph; degeneracy with within-run correlations
  and matching bins; raw intervals are aleatoric; coverage per fold; thermal-width explanation corrected; chi_v from CO; LVG paragraph;
  two of the author's red notes that were plain leftovers removed.
- **§5:** Serpens and Orion A data descriptions; Table 4 with fold ranges and correct r; model-to-model scatter and the two separate
  effects; chi_v paragraph rewritten (no conclusion about individual clouds); MWISP citations, subset wording, 22 of 55, large-tile
  statistics; dust control in all four clouds, three open explanations, raw values, no lower limits; offset paragraph.
- **§6:** dust caveat; 6-16 labels; chi as a run-level quantity; Schmaltz and Hu/Zhang statements; LTE cost; mask augmentation;
  limitations (scale, ceiling, shared seed).
- **Left as they are:** the author's red notes at the top of the dust subsection and in §6.4 (decisions for the author), and the older TODOs.

### Thesis (`chapter3` and `your_bib_file.bib` only)

Wrong citation keys repointed (s1, s3) and the eight entries added; 13CO thermal width (T-A1); Larson/Solomon relation (T-A2); Guo et al.
(s5); "thousands of distinct labels" (T-A3, line 27 only); dust control sentence (T-A4); Hu / Zhang & Hu and "all four suites" (s9).

---

## Changes made on 2026-11-12 (after the full-data trainings and checks finished)

Jobs finished: six `--fold full` models (final_proj, final_co13, final_coboth, final_mom0, final_co12, final_13co), `final_proj_s2` folds 0-4, `checks_final`. Submitted `obs_final` (22175751) with the full-data models.

### New findings from the finished jobs

- **N-8 (second seed, final_proj, same folds).** Pooled Ms scores reproduce (R2 lnMs 0.989, median 7.5%; Ms<0.5 bin 18% vs 56%). Image by image the two trainings differ by a median 7% at Ms>=1 (as much as the error); geometric mean of the two has 6.0% error. Per-run biases are only partly reproducible: the three runs near Ms 6.5 over-predicted 20-26% by seed 1 are within 5% in seed 2, which over-predicts xi_0_mach15.6 by 31% and under-predicts xi_0.25_mach112 by 15%. So the paper's "fold assignment" explanation for the Ms 6.5 outliers was wrong; changed.
- **N-9 (chi_v is seed-dependent).** R2(chi_v) at Ms>=1: 0.75 -> 0.63; at Ms 1-3: 0.68 -> 0.29; per-run medians 0.82 -> 0.64; per-run predicted chi_v differs between seeds by 0.04 median, up to 0.22. Coverage of the 90% Ms interval 0.93 -> 0.89.
- **N-10 (the CO offset is reproduced on simulations).** On held-out synthetic FCRAO-like run-2 maps, final_co13 / Ms_obs (same image) has median 1.09 at Ms 3-7, 1.19 at 7-20, 1.15 at Ms>=3 (16-84%: 0.99-1.30; per-run medians 0.96-1.42). The linewidth reads 10% low at Ms 7-20 and the network 7% high. The earlier statement "on synthetic maps Ms_obs is unbiased within 6%, so the offset arises in the step to real maps" was wrong for the nearby clouds (1.1-1.3). MWISP (1.3-1.6) remains above this.
- **N-11 (real-data path control).** 336 held-out images through 2 px/beam + moment-map upsampling: path/standard = 0.98 (0.2 K) and 0.98 (0.37 K), 0.96-0.99 in every bin. The upsampling is not the cause.
- **N-12 (non-LTE, final models out of fold).** Ms ratio LVG/LTE: 13CO 0.95-1.06, both 1.02-1.10, 12CO 1.04-1.40 (was up to 2.1 with the leaky pre-final models). chi_v shift: 13CO within 0.03; both +0.04 to +0.22; 12CO up to +0.64. The two-line model is NOT insensitive in chi_v.
- **N-13 (linewidth baseline, corrected).** Thermal + channel removed: 14% / 7% / 10% at Ms 1-3 / 3-7 / 7-20, R2 0.88; ratio 1.14 / 0.96 / 0.90. Against sqrt(3) x mass-weighted LOS Mach: 23% / 7% / 5%. What limits the linewidth at Ms 1-3 is not identified.
- **Data sources.** Serpens cube `serpens_13co_xyv.fit.gz` is listed in the COMPLETE FCRAO archive (checked the directory listing). Orion A `Fig1.fits` is from CDS J/A+A/564/A68 according to the note in `get_Observation_Data.py`; the CDS server was unreachable, so the catalogue ReadMe was not checked.

### Code

- `paper_numbers_net9010.py`: prints network/Ms_obs on held-out synthetic images (N-10). Output regenerated in `models/paper_numbers_final2.txt`.

### Paper

- §3.4: second-seed sentence. §4.1: new paragraph on the second seed; fold-assignment explanation corrected. §4.2: seed dependence of chi_v. §4.6: corrected linewidth baseline, linewidth bias stated per bin, network/Ms_obs on simulations. §4.7: Table 5 replaced by final models out of fold with a chi_v column for 13CO; text rewritten. §5.1: Serpens and Orion A cube sources. §5.5: offset paragraph rewritten (N-10, N-11). Abstract, conclusion, discussion (LTE item, new limitation "Single trainings") brought in line.

### Observation run with the full-data models (obs_final 22175751, 2026-11-12)

- **N-14.** 13CO run-2 model (all 42 runs): cloud medians / Ms_obs = 1.19 Perseus, 1.24 Ophiuchus, 1.03 Serpens, 1.26 Orion A; tiles >90% detected: 0.97-1.34, median 1.12. MWISP (22 clouds, tiles <=10 pc): 13CO ratio 1.17 (rho 0.90), two-line 1.56 (rho 0.86). The 13CO offset now matches the 1.15 on held-out simulations (N-10); only the two-line excess on MWISP is unexplained.
- **N-15.** chi_v of the all-runs projected model is outside the range of the five fold models for Serpens (0.46 vs 0.19-0.37) and Orion A (0.30 vs 0.17-0.24): further evidence that chi_v on real clouds is training-dependent.
- **N-16.** Column-density model on 13CO integrated intensity: all-runs model gives 1.2/1.0/0.9/0.5 of Ms_obs (Per/Oph/Ser/Ori), fold models 0.7-1.5, 0.7-1.8, 0.3-0.9, 0.2-0.3. Strongly training-dependent; Orion A low for every model.
- Paper: Table 4 first values, §5.2, §5.3, §5.4 control sentence, §5.5, abstract and conclusion updated; figures regenerated with `make_paper_figures.py` (it also copies one figure into thesis `Figures/chapter3`). Compiles, 28 pages, no undefined references.
- Remaining `\TODO`s are all older ones: Python package citations, a citation for Ms<20 in Galactic clouds, Herschel map releases, Fig. obs restyle, architecture-improvement number in §6.5, UAT IDs, and the decision on the dust subsection.
- Still open: thesis Ch. 3 refresh to these numbers; Orion A cube provenance to be confirmed against the CDS ReadMe (server down on 2026-11-12); MWISP-setup control for the two-line model; linewidth excess at Ms 1-3.

## Changes made on 2026-11-13 (user asked for the remaining tests; nothing committed or pushed)

### Jobs submitted

- `orch_nomask` (orchestrator 22201138, started 17:41): five trainings `final_proj_nomask_f0..4`, same as `final_proj` but without `--mask_aug`, for the mask-augmentation comparison in §4.1 (now `\prov{7.4\%}` vs `\prov{7.8\%}` from pre-final models). State file `~/projects/p79d_dataset/models/orch_nomask.json`; resubmit with `sh submit_orchestrator.sh <state>` if it stops after 11.5 h. Trainings wait on the per-user job limit (other project's jobs are running) and may straddle the 16-18 Nov outage.
- `checks2` (22201212, one shared job, 32 cores): (a) real-data-path control with the MWISP setup (6.7 -> 1.67 px/beam, 0.16 km/s channels, 0.25/0.47 K) for final_co13, final_coboth, final_co12, compared with the standard MWISP-like file; (b) the FCRAO-setup control for final_coboth; (c) `diag_linewidth_net9010.py`.

### Code

- `orchestrate_net9010.py`: `plan_nomask()` and `PLANS["nomask"]`.
- `control_realpath_net9010.py`: generalized to several lines (`--lines`, `--noise`), a reference file fed directly (`--std_data`, sample order asserted), velocity rebinning (`--chan_kms`), `--label`. Smoke test (final_coboth, MWISP setup, 16 images at Ms>=6): path/std 0.99, path/true 0.98, path/Ms_obs 1.08.
- `diag_linewidth_net9010.py` (new): tile linewidth after RT (noise-free) and after observation against sqrt(3) x mass-weighted and volume-weighted LOS Mach and the 3D Mach number, by Ms, n_H2 and xi.
- `submit_checks2.sh` (new).

### Findings

- **N-17 (`images` in the CO file).** In `p79d_mach_grid_256_co_v2_fcrao500.h5`, `images` is identical to `images_13co` (noise-free RT moments), not the projected fields; only the v1 projected file has Sigma-weighted moments. Not an error in any result, but the diagnostic cannot use it as a "projection" stage.
- **N-18 (linewidth excess at Ms 1-3, preliminary, 540 images).** RT linewidth / sqrt(3) Ms rises with density: 1.03 at n_H2 100-300, 1.13 at 300-1000, 1.29 at 1000-3000; the mass-weighted LOS dispersion is 0.93. Points to 13CO optical-depth broadening. To be confirmed on a third of the dataset by `checks2` before it goes into §4.6.
- **Resolution robustness.** Confirmed from `models/robustness_9010_final_proj_allmom.txt` (final models): f=0.4 gives 7.6% vs 7.4%; no rerun needed.
- **MWISP tile sizes.** §5.3 said tiles span 10-45 pc with spectral Mach up to ~47; outputs give 11-53 pc and ~46. Corrected.

### Estimates given for audit issues 1, 3, 5 (no decision yet)

- (1) mask augmentation for the CO/column-density models: 12 trainings for 13CO run 2 + two-line (about 1.5 days), 30 for all five (3-4 days).
- (3) excluding spin-up frames: all 36 trainings (about 4 days); alternative is excluding them from evaluation only.
- (5) second seed for the other five models: 25 trainings (about 3 days); 13CO run 2 only is 5.
- Recommendation given: (1) for the 13CO and two-line models, (3) evaluation-only, (5) for 13CO run 2 only.

### Still to do when the jobs finish

- `checks2`: update §5.3 (control ratio), §5.5 ("We have not run the corresponding control for the MWISP setup"), §4.6 ("We have not identified what limits the linewidth there").
- `final_proj_nomask`: pooled OOF numbers into §4.1, then remove the `\prov` macro and the header comment in `4_Results.tex`.

### Results of `checks2` (22201212, finished 2026-11-13 19:53) and paper edits

- **N-19 (MWISP-setup control).** 270 held-out images, Ms>=2. 13CO model: path/std 0.99, std/true 1.04, std/Ms_obs 1.14, path/Ms_obs 1.12. Two-line model: path/std 0.99, std/true 0.98, std/Ms_obs 1.06, path/Ms_obs 1.04. 12CO model: path/std 1.00, path/Ms_obs(12CO) 0.87. Two-line on FCRAO setup: path/std 0.99, path/Ms_obs 1.04. So the 13CO ratio on real MWISP clouds (1.17) is reproduced; the two-line ratio (1.56) is not, and neither the setup nor the upsampling path produces it. It appears only with real 12CO maps. Still unexplained.
- **N-20 (linewidth excess at Ms 1-3 = opacity broadening; confirms N-18).** 10834 images. Noise-free 13CO linewidth (thermal removed) / sqrt(3) Ms at Ms 1-3: 1.04 / 1.15 / 1.30 for n_H2 100-300 / 300-1000 / 1000-3000; mass-weighted LOS dispersion 0.93 at every density. Same ordering in every Ms bin (7-20: 0.92 / 0.94 / 0.99). sqrt(3) x volume-weighted LOS Mach / Ms = 1.00 in all bins (LOS scatter 16-84%: 0.95-1.06).
- Paper: §4.6 (opacity paragraph replaces "not identified"), §5.3 (old "median ratio of 1.05" replaced by final-model 1.04/1.14 and 0.98/1.06), §5.5 (MWISP control added; two-line excess appears only with real 12CO), conclusion items "Observed clouds" and "Linewidth baseline", abstract. Compiles, 28 pages, no undefined references.
- Outputs: `~/plots/control_realpath_mwisp/*.txt`, `~/plots/control_realpath/control_realpath_final_coboth.txt`, `models/diag_linewidth.txt`.

### No-mask-augmentation comparison (`final_proj_nomask`, five folds finished 2026-11-14 10:36)

- **N-21.** Pooled out of fold, same folds and suite as `final_proj`: R2(lnMs) 0.986, median error 7.7% (with mask aug: 0.980, 7.4%; second seed with mask aug: 0.989, 7.5%). Bins: 22% / 6.0% / 5.7% / 10.9% / 6.5% (with: 56% / 7.1% / 5.2% / 8.9% / 7.2%). Pooled chi_v R2 0.69 vs 0.56 (seed 2: 0.28), 90% Ms coverage 0.88 vs 0.93. All differences are within the seed-to-seed scatter, so the earlier claim that mask augmentation "slightly improves" the simulation results is not supported. Summary in `models/kfold_summary_final_proj_nomask_allmom.txt`.
- Paper: §4.1 sentence replaced with the final numbers (no cost in accuracy, difference within training scatter); §6.5 "Mask augmentation" item changed accordingly; `\prov` macro and its two header comments removed (no provisional numbers remain). Compiles, 28 pages, no undefined references.

## Changes made on 2026-11-16 (user asked to prepare audit issues 1, 3, 5 and to look into the two-line excess)

### Audit issues 1, 3, 5

- **Issues 1 and 5 (trainings).** `orchestrate_net9010.py`: new `plan_audit()` (17 trainings on run 2: `final_co13_mask` and `final_coboth_mask` with `--mask_aug_prob 0.5`, five folds plus the all-runs model each; `final_co13_s2` with `--seed 2`, five folds) and an `"after"` key, so an all-runs model is only submitted once the five fold files of its group exist (`--fold full` exits otherwise). State file `models/orch_audit.json`; orchestrator job 22259434 submitted on 2026-11-16 and left pending, so Slurm starts it when nodes are available; no cron job needed. Scope is the one recommended on 2026-11-13; the user did not narrow or widen it.
- **Issue 3 (spin-up frames), evaluation only.** `analyze_kfold_net9010.py --exclude_sims`. Leaving out xi_0.5_mach160 and xi_0.5_mach320 (21 images in every dataset) changes no pooled R2, median error or coverage by more than 0.003 for any of the six final models (largest: column-density model R2 0.8845 -> 0.8871). Summaries in `models/kfold_summary_<tag>_<mode>_nospinup.txt`. One sentence added to §4.1. Retraining without these frames is not needed.

### Two-line excess on MWISP (in progress)

- The resolution channel is not the cause: r = 1.00 for all 22 clouds with tiles <= 10 pc. Over these clouds the two-line/13CO prediction ratio rises with tile size (Spearman 0.64) and the excess falls with Ms_obs.
- `predict_observations_net9010.py --dump_inputs <npz>`: saves the network inputs and per-line detected fractions of every tile; outputs then go to `<output_dir>_dump` so the existing run is untouched.
- `diag_twoline_net9010.py` (new): compares per-line and between-line statistics of the inputs of real MWISP tiles with held-out synthetic MWISP-like images, and intervenes on the inputs (12CO width, centroid spread, contrast moved to the other set's 12/13 relation; 12CO maps replaced by 13CO ones) to see which moves the two-line prediction.

### Two-line excess on MWISP: result (`diag_twoline` job 22260895, 2026-11-17; `models/diag_twoline.txt`)

- **N-22.** 36 real tiles (22 clouds, tiles <= 10 pc) against 651 held-out synthetic MWISP-like images (23 runs, Ms >= 2). Tile-level two-line/Ms_obs: 1.46 real, 1.07 synthetic.
  - The real tiles are outside the training distribution: 13CO detected over a median 35% of the tile (synthetic: 100%, central 95% is 68-100%; every real tile outside), 12CO over 83%. Per-pixel 13CO dispersion 0.8 c_s (about one 0.166 km/s channel) against 2.6 c_s; 12CO/13CO dispersion-map ratio 3.3 against 1.46; 12CO intensity contrast (std ln W) 1.29 against 0.19 (97% of real tiles outside).
  - Interventions on the inputs: 12CO dispersion map divided by the real/synthetic difference (2.2) -> ratio 1.05; replaced by the 13CO dispersion map -> 0.96; the opposite change on synthetic images -> 2.00. Centroid or intensity maps changed likewise: < 0.1. So the excess enters through the 12CO linewidth input.
  - Not established: which estimate is nearer the truth (the 13CO width measured over 1-2 channels is itself truncated), and how much of the 3.3 is real 12CO broadening versus truncation of faint 13CO.
- Paper: §5.5 second part rewritten (was "not explained"); §5.3 pointer; conclusion and abstract. The two-line MWISP Mach numbers are now called extrapolations. Compiles, 28 pages, no undefined references.
- Possible follow-up (not done, needs the user's decision): re-observe the synthetic cubes at lower signal-to-noise (higher noise or lower abundance), so that training covers 13CO detected fractions of 20-60%; the mask-augmentation trainings now running (`final_co13_mask`, `final_coboth_mask`) cover partial detection but not truncated 13CO widths.
- Audit trainings: `orch_audit` started 2026-11-16 19:40 (the compute nodes were accepting jobs); fold 0 of the two mask-augmentation groups running.

### 2026-11-18: audit trainings hit a full file system

- Between about 2026-11-17 23:53 and 2026-11-18 05:30 writes to `/anvil/projects/x-phy240036` failed with "No space left on device" (quotas now show 21.5% used and writes succeed, so the condition was transient, cause not known). The orchestrator (22285047) died saving its state; `final_co13_s2_f2` (8.2 h in) and `final_co13_mask_f3` (6.5 h in) failed when saving the checkpoint; `final_coboth_mask_f3` and `final_co13_s2_f3` failed in 25-30 s (no log file could be written). In `sacct` they show the reason `QOSMaxJobsPerUserLimit`, but that is only the reason they had been pending, not the cause of the failure.
- Fix: the four jobs reset to queued in `orch_audit.json` (backup `orch_audit.json.bak_20261118`), resubmitted by one `run --once` cycle from the login node (22299213-16), orchestrator resubmitted (22299191). `orchestrate_net9010.py save()` now logs and continues on an OSError instead of dying.
- Status: 8 of 17 done (folds 0-2 of both mask groups, folds 0-1 of the seed-2 group); 4 pending; 5 queued (folds 4 and the two all-runs models).

### 2026-11-18: figure condensing (user request)

- Fig. 11 (tile maps): `make_paper_figures.py fig_obs_maps` now builds the figure from the data (13CO integrated intensity recomputed with the observation pipeline via `_obs_m0`, tiles from `observations_net9010_final_proj/*_tiles.csv`) instead of copying the diagnostic PNGs of `predict_observations_net9010.py`. Offsets in pc, colour bar, cloud names, labels in white boxes: predicted Ms of the all-runs model and (in brackets) Ms_obs; chi_v dropped from the labels (the paper says chi_v of observed clouds is not constrained). One file `fig_obs_maps.pdf`; caption rewritten (TODO for Ophiuchus/Serpens kept). The figure was never cited in the text; now cited at the start of §5.2.
- Figs. 14 + 15 merged: `fig_dust_scale` with `_dust_panel` (a) and `_linewidth_scale_panel` (b); `fig_dust` and `fig_linewidth_scale` removed. Paper: one figure* `fig:dust_scale`, refs changed to 14a / 14b. Old files `figures/fig_obs_{perseus,orion}.png`, `fig_dust.pdf`, `fig_linewidth_scale.pdf` are no longer referenced (not deleted).
- Figs. 9 + 10: ideas proposed to the user, nothing changed yet. Text use of the chi_v panels: Fig. 9 right is never cited for chi_v (the values in §4.6 are also in Table 2's last column); Fig. 10 right (MAE chi_v vs A_V) is not mentioned anywhere in the text. Fig. 10 left legend overlaps the 13CO curve.
- Paper compiles, 28 pages, no undefined references; 14 figures (was 15).

## Changes made on 2026-11-20 (user chose idea A for Figs. 9/10, asked for 14a with the all-runs model and a check for stale results)

### Audit trainings finished (`orch_audit`, all 17 done by 2026-11-20 00:29)

- **N-23 (mask augmentation and second seed, run 2, pooled out of fold).** 13CO: median error 9.1% (first training) / 8.4% (seed 2) / 8.5% (mask aug.); R2 lnMs 0.966 / 0.970 / 0.967; 90% coverage 0.889 / 0.881 / 0.902. Two-line: 6.5% / - / 7.1%; R2 0.980 / - / 0.964; coverage 0.894 / - / 0.901. R2(chi_v) per bin (Ms 1-3, 3-7, 7-20) moves by up to 0.15 either way between trainings (13CO: 0.40/0.46/0.66, seed 2 0.55/0.49/0.74, mask 0.44/0.58/0.55). Mask augmentation makes no difference beyond training scatter for the CO models either. Summaries `models/kfold_summary_final_{co13_mask,co13_s2,coboth_mask}_allmom.txt`.
- Paper: §3.4 sentence (CO models also trained with mask aug. as a test); §4.6 one sentence with these numbers.

### Figures and tables

- Figs. 9 + 10 merged (idea A): `fig_model_comparison` now (a) median Ms error by Ms bin, markers + lines, all five training sets, (b) run-2 Ms error vs median A_V (Ms >= 1). Both chi_v panels dropped. `fig_av` deleted from `make_paper_figures.py`; figure `fig:av` removed from §4.6, its references now `\ref{fig:model_comparison}b`. Colour per model fixed across panels (`MODEL_STYLE`), with a distinct marker per model.
- Table 2 (`tab:ms_metrics`): the chi_v range column replaced by three columns R2(chi_v) at Ms 1-3, 3-7, 7-20 (values from the k-fold summaries); table comment removed.
- Fig. 14a (`_dust_panel`): both estimates now from the all-runs models (dust: final_mom0 foldfull; 13CO: final_proj foldfull, joined from `observations_net9010_final_proj`), with the five-fold range as vertical bars; reference line 0.4:1. Was the five-fold geometric mean.
- `predict_dust_net9010.py`: adopted value is the all-runs model (`Ms_adopted` = foldfull, else the k-fold mean); summary gives the range of the per-fold medians next to it; matched tiles carry `Ms_CO_proj_adopted`. Rerun in `obs_audit` (below); the predictions themselves do not change.
- 4_Results header comment: `kfold_net9010.py` -> `analyze_kfold_net9010.py` (P-C14 / C-C4).

### Dust numbers with the all-runs model (paper §5.4, caption, §6, conclusion, abstract, intro)

- **N-24.** Matched tiles: dust/Ms_obs median 0.39 (range 0.19-0.74) with the all-runs model, 0.28-0.32 for the fold models (was quoted 0.3, range 0.1-0.5). Dust still below the 13CO projected-map model on every tile. Per-region medians (all-runs): Perseus 3.3, L1688 2.5, Orion A 9.5, Orion B 3.3, Aquila 2.6, L1495 1.8, Musca 1.1, Cham I-III 2.5/2.8/5.2 (was 2.8, 2.0, 7.1, 2.4, 2.4, 1.6, 1.5, 2.9/3.5/4.5 pooled over fold models). The all-runs value is outside the fold-model range in 9 of 10 regions: 10-40% above in six, 15-25% below in Musca and Cham I/II. Background subtraction: < 20% in five regions, 40% in Aquila, x1.6-2.3 in Musca/Chamaeleon.
- Text: "0.3 times" -> "0.4 times (0.3 for the fold models)"; "about a third" -> "0.3-0.4 times"; intro "several times lower" -> "two to three times lower"; "factor of three" -> "factor of 2.5".

### Job

- `obs_audit` (22328732, `submit_obs_audit.sh`): dust rerun with the new summary; `final_co13_mask` and `final_coboth_mask` on the nearby clouds and MWISP, to see whether mask augmentation changes the offsets on partly detected tiles and the two-line MWISP excess.

### `obs_audit` result (22328732; moved from shared to highmem after 2 h pending in a queue of ~11.7k jobs; ran 6 min)

- **N-25 (mask-augmented CO models on real data).** MWISP, 22 clouds with tiles <= 10 pc, median over clouds of (cloud-median prediction / cloud-median Ms_obs), all-runs model [five fold models]: 13CO 1.18 [1.16-1.32] -> mask 1.35 [1.12-1.52]; two-line 1.56 [1.32-1.53] -> mask 1.58 [1.34-1.85]; Spearman rho 0.90/0.90 and 0.86/0.89. Nearby clouds, 13CO model, tiles > 90% detected: median 1.12 (0.97-1.34) -> 1.21 (1.01-1.40); cloud medians 1.13/1.18/1.03/1.17 -> 1.29/1.21/1.12/1.24 (Per/Oph/Ser/Ori). So mask augmentation (partial detection, unchanged linewidths) neither removes the two-line excess nor reduces the 13CO offset; all shifts are within the scatter between trainings. Consistent with N-22: the excess enters through the 12CO linewidth input, not through the detected fraction. (The 1.18 here vs 1.17 in the paper is rounding of a slightly different median order; not changed.)
- Paper: §5.5 two sentences after the two-line diagnosis; §6.5 mask-augmentation item extended (CO models kept without mask aug.). Compiles, 28 pages, 13 figures, no undefined references.
- Dust rerun: `dust_summary.txt` now headlines the all-runs model; numbers match those put into §5.4 earlier today.
- Summary of the month's changes by paper section: `audit/CHANGES_SUMMARY_2026-11.md`.

## 2026-11-20 (later): read-through before the advisor's reading (user request; paper had been committed at 7b3a889)

- Deleted unreferenced figure files: `fig_av.pdf`, `fig_dust.pdf`, `fig_linewidth_scale.pdf`, `fig_obs_{perseus,orion}.png`, `fig_xi_confusion.png`; removed `fig_xi_confusion` (it only copied a thesis figure) from `make_paper_figures.py`.
- Consistency fix: conclusion said the non-LTE check moves Mach numbers by "at most 15%"; Table 5 and the abstract give 10% (13CO <= 6%, two-line <= 10%). Now 10%.
- Abstract rewritten from ~400 to ~255 words (ApJ limit 250), same numbers. Intro P1 shortened; intro P5 no longer says the offset "cannot yet be explained" (it is traced, §5.5).
- Cut to thesis: §3.2 Architecture development and Table 1 (net9001-9009 scores on an older dataset, unverified, with a TODO); replaced by one sentence in §3.2 (CNN stem beat a pure ViT in development runs). "net9010" removed from the figure caption. §2.3 dilatation/vorticity bullet, skewness/kurtosis and stored spectra (never used); §2.4 LVG cost history (one sentence, points to §6.5); wall-clock budget detail; batch-norm implementation note in §6.5.
- Moved to a new appendix (`9_Appendix.tex`, `\input` after acknowledgements): App. A Non-LTE check (former §4.7 text and Table 5); App. B two-line excess diagnosis (input comparison, interventions, mask-aug test). Main text keeps a short §4.7 summary and a short paragraph in §5.5.
- Condensed: §4.6 linewidth-baseline paragraph (same numbers, fewer words); §5.5 split from one ~1200-word paragraph into five; §6.2, §6.3 and the first paragraph of §6.4 (overlap with §1, §3.5, §4.2); §6.5 resolution item; conclusion item "Observed clouds".
- §6.4: the author's red note ("reads more like an introduction para") addressed by rewriting the paragraph as a comparison and removing the note. The red note at the top of §5.4 (dust) is left for the author.
- Small fixes: "(v2)" removed from the Fig. 12 caption; MWISP sentence grammar; §5.5 "however, is not" across a paragraph break reworded; Hu (2025) verb agreement.
- Result: 27 pages (was 28), no undefined references.
