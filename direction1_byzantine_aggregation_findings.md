# Direction 1 — Byzantine-Robust Federated Aggregation: Consolidated Findings

**Status:** Empirical investigation complete on real degradation data; ready to inform §3.5 of `chapter3_draft.md` and/or a dedicated results section. This document consolidates the session-by-session narrative in `experiment_log.md` (Session 8 onward) into a single, self-contained account: motivation, method, two experiments, results, and a recommendation.

**One-line summary:** reframing RFTL-S's existing federated-MAD/Huber machinery as one instance of a broader family of Byzantine-robust aggregation rules, benchmarking six such rules under two contamination models, and finding that **Multi-Krum is the clear winner** — while also surfacing a genuine methodological finding: theoretical robustness guarantees proven for a single aggregation step do not automatically survive when that step is embedded inside an iterative, non-convex, coupled optimization (as federated tensor factorization is).

---

## 1. Motivation and relationship to the existing chapter draft

`chapter3_draft.md` §3.2.2 already sketches **RFTL-U**, a server-side, whole-user reweighting scheme based on Grassmann-distance trimming, explicitly describing it as "a federated, Grassmann-manifold analogue of trimmed mean robust aggregation as used in robust federated learning [refs needed]." Direction 1 takes that connection and makes it the explicit object of study: rather than one bespoke consensus-weighting rule, we benchmark the actual family of aggregation rules from the Byzantine-robust FL literature — coordinate-wise median, trimmed mean, Krum, Multi-Krum, and geometric median — against each other and against plain (non-robust) aggregation, in the federated MPCA setting Chapter 2/3 already uses.

**Motivation for the pivot.** Every contamination model tried earlier in the chapter's empirical arc (Gaussian sample noise, physically-motivated structured artifacts — hot-pixel blocks, readout stripes) turned out to be largely absorbed by the estimation pipeline: real, physically-plausible corruption carries limited, incidentally-shaped energy that tends to settle into its own subspace direction rather than distorting the directions that matter downstream (documented at length in `experiment_log.md`, Sessions 6–7, for both the prediction and monitoring pipelines). The question this raised — is *any* aggregation rule choice meaningful here, or does everything just get absorbed regardless? — is what Direction 1 answers.

**Open item, not yet resolved:** how this empirical benchmark should be reconciled with the existing formal RFTL-U proposition (§3.2.2–3.3.3, §3.4). The two are complementary — RFTL-U's Grassmann-trimming scheme is itself a candidate rule that could be added to the benchmark — but integrating them into one section of the dissertation is a decision for Madi, not made here.

---

## 2. Experimental setup

### 2.1 Federated architecture

Federated MPCA as in Chapter 2/3: each client fits a local projection `V_m` from its own data; a shared global projection `U` is estimated from the clients' aggregated contributions. `code/mpca_byzantine.py` (`MPCA_FD_Robust`) makes the server-side aggregation step explicit and pluggable: each client computes a local scatter-matrix contribution to each tensor mode's shared subspace, and these are combined with a configurable rule *before* eigendecomposition, in place of the batch/incremental SVD `MPCA_FD` uses for the same target. Passing `agg_fn=sum` was verified to recover `MPCA_FD`'s behavior on clean data (principal angle 2.60°±5.81° across 5 repeats — the residual gap is the expected numerical difference between the two algorithms, not a discrepancy in target).

### 2.2 Synthetic client expansion (n=3 → n=8)

The real federated setting has only 3 clients (users A/B/C, distinct image crop sizes from Chapter 2's design). This is too small for the classical guarantees to apply: median/trimmed-mean degenerate at n=3 (a "coordinate-wise median" of 3 values is a raw selection, not a smoothed estimate), and Krum's guarantee requires n ≥ 2f+3 = 5 for a single Byzantine client.

`code/pilot_byzantine_agg.py::build_synthetic_clients()` splits each real user into sub-cohorts — split plan `[('A',2),('B',3),('C',3)]` — reaching **n=8 synthetic clients**, sizes 16–23 samples each. These are honestly synthetic (same underlying sensor data, randomly subsampled, not independent physical sites) but this expansion is exactly what a future non-IID/heterogeneity study (Direction 2) will also need, so it is directly reusable groundwork.

### 2.3 Aggregation rules under study

Implemented in `code/byzantine_agg.py`, operating on a list of n per-client matrices:

| Family | Rule | Mechanism | Requirement |
|---|---|---|---|
| Non-robust baseline | `sum` | Plain sum (≡ pooling all client data centrally) | — |
| Coordinate-wise | `median` | Per-entry median across clients | tolerates < n/2 Byzantine |
| Coordinate-wise | `trimmed_mean` | Drop top/bottom `n_trim` per entry, average rest | n > 2·n_trim |
| Distance-based | `geometric_median` | Weiszfeld iteration — a distance-weighted average, preserving cross-entry correlation | tolerates ~n/2 Byzantine (asymptotic) |
| Distance-based | `krum` | Keep the single client whose neighborhood is most mutually consistent | n ≥ 2f+3 |
| Distance-based | `multi_krum` | Average the (n−f) lowest-Krum-score clients | n ≥ 2f+3 |

Citations for these methods (Blanchard et al. for Krum, Yin et al. for coordinate median/trimmed mean, and the "small-magnitude attack" literature for the single-shot-vs-iterative point below) are well-known but have **not yet been pulled and verified** — flagged as a pre-submission task, not a blocker for the empirical findings themselves.

---

## 3. Experiment 1 — physically-motivated contamination (hot_block artifact)

**Design.** One client ("A0" at n=8; the single real client "user 0" at n=3) has a fraction π of its samples replaced with a fixed, spatially-coherent artifact (`contamination.py::contaminate`, `hot_block` mode — a saturated pixel block, same location/value across all contaminated samples, amplitude and block size matched to the earlier monitoring/prediction pilots). Each rule's dirty fit is compared to **its own clean fit** (not a shared reference — see the methodological note below, which applies here too, discovered independently for this simpler design first).

### 3.1 The n=3 degeneracy

At n=3, comparing every rule's dirty fit against a single *median*-based clean reference initially suggested median was catastrophically worse than sum. Re-running with each rule's own matched clean reference isolated the real effect:

| Check | n=3 result |
|---|---|
| sum vs. `MPCA_FD` (sanity) | 2.60° ± 5.81° |
| sum-clean vs. median-clean (**attack-free**) | **20.73° ± 6.95°** |
| sum: dirty vs. own clean, π=0.6/1.0 | 1.89° / 1.89° (flat) |
| median: dirty vs. own clean, π=0.6/1.0 | 38.19° → 60.61° |

Median was substantially biased relative to sum **even with zero attacker**, purely from cross-client heterogeneity: at n=3, a "coordinate-wise median" of 3 values is a raw per-entry pick among only 3 clients, not a smoothed estimate, so real heterogeneity (not just an attack) swings it. This also meant Krum (n≥5 needed) and a genuinely-averaging trimmed mean (n > 2·n_trim with slack) were not well-posed yet.

### 3.2 The n=8 fix and remaining findings

Expanding to n=8 and adding the two distance-based rules gave a full six-way comparison (`output/pilot_byzantine_agg.csv`, 5 repeats, hot_block π ∈ {0.6, 1.0}):

| Rule | Heterogeneity (attack-free) | Robustness (π=0.6, dirty vs. own clean) |
|---|---|---|
| sum | 0.00° ± 0.0004° | 1.07° ± 0.49° |
| median | 21.20° ± **19.44°** | 19.08° ± **20.27°** |
| trimmed_mean | **1.78° ± 0.47°** | **1.64° ± 0.93°** |
| geometric_median | 10.69° ± 4.06° | 15.34° ± 4.98° |
| krum | 30.27° ± 8.96° | 40.45° ± 10.34° |
| multi_krum | **4.23° ± 1.16°** | **4.48° ± 1.30°** |

Findings:

1. n=8 reduced median's magnitude (38–61° → ~19–21°) but not its instability: its robustness angle (19.08°) is not meaningfully larger than its own attack-free heterogeneity angle (21.20°) — median is simply a noisy, high-variance rule at this scale, attack or not.
2. **trimmed_mean and multi_krum stood out** — small, low-variance deviation from sum both attack-free and under attack, in sharp contrast to their nearest relatives (median, plain Krum).
3. Plain Krum was the *worst* performer of the six, both attack-free (30.27°) and under attack (40.45°) — it keeps only 1 of 8 clients and discards the rest, a well-known low-statistical-efficiency criticism of vanilla Krum, reproduced here empirically.
4. **None of the six rules showed a dramatic attack-specific effect beyond their own baseline noise.** Sum's near-total insensitivity (1.0–1.07°, unmoved even at π=1.0) is consistent with the "absorption" mechanism documented throughout the chapter's monitoring/prediction pilots: a large, low-rank, repeated artifact tends to claim its own eigenvector slot rather than distorting the directions that matter. This raised the question motivating Experiment 2: is this contamination model simply too benign to differentiate the rules at all?

---

## 4. Experiment 2 — the adversarial "hijack" attack

### 4.1 Threat model and attack construction

The classical Byzantine-robust FL literature does not constrain the attacker to derive its contribution from real (even corrupted) data — a compromised client can send an **arbitrary** contribution at the aggregation step. `code/byzantine_attacks.py::hijack_attack` implements this directly in scatter-matrix space: it eigendecomposes the honest clients' aggregated scatter matrix, targets the honest data's own **weakest** eigendirection (the direction real data says matters least — the classic "single large outlier hijacks unweighted averaging" construction), and injects a rank-1 malicious contribution there, scaled to `amplitude_mult × honest top eigenvalue`. A `target='random'` variant serves as a sanity check that the finding is not an artifact of targeting the specific weakest direction. `MPCA_FD_Robust` was extended (`adversarial_client`/`adversarial_fn`) to let one client's per-mode scatter contribution be overridden this way at every aggregation call, rather than derived from `client_data` at all.

This assumes an "omniscient" attacker (can observe/estimate the honest aggregate) — standard, if strong, in this literature for deriving worst-case guarantees.

### 4.2 A methodological pitfall, found and fixed

The first run of this experiment produced a confusing result: `geometric_median` was hijacked almost as badly as plain `sum` (~85–90° from a shared clean reference), contradicting its textbook 1/2 breakdown-point guarantee. This was investigated by directly instrumenting `MPCA_FD_Robust` to capture the exact matrices passed to `agg_fn` during live execution.

**Finding:** the aggregation step itself was correctly rejecting the malicious contribution at *every single call* — the captured inputs, run standalone, converged to within 0.01–0.1° of the true honest-clients' aggregate. The apparent "hijack" was an artifact of the *comparison*, not the defense: comparing a defended fit (which necessarily excludes the target client's real information once the defense works) against a clean reference that *includes* that client's real, legitimate data conflates two distinct things — whether the attack was rejected, and the unavoidable cost of losing one real, non-IID client's unique signal once it is correctly excluded. A direct no-attack control confirmed this: merely removing client 0's real data (zero attacker involved) shifted `geometric_median`'s converged fit by **33.87°**, versus **1.25°** for `sum` — i.e., `geometric_median` (and, to a lesser extent, `krum`) have real, attack-independent optimization instability in this non-convex, alternating (V, U) setting, purely from client-composition sensitivity.

**Fix — every comparison now reports three numbers:**

- **`angle_vs_loo_clean`** — dirty fit vs. a clean fit trained on the 7 *other* clients only (target absent entirely). This isolates whether the attack was actually rejected.
- **`angle_vs_full_clean`** — dirty fit vs. a clean fit on all 8 real clients (total real-world cost: attack + exclusion combined).
- **`no_attack_baseline`** — the *same* LOO comparison with zero attacker, giving each rule's own natural noise floor. A rule whose attacked-LOO angle doesn't exceed this baseline hasn't actually been beaten.

### 4.3 Results (5 repeats, `output/pilot_byzantine_hijack.csv`, amplitude 20×, target='weakest')

| Rule | No-attack baseline | Attacked (LOO) | **Residual attack effect** |
|---|---|---|---|
| sum | 0.97° | 89.64° | **+88.66°** (catastrophic, as classical theory predicts) |
| median | 30.44° | 27.49° | −2.94° (see caveat below — not real robustness) |
| trimmed_mean | 1.58° | 4.89° | **+3.31°** |
| geometric_median | 18.55° | 75.53° | **+56.98°** (worst of the five robust rules) |
| krum | 19.51° | 33.61° | +14.09° |
| **multi_krum** | **1.27°** | **4.22°** | **+2.94°** (best) |

**Random-direction sanity check** (target='random', amplitude 20×, LOO metric): confirms the finding is not an artifact of the specific 'weakest' target.

| Rule | Angle | Note |
|---|---|---|
| sum | 83.36° ± 0.85° | Consistently hijacked regardless of direction |
| median | 49.00° ± **33.16°** | Enormous variance — unusable as a robust rule |
| trimmed_mean | 48.99° ± **31.19°** | Good vs. the *targeted* attack, but fragile and inconsistent against a random direction |
| geometric_median | 79.07° ± 4.34° | Consistently bad (low variance) — a real, stable vulnerability, not noise |
| krum | 33.61° ± 18.29° | Matches the 'weakest' result; moderate variance |
| **multi_krum** | **4.22° ± 0.98°** | Small and stable — the only rule confirmed robust regardless of attack direction |

### 4.4 Why geometric_median and krum underperform their theoretical guarantees

Both rules' single-call aggregation is provably correct (verified directly, §4.2), yet both show large *cumulative* damage over the full 30-iteration fit. The mechanism: their higher inherent optimization instability in this iteratively-coupled, non-convex (V, U) alternating scheme compounds across rounds, against an attacker that gets to adaptively retarget its injection every round (the "weakest honest direction" is recomputed fresh each call, tracking the honest representation as it evolves). This is a genuine, citable point distinct from anything in the original benchmark plan: **classical Byzantine-robustness guarantees, derived for a single aggregation step, do not automatically transfer to settings where that step is embedded inside an iterative, coupled non-convex optimization** — which is exactly the setting federated tensor factorization lives in, and one the mostly single-shot/convex classical literature does not cover.

---

## 5. Synthesis and recommendation

**Multi-Krum is the carried-forward robust aggregation rule for Direction 1.** Across both experiments — the physically-motivated hot_block artifact and the literature-standard adversarial hijack attack, at multiple amplitudes and two attack directions — it is the only rule that is simultaneously low-baseline-noise, low-residual-damage, and consistent regardless of attack direction. Trimmed mean is a reasonable second choice against a *known, targeted* threat model but showed real fragility against an untargeted/random-direction attack; median, despite an apparently small residual effect, is disqualified by its own enormous baseline variance; plain Krum and geometric median, despite the strongest textbook guarantees among the distance-based family, underperform once the iterative-compounding effect is accounted for.

**Two findings worth foregrounding in the chapter write-up**, beyond "which rule wins":

1. **The "absorption" theme recurs.** Across the whole chapter's empirical arc (prediction pipeline, monitoring pipeline, and now this benchmark), physically-plausible, low-rank, structured contamination tends to be absorbed into its own subspace direction rather than damaging the directions that matter — visible here as every rule's near-flat response to the hot_block attack (Experiment 1). Differentiating the rules required abandoning the "physically plausible corruption" framing entirely in favor of a literature-standard *arbitrary* Byzantine contribution.
2. **Single-shot robustness ≠ iterative robustness.** This is the more novel methodological point: two rules with strong, provable single-aggregation-step guarantees (Krum's breakdown point, geometric median's 1/2 breakdown point) both failed to preserve that robustness once embedded in a 30-round alternating optimization against an adaptive attacker, for reasons unrelated to the aggregation step itself. Multi-Krum's advantage over plain Krum, and its clean win over geometric median, is consistent with lower inherent optimization variance being what actually matters once you're inside an iterative loop.

---

## 6. Limitations and open items

- **Citations not yet verified.** Krum (Blanchard et al.), coordinate median/trimmed mean (Yin et al.), and the small-magnitude/"stealthy" attack literature are referenced by well-known name/mechanism but have not been looked up and confirmed (exact venues/years) — needed before this goes into the dissertation.
- **Synthetic client caveat.** The n=8 split is real data, honestly subsampled, but not 8 independent physical sites — this should be stated plainly wherever these results are reported, and revisited if/when Direction 2's heterogeneity work suggests a more natural multi-site construction.
- **RFTL-U reconciliation.** How this benchmark relates to `chapter3_draft.md`'s existing RFTL-U (Grassmann-distance trimming) proposal is an open scope question for Madi — options include folding RFTL-U in as a seventh candidate rule, keeping the two as separate contributions, or superseding RFTL-U's aggregation scheme with Multi-Krum's.
- **Amplitude/direction coverage.** The hijack attack was tested at 3 amplitudes and 2 directions (weakest, random); a fuller sweep (more random-direction repeats, intermediate amplitudes, multiple simultaneous Byzantine clients at f>1) would strengthen the claim further but was not pursued here given time constraints.
- **Downstream task not yet connected.** This entire investigation operates at the subspace-recovery level (principal angle). It has not yet been connected to either the prediction pipeline (`rftl_s_real.py`) or the monitoring pipeline (`monitoring_real.py`) as a fourth method arm — a natural next step once the citation and scope questions above are resolved.

---

## 7. Reproducibility

| Artifact | Path |
|---|---|
| Aggregation rules | `code/byzantine_agg.py` |
| Robust federated MPCA (pluggable aggregation + adversarial injection) | `code/mpca_byzantine.py` |
| Hijack attack construction | `code/byzantine_attacks.py` |
| Experiment 1 pilot (n=8, 6 rules, hot_block) | `code/pilot_byzantine_agg.py` → `output/pilot_byzantine_agg.csv` |
| Experiment 2 pilot (hijack attack, LOO/full/baseline metrics) | `code/pilot_byzantine_hijack.py` → `output/pilot_byzantine_hijack.csv` |
| Full session-by-session narrative (including the n=3 degeneracy, now superseded on disk but preserved here and in the log) | `experiment_log.md`, Session 8 onward |

Note: the n=3 version of Experiment 1's output CSV was overwritten in place when the n=8 version was run (both used the same default output path); its numbers are preserved in §3.1 above and in `experiment_log.md`, not recoverable from a CSV on disk. Re-running `pilot_byzantine_agg.py` with `SPLIT_PLAN` reduced to the 3 real users would reproduce it if needed.
