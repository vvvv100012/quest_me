# Final independent consult: rolling address-profile stability

The question is useful, but **the design as proposed would mislead the owner if implemented without resolving the defects below**. It mixes conditional persistence, availability, duration-dependent summaries, and trading-volume-weighted aggregates under one stability label. Several proposed calculations are not uniquely defined. A good result would support a narrow descriptive statement about this address universe and this history; it would not validate sleeve selection.

I applied `REVIEW_POLICY.md`, including its materiality gate. This is the final review, not a request for another reviewer. No network or external service was used; package addresses stayed local. Protected files were read only. All new files are under `work/codex10/`. The only code written is a small diagnostic, not an evaluation implementation.

## 1. Findings, evidence, and disposition

### Material defects in the proposed specification

**B1 — “Pooled Spearman” does not identify a calculation, and one natural implementation gives the opposite answer.** Concatenating address–Monday rows ranks both calendar effects and addresses. In the reproducible diagnostic, 11 cross-sections of ten observations each have Spearman **−1 at every date**, while concatenation gives **+0.983635**. This failure needs neither overlap nor an attacker. Pool **within-date correlations**, with equal Monday weight; do not pool raw values or globally ranked observations. Overlap does not invalidate a descriptive average, but does invalidate treating its rows as independent evidence. This is a blocker to reproducible, correctly labelled output, not evidence that the existing data have that particular reversal.

**B2 — Complete-episode statistics change their observation filter when the window changes from 60 to 28 days.** For stationary starts and fixed duration d, the expected number of fully contained episodes in a window of length T is proportional to **max(T−d, 0)**. Thus their duration distribution is proportional to `f(d) max(T−d,0)`, not simply `f(d)`. With equally frequent 7-day and 21-day episodes, their observed long/short ratio is 39/53 in a 60-day window and 7/21 in a 28-day window: the forward relative representation is **0.453** of the lookback representation with no behaviour change. Durations above 28 days cannot enter a fully contained forward episode statistic at all.

This mechanism is present in the current data, not just hypothetical: the fixed fill-episode builder produces **45,548** full-window complete episodes, including **5,097 above seven days, 1,109 above 28 days, and 242 above 60 days**. Full-window p95/p99 duration is **16.001/46.519 days**. These are full-universe episode counts, not estimates for the rolling eligible cohort. Consequently, a change in holding summary cannot be called a change in the address's holding tendency without qualification. Preserve the specified complete-only summaries, label them accurately, add a matched 28-versus-28 diagnostic, and retain turnover time and censoring counts. Do not attempt to “fix” this by using future closes in L.

The same issue, without episode censoring, affects **number of symbols**: a 60-day distinct count and a 28-day distinct count have different opportunity sets. Quantiles, HHI, and active-day summaries also have different finite-sample properties. A duration sensitivity is not a calibrated reliability correction.

**B3 — The 90-day sensitivity is unavailable on four proposed dates.** Its first four starts are **July 1, July 8, July 15, and July 22**, all before the first package fill on **July 27 at 08:52:47.550 UTC**. Price prehistory does not supply missing address transactions. Filling that interval with zeros, or silently shortening it, changes the profile and eligibility. Use 90 days only on the seven dates **October 27–December 8**, and compare all lookback lengths on those dates with explicit common-cohort results.

**B4 — Universal rank/tercile rules are undefined for a category and misleading for heavily tied columns.** `top_symbol` has no ordinal scale. Alphabetical Spearman or terciles would be meaningless. Current full-window eligible data demonstrate the tie issue: **340/530** nonmissing `adds_per_ep_med` values are in one tie, and **335/541** `forced_ntl_share` values are in one tie. Arbitrarily assigning tied addresses to thirds creates distinctions absent from the profile; keeping ties together destroys the universal one-third chance baseline. Specify category agreement separately and preserve ties, with an explicit no-discrimination outcome. These full-window checks establish reachable data patterns, not the exact rolling tie counts.

These are concrete specification/output defects under the review policy. The remaining points are interpretation requirements or design choices, not additional code findings.

### Framing: agreement, disagreement, and overstatement

**Agree:** a Monday trailing-60-day description versus the next 28 days matches the **age of the input and holding horizon** of one newly selected sleeve. Every Monday is relevant because a different sleeve turns over each week. Four calendar phases represent the four sleeve schedules.

**Disagree:** this is sufficient validation of the operating plan. The study does not specify which profile columns choose addresses, the selection function, allocations, diversification, exit behaviour, or treatment of unavailable profiles. A cross-sectional rho over hundreds of addresses need not describe the few selected names or their tails. Persistent account size can dominate dollar-column ranks without preserving a practically useful description at fixed dollar levels. Stable marginal columns do not establish stability of a joint profile or of a selection decision.

**Overstated:** “profile stability” interpreted as an intrinsic trader trait. The observed result combines changing behaviour, market state, transaction opportunities, estimation noise, and censoring. Maker scores describe executed orders, not placement intent. Addresses are the observational units; neither independent people nor the full potential copy-trading universe are identified.

**Availability is part of the result, not a footnote.** Independent order/day counts reproduce the supplied medians: **362 L-eligible**, **344 trading in F**, and **212 reaching 30 orders/10 days in F**. The proposed relaxed 15-order/5-day gate admits a median **274**, substantially more than 212 but still a selected subset. Counts vary materially: September 29 has **320 eligible, 274 passing the relaxed F gate**; December 8 has **393 eligible, 206 passing**. A falling gated rho or changing rho can reflect changing composition. The supplied alignment counts imply still greater attrition, but this diagnostic did not recompute them.

Forward gates are defensible as **precision/reportability screens**, not as a missing-at-random mechanism. Conditioning on forward activity may inflate or reduce agreement; its direction is not identified from the gate alone. Do not assert inflation as an established numerical fact. Report conditional agreement, reportability by L tier, and the joint chance of being both reportable and retained. Do not equate no new fills with no exposure: a carried position can remain measurable on the grid.

### Measurement reliability

**Agree that sensitivity to limited observations deserves a separate diagnostic. Disagree that even/odd or interleaved halves identify a noise ceiling.** The halves share regimes, positions, nested return paths, and possibly snapshot brackets. Splitting orders/episodes also changes weighting, dependence, and information size. Selection on L and common estimation inputs violate the simple independent-error argument.

Under a classical model `X=theta_L+e_L`, `Y=theta_F+e_F`, independent errors would give Pearson correlation `corr(theta_L,theta_F) sqrt(R_L R_F)`. This is an assumption-dependent illustration, **not a Spearman deattenuation formula**. Here neither reliability is identified, and interleaving can look artificially reliable while chronological drift remains real.

The diagnostic simulates a perfectly unchanged Gaussian latent trait with independent observation noise, variance inversely proportional to window length. It produces:

| Assumed 60-day reliability | Observed rho | Same-tercile fraction | Extreme-switch fraction |
|---:|---:|---:|---:|
| 0: independent noise | −0.003 | 0.333 | 0.223 |
| 0.5 | 0.385 | 0.455 | 0.118 |
| 0.8 | 0.705 | 0.597 | 0.039 |
| 0.9 | 0.840 | 0.690 | 0.011 |
| 1.0 | 1.000 | 1.000 | 0.000 |

These are **illustrations, not fitted reliabilities or forecasts for this dataset**. They show why rho below .5 or retention below 50% cannot establish behavioural change. Conversely, noisy observation is an operational limitation even if the latent trait is constant.

The revised specification uses a transparent, matched-length historical diagnostic rather than estimating an unidentifiable ceiling. Effective notional counts, event/day counts, censoring, and coverage accompany each score. An interleaved comparison may be an explicitly exploratory supplement, but is neither required nor used to normalize agreement or label a column stable.

### Each measure

1. **Rank agreement:** retain per-Monday Spearman with average ranks for ties, plus an equal-Monday mean and temporal dispersion. Avoid global concatenation, averaging p-values, or presenting n(address–dates) as independent n. No Fisher transformation is needed for the descriptive estimand chosen here. Constant columns produce NA, not zero correlation.
2. **Level change:** retain, but define ratios at zero, two-sided factor-of-two changes, units, and the exact cohort. Signed alignment values require differences, not ratios. A 0.2 alignment-unit threshold is an arbitrary descriptive threshold; it has no common operational meaning with a 20-percentage-point share change. Absolute level retention and relative rank retention answer different questions.
3. **Terciles:** useful coarse resolution at these sample sizes, provided ties and availability are exposed. With independent, balanced thirds, same-tier probability is **1/3**, any top–bottom switch is **2/9**, and top→bottom conditional on initially top is **1/3**. Under general margins they are `sum(p_Lt p_Ft)` and `p_L1 p_F3+p_L3 p_F1`. Neither “50%” nor “one third” is universal once there are ties or fixed L tiers followed by attrition. At n=138, the independent balanced-null binomial standard error for retention is about **0.040**, at n=274 about **0.028**; these are only scale illustrations, not valid study error bars. Realistic measurement error can yield either side of 50% even with perfect latent persistence, as the simulation shows.
4. **Survival:** rename **forward reportability/activity**. An address failing a gate has not necessarily stopped existing or abandoned trading. HIP-3 is a separate scope limitation, not a terminal state for all columns. Supply a mutually exclusive primary reason and overlapping diagnostic flags. Reportability denominators must be the L-reportable population of the specific column as well as the common eligible universe.
5. **Tier-pooled values:** retain as an explicitly event/notional-weighted supplementary estimand. It does not recover the unobserved values of inactive addresses. Even without forward gates, the pool can be dominated by large or active addresses. For example, forward alignment scores +1 and −1 at weights 1 and 100 pool to −99/101, although their equal-address mean is zero. Show both summaries and concentration. Compute the L spread with **the identical pooling functional**, not differences between address tercile cut points or address medians. With ordinary transient measurement noise, sorting on L inflates apparent L separation, and regression to the mean can contract forward separation without latent drift. A ratio below one is not a calibrated failure. Compare raw spreads and zero separation; do not claim a universal expected spread ratio under persistence. A permutation null tests absence of association, not the amount of decay under a stationary noisy trait.
6. **Forward weeks:** useful for the profile's observed usefulness at successive ages, but varying week gates change the population and seven-day estimators are noisier. Reusing 15 orders/5 days leaves median samples **112, 107, 97, 94** in weeks 1–4, before any column-specific requirements. At December 8 they are only **72, 76, 64, 65**. Use explicitly lighter weekly gates plus a common-four-week cohort. Even then, an age pattern also contains the realized calendar regimes; it is not identified pure decay. Complete-episode weekly statistics have an additional seven-day truncation and cannot establish holding-style decay.

### Uncertainty

**Disagree that the proposed two-way percentile intervals provide credible temporal uncertainty.** A 28-day target implies dependence across at least four consecutive Mondays; dependence in the 60-day inputs is longer, and persistent positions/address effects extend further. Eleven dates cannot estimate a general dependence structure or calibrate a block-length choice. More resamples cannot create new regimes. “Mondays in blocks” also omits block length, edge handling, and the combination with address resampling.

There are **three disjoint 28-day forwards in each of three sleeve phases and two in the fourth**. Their entire forward union is September 29–January 5, **98 days**, or 3.5 intervals of 28 days. Even the disjoint forwards are not necessarily independent because their L windows and market conditions overlap. The late three-date group spans only **42 forward calendar days**.

Report the actual date series, four phase summaries, and ranges. If an address bootstrap is kept, resample whole address trajectories once per draw and label it **conditional address-resampling sensitivity on the realized dates**. It is not a future-regime interval, and the fixed sample is close to a census of the supplied universe. Cross-address dependence further limits a population-CI interpretation. The revised specification deliberately gives no calibrated temporal or unconditional 95% confidence interval. A time block bootstrap with 11 dates is not a better source of credible coverage.

### Design/confirmation split

**Agree with preserving a frozen late-period display; disagree with calling it an independent confirmation.** The earlier one-off split and alignment choice already used the late data. No temporal partition recovers that hold-out. Keep the labels **early / bridge / late historical replication**; freeze this specification before calculation, show all dates, and make no keep/drop decisions from any of these groups.

The proposed early/late forward boundary is arithmetically right: the last early F ends **November 24 exclusively**, the first late F starts then. Overlap between a late L and earlier observations is normal for sequential deployment and is **not look-ahead**. On November 24, the trailing L is September 25–November 24, so part of the earlier forward history legitimately becomes its input. Do not purge legitimate past information merely to make the picture look independent. If truly prospective validation is wanted, freeze now and collect future data and an explicit selection rule; neither is supplied here.

### Look-ahead audit

| Item | Assessment and exact resolution |
|---|---|
| Funding timestamp | Daily entries represent D but are available at D+1. Use `n_samples` to distinguish daily from hourly, not midnight alone. Include daily availability **<= cutoff**, hourly event time **< cutoff**. There are **1,956–2,306 daily rows**, covering **456–523 addresses**, becoming available exactly at each M; `<` versus `<=` materially changes inputs. Reconstruct the represented day retrospectively; do not shift its economic position exposure to the next day. |
| Next account snapshot | A snapshot later than a grid point but **before M** is permitted in L's historical max-bracket proxy. A snapshot at/after M is not. This is a profile available at M, not a historical series known at every grid point. F must use its own end cutoff, not the full package cutoff. |
| Funding-only and carried positions | Rebuild their classification at each cutoff from that prefix. A pair first filled after M is still funding-only at M. Do not use its later `gstart`, eventual symbol membership, or eventual funding-run endpoint to rewrite L. The full-window `hidden_positions.parquet` and aggregated daily exposure table are not safe substitutes for prefix reconstruction. |
| Episodes | Freeze censoring at the window end. Later closes cannot turn an L episode complete. Funding-only positions never have fill-observed complete endpoints. Current `exposure.py` explicitly excludes those runs from complete summaries; preserve that correction. |
| Clusters/orders | Slice fills before forming the window's order and cluster aggregates. Future fills of a previously observed order cannot enlarge its L notional or change its mixed-side/multi-episode classification. |
| Grid boundary and prices | The last bar of L opens before M and closes at M. Its closed price is permitted exactly at M; fills and snapshots at M belong to F. Specify these different boundary conventions rather than saying all inputs have timestamps strictly before M. |
| A alignment | A uses other closes in the decision's four-hour block, including ones after the decision. All listed window boundaries are 00:00 UTC and four-hour aligned, so each block's 48 candidate closes lie **before the profile cutoff**. This is not an M-level leak. It is not an online decision-time score. See the exact candidate rule below. |
| Cross-sectional cut points | L tiers for a selection input must use L only. F ranks/cuts are permissible for an explicitly retrospective relative-rank diagnostic; they cannot masquerade as the tiers deployed at M. Keep frozen-L and reranked comparisons separate. |
| HIP-3 | Earliest known fill or funding evidence, with funding's availability convention, determines the as-of flag. Never apply a final-window address flag backward. A new F HIP-3 flag affects account value/leverage reportability, not all other columns. |
| Alignment variant, universe, mapping | Choosing raw/A using the old post split spends that data for methodological selection. Re-running with prefixes does not undo it. The supplied address universe and fixed external symbol mapping are retrospective inputs; no claim of a historically discoverable complete opportunity universe is established. |

## 2. Revised implementation specification

This is the replacement specification. It deliberately retains the reviewed profile definitions where possible and makes every change explicit. It defines descriptive arithmetic, not a model of latent stability or a return backtest.

### 2.1 Fixed inputs, calendar, and information cutoffs

Use the 1,173 addresses with package perp fills as the fixed universe U. Preserve the package loader's spot exclusion, forced-direction flags, fill notional, maker flag, position epsilon `1e-9` coins, and same-timestamp group definitions in `work/v2dev/code/lib.py`. No non-perp rows enter any statistic. Fills in HIP-3 symbols remain in fill-based frequency/behaviour statistics unless a rule below explicitly excludes them. Do not remove addresses from U using later activity.

At implementation time, record SHA-256 hashes and schemas of all input/source files used, dependency versions, the external symbol map, this specification, and the seed. Hashes are reproducibility records; this is not tamper modelling. The numerical contract below overrides inconsistent old full-window aggregation or old gates. Do not load old full-window address scores and pretend they are rolling profiles.

Every timestamp denotes UTC; use integer nanoseconds internally, float64 arithmetic, no rounding before comparisons. Output dates/times explicitly as UTC. Use 24-hour days, not local calendar/DST durations. All divisions with undefined denominators produce NA except the level-ratio conventions in §2.5. Quantiles use linear/type-7 interpolation; an even-sized median is the mean of the two middle observations. No winsorization or sign rounding.

| M at 00:00 UTC | L60 start | F end, exclusive |
|---|---|---|
| 2025-09-29 | 2025-07-31 | 2025-10-27 |
| 2025-10-06 | 2025-08-07 | 2025-11-03 |
| 2025-10-13 | 2025-08-14 | 2025-11-10 |
| 2025-10-20 | 2025-08-21 | 2025-11-17 |
| 2025-10-27 | 2025-08-28 | 2025-11-24 |
| 2025-11-03 | 2025-09-04 | 2025-12-01 |
| 2025-11-10 | 2025-09-11 | 2025-12-08 |
| 2025-11-17 | 2025-09-18 | 2025-12-15 |
| 2025-11-24 | 2025-09-25 | 2025-12-22 |
| 2025-12-01 | 2025-10-02 | 2025-12-29 |
| 2025-12-08 | 2025-10-09 | 2026-01-05 |

For each M, `L=[M−60d,M)`, `F=[M,M+28d)`, `Fk=[M+7(k−1)d,M+7kd)` for k=1..4. Compute a window S=[a,b) at **its own cutoff C=b**; Fk does not borrow snapshots, funding records, cluster extensions, or episode closes from later forward weeks. Thus a week is an as-of-week-end historical profile, not a decomposition of the fully revised 28-day profile. Reconstruct inherited positions using all available history before C, not just fills inside S. The same rule defines the matched historical window R=[M−28d,M).

Fills, snapshots, and hourly funding need event time `<C`. A daily funding aggregate for D is usable when `D+1d<=C`; its represented day is D. Daily means `n_samples` is non-null, hourly means null. No additional publication latency is assumed; state this limitation. Closed price bars with close `<=C` are usable. For a price at grid close c, use only bars closed by c and fallback fills at `t<=c` that also satisfy `t<C`.

Maintain HIP-3 evidence as any HIP-3 fill before C or usable HIP-3 funding record. Evidence persists once observed; do not reset it at S's start. It suppresses **only account-value and leverage columns** as in `exposure.py`, not native fill statistics or exposure summaries. Exposure retains all replayed perp symbols, including HIP-3, with its stated price fallback; alignment excludes HIP-3 opening fills. Treat a newly known HIP-3 address as out of AV/leverage scope for the entire S, and report this separately from inactivity.

### 2.2 Window records and profile definitions

**Orders and clusters.** Restrict fills to S first. An order is `(address,symbol,oid)` among those fills; aggregate only its in-S fills. Day of an order is the UTC day of its first in-S fill. For all order/cluster notional statistics, include opening, closing, and forced fills. Partial orders at boundaries are window fragments; never import their out-of-window fills.

Use the **episode-aware** clustering in `style.py` for both frequency/size and alignment. This explicitly replaces the non-episode-aware join in the original `profile.py`. Build group episode labels on the prefix as `style.py` does: start a label on a new pair, a zero starting position, or a sign flip. Associate the flip's fills with its new label for clustering. An order has a unique side/episode only if its in-S fills do. Sort orders by address, symbol, first time, last time, oid, stably. Join an order to the immediately preceding order iff same pair, both orders have one side and one episode, sides/episodes match, and `first_current−last_previous<=60 seconds`. This is transitive adjacency chaining, not a maximum cluster duration. Mixed-side/multi-episode orders stand alone. Negative gaps from overlapping order lifetimes satisfy this rule, as in the fixed builder. A cluster's day is its first in-S order time's UTC day. Do not join across S's boundary.

**Position grid.** Use bars opening at o on the 5-minute UTC lattice with `a<=o<b`; c=o+5min. Rebuild the prefix as in the position, carried-start, funding-only, and price rules of `exposure.py` with the cutoff replaced by C and funding filtered by availability above. More explicitly:

- A pair with a fill before C uses the last group `gend` with event time `<=c` and `<C`. Before its first fill, a nonzero first `gstart` is carried only over the consecutive funding-day run ending the day before its first fill day; run start is 00:00 of its earliest consecutive day, bounded below by July 27. Use that `gstart`, not future information beyond C. If no such run, there is no inferred pre-fill segment under this proxy.
- A pair with **no fill before C** uses each represented day's last usable funding `szi` on that day's 288 bars, ordered by original funding timestamp. Ignore sizes with absolute value below `1e-9`. Do not forward-fill over days without records. A later fill does not change this prefix classification. A filled pair's pre-first-fill reconstruction follows the preceding carried-start rule; do not add its funding-only grid again.
- Synthetic starts affect the first bar **opening** at their start, matching the existing stated convention. Group events affect the first close at/after their timestamp, with the last event at that close winning. Once the first fill occurs, its replay supersedes the pre-fill approximation.
- Price priority at c is exact Binance five-minute close, latest closed Hyperliquid four-hour candle, then latest package fill price at/before c, with no interpolation from future prices. Use the fixed symbol map. A nonzero unpriced position makes that address/grid point invalid, not zero exposure; record source and missing-price counts. Existing review evidence found no such missing-price points, but implementation must not silently convert them to flat time.
- All valued perp positions give `G(c)=sum(abs(position*price))`, `N(c)=sum(position*price)`, and `P(c)=number of nonzero positions`. The observation span starts at the earliest prefix-known first fill, inferred carry start, or funding-only represented day for the address, intersected with S; retain the existing `bar_open>=first_appearance` convention, and continue the span to b even if trading stops. Times before first appearance are not asserted known-flat. Valid, observed flat points thereafter have G=N=P=0.

**Account value.** Each snapshot's AV is `pnl+net_deposit`. For c, use the latest snapshot time `s_prev<=c` and the next strictly later snapshot, both `<C`. If both exist and `s_next−s_prev<=48h`, use max of their values. If only a predecessor exists, use it when `c−s_prev<=24h`; if only a successor exists, use it when `s_next−c<=24h`. Both-sided gaps above 48h are invalid, even if the predecessor is nearby. Exact snapshot time follows this same predecessor/strict-successor rule, as in `av_grid`; do not silently special-case it to its own value. Leverage is valid at observed, priced, in-market points with finite AV>=100. This is an endpoint-based leverage proxy, not a guaranteed bound on actual intraday leverage. Account-value median is the median of **snapshot AVs whose timestamps are in S**, not the median of this grid proxy; preserve `exposure.py`'s distinction.

**Episodes.** Use `profile.py::build_episodes` on S's groups, with position state carried into S. An inherited episode is left-censored for window-contained statistics even if its older opening is known. An episode still open at b is right-censored; one closing exactly at b is not complete in S. Funding-only episodes and positions held through S with no fill are always censored, never complete. A flip closes the old episode and opens the new at the same timestamp; split increasing notional pro rata as in the fixed builder. Same-timestamp flat-to-flat groups have no positive-duration episode under that builder. Do not use future closes or funding-run ends to remove censoring. For behaviour requiring a known average entry, use only episodes opened inside S, including right-censored ones. This preserves the old window-local left-censor exclusion.

The following table fixes the scalar columns. `I` is the set of valid observed grid points, `J={c in I:G(c)>0}`; sums over a grid use a constant weight of 5/60 hours where units require integration. `n_fills`, `n_orders`, and active days refer to in-S perp fills; active day means at least one such fill. Every median uses the quantile rule above.

| Family / column | Exact statistic |
|---|---|
| Frequency / active-day share | Active fill days divided by the number of UTC days in `[max(a, first-prefix-fill-day),b)`. At F and Fk, L-eligible addresses have the full 28/7-day denominator, including zero-trade days. Funding-only days are not fill-active days. |
| Orders per active day | Median daily count of orders assigned by first in-S fill, over all fill-active days; include zero assigned orders on an active day sustained only by earlier orders. |
| Clusters per active day | Same construction with cluster first times; include zero assigned clusters on active days. |
| Orders per cluster | Number of orders divided by number of clusters. |
| Fills per order | Number of fills divided by number of orders. |
| Maker share | Maker fill notional divided by all fill notional. |
| Holding median | Median duration in hours of window-contained complete fill episodes. |
| Holding weighted mean | `sum(open_ntl_e * duration_h_e)/sum(open_ntl_e)` over those complete episodes; open_ntl includes all increasing notional, not just initial-entry notional. |
| Turnover time | `sum_I G(c)*(5/60) / (sum_S fill_notional/2)`, hours; denominator zero gives NA even if a position persists. Numerator includes the inherited/funding proxy. |
| Median order / cluster notional | Median total in-S fill notional per order / cluster. |
| Median turnover per active day | Median of daily total fill notional on fill-active days. |
| Time in market | `|J|/|I|`; zero is valid when the observed span is flat. |
| Gross exposure median / p95 | Quantiles of G on J, not of all points including flat time. |
| Account value median | Median snapshot AV in S; suppress for HIP-3-as-of-C. Negative/low snapshots remain in this median. |
| Leverage median / p95 | Quantiles of G/AV on valid leverage points; suppress for HIP-3-as-of-C. |
| Number of symbols | Distinct symbols with fills in S, not funding-only holdings. |
| Top symbol / share | Symbol with largest in-S fill notional; equal notional breaks by ascending symbol string. Share is its notional / total fill notional. |
| Symbol HHI | Sum of squared per-symbol fill-notional shares. |
| BTC+ETH share | Fill notional on exactly BTC and ETH divided by total fill notional. |
| Long opening share | Preserve `profile.py`: notional with literal dir `Open Long` / notional with dir `Open Long` or `Open Short`. This excludes flip directions. Label that restriction; do not silently substitute all opening legs. |
| Net bias | `sum_I N(c)/sum_I G(c)`; denominator zero gives NA. |
| Long-time share | Fraction of J with N(c)>0. Exactly net-flat in-market points are in the denominator and not the numerator. |
| Median simultaneous positions | Median of P on J. |
| Adds per episode | Median number of distinct increasing-order IDs in each complete episode, **including its initial opening order**, preserving `adds_per_ep_med`; this is not a count of later adds alone. |
| Single-shot share | Among complete episodes, fraction with exactly one distinct increasing and one distinct reducing order. A flip contributes the appropriate role to both adjacent episodes. |
| Add share of notional | On episodes opened inside S, later increasing notional / all increasing notional, including right-censored episodes. |
| Add-at-loss share | On those episodes, later increasing notional added at a strictly losing running average entry / all later increasing notional. For a long, add price < running average; for a short, add price > running average. Equal is not losing. Preserve the builder's VWAP update and reduction handling. Zero later adds gives NA, not zero. |
| Flip share | Fraction of complete episodes ending by a flip, preserving `profile.py`. Not flips / fills or flips / all episodes. |
| Forced-fill share | **Forced notional / all fill notional**, preserving `forced_ntl_share`; label it as notional share, not fill-count share. |

**Alignment:** 16 scalar cells: 2 execution classes (taker/maker) × 4 horizons (1h/4h/1d/7d) × 2 variants (raw/A). No `all` class or B variant is part of the requested primary table. Build positive opening legs and eligible opening clusters exactly as in `style.py`, on the window-fragment orders/clusters defined above. Exclude forced, HIP-3, mixed-group, and multi-episode-order opening fills; a flip contributes only its residual new-position leg. Cluster scoring time t is its first eligible opening-fill time, last is its last such time; exclude duration>300 seconds. A cluster may supply positive weights to both maker and taker, and these are not independent decisions. Class weight is that class's opening notional, not the whole cluster's weight.

Let `c(t)=floor_5min(t)`, `lp(c)=log(close)`, `r_h(c)=lp(c)−lp(c−h)`. Use the 288 consecutive five-minute returns ending at c, requiring all 289 closes, with `V(c)=sum r_5min^2`. Set `sigma_h(c)=sqrt(V(c)*hours(h)/24)` and `q_h=r_h/sigma_h`. Missing return endpoints, missing volatility-window closes, or V<=0 give no score; r_h itself requires its two endpoints, not every intervening close outside the volatility window. Raw per-decision z is `sign * q_h(c(t))`. Use the fixed Binance alignment files without HL/fill fallback, as `style.py` does; exposure and alignment price coverage intentionally differ.

For A, B is t floored to a four-hour UTC boundary. Candidate closes are **B+j*5min for j=0..47** (including B and excluding B+4h), exactly the fixed candidate convention. Retain finite q candidates with candidate sigma / actual sigma in **[2/3, 3/2] inclusive**. Require >=5 candidates; own point is included if eligible. `x_A=z−sign*mean(q_candidates)`. Compute address raw/A scores as class-notional-weighted means on their respective finite decision sets. A's eligibility counts **its own** finite scores/days/weight, never raw's reportable flag. Report class notional, decision/day counts, largest decision weight share, and Kish count `(sum w)^2/sum(w^2)` on each variant's scored set. Also report first-entry fractions by decision count and by class notional on that set: first means cluster t equals the observed episode opening timestamp, as in `style.py::is_first`; a left-censored opening time cannot be first. Scored/candidate notional coverage is each variant's scored class notional divided by class notional in eligible opening clusters before duration/price/score exclusions. Separately retain notional excluded for forced, HIP-3, and ambiguous opening fills. These diagnose concentration and coverage and are not independent-sample sizes.

### 2.3 Eligibility, supports, and gates

For the primary 60-day evaluation, base eligibility `B_M` is >=30 in-L orders on >=10 distinct in-L fill days. This uses L only. Do not require future activity for B_M. Publish its count, including addresses not reportable for particular columns.

For each column j, `A_jM` consists of B_M addresses with a finite numeric L value (or nonmissing top-symbol category) and the L support gate below. Thus all defined L terciles are **column-specific**; an alignment tercile is not a tercile of all B_M. `C_jM` consists of A_jM addresses passing the F gate for that column and with finite numeric F value (or nonmissing category). Week cohorts use the corresponding weekly gate. No common all-columns complete-case filter.

| Support family | L60 gate in addition to B_M | F28 gate | Fk7 gate |
|---|---|---|---|
| Fill/order/cluster columns; symbol category/concentration; long-opening share; forced/maker shares | Defined statistic/positive required denominator | >=15 orders on >=5 fill days; defined denominator | >=5 orders on >=2 fill days; defined denominator |
| Time in market | >=14 days of valid observed grid duration | >=14 days valid observed duration | >=5 days valid observed duration |
| Gross quantiles, net bias, long-time share, simultaneous positions | Previous grid gate plus >=10 distinct UTC bar-open days with in-market valid points | Grid gate plus >=5 such days | Grid gate plus >=2 such days |
| Leverage quantiles | Grid-span gate plus valid leverage points on >=10 days, no HIP-3 evidence | Grid-span gate plus valid leverage points on >=5 days, no HIP-3 | Weekly grid-span gate plus valid leverage points on >=2 days, no HIP-3 |
| Snapshot AV median | In-window finite snapshots on >=10 distinct UTC days; no HIP-3 | Snapshots on >=5 days; no HIP-3 | Snapshots on >=2 days; no HIP-3 |
| Turnover time | Grid-span gate and positive turnover | Grid-span gate and >=15 orders/5 days | Weekly grid-span gate and >=5 orders/2 days |
| Complete holding, adds/episode, single-shot, flip share | >=10 complete fill episodes; required positive weight denominator | >=5 complete fill episodes | >=3 complete fill episodes |
| Add-notional / add-at-loss shares | >=30 eligible increasing / later-add groups on >=10 distinct days, respectively, and positive appropriate denominator | >=15 on >=5 days | >=5 on >=2 days |
| Each alignment class/horizon/variant | >=30 finite positive-class-weight decisions on >=10 decision days | >=15 on >=5 days | >=5 on >=2 days |

The extra grid support and add-behaviour gates are explicit changes, not claims that the originally proposed gates have been empirically optimized. Here a support “day” means at least one relevant point/event that UTC day, **not 24 hours of independent information**. Grid observation duration is count(valid observed points)*5min; invalid prices remove points. Count all in-market/AV-invalid support separately. For add-notional support, the eligible groups are increasing groups of episodes opened inside S, including the initial group; for add-at-loss support, only subsequent increasing groups count. Use the builder's group classification, not number of partial fills. Report support and Kish counts even when a gate passes.

For AV/leverage, no HIP-3 is required at the relevant cutoff. HIP-3 does not automatically fail any other support family. No trading in F fails order/alignment/complete-episode gates but need not fail exposure/AV gates. Only time-in-market has a valid all-flat zero grid value; conditional in-market summaries are NA when no in-market points exist.

**Fixed gate sensitivity:** additionally show F28 order-derived columns at >=30 orders/10 days, and alignment at >=30 variant-specific decisions/10 days. For these panels show both the native gated cohort and the primary statistic restricted to the identical stricter cohort; do not interpret differences as pure measurement-noise effects. Do not choose a gate from the better result.

### 2.4 Rank and categorical agreement

On C_jM, define `rho_jM = Pearson(average_rank(L), average_rank(F))`. Require >=3 paired addresses and nonconstant ranks on both sides, otherwise NA with reason. Counts below 30 receive a small-cohort annotation but are not erased. For any fixed group of Mondays D, report **arithmetic mean of finite rho_jM**, median, min/max, K valid dates, all date-specific counts, and the dates omitted. This estimates an equal-date average of conditional cross-sectional associations. Do not weight by address count, precision, orders, or capital. Do not concatenate raw rows.

Top symbol uses the same order support family, but report `p_same=mean(1[top_L=top_F])`, the full symbol×symbol transition counts, and the independent-marginal baseline `p0=sum_s p_L(s)p_F(s)`. Report chance-adjusted `(p_same−p0)/(1−p0)` when p0<1. No rho, ratio, factor-of-two, numeric level difference, or terciles apply to this category. Its aggregate is the equal-date mean of each defined agreement statistic.

### 2.5 Level change

Use C_jM, the **same cohort as the column's rank panel**, even when rho itself is undefined due to ties. No rounding or trimming. Shares and signed scalars (all alignment scores and net bias included) use `d_i=F_i−L_i`: report median d, p10/p90 d, and fraction `|d_i|>0.2`. Share differences are fractional units: 0.2 means 20 percentage points. Alignment is in normalized-return units and must be labelled separately.

All other numeric columns except account value are nonnegative and use ratios r=F/L. If L=F=0 set r=1; if L=0,F>0 set r=+infinity; if L>0,F=0 set r=0. Report median r, p10/p90, fraction `(r>2 or r<1/2)` with strict inequalities, and counts of each zero case. Quantiles with infinity use order statistics with extended-real arithmetic; if linear interpolation has positive weight on infinity the result is infinity, and an exact finite order statistic remains finite. All numeric panels additionally provide median absolute change in native units.

Account-value medians can be nonpositive. For AV report median F−L in dollars on **all C**, with no arbitrary 0.2-dollar flag. Additionally report the ratio and factor-of-two summaries on the explicitly labelled subset `C_positive={i in C:L_i>0 and F_i>0}`, together with its count and the excluded nonpositive count. AV rank and tier panels continue to use all C. Aggregate dollar differences and positive-subset ratios separately, each by the rule below; never change the unit of a panel depending on a single member's value.

Aggregate level summaries are the median of per-date medians; change fractions are equal-date arithmetic means. Aggregate p10/p90 are not quantiles of concatenated addresses: retain them at the date level only. State the number of included dates. Complete holding and distinct-symbol comparisons carry the duration/opportunity warning in every aggregate display.

### 2.6 Ties, relative retention, and frozen-L retention

For any finite numeric sample x of size n define mid-percentile `u(x_i)=(average_rank(x_i)−0.5)/n`. Assign tier 1 if u<=1/3, tier 2 if 1/3<u<=2/3, tier 3 otherwise. **Never split ties by address ID, sort order, jitter, or random assignment.** Empty tiers are allowed. A constant column puts everyone in tier 2 and has no rank discrimination.

Produce two explicitly different panels:

1. **Conditional relative-rank retention:** on C_jM recompute both L and F tiers by this rule using that same paired cohort. Report the 3×3 count matrix divided by n, same-tier fraction, extreme fraction `Pr(|T_F−T_L|=2)`, and row-conditional transition fractions (including top→bottom and bottom→top). Baselines use the actual margins: `p0=sum_t p_Lt p_Ft`, `e0=p_L1 p_F3+p_L3 p_F1`. Report excess same-tier fraction `retention−p0`; chance-adjusted excess is optional and not an input to flags. If either side lacks three occupied tiers, publish the matrix and margins but mark the headline tercile-retention/threshold test **non-discriminating**, not stable. Re-ranking L on C is a retrospective comparison only, never the tiers assigned to a sleeve at M.
2. **Frozen-L tiers and absolute retention:** form deployed tiers on all A_jM using L only. For a forward scalar v, its position in that frozen distribution is `u_L(v)=(count(L<v)+0.5*count(L=v))/|A_jM|`. Apply the same tier thresholds to v. On measurable addresses, report the frozen-L→forward-frozen-bin matrix and same-bin fraction. Also publish four outcomes per deployed tier: measured in each of the three forward frozen bins, or unreportable. Their denominator is **all L addresses assigned to that tier**, not survivors. This tests scale-specific consistency of the original bins. The empirical L distribution is never updated using F.

For the frozen panel, the all-A joint fraction **reportable and in the same frozen bin** is S/N. Conditional same-bin fraction is S/G where G=|C| and N=|A|. The missing-outcome bounds `[S/N, (S+N−G)/N]` describe the range of full-cohort same-bin retention if missing values could be known; they are not confidence intervals and do not identify those missing values. At N=0/G=0 denominators are NA. Ties/category degeneracy do not receive a false stable label.

Aggregate transition matrices and fractions with equal date weight, using dates where each statistic is defined; retain each denominator and K. Do not sum address–date counts and imply an independent multinomial sample.

### 2.7 Forward reportability and activity

For every j,M show `|B|`, `|A|`, `|C|`, L coverage `|A|/|B|`, F reportability `|C|/|A|`, and by deployed-L-tier reportability. Also give, on **all B**, fractions with no F fills, any F fills, and each fixed order/day gate; these are not column-specific attrition reasons. No-fill fraction includes carried-position addresses and must not overwrite their measurable grid values.

For A\C assign exactly one primary reason, in this order:

1. **HIP-3 scope change**, only for AV/leverage.
2. **No relevant forward observations**: no finite snapshots for AV, no valid observed grid points for grid columns, no complete fill episodes for complete-episode columns, no positive eligible denominator events for add shares, no finite scored class/variant decisions for alignment, or no fills for order-family columns.
3. **Insufficient support**, when a nonempty relevant support fails its gate.
4. **Undefined value**, when counts pass but the required denominator/statistic remains undefined.

Retain overlapping flags for no fills, missing prices, AV<100/invalid brackets, no opening/later adds, raw/A coverage failure, and every gate shortfall. Primary counts sum to |A|−|C|. L failures B\A receive the analogous reasons, with existing HIP-3 labelled pre-existing scope rather than “started.” “HIP-3 started” here means **first became observable under these inputs**, not a proven first-ever trade time. Aggregate fractions as date means and show the full count series.

### 2.8 Tier forward summaries and spread

Use **deployed L tiers on A**, never tiers reconstructed on survivors. Include every such address's eligible F records even if it fails the F reportability gate. An address with zero eligible records contributes no event weight, not an invented zero score; count it among noncontributors. HIP-3 suppression, missing data, and score eligibility remain in force.

For each tier t compute native pooled values `P_t(S)` using this mapping for both S=L and S=F:

| Column | Pooled record / statistic |
|---|---|
| Each alignment cell | All its finite class/variant scored decisions; class-notional-weighted mean. |
| Holding median | All complete fill episodes; unweighted median duration. |
| Holding weighted mean | All complete fill episodes; increasing-notional-weighted mean duration. |
| Median cluster notional | All clusters; median total cluster notional. |
| Maker share | Sum maker fill notional / sum total fill notional. |
| Add-at-loss share | Sum loss-add notional / sum later-add notional on episodes opened inside S. |

Also compute `E_t(S)`, the **equal-address mean** of minimally defined per-address statistics for the same column (finite value and positive required denominator, with no minimum count/day gate). Show contributors / all tier addresses, total records/days/notional, largest contributing address's share of the native pooling weight, and Kish effective address count. For medians the native address weight is its number of pooled records; for weighted means/ratios it is the denominator weight. The equal-address display remains conditional on having at least one usable denominator/record; it does not fill missing addresses. Report its contributor counts separately.

Native spreads are `Delta_L=P_3(L)−P_1(L)` and `Delta_F=P_3(F)−P_1(F)`. Report all three values, both adjacent differences, both spreads, and whether `P_1(F)<P_2(F)<P_3(F)` strictly; ties are “tied,” inversions “not preserved,” empty tiers “not assessable.” No assumption makes a weighted L pool necessarily monotone merely because per-address L statistics defined its tiers. If `Delta_L<=0`, mark the native spread ratio **not interpretable**, retain its spreads, and do not flip signs to improve it. Otherwise report `Delta_F/Delta_L`. An extremely small positive denominator is reported with its actual value, not hidden with an unspecified epsilon. Repeat for the E summaries, using their own spreads.

Across dates, report equal-date mean P_t and E_t on dates with all three tiers defined, mean spreads on that same set, and ratio **mean Delta_F / mean Delta_L** only if mean Delta_L>0. Do not average individual spread ratios. Also report the fraction of assessable dates with strict forward ordering and the number of dates with tied/empty tiers or nonpositive L spread. State that the primary all-Monday ratio is an in-sample-sort/forward separation diagnostic, with no reference value of one. Zero forward separation is the descriptive no-separation reference, not a formal hypothesis test.

No permutation p-value, noise-corrected spread ratio, fitted errors-in-variables model, or stationary-noise calibration is required. Those would introduce additional assumptions unsupported by this short history. In particular, pooling more decisions is not proof that tier separation generalizes to typical members or to inactive members.

### 2.9 Weeks, length diagnostics, and the four sleeves

For each Fk, compute ranks, category agreement, tier summaries, and reportability using the gates and cutoffs above. Tier summaries always retain the **same deployed L tiers**. Provide the native per-week gated cohorts and, separately, rank agreement on `C_all4 = A intersect C_F1 intersect C_F2 intersect C_F3 intersect C_F4` for that column. On C_all4 ranks are computed afresh for L and the relevant week; show all four rhos and `rho_week4−rho_week1`. Publish cohort size and counts; NA if <3 or constant. This paired difference still mixes period effects and measurement noise and gets no “decay rate” interpretation. Weekly complete-episode summaries are labelled seven-day-contained summaries, not holding-style persistence estimates.

Mandatory matched-length diagnostic: for every M also build R=[M−28d,M) at cutoff M. Keep B_M defined from L60. Apply **the F28 support gates** to R, including A-specific counts and window-local episode censoring. On the cohort passing both R and F gates and having a primary L60-reportable value, show R→F rho, level/relative-tier measures and L60→F rho on that **identical cohort**. Do not use the R result to select addresses, columns, or windows. This reduces the unequal-length confound, does not remove noise or establish a latent reliability ceiling, and is particularly necessary for holding and distinct symbols.

Keep 60 as the operating window. For the proposed length sensitivity:

- L30 is available on all 11 dates; L90 only on the seven dates October 27–December 8. Never pad or shorten L90.
- For length-specific native panels, reapply base >=30 orders/10 days to that L and use the same L-support counts from §2.3; record eligible/reportable cohorts. This deliberately fixes counts rather than pretending information content is equal across lengths.
- On the seven shared dates, show 30/60/90 native-cohort results and a separate paired comparison on the **intersection of all three L-reportable sets and their F-reportable sets for that column**. Include unchanged F estimates and equal-Monday aggregation. Also show 30 versus 60 on their common cohort over all 11 dates. Differences outside common cohorts mix population changes with window changes.
- Report the main 60-day result on those seven dates as well as all 11. No winner, tuning, keep/drop rule, or confidence claim is selected from sensitivities.

Four sleeve phases are indexed from September 29, without inferring the owner's actual phase labels:

| Phase | Mondays |
|---|---|
| 0 | Sep 29, Oct 27, Nov 24 |
| 1 | Oct 6, Nov 3, Dec 1 |
| 2 | Oct 13, Nov 10, Dec 8 |
| 3 | Oct 20, Nov 17 |

Use the same equal-date summaries within each phase. Their F windows do not overlap within phase; their inputs still can. Do not average phase summaries equally as the primary result, since phase 3 has two dates rather than three. They are temporal sensitivity views, not four independent experiments.

### 2.10 Pooling, uncertainty, and historical groups

Unless otherwise specified above, every multi-Monday probability/correlation is an equal-date mean of its defined per-date values, and every level median summary is a median of per-date medians. Always record K and omitted dates. Native tier value pooling occurs **within a date/tier only**; do not concatenate duplicate events from overlapping windows across dates. A joint address–date ratio may be an explicitly secondary descriptive table, but is not a substitute for these primary summaries.

Report all 11 rows, the all-date summary, the four phase summaries, and three historical groups: **early** Sep 29–Oct 27 (5 dates); **bridge** Nov 3, Nov 10, Nov 17 (3); **late historical replication** Nov 24, Dec 1, Dec 8 (3). Make no claim that late is untouched confirmation. No column definitions, variants, gates, or flags may be selected using results in any group. Future prospective confirmation would require new observations; this specification does not promise it.

Primary temporal uncertainty presentation is the date series, min/max, and phase/group values, **without temporal confidence intervals or p-values**. Do not implement the proposed two-way time/address bootstrap as a calibrated interval.

If conditional address-resampling bands are produced, use exactly 2,000 draws from NumPy PCG64, seed **20261120**, sorted U, sampling 1,173 address indices with replacement per draw. Generate the index matrix as `Generator(PCG64(20261120)).integers(0,1173,size=(2000,1173),endpoint=False)`. Make one draw per whole address trajectory and use that same multiplicity for **all dates, weeks, lookback lengths, and records** in that replicate. The per-address values/supports are fixed; recompute cross-sectional ranks, tied tiers, all-A denominators, pools, and date summaries with duplicated copies. Do not collapse copies into one ranked observation or resample individual events. Pooling weights duplicate records with the address's multiplicity. The empirical distributions and medians use ordinary repeated observations with type-7 quantiles, not an unspecified alternative weighted-quantile rule. Missing gates do not become passes because an address was drawn twice.

Use the 2.5th/97.5th type-7 percentiles of finite replicate statistics; print invalid-draw counts and do not provide a band if fewer than 1,900 draws are finite. Keep dates fixed. Labels must say **“conditional address-resampling band; not future-time coverage; address independence unverified.”** It is acceptable to omit these bands entirely in favour of the required descriptive series; this does not change any primary point statistic. There is no multiplicity-adjusted discovery claim because no significance tests or keep/drop decisions are defined.

### 2.11 Flags, deliverables, and interpretation

No hard threshold drops a column. For numeric columns with at least one defined date-level rho, flag mean rho<.5 as **low observed conditional rank agreement**. For discriminating relative-tier panels flag mean same-tercile fraction<.5 as **low observed conditional retention**. These are the owner's descriptive conventions, not calibrated failure tests. Print the number of dates and cohort sizes supporting the flag. Undefined/constant/empty-tier panels instead say **insufficient discrimination or support**. Do not label an unflagged column “validated stable.” Do not apply numeric flags to top symbol; report its agreement and chance baseline. Also display missingness, duration/opportunity warnings, and zero-value cases independently of these flags.

The evaluator should emit:

1. A source/version/definition manifest and the exact date/window/gate table.
2. An address×date×window×column ledger containing scalar value, support counts, required denominators, censoring and coverage, base/L/F eligibility, and reasons. Address keys remain local; publication can use stable integer IDs. No future cutoff may revise an older ledger row.
3. Date×column measure tables with counts, all defined rank/level/relative and frozen-tier measures, and categorical exceptions; aggregate tables with explicit K and weighting.
4. Per-tier transition/reportability tables and both native-pooled and equal-address forward summaries with concentration.
5. Weekly native/common-cohort tables; matched-28-day diagnostics; 30/60/90 length and fixed-gate sensitivities; four sleeve phases and early/bridge/late historical displays.
6. Optional, explicitly conditional address-resampling bands; no general time/regime confidence intervals.

Required implementation acceptance checks are limited to the numerical contract: a prefix calculation must equal calculation after physically excluding post-cutoff inputs under their **availability** rule; a carried position with no new fills must stay measurable where appropriate; a later episode close must not complete its L episode; all attrition/transition counts must reconcile; tied/constant/category fixtures must yield the specified NA/baselines; each date aggregate must match the stated formulas. These are future implementation checks, not a request to build the evaluator in this consult.

## 3. What even excellent results cannot establish

- That the owner's actual selected addresses remain suitable, that four sleeves diversify each other, or that a joint profile/selection ranking persists. Those need the actual selection/weighting rule and selected-set evaluation.
- That copying is profitable, feasible at the owner's size, or safe under latency, fees, slippage, funding, drawdowns, liquidations, margin mechanics, or capacity. This evaluation deliberately has no such outcome.
- That profile values for inactive, unsupported, newly HIP-3, missing-price, or unobserved addresses persist. Tier pools do not recover them.
- That a change is true behavioural drift rather than finite-sample error, period effects, snapshot/funding approximations, or censoring; nor that a positive rho identifies a stable latent skill or intent.
- That persistent dollar scale implies persistent risk-taking, that the leverage proxy is actual equity-based leverage, or that daily funding sizes reveal the true intraday path.
- That the next market regime, the wider address population, or independent human traders resemble these retrospectively supplied addresses during these 98 forward calendar days.
- That the late period is an independent hold-out, or that .5/50% are universal suitability thresholds.

A defensible successful conclusion is: **“Within this supplied universe and realized history, profiles available on these Mondays retained the reported conditional ranks and levels over the next 28 days, with the stated reportability, concentration, and duration limitations.”** Anything about operational selection or future regimes needs additional evidence.

## 4. Evidence and reproduction

Run from the workspace root with the required interpreter:

```text
/Users/xc/Documents/trading_multi_exhcange/.venv/bin/python -B work/codex10/checks.py
```

`checks.py` reads a small subset of fill columns, existing groups, funding, and the full-window profile. It extracts only the pure fixed episode-builder function without importing modules that create output directories elsewhere. It does not run any existing profile pipeline, contact a network, or emit addresses. `activity_counts.csv` contains all 11 exact order/day gate counts; `checks.json` contains dates, episode/tie/funding-boundary diagnostics and the seeded synthetic examples. The Gaussian table uses 200,000 observations per assumed reliability and seed 20261120; it is explicitly illustrative. The current-profile tie counts and full-window episode counts are not passed off as rolling estimates. The diagnostic completed successfully; PyArrow printed restricted macOS cache-probe warnings, which did not prevent reading or computation.

Resolution: implement the replacement specification above. This consult does not require another review round and does not certify a future implementation that has not been written.

Verdict: BLOCK
Next review scope: B1–B4 only: within-date pooling, unequal-duration comparisons, unavailable L90 history, and category/tie handling.
