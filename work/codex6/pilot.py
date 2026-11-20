"""Read-only fills-only style feasibility pilot. Run with the repository venv.

All generated files stay beside this script. No network calls.
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path('/Users/xc/Documents/arkstream_quest')
OUT = Path(__file__).resolve().parent
os.environ['ARK_DATA_DIR'] = str(ROOT / 'ArkStream笔试题_v2')
os.environ['ARK_BUILD_DIR'] = str(OUT / 'build')
os.environ['ARK_TABLES_DIR'] = str(OUT / 'tables')
sys.path.insert(0, str(ROOT / 'work/v2dev/code'))
import lib

SEC = 10**9
DAY = 86400 * SEC
HORIZONS = {'1h': 3600, '4h': 14400, '1d': 86400, '7d': 604800}
START_TOL = {'1h': 300, '4h': 900, '1d': 3600, '7d': 21600}
SEED = 20261119


def read_and_decisions():
    p = lib.prep_perp_fills()
    p['fid'] = np.arange(len(p))
    g = pd.read_parquet(ROOT / 'work/v2dev/build/groups.parquet')
    g = g.sort_values(['trader', 'symbol', 'ts']).reset_index(drop=True)
    pair_new = g.trader.ne(g.trader.shift()) | g.symbol.ne(g.symbol.shift())
    flip = (g.gstart * g.gend < 0)
    new = pair_new | g.gstart.eq(0) | flip
    g['episode'] = new.cumsum()
    p = p.merge(g[['trader', 'symbol', 'ts', 'episode', 'mixed']], on=['trader', 'symbol', 'ts'], validate='many_to_one')
    # Opening quantity on each fill; a flip's closing leg contributes no entry weight.
    s0 = p.start_position.to_numpy().copy()
    s0[np.abs(s0) < lib.POS_EPS] = 0
    dq = p.signed.to_numpy()
    incq = np.where(s0 * dq >= 0, np.abs(dq), np.maximum(np.abs(dq) - np.abs(s0), 0))
    incq[incq < lib.POS_EPS] = 0
    p['inc_ntl'] = incq * p.price
    # Every order participates in clustering, including closing orders. Preserve the
    # original profile's rolling <=60s gap and opposite-side interruption semantics.
    keys = ['trader', 'symbol', 'oid']
    o = p.groupby(keys).agg(first_ts=('ts', 'min'), last_ts=('ts', 'max'),
        side=('side', 'first'), nside=('side', 'nunique'), nep=('episode', 'nunique'),
        episode=('episode', 'first')).reset_index()
    o = o.sort_values(['trader', 'symbol', 'first_ts', 'last_ts', 'oid']).reset_index(drop=True)
    same = o.trader.eq(o.trader.shift()) & o.symbol.eq(o.symbol.shift())
    join = same & o.side.eq(o.side.shift()) & o.nside.eq(1) & o.nside.shift().eq(1) & ((o.first_ts - o.last_ts.shift()).dt.total_seconds() <= 60)
    o['original_cid'] = (~join).cumsum()
    # Split original clusters at episode changes. Multi-episode orders and mixed
    # timestamp groups are excluded rather than assigning an invented decision time.
    join_ep = join & o.episode.eq(o.episode.shift()) & o.nep.eq(1) & o.nep.shift().eq(1)
    o['cid'] = (~join_ep).cumsum()
    p = p.merge(o[keys + ['cid', 'original_cid', 'nep']], on=keys, validate='many_to_one')
    opening = p.inc_ntl.gt(0) & ~p.is_forced & ~p.is_hip3
    ambiguous = p.mixed | p.nep.ne(1)
    op = p[opening & ~ambiguous].copy()
    op['maker_inc'] = op.inc_ntl * op.is_maker
    c = op.groupby('cid').agg(trader=('trader', 'first'), symbol=('symbol', 'first'),
        episode=('episode', 'first'), t=('ts', 'min'), last=('ts', 'max'),
        w=('inc_ntl', 'sum'), maker_w=('maker_inc', 'sum'), side=('sgn', 'first'),
        original_cid=('original_cid', 'first')).reset_index()
    first_ep = c.groupby('episode').t.transform('min').eq(c.t)
    # A first observed add to a left-censored position is not a first entry.
    genuine = g[g.gstart.eq(0) | flip].groupby('episode').ts.min()
    c['role'] = np.where(first_ep & c.t.eq(c.episode.map(genuine)), 'first', 'add')
    c['day'] = c.t.dt.floor('D')
    c['maker_fraction'] = c.maker_w / c.w
    profile = pd.read_csv(ROOT / 'work/profile/address_profile.csv')
    c['enough'] = c.trader.isin(profile.loc[profile.enough, 'trader'])
    original_mixed = op.groupby('original_cid').episode.nunique().gt(1)
    meta = dict(fills=len(p), symbols=int(p.symbol.nunique()), eligible=int(profile.enough.sum()),
        duplicate_prints=int(p.duplicated(['symbol', 'tid']).sum()),
        mixed_groups=int(g.mixed.sum()), multi_episode_orders=int(o.nep.gt(1).sum()),
        original_clusters_spanning_episodes=int(original_mixed.sum()),
        original_spanning_open_ntl=float(op.loc[op.original_cid.isin(original_mixed[original_mixed].index), 'inc_ntl'].sum()),
        all_perp_turnover=float(p.notional.sum()), hip3_turnover=float(p.loc[p.is_hip3, 'notional'].sum()),
        all_open_ntl=float(p.loc[p.inc_ntl.gt(0) & ~p.is_forced, 'inc_ntl'].sum()),
        ambiguous_open_ntl=float(p.loc[opening & ambiguous, 'inc_ntl'].sum()),
        decision_count=len(c), decision_ntl=float(c.w.sum()),
        eligible_decision_count=int(c.enough.sum()), eligible_decision_ntl=float(c.loc[c.enough, 'w'].sum()),
        cluster_duration_gt1h=int(((c['last'] - c.t).dt.total_seconds() > 3600).sum()),
        max_cluster_hours=float((c['last'] - c.t).dt.total_seconds().max() / 3600))
    return p, c, profile, meta


def price_arrays(rows):
    # One print per exchange trade, then a size-weighted price at each timestamp.
    q = rows.drop_duplicates(['tid']).copy()
    q['pv'] = q.price * q['size']
    a = q.groupby('ts').agg(pv=('pv', 'sum'), qty=('size', 'sum'))
    return a.index.asi8, np.log((a.pv / a.qty).to_numpy())


def at(times, prices, query, tol):
    # Strictly before every query, even the start. Thus no same-ms execution leak.
    ix = np.searchsorted(times, query, side='left') - 1
    good = ix >= 0
    ix = np.maximum(ix, 0)
    age = query - times[ix]
    good &= age <= tol * SEC
    return np.where(good, prices[ix], np.nan)


def features(times, prices, query, horizon, start_tol, end_tol=60):
    end = at(times, prices, query, end_tol)
    start = at(times, prices, query - horizon * SEC, start_tol)
    r = end - start
    # Past 24h volatility from hourly samples, each at most 15m stale. Require
    # 18 valid adjacent hourly increments. Missing gaps are never bridged.
    grid = query[:, None] - np.arange(25)[None, :] * 3600 * SEC
    px = at(times, prices, grid.ravel(), 900).reshape(-1, 25)
    ret = np.diff(px, axis=1)
    n = np.isfinite(ret).sum(axis=1)
    vol = np.sqrt(np.nansum(ret**2, axis=1) / np.maximum(n, 1) * (horizon / 3600))
    vol[(n < 18) | (vol <= 0)] = np.nan
    return r, vol


def build_scores(p, c):
    results = []
    density = []
    n_pair = 0
    for sym, ps in p[~p.is_hip3].groupby('symbol', sort=True):
        cs = c[c.symbol.eq(sym)]
        if cs.empty:
            continue
        pooled_t, pooled_px = price_arrays(ps)
        gaps = np.diff(pooled_t) / SEC
        density.append(dict(symbol=sym, prints=len(pooled_t), traders=ps.trader.nunique(),
            median_gap_s=float(np.median(gaps)) if len(gaps) else np.nan,
            p90_gap_s=float(np.quantile(gaps, .9)) if len(gaps) else np.nan,
            p99_gap_s=float(np.quantile(gaps, .99)) if len(gaps) else np.nan))
        for trader, ct in cs.groupby('trader', sort=True):
            n_pair += 1
            own_tids = ps.loc[ps.trader.eq(trader), 'tid'].unique()
            lo_t, lo_px = price_arrays(ps[~ps.tid.isin(own_tids)])
            t = ct.t.astype('int64').to_numpy()
            if not len(lo_t):
                # Keep uncovered rows so all coverage denominators remain intact.
                for h in HORIZONS:
                    out = ct.copy(); out['h'] = h
                    for col in ['raw', 'z', 'vol', 'base_z', 'x']:
                        out[col] = np.nan
                    out['coverage'] = False
                    pooled_r = at(pooled_t, pooled_px, t, 60) - at(pooled_t, pooled_px, t - HORIZONS[h]*SEC, START_TOL[h])
                    out['pooled_coverage'] = np.isfinite(pooled_r)
                    out['strict_coverage'] = False; out['relaxed_coverage'] = False
                    out['n_candidates'] = 0
                    results.append(out)
                continue
            # A uniform 5-minute grid in the same UTC 4h block as the decision.
            block = t // (14400 * SEC)
            ub = np.unique(block)
            candidates = (ub[:, None] * 14400 * SEC + np.arange(48)[None, :] * 300 * SEC).ravel()
            for h, hs in HORIZONS.items():
                r, vol = features(lo_t, lo_px, t, hs, START_TOL[h])
                cr, cv = features(lo_t, lo_px, candidates, hs, START_TOL[h])
                cz = cr / cv
                z = r / vol
                base = np.full(len(ct), np.nan)
                nc = np.zeros(len(ct), dtype=int)
                for b in ub:
                    ii = np.flatnonzero(block == b)
                    jj = np.flatnonzero(candidates // (14400 * SEC) == b)
                    # Ratio matching removes gross volatility mismatch without
                    # matching the sign of the very return we are measuring.
                    ratio = cv[jj][None, :] / vol[ii][:, None]
                    ok = np.isfinite(cz[jj])[None, :] & (ratio >= 2/3) & (ratio <= 1.5)
                    count = ok.sum(axis=1)
                    mean = (np.where(ok, cz[jj][None, :], 0).sum(axis=1) / np.maximum(count, 1))
                    base[ii] = np.where(count >= 5, mean, np.nan)
                    nc[ii] = count
                pooled_r = at(pooled_t, pooled_px, t, 60) - at(pooled_t, pooled_px, t - hs*SEC, START_TOL[h])
                strictr = at(lo_t, lo_px, t, 30) - at(lo_t, lo_px, t - hs*SEC, START_TOL[h] / 2)
                relaxedr = at(lo_t, lo_px, t, 300) - at(lo_t, lo_px, t - hs*SEC, START_TOL[h] * 2)
                out = ct.copy(); out['h'] = h
                out['raw'] = ct.side.to_numpy() * r
                out['vol'] = vol
                out['z'] = ct.side.to_numpy() * z
                out['base_z'] = ct.side.to_numpy() * base
                out['x'] = out.z - out.base_z
                out['coverage'] = np.isfinite(r)
                out['pooled_coverage'] = np.isfinite(pooled_r)
                out['strict_coverage'] = np.isfinite(strictr)
                out['relaxed_coverage'] = np.isfinite(relaxedr)
                out['n_candidates'] = nc
                results.append(out)
        print(f'{sym}: {len(cs)} decisions; {n_pair} address-symbol pairs done', flush=True)
    pd.DataFrame(density).to_csv(OUT / 'density.csv', index=False)
    result = pd.concat(results, ignore_index=True)
    result.to_parquet(OUT / 'decision_scores.parquet', index=False)
    return result


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    vals = p[order] * len(p) / np.arange(1, len(p)+1)
    vals = np.minimum.accumulate(vals[::-1])[::-1].clip(0, 1)
    result = np.empty(len(p)); result[order] = vals
    return result


def wmean(x, w):
    ok = np.isfinite(x) & (w > 0)
    return float(np.sum(x[ok]*w[ok])/np.sum(w[ok])) if ok.any() else np.nan


def summarize(s, profile, meta):
    cov = []
    for (sym, h), d in s.groupby(['symbol', 'h']):
        row = dict(symbol=sym, h=h, n=len(d), ntl=float(d.w.sum()))
        for col in ['coverage', 'pooled_coverage', 'strict_coverage', 'relaxed_coverage']:
            row[col] = float(d.loc[d[col], 'w'].sum() / d.w.sum())
        row['scored_coverage'] = float(d.loc[d.x.notna(), 'w'].sum()/d.w.sum())
        row['scored_n'] = int(d.x.notna().sum())
        row['usable'] = row['coverage'] >= .8 and row['scored_coverage'] >= .7 and row['scored_n'] >= 100
        cov.append(row)
    pd.DataFrame(cov).to_csv(OUT / 'coverage_by_symbol.csv', index=False)
    overall = []
    for population, dd in [('all', s), ('eligible', s[s.enough])]:
        for h, d in dd.groupby('h'):
            row = dict(population=population, h=h, n=len(d), ntl=float(d.w.sum()))
            for col in ['coverage', 'pooled_coverage', 'strict_coverage', 'relaxed_coverage']:
                row[col] = float(d.loc[d[col], 'w'].sum()/d.w.sum())
            row['vol_coverage'] = float(d.loc[d.z.notna(), 'w'].sum()/d.w.sum())
            row['scored_coverage'] = float(d.loc[d.x.notna(), 'w'].sum()/d.w.sum())
            row['scored_n'] = int(d.x.notna().sum())
            overall.append(row)
    pd.DataFrame(overall).to_csv(OUT / 'coverage_overall.csv', index=False)
    rows = []
    rng = np.random.default_rng(SEED)
    for (tr, h), ds in s[s.enough].groupby(['trader', 'h'], sort=True):
        for split in ['all', 'first', 'add', 'maker', 'taker', 'pure_maker', 'pure_taker']:
            d = ds.copy()
            if split in ['first', 'add']:
                d = d[d.role.eq(split)]
            if split == 'maker':
                d['w'] = d.maker_w
            if split == 'taker':
                d['w'] = d.w - d.maker_w
            if split == 'pure_maker':
                d = d[d.maker_fraction >= .9]
            if split == 'pure_taker':
                d = d[d.maker_fraction <= .1]
            d = d[d.x.notna() & d.w.gt(0)].sort_values(['t', 'symbol', 'cid'])
            if d.empty:
                continue
            w, x = d.w.to_numpy(), d.x.to_numpy()
            n = len(d); nd = d.day.nunique(); neff = w.sum()**2/(w*w).sum()
            block_days = 14 if h == '7d' else 7
            d['block'] = (d.t.astype('int64') // DAY - pd.Timestamp('2025-07-27').value // DAY) // block_days
            nb = d.block.nunique()
            sufficient = n >= 30 and nd >= 10 and neff >= 10 and nb >= (6 if h == '7d' else 8)
            vals = (d.w * d.x).groupby(d.block).sum().to_numpy()
            # Approximate cluster sign-flip diagnostic, not an identified causal test.
            pvalue = np.nan
            if sufficient:
                signs = rng.choice([-1, 1], size=(1999, nb))
                pvalue = (1 + (np.abs(signs @ vals) >= abs(vals.sum()) - 1e-12).sum()) / 2000
            mid = n//2
            rows.append(dict(trader=tr, h=h, split=split, n=n, days=nd, neff=neff, blocks=nb,
                sufficient=sufficient, weight=float(w.sum()), mean=wmean(x,w),
                raw_mean=wmean(d.raw.to_numpy(),w), z_mean=wmean(d.z.to_numpy(),w),
                baseline=wmean(d.base_z.to_numpy(),w), positive_share=wmean((d.raw.to_numpy()>0).astype(float),w),
                decision_positive_share=float((d.raw>0).mean()),
                first_half=wmean(x[:mid],w[:mid]) if mid>=20 else np.nan,
                second_half=wmean(x[mid:],w[mid:]) if mid>=20 else np.nan,
                odd=wmean(x[::2],w[::2]) if mid>=20 else np.nan,
                even=wmean(x[1::2],w[1::2]) if mid>=20 else np.nan,
                first_raw=wmean(d.raw.to_numpy()[:mid],w[:mid]) if mid>=20 else np.nan,
                second_raw=wmean(d.raw.to_numpy()[mid:],w[mid:]) if mid>=20 else np.nan,
                first_z=wmean(d.z.to_numpy()[:mid],w[:mid]) if mid>=20 else np.nan,
                second_z=wmean(d.z.to_numpy()[mid:],w[mid:]) if mid>=20 else np.nan,
                p=pvalue))
    a = pd.DataFrame(rows); a['q'] = np.nan
    primary = a.split.eq('all') & a.p.notna()
    a.loc[primary,'q'] = bh(a.loc[primary,'p'])
    a.to_csv(OUT / 'address_scores.csv', index=False)
    st = []
    for (h, split), d in a.groupby(['h','split']):
        ds = d[d.sufficient]
        stable_ds = d[(d.n >= 30) & (d.days >= 10) & (d.neff >= 10)]
        valid = stable_ds.dropna(subset=['first_half','second_half'])
        def rho(cols):
            v = valid.dropna(subset=cols)
            return float(spearmanr(v[cols[0]],v[cols[1]]).statistic) if len(v)>=3 else np.nan
        st.append(dict(h=h, split=split, any_score=len(d), eligible=len(ds), stable_eligible=len(stable_ds),
            nominal_p05=int(ds.p.lt(.05).sum()), bh_q05=int(ds.q.lt(.05).sum()),
            bh_trend=int(((ds.q<.05)&(ds['mean']>0)).sum()),
            bh_reversal=int(((ds.q<.05)&(ds['mean']<0)).sum()),
            stable_n=len(valid), half_rho=rho(['first_half','second_half']),
            odd_even_rho=rho(['odd','even']), raw_half_rho=rho(['first_raw','second_raw']),
            z_half_rho=rho(['first_z','second_z']),
            half_sign_agreement=float((np.sign(valid.first_half)==np.sign(valid.second_half)).mean()),
            median_mean=float(ds['mean'].median()), median_raw_bp=float(ds.raw_mean.median()*10000)))
    pd.DataFrame(st).to_csv(OUT / 'stability.csv', index=False)
    paired = []
    for h, dd in a.groupby('h'):
        d = dd[dd.sufficient]
        maker = d[d.split.eq('maker')].set_index('trader')
        taker = d[d.split.eq('taker')].set_index('trader')
        pp = maker[['mean','z_mean','raw_mean']].join(taker[['mean','z_mean','raw_mean']],lsuffix='_m',rsuffix='_t',how='inner')
        main = d[d.split.eq('all')].merge(profile[['trader','maker_share','add_loser_share']],on='trader')
        row = dict(h=h, paired_n=len(pp),
            paired_rho=float(spearmanr(pp.mean_m,pp.mean_t).statistic) if len(pp)>2 else np.nan,
            median_m_minus_t=float((pp.mean_m-pp.mean_t).median()),
            median_raw_m_minus_t_bp=float((pp.raw_mean_m-pp.raw_mean_t).median()*10000),
            sign_disagree=float((np.sign(pp.mean_m)!=np.sign(pp.mean_t)).mean()),
            maker_correlation=float(spearmanr(main['mean'],main.maker_share,nan_policy='omit').statistic),
            add_loser_correlation=float(spearmanr(main['mean'],main.add_loser_share,nan_policy='omit').statistic))
        paired.append(row)
    pd.DataFrame(paired).to_csv(OUT / 'maker_comparison.csv', index=False)
    meta['seed'] = SEED
    meta['baseline_min_candidates'] = 5
    (OUT / 'metadata.json').write_text(json.dumps(meta,indent=2))


if __name__ == '__main__':
    p, c, profile, meta = read_and_decisions()
    print(json.dumps(meta, indent=2), flush=True)
    c.to_parquet(OUT / 'decisions.parquet', index=False)
    s = build_scores(p, c)
    summarize(s, profile, meta)
    print('DONE', flush=True)
