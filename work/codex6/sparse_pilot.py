"""Sensitivity: sparse-print realized-vol proxy and unscaled matched-time E.

Uses identical decisions/endpoints as pilot.py. Computes both variants in one pass.
"""
import json
import numpy as np
import pandas as pd
import pilot as m


def sparse_vol(t, px, query):
    end=np.searchsorted(t,query,side='left')-1
    start=np.searchsorted(t,query-24*3600*m.SEC,side='left')-1
    ee=np.maximum(end,0); ss=np.maximum(start,0)
    gap=np.r_[np.nan,np.diff(t)/m.SEC]
    # Conservative gap screen; the extra hour avoids ignoring the boundary gap.
    maxgap=pd.Series(gap,index=pd.to_datetime(t)).rolling('25h').max().to_numpy()
    rv=np.r_[0,np.cumsum(np.diff(px)**2)]
    elapsed=(t[ee]-t[ss])/m.SEC/3600
    variance=(rv[ee]-rv[ss])/np.maximum(elapsed,1e-9)
    ok=(end>=0)&(start>=0)&(query-t[ee]<=3600*m.SEC)&(query-24*3600*m.SEC-t[ss]<=3600*m.SEC)&(ee-ss>=24)&(maxgap[ee]<=10800)&(variance>0)
    return np.where(ok,np.sqrt(variance),np.nan)


def match(qr,qv,cr,cv,block,cblock,scaled):
    z=qr/qv if scaled else qr
    cz=cr/cv if scaled else cr
    base=np.full(len(qr),np.nan);nc=np.zeros(len(qr),int)
    for b in np.unique(block):
        ii=np.flatnonzero(block==b);jj=np.flatnonzero(cblock==b)
        ok=np.broadcast_to(np.isfinite(cz[jj]),(len(ii),len(jj))).copy()
        if scaled:
            ratio=cv[jj][None,:]/qv[ii][:,None]
            ok&=(ratio>=2/3)&(ratio<=1.5)
        count=ok.sum(axis=1)
        mean=np.where(ok,cz[jj][None,:],0).sum(axis=1)/np.maximum(count,1)
        base[ii]=np.where(count>=5,mean,np.nan);nc[ii]=count
    return z,base,nc


def main():
    p,c,profile,meta=m.read_and_decisions()
    cached=pd.read_parquet(m.OUT/'decision_scores.parquet').set_index(['cid','h'])
    outputs={'sparse':[],'unscaled':[],'horizon_sparse':[],'horizon_unscaled':[]}
    npair=0
    for sym,ps in p[~p.is_hip3].groupby('symbol',sort=True):
        cs=c[c.symbol.eq(sym)]
        if cs.empty:continue
        pooled_t,pooled_px=m.price_arrays(ps)
        for trader,ct in cs.groupby('trader',sort=True):
            npair+=1
            own=ps.loc[ps.trader.eq(trader),'tid'].unique()
            t,px=m.price_arrays(ps[~ps.tid.isin(own)])
            query=ct.t.astype('int64').to_numpy();block=query//(14400*m.SEC)
            blocks=np.unique(block)
            cand=(blocks[:,None]*14400*m.SEC+np.arange(48)[None,:]*300*m.SEC).ravel()
            if len(t):
                qv=sparse_vol(t,px,query);cv=sparse_vol(t,px,cand)
            else:qv=np.full(len(query),np.nan);cv=np.full(len(cand),np.nan)
            for h,hs in m.HORIZONS.items():
                template=cached.loc[[(cid,h) for cid in ct.cid]].reset_index()
                for mode in outputs:
                    scaled=mode in ['sparse','horizon_sparse']
                    end_tol=m.START_TOL[h] if mode.startswith('horizon') else 60
                    if len(t):
                        qr=m.at(t,px,query,end_tol)-m.at(t,px,query-hs*m.SEC,m.START_TOL[h])
                        cr=m.at(t,px,cand,end_tol)-m.at(t,px,cand-hs*m.SEC,m.START_TOL[h])
                    else:qr=np.full(len(query),np.nan);cr=np.full(len(cand),np.nan)
                    z,base,nc=match(qr,qv*np.sqrt(hs/3600),cr,cv*np.sqrt(hs/3600),block,cand//(14400*m.SEC),scaled)
                    out=template.copy();side=ct.side.to_numpy()
                    out['vol']=qv*np.sqrt(hs/3600) if scaled else 1.
                    out['z']=side*z;out['base_z']=side*base
                    out['x']=out.z-out.base_z;out['n_candidates']=nc
                    out['role']=ct.role.to_numpy()
                    out['raw']=side*qr
                    if mode.startswith('horizon'):
                        out['coverage']=np.isfinite(qr)
                        pr=m.at(pooled_t,pooled_px,query,end_tol)-m.at(pooled_t,pooled_px,query-hs*m.SEC,m.START_TOL[h])
                        out['pooled_coverage']=np.isfinite(pr)
                        if len(t):
                            sr=m.at(t,px,query,end_tol/2)-m.at(t,px,query-hs*m.SEC,m.START_TOL[h]/2)
                            rr=m.at(t,px,query,end_tol*2)-m.at(t,px,query-hs*m.SEC,m.START_TOL[h]*2)
                            out['strict_coverage']=np.isfinite(sr);out['relaxed_coverage']=np.isfinite(rr)
                    outputs[mode].append(out)
        print(f'{sym}: {npair} pairs done',flush=True)
    original=m.OUT
    for mode,data in outputs.items():
        m.OUT=original/mode;m.OUT.mkdir(exist_ok=True)
        result=pd.concat(data,ignore_index=True)
        result.to_parquet(m.OUT/'decision_scores.parquet',index=False)
        c.to_parquet(m.OUT/'decisions.parquet',index=False)
        md=dict(meta,volatility_method=mode)
        m.summarize(result,profile,md)
        print(f'{mode}: DONE',flush=True)


if __name__=='__main__':main()
