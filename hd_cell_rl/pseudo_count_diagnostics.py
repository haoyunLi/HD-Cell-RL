"""Count-conditioned diagnostics only; never generate benchmark inference inputs."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import gammaln
from scipy.stats import spearmanr


def integer_csr(matrix):
    x=sparse.csr_matrix(matrix,dtype=np.float64,copy=True)
    x.sum_duplicates();x.eliminate_zeros();x.sort_indices()
    if not np.isfinite(x.data).all() or np.any(x.data<0) or np.any(x.data!=np.floor(x.data)):
        raise ValueError('diagnostics require finite nonnegative integer counts')
    return x.astype(np.int64)


def rarefy_exact(matrix, budgets, rng):
    """Uniform sampling WITHOUT replacement, at an exact integer count per row.

    No UMI expansion or multinomial resynthesis. Original zero genes stay zero.
    This conditions on observed counts and must not be used as held-out inference.
    """
    x=integer_csr(matrix);budgets=np.asarray(budgets)
    totals=np.asarray(x.sum(1)).ravel()
    if budgets.shape!=totals.shape or not np.isfinite(budgets).all() or np.any(budgets<0) or np.any(budgets!=np.floor(budgets)) or np.any(budgets>totals):
        raise ValueError('budgets must be integer row counts between zero and observed total')
    out=x.copy()
    for i,n in enumerate(budgets.astype(np.int64)):
        s,e=x.indptr[i:i+2]
        if n==0:out.data[s:e]=0
        elif n<totals[i]:out.data[s:e]=rng.multivariate_hypergeometric(x.data[s:e],int(n),method='marginals')
    out.eliminate_zeros()
    if not np.array_equal(np.asarray(out.sum(1)).ravel(),budgets):raise AssertionError('exact budget mismatch')
    return out


def expected_genes(matrix,budgets):
    """Analytic expected richness under exact rarefaction, avoiding seed noise."""
    x=integer_csr(matrix);n=np.asarray(x.sum(1)).ravel();m=np.asarray(budgets)
    if m.shape!=n.shape or np.any(m<0) or np.any(m>n) or not np.isfinite(m).all() or np.any(m!=np.floor(m)):
        raise ValueError('invalid exact rarefaction budgets')
    row=np.repeat(np.arange(x.shape[0]),np.diff(x.indptr));total=n[row];draw=m[row];absent=total-x.data
    p=np.ones(len(row));possible=absent>=draw
    a=absent[possible];b=total[possible];d=draw[possible]
    logp=gammaln(a+1)-gammaln(a-d+1)-gammaln(b+1)+gammaln(b-d+1)
    p[possible]=-np.expm1(np.minimum(logp,0))
    p[draw==0]=0
    return np.bincount(row,weights=p,minlength=x.shape[0])


def row_features(matrix,theta=None,markers=None):
    x=integer_csr(matrix);n=np.asarray(x.sum(1)).ravel();den=np.maximum(n,1)
    top1=np.zeros(len(n));top10=np.zeros(len(n));entropy=np.zeros(len(n))
    for i in range(len(n)):
        s,e=x.indptr[i:i+2];a=x.data[s:e]
        if not len(a):continue
        top1[i]=a.max()/den[i];top10[i]=np.partition(a,max(0,len(a)-10))[-10:].sum()/den[i]
        p=a/den[i];entropy[i]=-np.dot(p,np.log(p))
    result={'genes':np.diff(x.indptr).astype(float),'top1_fraction':top1,'top10_fraction':top10,
            'simpson_concentration':np.asarray(x.astype(float).power(2).sum(1)).ravel()/den**2,
            'gene_entropy':entropy}
    if theta is not None:
        # Same normalized bin LL definition as reward.py; sparse multiplication.
        ll=np.asarray(x@np.log(theta).T)/den[:,None];best=np.sort(ll,axis=1)
        result['type_ll_gap']=best[:,-1]-best[:,-2]
        result['max_marker_fraction']=np.asarray(x@markers.T).max(1)/den
    return result


def summarize_matched(real,sim,seed,theta=None,markers=None):
    """Compare the same physical regions with identical post-rarefaction counts."""
    real=integer_csr(real);sim=integer_csr(sim)
    if real.shape!=sim.shape:raise ValueError('unaligned expression axes')
    rn=np.asarray(real.sum(1)).ravel();sn=np.asarray(sim.sum(1)).ravel();target=np.minimum(rn,sn)
    rr=rarefy_exact(real,target,np.random.default_rng(seed))
    ss=rarefy_exact(sim,target,np.random.default_rng(seed+1))
    rf=row_features(rr,theta,markers);sf=row_features(ss,theta,markers)
    rg=expected_genes(real,target);sg=expected_genes(sim,target)
    rows=[]
    for lo,hi in [(1,np.inf),(1,5),(5,20),(20,100),(100,500),(500,2000),(2000,np.inf)]:
        keep=(target>=lo)&(target<hi);size=int(keep.sum())
        if not size:continue
        row=dict(count_band=f'{lo}-'+('inf' if not np.isfinite(hi) else str(hi)),n_regions=size,
            target_mean=float(target[keep].mean()),expected_real_genes=float(rg[keep].mean()),
            expected_sim_genes=float(sg[keep].mean()),expected_genes_ratio=float(sg[keep].sum()/rg[keep].sum()))
        for key in rf:
            valid=keep&(target>=20) if key in ['type_ll_gap','max_marker_fraction'] else keep
            row[key+'_n']=int(valid.sum())
            row['real_'+key]=float(rf[key][valid].mean()) if valid.any() else np.nan
            row['sim_'+key]=float(sf[key][valid].mean()) if valid.any() else np.nan
        rows.append(row)
    supported=target>0
    evidence=dict(n_total=len(target),n_both_positive=int(supported.sum()),n_real_only=int(((rn>0)&(sn==0)).sum()),
        n_sim_only=int(((sn>0)&(rn==0)).sum()),real_retained_fraction=float(target.sum()/max(rn.sum(),1)),
        sim_retained_fraction=float(target.sum()/max(sn.sum(),1)),raw_counts_ratio=float(sn.sum()/max(rn.sum(),1)),
        raw_genes_ratio=float(real.shape[0] and sim.nnz/max(real.nnz,1)),
        matched_exact_counts=True)
    rt=np.asarray(rr.sum(0)).ravel();st=np.asarray(ss.sum(0)).ravel();active=(rt+st)>0
    evidence['matched_gene_abundance_spearman']=float(spearmanr(rt[active],st[active]).statistic) if active.sum()>1 and min(rt[active].std(),st[active].std())>0 else np.nan
    return rows,evidence
