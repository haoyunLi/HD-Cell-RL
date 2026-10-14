"""Calibration-only regional observation-depth models for pseudo-HD controls.

Depth is not identified as capture efficiency. Geometry/type covariates explain
part of the conditional mean; residual spatial fields are generative hypotheses,
not reconstructions of a held-out real expression map.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def spatial_blocks(grid, patch_ids, width_bins):
    grid=np.asarray(grid,dtype=np.int64);patch=np.asarray(patch_ids).astype(str)
    if grid.shape!=(len(patch),2) or int(width_bins)!=width_bins or width_bins<1:
        raise ValueError('aligned 2D bin grid and positive integer width required')
    patches, codes=np.unique(patch,return_inverse=True)
    unique,inverse,n=np.unique(np.column_stack((codes,grid//width_bins)),axis=0,return_inverse=True,return_counts=True)
    mapping=sparse.csr_matrix((np.ones(len(grid)),(inverse,np.arange(len(grid)))),shape=(len(unique),len(grid)))
    return mapping, inverse, unique, patches[unique[:,0]], n


def geometry_type_features(coordinates, patch_ids, nuclei_xy, nuclei_types, *, coverage, nuclear_coverage,
                           density_sigma_um=16.):
    """Expression-free covariates, conditional on existing segmentation/type labels.

    Type annotations themselves may have originated in real expression; this is
    explicitly not a morphology-only or independently annotated validation.
    """
    xy=np.asarray(coordinates,float);centers=np.asarray(nuclei_xy,float);types=np.asarray(nuclei_types).astype(str)
    cov=np.asarray(coverage,float);nuc=np.asarray(nuclear_coverage,float)
    if xy.shape!=(len(cov),2) or nuc.shape!=cov.shape or centers.shape!=(len(types),2):
        raise ValueError('unaligned geometry features')
    if not all(np.isfinite(x).all() for x in [xy,centers,cov,nuc]) or np.any(cov<0) or np.any(nuc<0):
        raise ValueError('invalid geometry values')
    if not np.isfinite(density_sigma_um) or density_sigma_um<=0:raise ValueError('invalid density bandwidth')
    names=np.unique(types);lookup={t:i for i,t in enumerate(names)}
    tree=cKDTree(centers);densities=np.zeros((len(xy),len(names)))
    for i,indices in enumerate(tree.query_ball_point(xy,3*density_sigma_um)):
        if not indices:continue
        indices=np.asarray(indices,int)
        weight=np.exp(-np.sum((centers[indices]-xy[i])**2,axis=1)/(2*density_sigma_um**2))
        # Nucleus equivalents per 100 square micrometers; absolute units retained.
        np.add.at(densities[i],[lookup[t] for t in types[indices]],weight*100/(2*np.pi*density_sigma_um**2))
    return np.column_stack((cov,nuc,densities.sum(1),densities)), (
        ['modeled_coverage','nuclear_coverage','nucleus_density_per_100um2']+
        ['type_density:'+t for t in names])


@dataclass
class RegionalDepthModel:
    feature_names: list[str]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    training_mean: float
    ridge: float
    converged: bool

    def predict(self, features):
        x=np.asarray(features,float)
        if x.ndim!=2 or x.shape[1]!=len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError('invalid prediction features')
        eta=self.intercept+((x-self.feature_mean)/self.feature_scale)@self.coefficient
        # Numerical safety only, not an audit-target-derived depth bound.
        return self.training_mean*np.exp(np.clip(eta,-15,15))

    def to_dict(self):
        return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(self).items()}


def fit_regional_depth(features, mean_counts, n_bins, calibration, feature_names, *, ridge=0.1):
    """Fit penalized Poisson conditional means, using calibration rows only.

    Held-out responses are never used for scaling, fitting or validation. Mean
    counts are scaled by the training mean so a declared ridge has stable units.
    """
    keep=np.asarray(calibration,bool)
    x=np.asarray(features,float)[keep];y=np.asarray(mean_counts,float)[keep];n=np.asarray(n_bins,float)[keep]
    if x.ndim!=2 or x.shape[1]!=len(feature_names) or len(x)<2 or y.shape!=(len(x),) or n.shape!=y.shape:
        raise ValueError('aligned calibration rows and at least two blocks required')
    if not all(np.isfinite(v).all() for v in [x,y,n]) or np.any(y<0) or np.any(n<=0):
        raise ValueError('invalid calibration observations')
    if not np.isfinite(ridge) or ridge<=0:raise ValueError('positive ridge required')
    weights=n/n.sum();average=float(weights@y)
    if average<=0:raise ValueError('calibration has no observed counts')
    center=weights@x;scale=np.sqrt(weights@((x-center)**2))
    scale=np.where(scale>1e-8,scale,1.)
    design=np.column_stack((np.ones(len(x)),(x-center)/scale));response=y/average
    def objective(beta):
        eta=design@beta
        mu=np.exp(eta)
        loss=float(weights@(mu-response*eta)+ridge/2*np.dot(beta[1:],beta[1:]))
        gradient=design.T@(weights*(mu-response));gradient[1:]+=ridge*beta[1:]
        return loss,gradient
    fit=minimize(objective,np.zeros(design.shape[1]),jac=True,method='L-BFGS-B',
                 bounds=[(-10,10)]*design.shape[1],options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    if not fit.success:raise RuntimeError('regional Poisson fit did not converge: '+str(fit.message))
    return RegionalDepthModel(list(feature_names),center,scale,fit.x[1:],float(fit.x[0]),average,float(ridge),True)


def fit_residual_field(mean_counts, prediction, n_bins, block_keys, calibration, *, block_width_um,
                       sigma_grid_um=(4.,8.,16.,32.)):
    """Estimate training residual spread and correlation, not technical capture.

    Within-patch variation and patch-intercept variation are separated. These
    in-sample residual estimates are descriptive and may understate uncertainty
    with only two or three calibration patches.
    """
    keep=np.asarray(calibration,bool);keys=np.asarray(block_keys)[keep]
    y=np.asarray(mean_counts,float)[keep];mu=np.asarray(prediction,float)[keep];n=np.asarray(n_bins,float)[keep]
    if len(y)<2 or np.any(mu<=0) or np.any(n<=0) or not np.isfinite(np.r_[y,mu,n]).all():
        raise ValueError('invalid residual calibration')
    residual=np.log((y+0.5/n)/(mu+0.5/n))
    means=np.array([np.average(residual[keys[:,0]==p],weights=n[keys[:,0]==p]) for p in np.unique(keys[:,0])])
    centered=residual.copy()
    for p,m in zip(np.unique(keys[:,0]),means):centered[keys[:,0]==p]-=m
    noise=float(np.average(1/np.maximum(n*mu,1),weights=n))
    variance=max(0.,float(np.average(centered**2,weights=n))-noise)
    patch_variance=float(np.var(means,ddof=1)) if len(means)>1 else 0.
    lookup={tuple(k):i for i,k in enumerate(keys)};correlations=[]
    for lag in [1,2,3]:
        pairs=[]
        for i,k in enumerate(keys):
            for offset in [(0,lag,0),(0,0,lag)]:
                j=lookup.get(tuple(k+offset))
                if j is not None:pairs.append((i,j))
        if len(pairs)<4 or variance<=1e-12:continue
        a,b=np.asarray(pairs).T
        denom=np.sqrt(np.mean(centered[a]**2)*np.mean(centered[b]**2))
        rho=float(np.mean(centered[a]*centered[b])/max(denom,1e-12))
        correlations.append({'distance_um':lag*block_width_um,'correlation':rho,'n_pairs':len(pairs)})
    if correlations:
        loss=[sum((r['correlation']-np.exp(-r['distance_um']**2/(4*s*s)))**2*r['n_pairs']
                  for r in correlations) for s in sigma_grid_um]
        sigma=float(sigma_grid_um[int(np.argmin(loss))])
    else:sigma=float(sigma_grid_um[0])
    return {'within_patch_log_sd':float(np.sqrt(variance)),'patch_log_sd':float(np.sqrt(patch_variance)),
            'sigma_um':sigma,'n_calibration_patches':len(means),'lag_correlations':correlations,
            'interpretation':'residual observed-depth variability; not identified technical capture'}


def draw_depth_field(grid, patch_ids, parameters, rng, *, spatial=True, patch_effect=False):
    """Independent smooth realizations; never interpolate a held-out real map."""
    grid=np.asarray(grid,int);patch=np.asarray(patch_ids).astype(str);out=np.ones(len(grid))
    for p in sorted(set(patch)):
        keep=np.flatnonzero(patch==p);local=grid[keep]-grid[keep].min(0)
        if spatial and parameters['within_patch_log_sd']>0:
            noise=rng.normal(size=tuple(local.max(0)+1))
            smooth=gaussian_filter(noise,float(parameters['sigma_um'])/2,mode='reflect')[local[:,0],local[:,1]]
            smooth=(smooth-smooth.mean())/max(smooth.std(),1e-12)
            field=np.exp(np.clip(parameters['within_patch_log_sd']*smooth,-15,15))
            out[keep]=field/field.mean()  # simulated-only normalization separates the patch effect
        if patch_effect and parameters['patch_log_sd']>0:
            sd=parameters['patch_log_sd'];out[keep]*=np.exp(np.clip(rng.normal(-sd*sd/2,sd),-15,15))
    return out


def interpolate_block_values(values, coordinates, block_centers, block_patch_ids, patch_ids, *, sigma_um):
    """Smooth synthetic/model block rates to bins without crossing patch contexts."""
    values=np.asarray(values,float);xy=np.asarray(coordinates,float);centers=np.asarray(block_centers,float)
    block_patch=np.asarray(block_patch_ids).astype(str);patch=np.asarray(patch_ids).astype(str)
    if values.shape!=(len(centers),) or np.any(values<=0) or not np.isfinite(values).all() or sigma_um<=0:
        raise ValueError('positive block values and bandwidth required')
    out=np.empty(len(xy))
    for p in sorted(set(patch)):
        b=np.flatnonzero(block_patch==p);rows=np.flatnonzero(patch==p)
        if not len(b):raise ValueError('missing model blocks for patch')
        d2=np.sum((xy[rows,None,:]-centers[None,b,:])**2,axis=2)
        logw=-d2/(2*sigma_um*sigma_um);logw-=logw.max(1,keepdims=True)
        w=np.exp(logw);out[rows]=(w@values[b])/w.sum(1)
    return out
