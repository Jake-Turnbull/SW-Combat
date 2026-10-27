#!/usr/bin/env python3
"""Three-figure benchmark of feature-block oracle vs learned local ComBat priors.

Run: python oracle_similarity_benchmark.py --output results/oracle_similarity
The two oracle matrices are *nonnegative affinity targets* (not Pearson
correlations): same-block features have affinity 1, others 0.05.
"""
from __future__ import annotations
import argparse
import contextlib
import importlib.util
import io
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODULE = HERE / 'HarmonisationFunctions.py'
spec = importlib.util.spec_from_file_location('Harmonisation_oracle', MODULE)
harm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harm)

METHODS = {
    'Global': ("global", []),
    'Correlation': ("local", ['correlation_similarity']),
    'Magnitude': ("local", ['magnitude_similarity']),
    'Variance': ("local", ['variance_similarity']),
    'Oracle': ("local", ['custom']),
}


def row_normalize(a):
    return a / np.maximum(a.sum(axis=-1, keepdims=True), 1e-12)


def simulate(n, seed, features=20, batches=8, blocks=5):
    rng = np.random.default_rng(seed)
    groups = np.concatenate([np.full(len(ix), i) for i, ix in enumerate(np.array_split(np.arange(features), blocks))])
    # Two deliberately different partitions: distinct mean and scale oracles.
    scale_groups = (groups + np.arange(features) % 3) % blocks
    mean_sim = np.where(groups[:, None] == groups[None, :], 1.0, 0.05)
    scale_sim = np.where(scale_groups[:, None] == scale_groups[None, :], 1.0, 0.1)
    total = n * batches
    batch = np.repeat(np.arange(batches), n)
    # Correlated biological noise within the mean-effect blocks; its structure
    # need not coincide with the multiplicative-effect blocks.
    latent = rng.normal(size=(blocks, total))
    residual = np.sqrt(0.65) * latent[groups] + np.sqrt(0.35) * rng.normal(size=(features, total))
    clean = residual.copy()
    gamma = rng.normal(0, 0.65, (batches, blocks))[:, groups] + rng.normal(0, 0.12, (batches, features))
    gamma -= gamma.mean(axis=0, keepdims=True)
    log_delta = rng.normal(0, 0.32, (batches, blocks))[:, scale_groups] + rng.normal(0, 0.1, (batches, features))
    log_delta -= log_delta.mean(axis=0, keepdims=True)
    delta = np.exp(log_delta)
    observed = gamma[batch].T + np.sqrt(delta[batch].T) * residual
    return dict(observed=observed, clean=clean, batch=batch, gamma=gamma, delta=delta,
                mean_sim=mean_sim, scale_sim=scale_sim, groups=groups, scale_groups=scale_groups)


def fit(sim, method, trust=0.85):
    mode, weight_methods = METHODS[method]
    opts = {'mean_trust': trust, 'scale_trust': trust, 'min_effective': 1,
            'variance_similarity_kernel': 'gaussian_log', 'variance_kernel_bandwidth': 0.5}
    if method == 'Oracle':
        opts['similarity_matrix'] = sim['mean_sim']
        opts['scale_similarity_matrix'] = sim['scale_sim']
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore', RuntimeWarning)
        output = harm.combat_modular(sim['observed'], sim['batch'], mod=None,
                prior_mode=mode, prior_weight_methods=weight_methods,
                prior_weight_opts=opts if mode == 'local' else None)
    p = output['priors']
    v = np.asarray(output['standardisation']['var_pooled'])
    # Truth in fitted pooled-standardized ComBat units, as in existing benchmark.
    true_gamma = sim['gamma'] / np.sqrt(v)[None, :]
    true_delta = sim['delta'] / v[None, :]
    if mode == 'global':
        uniform = np.full_like(sim['mean_sim'], 1.0 / len(v))
        w_mean, w_scale = uniform, uniform
    else:
        local = p['local_priors']
        w_mean = np.mean(local['weights'], axis=0)
        w_scale = np.mean(local['scale_weights'], axis=0)
    return dict(gamma=np.asarray(p['gamma_star']), delta=np.asarray(p['delta_star']),
                true_gamma=true_gamma, true_delta=true_delta,
                w_mean=w_mean, w_scale=w_scale,
                Harmonised=np.asarray(output['bayesdata']))


def rmse(a, b):
    return float(np.sqrt(np.mean((a-b)**2)))


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    sizes = args.sample_sizes
    rows, weights, effects = [], [], []
    representative = None
    for n in sizes:
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            sim = simulate(n, seed, args.features, args.batches, args.blocks)
            if n == sizes[len(sizes)//2] and repeat == 0:
                representative = (sim, {})
            for method in METHODS:
                fit_result = fit(sim, method, args.trust)
                rows.append(dict(n_per_batch=n, repeat=repeat, seed=seed, method=method,
                    gamma_rmse=rmse(fit_result['gamma'], fit_result['true_gamma']),
                    log_delta_rmse=rmse(np.log(np.maximum(fit_result['delta'], 1e-12)), np.log(fit_result['true_delta'])),
                    Harmonisation_rmse=rmse(fit_result['Harmonised'], sim['clean']),
                    mean_weight_rmse=rmse(fit_result['w_mean'], row_normalize(sim['mean_sim'])),
                    scale_weight_rmse=rmse(fit_result['w_scale'], row_normalize(sim['scale_sim']))))
                if representative is not None and n == sizes[len(sizes)//2] and repeat == 0:
                    representative[1][method] = fit_result
                if repeat == 0 and n == sizes[len(sizes)//2]:
                    for kind in ('mean', 'scale'):
                        mat = fit_result['w_mean' if kind == 'mean' else 'w_scale']
                        for i in range(args.features):
                            for j in range(args.features):
                                weights.append(dict(method=method, effect=kind, target_feature=i,
                                                    source_feature=j, weight=float(mat[i,j])))
                    for b in range(args.batches):
                        for f in range(args.features):
                            effects.append(dict(method=method, batch=b, feature=f,
                                gamma_true=float(fit_result['true_gamma'][b,f]), gamma_estimated=float(fit_result['gamma'][b,f]),
                                delta_true=float(fit_result['true_delta'][b,f]), delta_estimated=float(fit_result['delta'][b,f])))
        print(f'Finished n={n} per batch', flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output/'replicate_metrics.csv', index=False)
    summary = metrics.groupby(['n_per_batch','method'],sort=False).agg(
        **{k+'_mean':(k,'mean') for k in ['gamma_rmse','log_delta_rmse','Harmonisation_rmse','mean_weight_rmse','scale_weight_rmse']},
        **{k+'_sem':(k,lambda x: x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0) for k in ['gamma_rmse','log_delta_rmse','Harmonisation_rmse','mean_weight_rmse','scale_weight_rmse']}).reset_index()
    summary.to_csv(args.output/'summary.csv',index=False)
    pd.DataFrame(weights).to_csv(args.output/'representative_weights.csv',index=False)
    pd.DataFrame(effects).to_csv(args.output/'representative_effect_estimates.csv',index=False)
    sim, fits = representative
    np.savez_compressed(args.output/'oracle_matrices.npz', mean=sim['mean_sim'], scale=sim['scale_sim'],
                        mean_groups=sim['groups'], scale_groups=sim['scale_groups'])
    (args.output/'config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2))
    # FIGURE 1: affinity oracles (top row) versus normalized effective prior weights.
    fig, ax = plt.subplots(2, 6, figsize=(17,6), constrained_layout=True)
    names = ['True oracle'] + list(METHODS)
    for r, (effect, oracle) in enumerate([('mean',sim['mean_sim']),('scale',sim['scale_sim'])]):
        for c, name in enumerate(names):
            mat = row_normalize(oracle) if c == 0 else fits[name]['w_'+effect]
            im = ax[r,c].imshow(mat,cmap='viridis',vmin=0,vmax=0.12,interpolation='nearest')
            ax[r,c].set_title(f'{effect.title()}: {name}',fontsize=10)
            ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
    fig.colorbar(im,ax=ax.ravel().tolist(),label='Row-normalized weight',shrink=0.75)
    fig.savefig(args.output/'figure1_similarity_heatmaps.png',dpi=180); plt.close(fig)
    # FIGURE 2: one panel per method, gamma upper row, delta lower row.
    fig, ax = plt.subplots(2,len(METHODS),figsize=(16,7),constrained_layout=True)
    for c, (name, f) in enumerate(fits.items()):
        for r, (truth, estimate, title) in enumerate([
            (f['true_gamma'],f['gamma'],'Gamma'),(f['true_delta'],f['delta'],'Delta')]):
            x,y=truth.ravel(),estimate.ravel()
            ax[r,c].scatter(x,y,s=9,alpha=0.4)
            lo=min(x.min(),y.min()); hi=max(x.max(),y.max())
            ax[r,c].plot([lo,hi],[lo,hi],'k--',lw=1)
            ax[r,c].set(title=f'{name}: {title}',xlabel='True',ylabel='Estimated')
            # Add lable showing the correlation coefficient
            corr = np.corrcoef(x, y)[0, 1]
            ax[r,c].text(0.05, 0.95, f'Corr: {corr:.2f}', transform=ax[r,c].transAxes,
                         verticalalignment='top', fontsize=8)
    fig.savefig(args.output/'figure2_effect_recovery.png',dpi=180); plt.close(fig)
    # FIGURE 3: recovery and Harmonisation errors across n.
    fig, ax = plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
    for name in METHODS:
        sub=summary[summary.method==name].sort_values('n_per_batch')
        x=sub.n_per_batch.to_numpy()
        for a,key,title in zip(ax,['gamma_rmse','log_delta_rmse','Harmonisation_rmse'],
                                ['Gamma RMSE','Log-delta RMSE','Harmonisation RMSE']):
            y=sub[key+'_mean'].to_numpy(); sem=sub[key+'_sem'].to_numpy()
            a.plot(x,y,marker='o',label=name)
            a.fill_between(x,y-1.96*sem,y+1.96*sem,alpha=0.10)
            a.set(xlabel='Samples per batch',ylabel='RMSE',title=title)
            a.grid(alpha=0.2)
    ax[-1].legend(frameon=False,fontsize=8)
    fig.savefig(args.output/'figure3_sample_size_performance.png',dpi=180); plt.close(fig)
    print(f'Wrote three figures and numerical results to {args.output.resolve()}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'oracle_similarity_results')
    parser.add_argument('--sample-sizes',nargs='+',type=int,default=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150])
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--features',type=int,default=50)
    parser.add_argument('--batches',type=int,default=2)
    parser.add_argument('--blocks',type=int,default=5)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--trust',type=float,default=0.5)
    args=parser.parse_args()
    if min(args.sample_sizes)<2 or args.repeats<1 or args.batches<2 or args.blocks<2 or args.features<args.blocks:
        parser.error('Invalid sample sizes, repeats, batches, features or blocks')
    run(args)

if __name__=='__main__':
    main()
