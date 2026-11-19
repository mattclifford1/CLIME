'''Figures from results/*.csv. Writes figures/*.pdf.

usage: uv run python make_figures.py [name ...]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
from common import setup   # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thesis_figures import roles
from common import paths, style
from common.style import (ARM_COLOURS, ARM_LABELS, EXPLAINER_COLOURS, EXPLAINER_LABELS,
                          EXPLAINER_MARKERS, FAMILY_SHORT, INK, INK2, MUTED, SETTING_LABELS,
                          TEXTWIDTH)

MAIN = ['LIME', 'CLIME', 'LIME + re-threshold']
SETTINGS = ['native', 'balanced', 'imbalanced']
FAMILIES = list(FAMILY_SHORT)
ARMS = ['plain', 'weighted', 'SMOTE', 're-thresholded', 'CP Minimax', 'Slacks Deltas',
        'Gaussian Predictive']
MINORITY = ['none', 'under 1%', '1-5%', '5-20%', '20-50%']
SETTING_SHORT = {'native': 'own\nratio', 'balanced': '110:110', 'imbalanced': '200:20'}
FAMILY_COLOURS = dict(zip(FAMILIES, [roles.C[c] for c in
                                     ('aqua', 'blue', 'plum', 'orange', 'brown', 'red')]))


def read(name):
    return pd.read_csv(paths.RESULTS/f'{name}.csv')


def settings_present(frame):
    return [s for s in SETTINGS if s in set(frame.setting)]


def explainer_legend(fig, names=MAIN, y=-0.02, **kwargs):
    handles = [plt.Line2D([], [], color=EXPLAINER_COLOURS[n], marker=EXPLAINER_MARKERS[n],
                          label=EXPLAINER_LABELS[n]) for n in names]
    fig.legend(handles=handles, ncol=len(names), loc='lower center', bbox_to_anchor=(0.5, y),
               **kwargs)


def reproduce():
    '''the published configurations, rerun'''
    frame = read('reproduce')
    names = list(dict.fromkeys(frame.configuration))
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, TEXTWIDTH*0.5), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), names):
        part = frame[frame.configuration == name]
        for explainer in ('LIME', 'CLIME'):
            line = part[part.explainer == explainer]
            ax.plot(line['index'], line.fidelity, color=EXPLAINER_COLOURS[explainer],
                    marker=EXPLAINER_MARKERS[explainer], ms=2.5)
        ax.set_title(name, fontsize=7.5)
        ax.set(ylim=(0.4, 1.02), xticks=[0, 5, 10, 15])
    for ax in axes[-1]:
        ax.set_xlabel('query point on the line')
    for ax in axes[:, 0]:
        ax.set_ylabel('local fidelity')
    explainer_legend(fig, ('LIME', 'CLIME'), y=-0.07)
    style.save(fig, 'reproduce')


def replication():
    '''LIME against CLIME per dataset and family, by accuracy and by balanced fidelity'''
    frame = read('datasets')
    frame = frame[frame.kind == 'line']
    present = settings_present(frame)
    metrics = [('test_fidelity', 'local fidelity'), ('test_balanced', 'balanced local fidelity')]
    fig, axes = plt.subplots(2, len(present), figsize=(TEXTWIDTH, TEXTWIDTH*0.62), squeeze=False)
    for row, (metric, label) in enumerate(metrics):
        for ax, setting in zip(axes[row], present):
            wide = frame[frame.setting == setting].pivot_table(
                index=['dataset', 'family'], columns='explainer', values=metric).reset_index()
            for family in FAMILIES:
                part = wide[wide.family == family]
                ax.scatter(part.LIME, part.CLIME, s=9, color=FAMILY_COLOURS[family], alpha=0.75,
                           linewidths=0, label=FAMILY_SHORT[family])
            ax.plot([0.4, 1], [0.4, 1], color=MUTED, lw=0.8, ls=':')
            ax.set(xlim=(0.4, 1.01), ylim=(0.4, 1.01), aspect='equal')
            above = (wide.CLIME > wide.LIME + 0.005).mean()
            ax.text(0.97, 0.05, f'CLIME higher in {above:.0%}', transform=ax.transAxes, fontsize=7,
                    color=INK, ha='right', bbox=dict(fc='white', ec='none', pad=1.5, alpha=0.8))
            if row == 0:
                ax.set_title(SETTING_LABELS[setting], fontsize=8)
            if row == 1:
                ax.set_xlabel('LIME')
        axes[row][0].set_ylabel(f'CLIME\n{label}')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.07),
               markerscale=1.6, handletextpad=0.2, columnspacing=1.2)
    style.save(fig, 'replication')


def profile():
    '''along the line: fidelity, balanced fidelity and how one-sided the neighbourhood is'''
    frame = read('profile')
    frame = frame[frame.family == 'all']
    present = settings_present(frame)
    rows = [('test_fidelity', 'local fidelity'), ('test_balanced', 'balanced\nlocal fidelity')]
    fig, axes = plt.subplots(3, len(present), figsize=(TEXTWIDTH, TEXTWIDTH*0.66), sharex=True,
                             squeeze=False)
    for col, setting in enumerate(present):
        part = frame[frame.setting == setting]
        for row, (metric, label) in enumerate(rows):
            ax = axes[row][col]
            for explainer in MAIN:
                line = part[part.explainer == explainer].sort_values('index')
                ax.plot(line['index'], line[metric], color=EXPLAINER_COLOURS[explainer],
                        marker=EXPLAINER_MARKERS[explainer], ms=2.5)
            ax.set_ylim(0.45, 1.0)
            if col == 0:
                ax.set_ylabel(label)
        ax = axes[2][col]
        line = part[part.explainer == 'LIME'].sort_values('index')
        ax.plot(line['index'], line.f_query, color=INK, label='$f(q)$')
        ax.plot(line['index'], line.sample_minority, color=roles.SAMPLE, ls='--',
                label='rarer class in the sample')
        ax.plot(line['index'], line.test_effective_share, color=MUTED, ls=':',
                label='effective share of test set')
        ax.set(ylim=(0, 1), xlabel='query point', xticks=[0, 5, 10, 15])
        axes[0][col].set_title(SETTING_LABELS[setting], fontsize=8)
    axes[2][0].set_ylabel('share')
    handles, labels = axes[2][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.1),
               fontsize=7)
    explainer_legend(fig, y=-0.05)
    style.save(fig, 'profile')


def triad():
    '''operating point, ranking and calibration of the three surrogates'''
    frame = read('triad')
    frame = frame[(frame.family == 'all') & (frame.arm == 'plain') & (frame.kind == 'line')
                  & frame.explainer.isin(MAIN)]
    present = settings_present(frame)
    metrics = [('test_fidelity', 'fidelity\n(cut at 0.5 or moved)'),
               ('test_balanced', 'balanced fidelity'), ('test_auc', 'AUC (ranking)'),
               ('test_brier', 'Brier (calibration,\nlower is better)')]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, TEXTWIDTH*0.3))
    width = 0.25
    for ax, (metric, label) in zip(axes, metrics):
        for i, explainer in enumerate(MAIN):
            values = [frame[(frame.setting == s) & (frame.explainer == explainer)][metric].mean()
                      for s in present]
            ax.bar(np.arange(len(present)) + (i - 1)*width, values, width,
                   color=EXPLAINER_COLOURS[explainer], label=EXPLAINER_LABELS[explainer])
        ax.set_xticks(np.arange(len(present)))
        ax.set_xticklabels([SETTING_SHORT[s] for s in present], fontsize=6.5)
        ax.set_title(label, fontsize=7.5)
        if metric != 'test_brier':
            ax.set_ylim(0.4, 1.0)
        ax.grid(axis='x', visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.2))
    fig.subplots_adjust(wspace=0.4)
    style.save(fig, 'triad')


def one_sided():
    '''scores by the share of the neighbourhood sample in the rarer black-box class'''
    frame = read('one_sided')
    frame = frame[(frame.arm == 'plain') & (frame.kind == 'line')]
    present = settings_present(frame)
    rows = [('test_fidelity', 'local fidelity'), ('test_balanced', 'balanced\nlocal fidelity'),
            ('share', 'share of\nquery points')]
    fig, axes = plt.subplots(4, len(present), figsize=(TEXTWIDTH, TEXTWIDTH*0.74), sharex=True,
                             squeeze=False, height_ratios=(1, 1, 0.6, 0.6))
    x = np.arange(len(MINORITY))
    for col, setting in enumerate(present):
        part = frame[frame.setting == setting].set_index(['explainer', 'minority'])
        for row, (metric, label) in enumerate(rows[:2]):
            ax = axes[row][col]
            for explainer in MAIN:
                values = [part[metric].get((explainer, m), np.nan) for m in MINORITY]
                ax.plot(x, values, color=EXPLAINER_COLOURS[explainer],
                        marker=EXPLAINER_MARKERS[explainer])
            ax.set_ylim(0.45, 1.01)
            if col == 0:
                ax.set_ylabel(label)
        ax = axes[2][col]
        ax.plot(x, [part['cosine'].get(('LIME', m), np.nan) for m in MINORITY], color=INK,
                marker='o')
        ax.set_ylim(0.5, 1.02)
        ax = axes[3][col]
        ax.bar(x, [part['share'].get(('LIME', m), 0) for m in MINORITY], color=MUTED, width=0.6)
        ax.set(ylim=(0, 0.6), xticks=x)
        ax.set_xticklabels(['none', '<1%', '1-5%', '5-20%', '20-50%'], fontsize=6.5)
        ax.set_xlabel('rarer class in the sample', fontsize=8)
        ax.grid(axis='x', visible=False)
        axes[0][col].set_title(SETTING_LABELS[setting], fontsize=8)
    axes[2][0].set_ylabel('cosine of LIME\nand CLIME\ncoefficients')
    axes[3][0].set_ylabel(rows[2][1])
    explainer_legend(fig, y=-0.04)
    style.save(fig, 'one_sided')


def arms():
    '''what each correction of the black box does to the classifier and to its explanations'''
    frame = read('arms')
    frame = frame[(frame.family == 'all') & (frame.kind == 'test') & frame.arm.isin(ARMS)]
    present = [s for s in ('imbalanced', 'native') if s in set(frame.setting)]
    panels = [('balanced_accuracy', "black box's\nbalanced accuracy"),
              ('one_sided', 'one-sided\nneighbourhoods'),
              ('fidelity: LIME', 'LIME\nlocal fidelity'),
              ('balanced: LIME', 'LIME balanced\nlocal fidelity'),
              ('balanced: CLIME', 'CLIME balanced\nlocal fidelity')]
    fig, axes = plt.subplots(len(present), len(panels), squeeze=False, sharey=True,
                             figsize=(TEXTWIDTH, TEXTWIDTH*0.26*len(present) + 0.5))
    for row, setting in enumerate(present):
        part = frame[frame.setting == setting].set_index('arm')
        order = [a for a in ARMS if a in part.index][::-1]
        for ax, (metric, label) in zip(axes[row], panels):
            ax.barh(range(len(order)), part.loc[order, metric],
                    color=[ARM_COLOURS[a] for a in order], height=0.7)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([ARM_LABELS[a] for a in order], fontsize=6.5)
            ax.set_xlim(0 if metric == 'one_sided' else 0.4, 1.0)
            ax.set_xticks([0, 0.5] if metric == 'one_sided' else [0.5, 0.75])
            ax.grid(axis='y', visible=False)
            if row == 0:
                ax.set_title(label, fontsize=7.5)
        axes[row][0].set_ylabel(SETTING_LABELS[setting], fontsize=8)
    fig.subplots_adjust(wspace=0.12, hspace=0.25)
    style.save(fig, 'arms')


def families():
    '''CLIME minus LIME by model family and class ratio'''
    frame = read('replication')
    frame = frame[(frame.kind == 'line') & (frame.explainer == 'CLIME') & (frame.family != 'all')]
    present = settings_present(frame)
    metrics = [('test_fidelity', 'local fidelity'), ('worst', 'worst query point'),
               ('test_balanced', 'balanced local fidelity')]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH*0.3), sharey=True)
    for ax, (metric, label) in zip(axes, metrics):
        part = frame[frame.metric == metric].pivot_table(index='family', columns='setting',
                                                         values='diff')
        part = part.reindex(index=FAMILIES, columns=present)
        limit = max(0.05, np.nanmax(np.abs(part.values)))
        ax.imshow(part.values, cmap=roles.DIVERGING_CMAP, vmin=-limit, vmax=limit, aspect='auto')
        for (i, j), value in np.ndenumerate(part.values):
            if np.isfinite(value):
                ax.text(j, i, f'{value:+.2f}', ha='center', va='center', fontsize=7)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([SETTING_SHORT[s] for s in present], fontsize=6.5)
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_yticklabels([FAMILY_SHORT[f] for f in FAMILIES])
        ax.set_title(f'CLIME $-$ LIME:\n{label}', fontsize=7.5)
        ax.grid(False)
    style.save(fig, 'families')


def grid():
    '''maps of LIME's fidelity and CLIME's gain on the first two principal components'''
    frame, test = read('grid'), read('grid_test')
    configurations = list(dict.fromkeys(zip(frame.dataset, frame.setting)))
    columns = [('test_fidelity', 'LIME', 'LIME local fidelity', 'viridis', (0.4, 1)),
               ('test_fidelity', 'CLIME', 'CLIME local fidelity', 'viridis', (0.4, 1)),
               ('gain', None, 'CLIME $-$ LIME', roles.DIVERGING_CMAP, (-0.4, 0.4)),
               ('sample_minority', 'LIME', 'rarer class in the sample', 'magma_r', (0, 0.5))]
    fig, axes = plt.subplots(len(configurations), 5, squeeze=False,
                             figsize=(TEXTWIDTH, TEXTWIDTH*0.2*len(configurations)))
    for row, (dataset, setting) in enumerate(configurations):
        part = frame[(frame.dataset == dataset) & (frame.setting == setting)]
        points = test[(test.dataset == dataset) & (test.setting == setting)]
        wide = part.pivot_table(index=['pc2', 'pc1'], columns='explainer', values='test_fidelity')
        gain = (wide['CLIME'] - wide['LIME']).unstack('pc1')
        extent = (part.pc1.min(), part.pc1.max(), part.pc2.min(), part.pc2.max())
        ax = axes[row][0]
        for label in (0, 1):
            keep = points.y == label
            ax.scatter(points.pc1[keep], points.pc2[keep], s=2, color=roles.CLASS[label],
                       alpha=0.6, linewidths=0)
        f = part[part.explainer == 'LIME'].pivot_table(index='pc2', columns='pc1', values='f_query')
        ax.contour(f.columns, f.index, f.values, levels=[0.5], colors=INK, linewidths=0.8)
        ax.set_ylabel(f'{dataset.split()[0]}\n{SETTING_LABELS[setting]}', fontsize=6.5)
        for ax, (value, explainer, title, cmap, limits) in zip(axes[row][1:], columns):
            if value == 'gain':
                image = gain
            else:
                image = part[part.explainer == explainer].pivot_table(index='pc2', columns='pc1',
                                                                      values=value)
            shown = ax.imshow(image.values, origin='lower', extent=extent, aspect='auto',
                              cmap=cmap, vmin=limits[0], vmax=limits[1])
            ax.contour(f.columns, f.index, f.values, levels=[0.5], colors=INK, linewidths=0.8)
            if row == 0:
                ax.set_title(title, fontsize=7)
            if row == len(configurations) - 1:
                fig.colorbar(shown, ax=list(axes[:, list(axes[row]).index(ax)]),
                             orientation='horizontal', fraction=0.03, pad=0.03)
        if row == 0:
            axes[row][0].set_title('test data and boundary', fontsize=7)
        for ax in axes[row]:
            ax.set(xticks=[], yticks=[], xlim=extent[:2], ylim=extent[2:])
            ax.grid(False)
    style.save(fig, 'grid')


def operating_point():
    '''every arm as a point: explanations follow where the black box is cut'''
    frame = read('arms')
    frame = frame[(frame.family == 'all') & (frame.kind == 'test')]
    present = [s for s in ('imbalanced', 'native') if s in set(frame.setting)]
    panels = [('one_sided', 'one-sided neighbourhoods'),
              ('fidelity: LIME', 'LIME local fidelity'),
              ('balanced: LIME', 'LIME balanced local fidelity')]
    fig, axes = plt.subplots(len(present), 3, squeeze=False, sharex='row',
                             figsize=(TEXTWIDTH, TEXTWIDTH*0.3*len(present) + 0.3))
    kinds = {'training': ('s', 22), 'validation': ('D', 22), 'deltas': ('o', 12)}
    for row, setting in enumerate(present):
        part = frame[frame.setting == setting]
        for ax, (metric, label) in zip(axes[row], panels):
            for placement, (marker, size) in kinds.items():
                sub = part[part.placement == placement]
                colours = [ARM_COLOURS.get(a, MUTED) for a in sub.arm]
                ax.scatter(sub.positive_rate, sub[metric], s=size, marker=marker, c=colours,
                           edgecolors=INK, linewidths=0.3, alpha=0.9, zorder=3)
            ax.set_ylim(0 if metric == 'one_sided' else 0.4, 1.0)
            if row == 0:
                ax.set_title(label, fontsize=7.5)
        axes[row][0].set_ylabel(SETTING_LABELS[setting], fontsize=8)
    axes[-1][1].set_xlabel('share of test points the black box calls minority')
    handles = [plt.Line2D([], [], ls='', marker='s', color=ARM_COLOURS[a], markeredgecolor=INK,
                          markeredgewidth=0.3, label=ARM_LABELS[a])
               for a in ('plain', 'weighted', 'SMOTE')]
    handles.append(plt.Line2D([], [], ls='', marker='D', color=ARM_COLOURS['re-thresholded'],
                              markeredgecolor=INK, markeredgewidth=0.3,
                              label=ARM_LABELS['re-thresholded']))
    handles.append(plt.Line2D([], [], ls='', marker='o', color=MUTED, markeredgecolor=INK,
                              markeredgewidth=0.3, label='a deltas method'))
    fig.legend(handles=handles, ncol=5, loc='lower center', bbox_to_anchor=(0.5, -0.13),
               handletextpad=0.2, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.28, hspace=0.3)
    style.save(fig, 'operating_point')


def by_class():
    '''test points of each true class under each correction of the black box'''
    frame = read('by_class')
    frame = frame[(frame.setting == 'imbalanced') & frame.explainer.isin(MAIN)]
    order = [a for a in ('plain', 'weighted', 'SMOTE', 're-thresholded', 'CP Minimax')
             if a in set(frame.arm)]
    panels = [('one_sided', 'one-sided\nneighbourhoods', None),
              ('test_fidelity', 'local fidelity', 'LIME'),
              ('test_balanced', 'balanced local fidelity', 'LIME'),
              ('test_balanced', 'balanced local fidelity', 'CLIME')]
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXTWIDTH, TEXTWIDTH*0.3), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (metric, label, explainer) in zip(axes, panels):
        part = frame[frame.explainer == (explainer or 'LIME')]
        for label_, offset, name in ((0, 0.18, 'majority'), (1, -0.18, 'minority')):
            values = part[part.label == label_].set_index('arm')[metric].reindex(order)
            ax.barh(y + offset, values, height=0.34, color=roles.CLASS[label_],
                    label=f'{name} query points')
        ax.set_yticks(y)
        ax.set_yticklabels([ARM_LABELS[a] for a in order], fontsize=6.5)
        ax.set_xlim(0 if metric == 'one_sided' else 0.4, 1.0)
        ax.set_title(label if explainer is None else f'{explainer}:\n{label}', fontsize=7.5)
        ax.grid(axis='y', visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.1))
    style.save(fig, 'by_class')


def sweep():
    '''the imbalance ratio'''
    frame = read('sweep_arms')
    frame = frame[(frame.family == 'all') & (frame.kind == 'line')
                  & frame.arm.isin(['plain', 'weighted', 're-thresholded', 'CP Minimax'])]
    panels = [('balanced_accuracy', "black box's\nbalanced accuracy"),
              ('one_sided', 'one-sided\nneighbourhoods'),
              ('fidelity: LIME', 'LIME local fidelity'),
              ('balanced: LIME', 'LIME balanced\nlocal fidelity'),
              ('balanced: CLIME', 'CLIME balanced\nlocal fidelity')]
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXTWIDTH, TEXTWIDTH*0.27))
    for ax, (metric, label) in zip(axes, panels):
        for arm, part in frame.groupby('arm'):
            part = part.sort_values('setting')
            ax.plot(part.setting, part[metric], color=ARM_COLOURS[arm], marker='o',
                    label=ARM_LABELS[arm])
        ax.set_xscale('log')
        ax.set_xticks([4, 10, 20, 40])
        ax.set_xticklabels(['4', '10', '20', '40'])
        ax.minorticks_off()
        ax.set_title(label, fontsize=7.5)
        ax.tick_params(labelsize=7)
    axes[2].set_xlabel('majority rows per minority row in the development set')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.22))
    fig.subplots_adjust(wspace=0.5)
    style.save(fig, 'sweep')


def kernel():
    '''the locality kernel width'''
    frame = read('kernel')
    frame = frame[frame.kind == 'line']
    present = [s for s in ('native', 'imbalanced') if s in set(frame.setting)]
    fig, axes = plt.subplots(len(present), 3, squeeze=False, sharex=True,
                             figsize=(TEXTWIDTH, TEXTWIDTH*0.26*len(present) + 0.3))
    for row, setting in enumerate(present):
        part = frame[frame.setting == setting]
        pooled = part.groupby(['kernel_width', 'explainer'])[
            ['test_fidelity', 'test_balanced', 'test_effective_share']].mean().reset_index()
        for ax, metric in zip(axes[row][:2], ('test_fidelity', 'test_balanced')):
            for explainer in MAIN:
                line = pooled[pooled.explainer == explainer]
                ax.plot(line.kernel_width, line[metric], color=EXPLAINER_COLOURS[explainer],
                        marker=EXPLAINER_MARKERS[explainer])
            ax.set_ylim(0.45, 1.0)
        line = pooled[pooled.explainer == 'LIME']
        axes[row][2].plot(line.kernel_width, line.test_effective_share, color=INK, marker='o')
        axes[row][2].set_ylim(0, 1)
        axes[row][0].set_ylabel(SETTING_LABELS[setting], fontsize=8)
        for ax in axes[row]:
            ax.set_xscale('log')
            ax.set_xticks([0.25, 0.5, 0.75, 1.5, 3])
            ax.set_xticklabels(['0.25', '0.5', '0.75', '1.5', '3'])
            ax.minorticks_off()
            ax.axvline(0.75, color=MUTED, lw=0.8, ls=':')
    for ax, title in zip(axes[0], ('local fidelity', 'balanced local fidelity',
                                   'effective share of the test set')):
        ax.set_title(title, fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel(r'kernel width / $\sqrt{d}$')
    explainer_legend(fig, y=-0.07)
    style.save(fig, 'kernel')


def matched():
    '''CLIME's gain over LIME split into a change of cut and a change of fit'''
    frame = read('matched_parts')
    frame = frame[frame.kind == 'line']
    present = settings_present(frame)
    metrics = [('test_fidelity', 'local fidelity'), ('worst', 'worst query point'),
               ('test_balanced', 'balanced local fidelity')]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH*0.3))
    x = np.arange(len(present))
    for ax, (metric, label) in zip(axes, metrics):
        part = frame[frame.metric == metric].pivot_table(index='setting', columns='part',
                                                         values='diff').reindex(present)
        ax.bar(x - 0.2, part['cut'], 0.38, color=EXPLAINER_COLOURS['LIME + re-threshold'],
               label="cut: LIME moved to CLIME's operating point")
        ax.bar(x + 0.2, part['fit'], 0.38, color=EXPLAINER_COLOURS['CLIME'],
               label="fit: CLIME against that cut")
        ax.plot(x, part['total'], ls='', marker='_', ms=22, color=INK, label='CLIME $-$ LIME')
        ax.axhline(0, **style.ZERO_LINE)
        ax.set_xticks(x)
        ax.set_xticklabels([SETTING_SHORT[s] for s in present], fontsize=6.5)
        ax.set_title(label, fontsize=7.5)
        ax.grid(axis='x', visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.24),
               fontsize=7)
    fig.subplots_adjust(wspace=0.35)
    style.save(fig, 'matched')


SIX = ['LIME', 'CLIME', 'Logit-LIME', 'Logit-CLIME', 'Soft logistic', 'Soft logistic CLIME']
SMOOTH = ['Logistic', 'SVM', 'MLP']


def cuts():
    """one surrogate scored at every cut: the scores that move and the explanation that does not"""
    frame = read('explanations_cuts')
    frame = frame[(frame.family == 'all') & (frame.kind == 'line')]
    present = settings_present(frame)
    fig, axes = plt.subplots(1, len(present), figsize=(TEXTWIDTH, TEXTWIDTH*0.3), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, setting in zip(axes, present):
        for explainer in ('LIME', 'CLIME'):
            part = frame[(frame.setting == setting) & (frame.explainer == explainer)].sort_values('cut')
            colour = EXPLAINER_COLOURS[explainer]
            ax.plot(part.cut, part.test_balanced, color=colour, marker='')
            ax.plot(part.cut, part.test_fidelity, color=colour, marker='', ls=':')
            at = part[np.isclose(part.cut, 0.5)]
            ax.plot(at.cut, at.test_balanced, ls='', marker=EXPLAINER_MARKERS[explainer],
                    color=colour, ms=5)
        ax.axvline(0.5, color=MUTED, lw=0.8, ls='--', zorder=1)
        ax.set_title(SETTING_LABELS[setting], fontsize=7.5)
        ax.set_xlabel("cut on the surrogate's score")
    axes[0].set_ylabel('agreement with the black box')
    handles = [plt.Line2D([], [], color=EXPLAINER_COLOURS[n], label=EXPLAINER_LABELS[n])
               for n in ('LIME', 'CLIME')]
    handles += [plt.Line2D([], [], color=INK, label='balanced fidelity'),
                plt.Line2D([], [], color=INK, ls=':', label='fidelity')]
    fig.legend(handles=handles, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.3))
    style.save(fig, 'cuts')


def truth():
    """cosine between each surrogate's coefficients and the black box's log-odds gradient"""
    frame = read('explanations_triad')
    frame = frame[(frame.arm == 'plain') & (frame.kind == 'test') & frame.explainer.isin(SIX)]
    present = settings_present(frame)
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH*0.32), sharey=True)
    width = 0.8/len(SIX)
    for ax, family in zip(axes, SMOOTH):
        for i, explainer in enumerate(SIX):
            part = frame[(frame.family == family) & (frame.explainer == explainer)]
            values = [part[part.setting == s].truth_point_cosine.mean() for s in present]
            ax.bar(np.arange(len(present)) + (i - (len(SIX) - 1)/2)*width, values, width,
                   color=EXPLAINER_COLOURS[explainer], label=EXPLAINER_LABELS[explainer])
        ax.set_xticks(np.arange(len(present)))
        ax.set_xticklabels([SETTING_SHORT[s] for s in present], fontsize=6.5)
        ax.set_title(FAMILY_SHORT[family], fontsize=7.5)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('cosine with the gradient\nat the query point')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.25))
    style.save(fig, 'truth')


def moves():
    """how far each explanation moves when the black box is corrected, at 200:20"""
    frame = read('explanations_moves')
    frame = frame[(frame.setting == 'imbalanced') & (frame.kind == 'test')
                  & (frame.neighbourhood == 'all')]
    frame = frame.assign(top_changed=100*(1 - frame.same_top))
    shown = ['re-thresholded', 'weighted', 'SMOTE']
    panels = [('all', 'top_changed', 'query points whose\ntop feature changes (%)'),
              ('all', 'median_norm_ratio', 'size of the coefficients\n(ratio to the plain model)'),
              ('smooth', 'top_changed', 'top feature changes (%),\nmodels with a gradient')]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH*0.32))
    for ax, (family, column, label) in zip(axes, panels):
        names = SIX + (['gradient'] if family == 'smooth' else [])
        width = 0.8/len(names)
        for i, explainer in enumerate(names):
            part = frame[(frame.family == family) & (frame.explainer == explainer)].set_index('arm')
            values = [part[column].get(arm, np.nan) for arm in shown]
            ax.bar(np.arange(len(shown)) + (i - (len(names) - 1)/2)*width, values, width,
                   color=EXPLAINER_COLOURS[explainer], label=EXPLAINER_LABELS[explainer])
        ax.set_xticks(np.arange(len(shown)))
        ax.set_xticklabels(['threshold\nmoved', 'class\nweighted', 'SMOTE'], fontsize=6.5)
        ax.set_title(label, fontsize=7.5)
        ax.grid(axis='x', visible=False)
        if column == 'median_norm_ratio':
            ax.axhline(1, **style.ZERO_LINE)
            ax.set_ylim(bottom=0.8)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.25))
    fig.subplots_adjust(wspace=0.3)
    style.save(fig, 'moves')


def losses():
    """CLIME minus LIME in Brier score, kernel weighted and class balanced, on each evaluation set"""
    frame = read('explanations_paired')
    frame = frame[(frame.family == 'all') & (frame.kind == 'test') & (frame.explainer == 'CLIME')]
    present = settings_present(frame)
    panels = [('local', "on fresh samples around $q$"), ('test', 'on the test set')]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH*0.8, TEXTWIDTH*0.3), sharey=True)
    x = np.arange(len(present))
    for ax, (where, label) in zip(axes, panels):
        for i, (suffix, name, colour) in enumerate((('', 'kernel weighted', EXPLAINER_COLOURS['LIME']),
                                                    ('_balanced', 'kernel weighted, class balanced',
                                                     EXPLAINER_COLOURS['CLIME']))):
            part = frame[frame.metric == f'{where}_brier{suffix}'].set_index('setting').reindex(present)
            ax.bar(x + (i - 0.5)*0.38, part['diff'], 0.36, yerr=part['ci'], color=colour, label=name,
                   error_kw=dict(lw=0.8, ecolor=INK))
        ax.axhline(0, **style.ZERO_LINE)
        ax.set_xticks(x)
        ax.set_xticklabels([SETTING_SHORT[s] for s in present], fontsize=6.5)
        ax.set_title(label, fontsize=7.5)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Brier score, CLIME $-$ LIME\n(below zero: CLIME better)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.28))
    style.save(fig, 'losses')


def instruments():
    """rank correlation of each score with the cosine to the gradient, over query points"""
    frame = read('explanations_instruments')
    frame = frame[(frame.family == 'smooth') & (frame.kind == 'test') & (frame.explainer == 'all')
                  & (frame['where'] == 'test')]
    present = settings_present(frame)
    order = ['fidelity', 'balanced', 'auc', 'brier', 'brier_balanced', 'kl', 'kl_balanced']
    names = ['fidelity', 'balanced fidelity', 'AUC', 'Brier', 'Brier, balanced', 'KL', 'KL, balanced']
    fig, ax = plt.subplots(figsize=(TEXTWIDTH*0.7, TEXTWIDTH*0.3))
    width = 0.8/len(present)
    for i, setting in enumerate(present):
        part = frame[frame.setting == setting].set_index('score').reindex(order)
        ax.bar(np.arange(len(order)) + (i - (len(present) - 1)/2)*width, part.rho, width,
               color=[INK2, roles.C['blue'], roles.C['orange']][i], label=SETTING_LABELS[setting])
    ax.axhline(0, **style.ZERO_LINE)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(names, fontsize=6.5, rotation=20, ha='right')
    ax.set_ylabel("rank correlation with\nthe cosine to the gradient")
    ax.grid(axis='x', visible=False)
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.6))
    style.save(fig, 'instruments')


def samples():
    """cosine with a logistic regression's gradient against the size of the neighbourhood sample"""
    frame, per_dataset = read('logistic_samples'), read('logistic_samples_datasets')
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH*0.9, TEXTWIDTH*0.34),
                             gridspec_kw=dict(wspace=0.42))
    for explainer in SIX:
        part = frame[frame.explainer == explainer].sort_values('samples')
        axes[0].plot(part.samples, part['mean'], color=EXPLAINER_COLOURS[explainer],
                     marker=EXPLAINER_MARKERS[explainer], ms=3, label=EXPLAINER_LABELS[explainer])
    axes[0].axvline(10_000, color=MUTED, lw=0.8, ls=':')
    axes[0].set(xscale='log', ylim=(0.85, 1.003), ylabel='cosine with the gradient')
    axes[0].set_title('mean over datasets', fontsize=7.5)
    lime = per_dataset[per_dataset.explainer == 'LIME']
    wide = lime.features > lime.features.median()
    floor = 1e-5   # the precision of the results file
    for mask, colour, label in ((~wide, INK2, 'fewer features'),
                                (wide, EXPLAINER_COLOURS['LIME'], 'more features')):
        for i, (_, line) in enumerate(lime[mask].groupby('dataset')):
            line = line.sort_values('samples')
            axes[1].plot(line.samples, np.maximum(1 - line['mean'], floor), color=colour, lw=0.8,
                         alpha=0.8, label=label if i == 0 else None)
    axes[1].set(xscale='log', yscale='log', ylabel='1 $-$ cosine')
    axes[1].set_title('LIME, one line per dataset', fontsize=7.5)
    axes[1].legend(fontsize=6.5, loc='lower left')
    for ax in axes:
        ax.set_xlabel('samples in the neighbourhood')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.38))
    style.save(fig, 'samples')


IMAGE_RATIOS = {1: '4000:4000', 10: '5000:500', 100: '5000:50'}


def images_truth():
    """cosine of each surrogate's patch coefficients with the two truths, ResNet-20 as trained"""
    frame = read('images_triad')
    frame = frame[(frame.network == 'scratch') & (frame.arm == 'plain')]
    ratios = [r for r in IMAGE_RATIOS if r in set(frame.ratio)]
    panels = [('truth_leave_one_out_cosine', 'leave-one-out effect of each patch'),
              ('truth_shapley_cosine', 'Shapley value of each patch')]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH*0.85, TEXTWIDTH*0.3), sharey=True)
    width = 0.8/len(SIX)
    for ax, (column, label) in zip(axes, panels):
        for i, explainer in enumerate(SIX):
            part = frame[frame.explainer == explainer].set_index('ratio')
            ax.bar(np.arange(len(ratios)) + (i - (len(SIX) - 1)/2)*width,
                   [part[column].get(r, np.nan) for r in ratios], width,
                   color=EXPLAINER_COLOURS[explainer], label=EXPLAINER_LABELS[explainer])
        ax.set_xticks(np.arange(len(ratios)))
        ax.set_xticklabels([IMAGE_RATIOS[r] for r in ratios], fontsize=6.5)
        ax.set_title(label, fontsize=7.5)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('cosine with the truth')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.3))
    style.save(fig, 'images_truth')


def images_samples():
    """cosine between a fit on n masks and the surrogate's exact fit on all 65,536"""
    frame = read('images_samples')
    frame = frame[(frame.network == 'scratch') & (frame.arm == 'plain')]
    ratios = [r for r in IMAGE_RATIOS if r in set(frame.ratio)]
    fig, axes = plt.subplots(1, len(ratios), figsize=(TEXTWIDTH, TEXTWIDTH*0.3), sharey=True,
                             squeeze=False)
    for ax, ratio in zip(axes[0], ratios):
        for explainer in SIX:
            part = frame[(frame.ratio == ratio) & (frame.explainer == explainer)].sort_values('samples')
            ax.plot(part.samples, part.cosine, color=EXPLAINER_COLOURS[explainer],
                    marker=EXPLAINER_MARKERS[explainer], ms=3, label=EXPLAINER_LABELS[explainer])
        ax.axvline(1000, color=MUTED, lw=0.8, ls=':')
        ax.set(xscale='log', xlabel='masks sampled')
        ax.set_title(IMAGE_RATIOS[ratio], fontsize=7.5)
    axes[0, 0].set_ylabel('cosine with the\nsurrogate\'s exact fit')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.38))
    style.save(fig, 'images_samples')


def images_moves():
    """how far each patch explanation, and each truth, moves when the network is corrected, at 5000:50"""
    frame = read('images_moves')
    frame = frame[(frame.network == 'scratch') & (frame.ratio == 100) & (frame.neighbourhood == 'all')
                  & frame.fit.isin(['sample', 'truth'])]
    frame = frame.assign(top_changed=100*(1 - frame.same_top))
    arms = {'prior shift': 'threshold\nmove', 'reseeded': 'other\nseed',
            'weighted': 'class\nweights', 'logit adjusted': 'logit\nadjusted'}
    shown = [a for a in arms if a in set(frame.arm)]
    truths = {'leave one out': (INK2, 'leave-one-out effects'), 'Shapley': (MUTED, 'Shapley values')}
    panels = [('top_changed', 'query images whose\ntop patch changes (%)'),
              ('median_norm_ratio', 'size of the coefficients\n(ratio to the plain network)'),
              ('median_cosine', 'cosine with the\nplain network\'s')]
    names = [*SIX, *truths]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH*0.34))
    width = 0.8/len(names)
    for ax, (column, label) in zip(axes, panels):
        for i, explainer in enumerate(names):
            part = frame[frame.explainer == explainer].set_index('arm')
            colour, name = truths.get(explainer, (EXPLAINER_COLOURS.get(explainer),
                                                  EXPLAINER_LABELS.get(explainer)))
            heights = np.array([part[column].get(arm, np.nan) for arm in shown])
            base = 1.0 if column == 'median_norm_ratio' else 0.0
            ax.bar(np.arange(len(shown)) + (i - (len(names) - 1)/2)*width, heights - base, width,
                   bottom=base, color=colour, label=name)
        ax.set_xticks(np.arange(len(shown)))
        ax.set_xticklabels([arms[a] for a in shown], fontsize=6)
        ax.set_title(label, fontsize=7.5)
        ax.grid(axis='x', visible=False)
        if column == 'median_norm_ratio':
            ax.axhline(1, **style.ZERO_LINE)
            ax.set_yscale('log')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.25))
    fig.subplots_adjust(wspace=0.3)
    style.save(fig, 'images_moves')


def images_example(dataset='airplane', seed=0, ratio=100):
    """one image's patch explanations before and after the prior shift"""
    from common import patches
    from explain_images import DEFAULT, draws
    stored = np.load(paths.raw('images')/f'scratch_{ratio}_{dataset}_{seed}_plain.npz')
    pictures = np.load(paths.raw('images_pictures')/f'{ratio}_{dataset}_{seed}.npz')
    M = patches.masks()
    w = patches.kernel(M)
    shift = np.log(ratio)

    def fitted(name, z, query):
        rows = draws(seed, query)[0][:DEFAULT]
        return patches.fit(name, M[rows], z[rows], w[rows])[0]

    found = {}
    for i in np.flatnonzero(stored['y_query'] == 1):
        z, query = stored['masks'][i].astype(np.float64), stored['query'][i]
        found[i] = {'Shapley values\nof the logit': patches.shapley(z),
                    'Logit-LIME,\nbefore and after': fitted('Logit-LIME', z, query),
                    'LIME,\nas trained': fitted('LIME', z, query),
                    'LIME, threshold\nmoved': fitted('LIME', z + shift, query)}
    growth = {i: np.linalg.norm(v['LIME, threshold\nmoved'])/np.linalg.norm(v['LIME,\nas trained'])
              for i, v in found.items()}
    chosen = sorted(growth, key=growth.get)[len(growth)//2]   # the median growth
    image = pictures['images'][chosen].transpose(1, 2, 0)
    side = image.shape[0]//4
    hidden = M[draws(seed, stored['query'][chosen])[0][1]].reshape(4, 4).astype(bool)
    masked = image.copy()
    for r, c in zip(*np.nonzero(~hidden)):
        block = masked[r*side:(r + 1)*side, c*side:(c + 1)*side]
        block[:] = block.reshape(-1, 3).mean(axis=0).round()
    fig, axes = plt.subplots(1, 6, figsize=(TEXTWIDTH, TEXTWIDTH*0.22))
    for ax, picture, title in zip(axes, (image, masked), ('query image', 'one mask')):
        ax.imshow(picture, interpolation='nearest')
        ax.set_title(title, fontsize=7)
        for edge in np.arange(1, 4)*side - 0.5:
            ax.axhline(edge, color='white', lw=0.4)
            ax.axvline(edge, color='white', lw=0.4)
    for ax, (title, coef) in zip(axes[2:], found[chosen].items()):
        limit = np.abs(coef).max()
        ax.imshow(coef.reshape(4, 4), cmap=roles.DIVERGING_CMAP, vmin=-limit, vmax=limit)
        ax.set_title(title, fontsize=7)
        ax.set_xlabel(f'largest {limit:.2g}', fontsize=6.5)
    for ax in axes:
        ax.set(xticks=[], yticks=[])
        ax.grid(False)
    style.save(fig, 'images_example')


FIGURES = {f.__name__: f for f in (reproduce, replication, profile, triad, one_sided, arms,
                                   families, grid, operating_point, by_class, sweep, kernel,
                                   matched, cuts, truth, moves, losses, instruments,
                                   samples, images_truth, images_samples, images_moves,
                                   images_example)}

if __name__ == '__main__':
    style.use()
    for name in sys.argv[1:] or FIGURES:
        try:
            FIGURES[name]()
        except FileNotFoundError as e:
            print(f'skipped {name}: {e.filename} not there yet')
