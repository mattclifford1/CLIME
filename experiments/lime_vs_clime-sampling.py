import matplotlib.pyplot as plt
import clime
import os
from tqdm import tqdm

file_path = os.path.dirname(os.path.abspath(__file__))

datasets = clime.data.AVAILABLE_DATASETS
datasets.pop('Credit Scoring 1')
datasets.pop('Credit Scoring 2')
datasets.pop('Direct Marketing')
datasets.pop('Gaussian')
datasets.pop('Moons')
datasets.pop('Abalone Gender')
datasets.pop('Iris')
datasets.pop('Wine')
datasets.pop('Sonar Rocks vs Mines')
datasets.pop('Ionosphere')
datasets.pop('Circles')
datasets.pop('Blobs')
datasets.pop('Wheat Seeds')

all_normal = []
all_CB = []

model = 'Random Forest'

# datasets = ['moons', 'breast cancer']

for dataset in tqdm(datasets):

    params_normal = {'data params': {'class_samples': (200, 200), 'percent of data': 0.11, 'moons_noise': 0.2, 'gaussian_means': [[-1, -1], [1, 1]], 'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [
        0, 1]]]}, 'dataset': dataset, 'dataset rebalancing': 'none', 'model': model, 'model balancer': 'none', 'explainer': 'bLIMEy (normal)', 'evaluation metric': 'fidelity (local)', 'evaluation run': 'between_class_means'}
    params_class_bal = {'data params': {'class_samples': (200, 200), 'percent of data': 0.11, 'moons_noise': 0.2, 'gaussian_means': [[-1, -1], [1, 1]], 'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [
        0, 1]]]}, 'dataset': dataset, 'dataset rebalancing': 'none', 'model': model, 'model balancer': 'none', 'explainer': 'bLIMEy (cost sensitive sampled)', 'evaluation metric': 'fidelity (local)', 'evaluation run': 'between_class_means'}

    result_normal = clime.pipeline.run_pipeline(
        params_normal, parallel_eval=True)
    result_bal = clime.pipeline.run_pipeline(
        params_class_bal, parallel_eval=True)

    fig_single, ax_single = plt.subplots(1, 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(
        18, 9), gridspec_kw={'height_ratios': [1, 1.7]},
        linewidth=4, edgecolor="black")

    scores = {'Class Balanced Weights: $w_{xc}$':
              result_bal['score'], 'Standard Weights: $w_x$': result_normal['score']}
    # scores = {'Standard Weights: $w_x$': result_normal['score'], 'Class Balanced Weights: $w_{xc}$': result_bal['score']}

    all_normal.append(result_normal['score']['scores'])
    all_CB.append(result_bal['score']['scores'])

    for ax in [ax_single, ax1]:
        clime.utils.plots.plot_line_graphs_on_one_graph(
            scores, 
            ylabel='Local Fidelity', 
            ax=ax,
            query_values=False,
            ylims=[0.4, 1]
            )

        # ax.set_title(f'Dataset: {dataset}')
    # fig_single.savefig(os.path.join(file_path, 'figs', 'sampling',
    #             f'CB-{dataset}.png'), bbox_inches="tight")
    # plt.show()
    fig_single.clf()

    for ax, figure in zip([ax_single, ax2], [fig_single, fig]):
        # save clf decision surface
        plt_data = {
            0: {'data': result_normal['test_data'], 
                'model': result_normal['clf'],
                'query_points': result_normal['score']['eval_points']}}
        clime.utils.plots.plot_clfs(plt_data, 
                                    title=False, 
                                    axs=[ax],
                                    fig=figure)
        # ax.set_title(f'Random Forest trained on {dataset}')
    # fig_single.savefig(os.path.join(file_path, 'figs', 'sampling',
    #             f'RF-{dataset}.png'), bbox_inches="tight")
    # plt.show()
    fig_single.clf()

    fig.savefig(os.path.join(file_path, 'figs', 'sampling', f'combined-{dataset}.png'), bbox_inches="tight")

all_scores = {
    'Class Balanced Sampling': all_CB, 'Normal Sampling': all_normal}


# ax = plt.gca()
# clime.utils.plots.plot_mean_std_graphs(all_scores, 
#                                        ax=ax,
#                                        ylabel='Local Fidelity (Accuracy)',
#                                     #    ylims=[0.5, 1]
#                                        )
# ax.set_title('Mean/Std over all datasets')
# plt.savefig(os.path.join(file_path, 'figs', 'sampling', f'mean-std-over-all-datasets.png'), bbox_inches="tight")
# plt.show()


# plot all normal on one graph
# fig, ax = plt.subplots(1, 1)
# x = [i for i in range(all_normal[0].shape[0])]
# for i in all_normal:
#     ax.plot(x, i)
# plt.show()

