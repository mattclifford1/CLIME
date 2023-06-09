import matplotlib.pyplot as plt
import clime
import os
from tqdm import tqdm

file_path = os.path.dirname(os.path.abspath(__file__))

datasets = clime.data.AVAILABLE_DATASETS
datasets.pop('credit scoring 1')
datasets.pop('credit scoring 2')
datasets.pop('direct marketing')

all_normal = []
all_CB = []

# datasets = ['moons', 'breast cancer']

for dataset in tqdm(datasets):

    params_normal = {'data params': {'class_samples': (200, 200), 'percent of data': 0.11, 'moons_noise': 0.2, 'gaussian_means': [[0, 0], [1, 1]], 'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [
        0, 1]]]}, 'dataset': dataset, 'dataset rebalancing': 'none', 'model': 'Random Forest', 'model balancer': 'none', 'explainer': 'bLIMEy (normal)', 'evaluation metric': 'fidelity (local)', 'evaluation run': 'between_class_means'}
    params_class_bal = {'data params': {'class_samples': (200, 200), 'percent of data': 0.11, 'moons_noise': 0.2, 'gaussian_means': [[0, 0], [1, 1]], 'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [
        0, 1]]]}, 'dataset': dataset, 'dataset rebalancing': 'none', 'model': 'Random Forest', 'model balancer': 'none', 'explainer': 'bLIMEy (cost sensitive sampled)', 'evaluation metric': 'fidelity (local)', 'evaluation run': 'between_class_means'}

    result_normal = clime.pipeline.run_pipeline(
        params_normal, parallel_eval=True)
    result_bal = clime.pipeline.run_pipeline(
        params_class_bal, parallel_eval=True)

    ax = plt.gca()

    scores = {'Class Balanced Sampling': result_bal['score'], 'Normal Sampling': result_normal['score']}
    all_normal.append(result_normal['score']['scores'])
    all_CB.append(result_bal['score']['scores'])


    clime.utils.plots.plot_line_graphs_on_one_graph(
        scores, 
        ylabel='Local Fidelity (Accuracy)', 
        ax=ax,
        query_values=False,
        # ylims=[0.5, 1]
        )

    ax.set_title(f'Dataset: {dataset}')
    plt.savefig(os.path.join(file_path, 'figs', 'sampling',
                f'CB-{dataset}.png'), bbox_inches="tight")
    # plt.show()
    plt.clf()

    # save clf decision surface
    plt_data = {
        0: {'data': result_normal['train_data'], 
            'model': result_normal['clf'],
            'query_points': result_normal['score']['eval_points']}}
    clime.utils.plots.plot_clfs(plt_data, 
                                title=False, 
                                ax_x=1)
    plt.title(f'Random Forest on {dataset}')
    plt.savefig(os.path.join(file_path, 'figs', 'sampling',
                f'RF-{dataset}.png'), bbox_inches="tight")
    # plt.show()
    plt.clf()

all_scores = {
    'Class Balanced Sampling': all_CB, 'Normal Sampling': all_normal}


ax = plt.gca()
clime.utils.plots.plot_mean_std_graphs(all_scores, 
                                       ax=ax,
                                       ylabel='Local Fidelity (Accuracy)',
                                    #    ylims=[0.5, 1]
                                       )
ax.set_title('Mean/Std over all datasets')
plt.savefig(os.path.join(file_path, 'figs', 'sampling', f'mean-std-over-all-datasets.png'), bbox_inches="tight")
# plt.show()