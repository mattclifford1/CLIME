'''shared matplotlib style for the paper figures'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# slots 1-3 of the reference categorical palette (light mode), used unmodified.
# documented as passing the all-pairs gates: CVD dE 9.2, normal-vision dE 24.0
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK_2, MUTED = '#0b0b0b', '#52514e', '#8a8985'

plt.rcParams.update({
    'figure.dpi': 160, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
    'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.labelcolor': INK, 'text.color': INK,
    'legend.frameon': False, 'lines.linewidth': 1.6,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
