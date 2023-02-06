import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

os.makedirs('images', exist_ok=True)

def process_array(a):
    print(a.shape)
    n = a.shape[0]
    m = max([len(a[i]) for i in range(n)])
    b = np.zeros((n, m), dtype=type(a[0][0]))
    b[:, :] = np.nan
    for i in range(n):
        t = len(a[i])
        q = a[i]
        if len(np.array(q).shape) > 1:
            q = np.mean(q, axis=1)
        b[i, :t] = np.array(q)
    print('new', b.shape)
    return b.astype(np.longdouble)

# parser = argparse.ArgumentParser()
# parser.add_argument('dataset', choices=['gaussians', 'mnist', 'frogs'])
# args = parser.parse_args()
#
# dataset = args.dataset

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
# parser.add_argument('dataset', choices=['gaussian', 'gaussians', 'mnist', 'mnist1', 'frogs', 'mf', 'sphere'])
parser.add_argument('--reduce', choices=['none', 'pca', 'isomap', 'hog', 'resize', 'pick'], default='none')
parser.add_argument('--kind', choices=['r', 'e', 'g'], default='r')  # g = gaussian1/1
parser.add_argument('--dim', type=int, default=None)
parser.add_argument('--yperc', type=float, default=10)
parser.add_argument('--logscale', action='store_true')
parser.add_argument('--no-logscale', dest='logscale', action='store_false')
parser.set_defaults(logscale=False)
args = parser.parse_args()

kind0 = args.kind
dataset = args.dataset
embedding = args.reduce
dim = args.dim
y_percentile = args.yperc
logx = args.logscale

# dataset = 'gaussians'  # gaussians | mnist | frogs
# embedding = 'none'  # none | pca | hog | isomap | resize
# kind0 = 'g'

# dim = 10

if embedding in ['pca', 'pick', 'isomap'] and dim is None:  # in case we forgot to set
    dim = 10
if embedding == 'resize' and dim is None:  # in case we forgot to set
    dim = 100
if embedding == 'none':
    dim = None

show = True
# y_percentile = 20
legend = False
bvde_comparison = False
# logx = True

dir = 'results'

estimators = ['contvde', 'cvde', 'kde', 'awkde', 'orig_kde', 'orig_awkde']
labels = ['BVDE', 'CVDE', 'KDE', 'AdaKDE', 'KDE (original)', 'AdaKDE (original)']
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:green', 'tab:blue']

kinds = [kind0]
kind_style = ['-']
# kinds = ['r', 'e', 'g']
# kind_style = ['-', '--', '-.']

alpha = 0.2

all_values = []

max_band_list = None

for ki, kind, ls in zip(range(len(kinds)), kinds, kind_style):
    for est, label, col in zip(estimators, labels, colors):
        fname = f'{dir}/{est}_{kind}_{dataset}_{embedding}_{dim}.npz'
        if not os.path.exists(fname):
            print(f'Missing {fname}')
            continue
        npz = np.load(fname, allow_pickle=True)
        band_list = npz['band']
        data = npz['scores']
        if logx:
            band_list = np.log(band_list)
        max_band_list = band_list
        data = process_array(data)
        data[np.any(np.isinf(data), axis=0, keepdims=True).repeat(data.shape[0], axis=0)] = -10000000000
        print(f'{fname} dim={npz["dim"]} runs={npz["runs"]}')

        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)

        band_list = band_list[:mean.shape[0]]
        if dataset in ['mnist', 'frogs'] and est == 'awkde' and kind0 == 'r' \
                or dataset in ['mnist'] and est == 'kde' and kind0 == 'g' and embedding != 'pca':
            lscur = '--'
            dashes = (5, 5)
            plt.plot(band_list, mean, lscur, dashes=dashes, label=label, c=col, linewidth=2, zorder=2)
        else:
            plt.plot(band_list, mean, ls, label=label, c=col, linewidth=2, zorder=2 if est != 'cvde' else 10)
        plt.fill_between(band_list, mean - std, mean + std, alpha=alpha, color=col, zorder=1)

        all_values.append(mean)

loaded_list = False
try:
    for est, label, col in zip(['contvde', 'cvde', 'kde', 'awkde', 'orig_kde', 'orig_awkde'], labels, colors):
        if est not in ['contvde']:
            continue
        alpha_fname = f'{dir}/alphas_{est}_{kind0}_{dataset}_{embedding}_{dim}.npz'
        alpha_npz = np.load(alpha_fname)
        bw_m = alpha_npz['suggested_bw_m']
        bw_l = alpha_npz['suggested_bw_l']
        bw_r = alpha_npz['suggested_bw_r']
        if not bvde_comparison:
            plt.plot([bw_m, bw_m], [-10000, 100000], ':', c=col, alpha=0.7)
            plt.fill_between([bw_l, bw_r], [-10000, -10000], [10000, 10000], alpha=0.1, color=col, zorder=2)

        alpha_list = alpha_npz['alpha_list']
        data_dim = alpha_npz['data_dim']

        loaded_list = True
except:
    pass

# if suggested_bw is not None:
#     plt.plot([suggested_bw, suggested_bw], [-10000, 100000], '--', c='gray')

all_values = np.concatenate(all_values)
ymin, ymax = {
    'gaussian': (-19.2, -14.55),
    'gaussians': (-13.81, -4.07),
    'frogs': (29.35, 34.12),
    'mnist': (90.23, 111.05)
}[dataset]

ymin = np.nanpercentile(all_values, y_percentile)
ymax = np.nanmax(all_values) * 1.1 - ymin * 0.1

print(f'y bounds: {ymin} {ymax}')

ax = plt.gca()
# ax.set_ylabel('Hellinger')
ax.set_ylabel('avg. log-likelihood')
ax.set_xlabel('bandwidth')
plt.xlim(max_band_list[0], max_band_list[-1])
plt.ylim(ymin, ymax)
# plt.ylim(-600, 600)

try:
    if bvde_comparison or not loaded_list:
        ax2 = ax
    else:
        ax2 = ax.twiny()
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xbound(ax.get_xbound())
        [t.set_color('darkred') for t in ax2.xaxis.get_ticklabels()]
        ax2.xaxis.label.set_color('darkred')
    labels = [alpha_list[np.argmin(np.abs(np.array(band_list) - x))] for x in ax.get_xticks()]
    labels = [a**(1/data_dim) for a in labels]
    labels = [f'{f:.3f}' for f in labels]
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('$\\alpha^{\\frac{1}{n}}$')
except:
    pass

lines, names = ax.get_legend_handles_labels()
# lines, names = reversed(lines), reversed(names)
if legend:
    plt.legend(lines, names, frameon=False)

# plt.xlim()

plt.tight_layout()
plt.savefig(f'images/vskde_{dataset}_{embedding}_{dim}_{kind0}.png')
if show:
    plt.show()

fig = plt.gcf()
size = fig.get_size_inches()
# size[0] = 1

plt.clf()

if not legend:
    plt.figure(figsize=(size[0], size[1] * .1))
    plt.axis(False)
    plt.legend((lines), (names), loc="center", frameon=False, ncol=len(lines))
    plt.tight_layout()
    plt.savefig("images/vskde_labels.png")
