import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

cmap = cm.get_cmap('binary')

figsize = (6.75, 3.375)
p1_list = [0.2, 0.4, 0.6, 0.8, 1.0]
filename_suffix_list = ['_50_%s.csv' % p1 for p1 in p1_list]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
for i,p1 in enumerate(p1_list):
    t_bin_edges, t_hist = np.loadtxt('lifetime'+filename_suffix_list[i], delimiter=',')
    s_bin_edges, s_hist = np.loadtxt('num_topples'+filename_suffix_list[i], delimiter=',')
    c = i/2/len(p1_list)+.4
    ax2.plot(t_bin_edges, t_hist, label='p = %s' % p1, linestyle='-', marker='s', color=cmap(c), markerfacecolor=cmap(c-0.2))
    ax1.plot(s_bin_edges, s_hist, label='p = %s' % p1, linestyle='-', marker='s', color=cmap(c), markerfacecolor=cmap(c-0.2))

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Avalanche lifetime')
ax2.set_ylabel('Number of avalanches')
ax2.legend()

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Avalanche topples')
ax1.set_ylabel('Number of avalanches')
ax1.legend()

plt.tight_layout()
plt.savefig('avalanches_p.pdf')
plt.show()

