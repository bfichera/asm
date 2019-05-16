import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit

def plot_data(p1, filename_suffix, plots_filename_suffix):

    generate_data = False
    save_data = False
    load_data = not generate_data
    plot_data = True
    save_figures = True
    exp_cutoff = 10**1.3
    get_exponents = False
    figsize = (2.2, 2.2)

    L = 50
    p1 = 0.2
    p2 = 1-p1
    zc = 4
    save_grid = False
    pull_grid = False

    pull_filename = 'initialized_grid_50_%s.npy' % str(p1)
    print('INITIALIZING %s' % pull_filename)

    if pull_grid is False:
        initial_grid = np.full((L, L, 2), 0)
    else:
        initial_grid = np.load(pull_filename, allow_pickle=True)
        print('Pulling grid of shape %s.' % str(initial_grid.shape))

    grid = initial_grid
    num_steps = 3000
    t = 0
    ts = []
    num_toppless = []
    lifetimes = []
    densities = []
    num_unique_sitess = []

    def critical_exponent(hist, bin_edges_cropped, cutoff):
        def f(x, a, b):
            return a*x+b
        last_idx = np.array([i for i in range(len(bin_edges_cropped)) if bin_edges_cropped[i] < cutoff])[-1]
        xdata = np.log10(bin_edges_cropped[0:last_idx])
        ydata = np.log10(hist[0:last_idx])
        (a,b), covar = curve_fit(f, xdata, ydata)
        return a,b

    def drive(grid):
        ans = np.copy(grid)
        rank = len(grid.shape)-1
        rand_idx = []
        for i in range(rank):
            rand_idx.append(random.randint(0, grid.shape[i]-1))
        if random.random() < p1:
            ans[tuple(rand_idx)] = ans[tuple(rand_idx)]+np.array([1,0])
        else:
            ans[tuple(rand_idx)] = ans[tuple(rand_idx)]+np.array([0,1])
            
        return ans

    def height(val):
        return val[0]+2*val[1]

    def update(grid):
        ans = np.copy(grid)
        num_topples = 0
        topple_idxs = []
        sites = []
        for i in range(L):
            for j in range(L):
                if height(grid[i][j]) >= 4:
                    topple_idxs.append([i,j])
        for topple_idx in topple_idxs:
            i,j = topple_idx
            sites.append(topple_idx)
            num_topples += 1
            neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            num_lost = 0
            while num_lost < 4:
                neighbor = neighbors.pop(neighbors.index(random.choice(neighbors)))
                dx, dy = neighbor
                if grid[i][j][1] > 0:
                    ans[i][j] = ans[i][j] - np.array([0, 1])
                    grid[i][j] = grid[i][j] - np.array([0, 1])
                    num_lost += 2
                    if i+dx >= 0 and i+dx < L and j+dy >= 0 and j+dy < L:
                        ans[i+dx][j+dy] = ans[i+dx][j+dy]+np.array([0, 1])

                else:
                    ans[i][j] = ans[i][j] - np.array([1, 0])
                    num_lost += 1
                    if i+dx >= 0 and i+dx < L and j+dy >= 0 and j+dy < L:
                        ans[i+dx][j+dy] = ans[i+dx][j+dy]+np.array([1, 0])
                            
        return num_topples, sites, ans

    def density(grid):
        ans = 0
        n = 0
        for i in range(L):
            for j in range(L):
                ans += height(grid[i][j])
                n += 1
        return ans/n

    if generate_data is True:
        while t < num_steps:
            if t % 100 == 0:
                print(t)
            grid = drive(grid)
            num_topples = 0
            lifetime = 0
            check = False
            all_sites = [[L+1, L+1]]
            while check is False:
                num_topples0, sites, grid = update(grid)
                all_sites += sites
                if num_topples0 == 0:
                    check = True 
                else:
                    num_topples += num_topples0
                    lifetime += 1
            ts.append(t)
            num_unique_sites = len(np.unique(np.array(all_sites), axis=0))-1
            if num_topples > 0:
                num_toppless.append(num_topples)
            if lifetime > 0:
                lifetimes.append(lifetime)
            if num_unique_sites > 0:
                num_unique_sitess.append(num_unique_sites)
            densities.append(density(grid))
            t += 1
    ##         print('all sites:', all_sites) 
    ##         print('s', num_topples)
    ##         print('t', lifetime)
    ##         print('n', num_unique_sites)
    ##         print(np.array([[height(grid[i][j]) for j in range(L)] for i in range(L)]))
    ##         input()

        if save_grid is True:
            np.save(pull_filename, grid, allow_pickle=True)

        num_topples_hist, num_topples_bin_edges = np.histogram(num_toppless, bins=max(num_toppless))
        lifetime_hist, lifetime_bin_edges = np.histogram(lifetimes, bins=int(max(lifetimes)))
        num_unique_sites_hist, num_unique_sites_bin_edges = np.histogram(num_unique_sitess, bins=max(num_unique_sitess))
        num_topples_bin_edges = num_topples_bin_edges[0:-1]
        lifetime_bin_edges = lifetime_bin_edges[0:-1]
        num_unique_sites_bin_edges = num_unique_sites_bin_edges[0:-1]

        if save_data is True:
            np.savetxt('density.csv', np.array([np.array(ts), np.array(densities)]), delimiter=',')
            np.savetxt('num_topples.csv', np.array([num_topples_bin_edges, num_topples_hist]), delimiter=',')
            np.savetxt('lifetime.csv', np.array([lifetime_bin_edges, lifetime_hist]), delimiter=',')
            np.savetxt('num_unique_sites.csv', np.array([num_unique_sites_bin_edges, num_unique_sites_hist]), delimiter=',')

    if load_data is True:

        ts, densities = np.loadtxt('density'+filename_suffix+'.csv', delimiter=',')
        num_topples_bin_edges, num_topples_hist = np.loadtxt('num_topples'+filename_suffix+'.csv', delimiter=',')
        lifetime_bin_edges, lifetime_hist = np.loadtxt('lifetime'+filename_suffix+'.csv', delimiter=',')
        num_unique_sites_bin_edges, num_unique_sites_hist = np.loadtxt('num_unique_sites'+filename_suffix+'.csv', delimiter=',')

    if get_exponents is True:
        t_exponent, bt = critical_exponent(lifetime_hist, lifetime_bin_edges, exp_cutoff)
        s_exponent, bs = critical_exponent(num_topples_hist, num_topples_bin_edges, exp_cutoff)
        n_exponent, bn = critical_exponent(num_unique_sites_hist, num_unique_sites_bin_edges, exp_cutoff)

    fig = plt.figure(figsize=figsize)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\rho$')
    plt.plot(ts, densities)
    plt.tight_layout()
    plt.savefig('density.pdf')
    if plot_data is True:
        plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.75, 2.2))

    params_dict = {'linestyle':'-', 'marker':'s', 'color':'black', 'markerfacecolor':'white'}
    ax1.plot(num_topples_bin_edges, num_topples_hist, **params_dict)
    if get_exponents is True:
        f = lambda x: 10**bs*x**s_exponent
        xdata = np.linspace(1, exp_cutoff, 100)
        ax1.plot(xdata, f(xdata), color='r')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Avalanche topples')
    ax1.set_ylabel('Number of avalanches')

    params_dict = {'linestyle':'-', 'marker':'s', 'color':'black', 'markerfacecolor':'white'}
    ax2.plot(lifetime_bin_edges, lifetime_hist, **params_dict)
    if get_exponents is True:
        f = lambda x: 10**bt*x**t_exponent
        xdata = np.linspace(1, exp_cutoff, 100)
        ax2.plot(xdata, f(xdata))
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Avalanche lifetime')
    ax2.set_ylabel('Number of avalanches')

    params_dict = {'linestyle':'-', 'marker':'s', 'color':'black', 'markerfacecolor':'white'}
    ax3.plot(num_unique_sites_bin_edges, num_unique_sites_hist, **params_dict)
    if get_exponents is True:
        f = lambda x: 10**bn*x**n_exponent
        xdata = np.linspace(1, exp_cutoff, 100)
        ax3.plot(xdata, f(xdata))
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Unique sites')
    ax3.set_ylabel('Number of avalanches')

    plt.tight_layout()
    if save_figures is True:
        plt.savefig('avalanches.pdf')
    if plot_data is True:
        plt.show()

if get_exponents is True:
    t_exponent, bt = critical_exponent(lifetime_hist, lifetime_bin_edges, exp_cutoff)
    s_exponent, bs = critical_exponent(num_topples_hist, num_topples_bin_edges, exp_cutoff)
    n_exponent, bn = critical_exponent(num_unique_sites_hist, num_unique_sites_bin_edges, exp_cutoff)
    
    print('t \\propto', t_exponent)
    print('s \\propto', s_exponent)
    print('n \\propto', n_exponent)
    
