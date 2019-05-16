import numpy as np
import matplotlib.pyplot as plt
import random

generate_data = False
save_data = True
load_data = not generate_data
plot_data = True
save_figures = True
figsize = (2.2, 2.2)

L = 20
save_grid = False
pull_grid = True

pull_filename = 'initialized_grid.npy'

if pull_grid is False:
    initial_grid = np.full((L, L), 0)
else:
    initial_grid = np.load(pull_filename, allow_pickle=True)

grid = initial_grid
num_steps = 10000
t = 0
ts = []
num_toppless = []
lifetimes = []
densities = []

Delta = np.zeros((L, L, L, L))
for i in range(L):
    for j in range(L):
        for k in range(L):
            for l in range(L):
                if i == k and j == l:
                    Delta[i][j][k][l] = 4
                if (abs(i-k) == 1 and abs(j-l) == 0) or (abs(i-k) == 0 and abs(j-l) == 1):
                    Delta[i][j][k][l] = -1

def drive(grid):
    ans = np.copy(grid)
    rank = len(grid.shape)
    rand_idx = []
    for i in range(rank):
        rand_idx.append(random.randint(0, grid.shape[i]-1))
    ans[tuple(rand_idx)] = ans[tuple(rand_idx)]+1
    return ans

def update(grid):
    ans = np.copy(grid)
    num_topples = 0
    for i in range(L):
        for j in range(L):
            if grid[i][j] >= Delta[i][j][i][j]:
                num_topples += 1
                for k in range(L):
                    for l in range(L):
                        ans[k][l] -= Delta[i][j][k][l]
    return num_topples, ans

def density(grid):
    ans = 0
    n = 0
    for i in range(L):
        for j in range(L):
            ans += grid[i][j]
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
        while check is False:
            num_topples0, grid = update(grid)
            if num_topples0 == 0:
                check = True 
            else:
                num_topples += num_topples0
                lifetime += 1
        ts.append(t)
        num_toppless.append(num_topples)
        lifetimes.append(lifetime)
        densities.append(density(grid))
        t += 1

    if save_grid is True:
        np.save(pull_filename, grid, allow_pickle=True)

    num_topples_hist, num_topples_bin_edges = np.histogram(num_toppless, bins=max(num_toppless))
    lifetime_hist, lifetime_bin_edges = np.histogram(lifetimes, bins=max(lifetimes))
    num_topples_bin_edges = num_topples_bin_edges[:-1]
    lifetime_bin_edges = lifetime_bin_edges[:-1]

    if save_data is True:
        np.savetxt('density.csv', np.array([np.array(ts), np.array(densities)]), delimiter=',')
        np.savetxt('num_topples.csv', np.array([num_topples_bin_edges, num_topples_hist]), delimiter=',')
        np.savetxt('lifetime.csv', np.array([lifetime_bin_edges, lifetime_hist]), delimiter=',')

if load_data is True:

    ts, densities = np.loadtxt('density.csv', delimiter=',')
    num_topples_bin_edges, num_topples_hist = np.loadtxt('num_topples.csv', delimiter=',')
    lifetime_bin_edges, lifetime_hist = np.loadtxt('lifetime.csv', delimiter=',')

if plot_data is True:

    fig = plt.figure(figsize=figsize)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\rho$')
    plt.plot(ts, densities)
    plt.tight_layout()
    plt.savefig('density.pdf')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.5, 2.2))

    params_dict = {'linestyle':'-', 'marker':'s', 'color':'black', 'markerfacecolor':'white'}
    ax1.plot(num_topples_bin_edges, num_topples_hist, **params_dict)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Avalanche topples')
    ax1.set_ylabel('Number of avalanches')

    params_dict = {'linestyle':'-', 'marker':'s', 'color':'black', 'markerfacecolor':'white'}
    ax2.plot(lifetime_bin_edges, lifetime_hist, **params_dict)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Avalanche lifetime')
    ax2.set_ylabel('Number of avalanches')

    plt.tight_layout()
    plt.savefig('avalanches.pdf')
    plt.show()
