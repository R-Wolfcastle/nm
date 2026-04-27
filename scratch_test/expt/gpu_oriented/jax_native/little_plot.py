import matplotlib.pyplot as plt
import numpy as np



def plot_with_diagonal_grid(x, y, title="X vs Y with Diagonal Grid", xlabel="x", ylabel="y"): 
    plt.figure(figsize=(8, 5)) 
    ax = plt.gca() 
 
    # Plot the data 
    plt.plot(x, y, marker='o', linestyle='-', color='royalblue', label='Data') 
 
    # Get actual data limits (including padding) 
    xmin, xmax = np.min(x), np.max(x) 
    ymin, ymax = np.min(y), np.max(y) 
 
    # Expand limits a bit for aesthetics 
    pad_x = (xmax - xmin) * 0.05 
    pad_y = (ymax - ymin) * 0.05 
    xmin -= pad_x 
    xmax += pad_x 
    ymin -= pad_y 
    ymax += pad_y 
 
    ax.set_xlim(xmin, xmax) 
    ax.set_ylim(ymin, ymax) 
 
    # Compute diagonal gridline offsets (c in y = x + c) that intersect visible region 
    c_min = ymin - xmax 
    c_max = ymax - xmin 
    offsets = np.arange(np.floor(c_min), np.ceil(c_max) + 1) 
 
    for c in offsets: 
        # y = x + c 
        xs = np.array([xmin, xmax]) 
        ys = xs + c 
        ax.plot(xs, ys, color='lightgray', linestyle='--', linewidth=1, zorder=0) 
 
        # y = 2x + c 
        xs = np.array([xmin, xmax]) 
        ys = 2*(xs + c) 
        ax.plot(xs, ys, color='lightgray', linestyle=':', linewidth=1, zorder=0) 
 
 
    # Standard grid, labels, etc. 
    #plt.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6) 
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel) 
    plt.title(title) 
    #plt.legend() 
    plt.tight_layout() 
    plt.show() 



dofs_log2 = np.array([ 12, 14, 16, 18, 20])
times_log2 = np.log2(np.array([5.939, 6.527, 9.998, 20.818, 87.003]))

#dofs_log2 = np.array([ 6.,  8., 10., 12., 14., 16., 18., 19., 20., 21., 22., 23., 24.])
#times_log2 = np.array([1.52356196, 1.36120689, 1.39999135, 1.41792001, 1.49005662,
#              1.67671874, 2.12465899, 2.60715274, 3.26828467, 3.98713893,
#              5.23817542, 6.20045727, 7.50049934])

plot_with_diagonal_grid(dofs_log2, times_log2, title="", xlabel="log2(n_dofs)", ylabel="log2(time)")
