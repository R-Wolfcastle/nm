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
    
    
def plot_two_lines_with_diagonal_grid(
    x, y1, y2,
    title="",
    xlabel="log_2(DOFs)",
    ylabel="log_2(time (s))",
    labels=("GPU", "CPU")
):
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    # Plot the data
    plt.plot(x, y1, marker='o', linestyle='-', color='royalblue', label=labels[0])
    plt.plot(x, y2, marker='s', linestyle='-', color='darkorange', label=labels[1])

    # Get limits from both series
    xmin, xmax = np.min(x), np.max(x)
    ymin = min(np.nanmin(y1), np.nanmin(y2))
    ymax = max(np.nanmax(y1), np.nanmax(y2))

    # Padding
    pad_x = (xmax - xmin) * 0.4
    pad_y = (ymax - ymin) * 0.4
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Diagonal gridlines
    c_min = ymin - xmax
    c_max = ymax - xmin
    offsets = np.arange(np.floor(c_min), np.ceil(c_max) + 1)

    for c in offsets:
        xs = np.array([xmin, xmax])

        # y = x + c
        ys1 = xs + c
        ax.plot(xs, ys1, color='lightgray', linestyle='--', linewidth=1, zorder=0)

        # y = 2x + c
        ys2 = 2 * (xs + c)
        ax.plot(xs, ys2, color='lightgray', linestyle=':', linewidth=0.5, zorder=0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


#dofs_log2 = np.array([ 12, 14, 16, 18, 20])
#times_log2 = np.log2(np.array([5.939, 6.527, 9.998, 20.818, 87.003]))

dofs_log2 = np.array([10, 12, 14, 16, 18, 20])

#times_log2_gpu = np.log2(np.array([6.48, 9.31, 10.01, 13.74, 34, 146]))
times_log2_gpu = np.log2(np.array([6.48, 9.31, 10.01, 13.74, 34, 146])/10)
times_log2_cpu = np.log2(np.array([3.31, 12.25, 59.5, 231.5, 1005.4, np.nan]))

#plot_with_diagonal_grid(dofs_log2, times_log2, title="", xlabel="log2(n_dofs)", ylabel="log2(time)")
plot_two_lines_with_diagonal_grid(dofs_log2, times_log2_gpu, times_log2_cpu)#, title="", xlabel="log2(n_dofs)", ylabel="log2(time)")
