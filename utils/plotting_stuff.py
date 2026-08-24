#1st party
from pathlib import Path
import sys
import os

#3rd party
from PIL import Image, ImageOps
import imageio
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#local apps
nm_home = os.environ['NM_HOME']   

sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c


rho = 900
rho_w = 1000

g = 9.8


def create_gif_from_png_fps(png_paths, output_path, duration=200, loop=0):
    frames = [Image.open(p) for p in png_paths]
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=loop)


def create_high_quality_gif_from_pngfps(png_paths, output_path, duration=200, loop=0):
    frames = [Image.open(p).convert("RGBA") for p in png_paths]
    frames = [f.quantize(colors=256, method=Image.MEDIANCUT) for f in frames]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=True,
        disposal=2
    )

def create_imageio_gif(png_paths, output_path):
    images = []
    for filename in png_paths:
        images.append(imageio.imread(filename))
    imageio.mimsave(output_path, images)



def create_gif_global_palette(png_paths, output_path, duration=200, loop=0,
                                  dither=True, background=(255, 255, 255)):
    # Load RGBA frames
    rgba = [Image.open(p).convert("RGBA") for p in png_paths]
    w, h = rgba[0].size

    # Composite onto a solid background -> RGB
    bg = Image.new("RGB", (w, h), background)
    rgb_frames = [Image.alpha_composite(bg.copy(), f).convert("RGB") for f in rgba]

    # Derive ONE global palette from a mosaic of downscaled frames
    scale = 4
    thumbs = [fr.resize((max(1, w//scale), max(1, h//scale)), Image.BILINEAR) for fr in rgb_frames]
    cols = max(1, int(len(thumbs) ** 0.5))
    rows = (len(thumbs) + cols - 1) // cols
    mosaic = Image.new("RGB", (cols * thumbs[0].width, rows * thumbs[0].height))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        mosaic.paste(t, (c * t.width, r * t.height))

    # Global adaptive palette (256 colors)
    palette_img = mosaic.convert("P", palette=Image.ADAPTIVE, colors=256)

    # Quantize each RGB frame to the SAME palette
    dither_mode = Image.FLOYDSTEINBERG if dither else Image.NONE
    qframes = [fr.quantize(palette=palette_img, dither=dither_mode) for fr in rgb_frames]

    # Save GIF
    qframes[0].save(
        output_path,
        save_all=True,
        append_images=qframes[1:],
        duration=duration,
        loop=loop,
        optimize=True,
        disposal=2,
    )


def create_webp_from_pngs(png_paths, output_path, duration=200, loop=0, quality=95):
    frames = [Image.open(p).convert("RGBA") for p in png_paths]
    frames[0].save(
        output_path,              # e.g. "out.webp"
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        quality=quality,          # 80–95 typically looks great
        method=6,                 # slowest/best compression
        lossless=False            # set True for lossless (larger files)
    )



def make_gif(arrays, filename="animation.gif", interval=200, cmap="viridis", vmin=None, vmax=None):
    images = []
    for arr in arrays:
        arr_np = np.array(arr)
        fig, ax = plt.subplots()
        im = ax.imshow(arr_np, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)

        # Draw the figure so the renderer is ready
        fig.canvas.draw()

        # Get RGBA buffer (portable across backends)
        buf = np.asarray(fig.canvas.buffer_rgba())  # shape (H, W, 4)
        images.append(buf[..., :3])  # drop alpha channel

        plt.close(fig)

    # Save gif
    imageio.mimsave(filename, images, duration=interval/1000.0)
    print(f"Saved gif to {filename}")


def show_vel_field(u, v, spacing=1, cmap='RdYlBu_r', vmin=None, vmax=None, showcbar=True, savepath=None, show=True, title=None):
    """
    Displays the magnitude of a 2D vector field and overlays flow direction lines.

    Parameters:
        u (2D array): x-component of the vector field.
        v (2D array): y-component of the vector field.
        spacing (int): step size for streamlines (larger means fewer lines).
        cmap (str): colormap to use for magnitude.
    """
    assert u.shape == v.shape, "u and v must have the same shape"
   
    u = jnp.flipud(u)
    v = jnp.flipud(v)

    magnitude = np.sqrt(u**2 + v**2)
    ny, nx = u.shape

    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(8, 6))
    plt.imshow(magnitude, origin='lower', cmap=cmap, extent=(0, nx, 0, ny), vmin=vmin, vmax=vmax)
    if showcbar:
        plt.colorbar(label='Speed (m/yr)')

    plt.streamplot(
        X, Y, u, v,
        color='k',
        density= 1/(spacing),
        linewidth=0.25,
        arrowstyle='-'
    )

    plt.tight_layout()

    plt.title(title)

    if savepath is not None:
        plt.savefig(savepath, dpi=100)

    if show:
        plt.show()


def show_vel_field_2(
    u,
    v,
    spacing=1,
    cmap='RdYlBu_r',
    vmin=None,
    vmax=None,
    showcbar=True,
    savepath=None,
    show=True,
    title=None,
):
    """
    Display the magnitude of a 2D velocity field with streamlines.

    Parameters
    ----------
    u : 2D array
        x-component of the velocity field.
    v : 2D array
        y-component of the velocity field.
    spacing : float
        Controls streamline density. Larger values give fewer lines.
    cmap : str
        Matplotlib colormap.
    """
    assert u.shape == v.shape, "u and v must have the same shape"

    u = jnp.flipud(u)
    v = jnp.flipud(v)

    magnitude = np.sqrt(u**2 + v**2)

    ny, nx = u.shape

    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        magnitude,
        origin='lower',
        cmap=cmap,
        extent=(0, nx, 0, ny),
        vmin=vmin,
        vmax=vmax,
    )

    if showcbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, label="Speed (m/yr)")

    ax.streamplot(
        X,
        Y,
        u,
        v,
        color='k',
        density=1 / spacing,
        linewidth=0.25,
        arrowstyle='-'
    )

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=100, bbox_inches='tight')

    if show:
        plt.show()

    return fig, ax



def show_damage_field(d, spacing=1, cmap='cubehelix_r', vmin=0, vmax=1, showcbar=True, savepath=None, show=True, title=None):
    d = jnp.flipud(d)

    ny, nx = d.shape

    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(8, 6))
    plt.imshow(d, origin='lower', cmap=cmap, extent=(0, nx, 0, ny), vmin=vmin, vmax=vmax)
    if showcbar:
        plt.colorbar(label='Damage')

    plt.tight_layout()

    plt.title(title)

    if savepath is not None:
        plt.savefig(savepath, dpi=100)

    if show:
        plt.show()



def show_vel_with_quiver(u, v, step=5, line_length=50, scale=50, cmap='RdYlBu_r'):
    """
    Displays the magnitude of a 2D vector field with short directional arrows at regular intervals.

    Parameters:
        u (2D array): x-component of the vector field.
        v (2D array): y-component of the vector field.
        step (int): spacing between arrows (larger means fewer arrows).
        scale (float): scaling factor for arrow length (larger means shorter arrows).
        cmap (str): colormap for the magnitude background.
    """
    assert u.shape == v.shape, "u and v must have the same shape"

    magnitude = np.sqrt(u**2 + v**2)
    ny, nx = u.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    # Downsample for quiver to avoid clutter
    Xq = X[::step, ::step]
    Yq = Y[::step, ::step]
    Uq = u[::step, ::step]
    Vq = v[::step, ::step]

    # Normalize vectors to unit direction
    norms = np.sqrt(Uq**2 + Vq**2) + 1e-9
    Udir = Uq / norms
    Vdir = Vq / norms

    # Compute segment endpoints
    dx = 0.5 * line_length * Udir
    dy = 0.5 * line_length * Vdir

    x_start = Xq - dx
    y_start = Yq - dy
    x_end = Xq + dx
    y_end = Yq + dy

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(magnitude, origin='lower', cmap=cmap, extent=(0, nx, 0, ny))
    plt.colorbar(label='Speed (m/yr)')

    # Plot quiver arrows
    plt.quiver(Xq, Yq, Uq, Vq, color='k', scale=scale, pivot='middle', headwidth=3)

    plt.tight_layout()
    plt.show()


def plotgeom(thk, b):

    s_gnd = b + thk
    s_flt = thk*(1-rho/rho_w)
    s = jnp.maximum(s_gnd, s_flt)

    base = s-thk

    #plot b, s and base on lhs y axis, and C on rhs y axis
    fig, ax1 = plt.subplots(figsize=(10,5))

    ax1.plot(s, label="surface")
    # ax1.plot(base, label="base")
    ax1.plot(base, label="base")
    ax1.plot(b, label="bed")

    #legend
    ax1.legend(loc='upper right')

    #axis labels
    ax1.set_xlabel("x")
    ax1.set_ylabel("elevation")

    plt.show()



def plotboth(x, thk, b, speed, title=None, savepath=None, axis_limits=None, show_plots=False):
    s_gnd = b + thk
    s_flt = thk*(1-c.RHO_O/c.RHO_W)
    s = jnp.maximum(s_gnd, s_flt)

    base = s-thk

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    ax1.plot(x / 1000, b, color="saddlebrown", label="bed")
    ax1.plot(x / 1000, jnp.where(thk>0, base, jnp.nan), color="teal", label="base")
    ax1.plot(x / 1000, jnp.where(thk>0, surface, jnp.nan), color="steelblue", label="surface")
    ax1.fill_between(x / 1000, base, surface,
                     where=(thk > 0),
                     color="lightblue", alpha=0.5)
    ax1.set_ylabel("elevation (m)")
    ax1.legend(fontsize=8)
    #ax1.set_title(f"Transect profile, {tag} ({resolution} m)")

    ax2.plot(x / 1000, jnp.where(speed>1e-10, speed, jnp.nan), color='k', marker=".", linewidth=0, label="speed")
    
    if axis_limits is not None:
        if axis_limits[0] is not None:
            ax1.set_ylim(axis_limits[0])
        if axis_limits[1] is not None:
            ax2.set_ylim(axis_limits[1])

    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.savefig(savepath)

    if show_plots:
        plt.show()


def plotboths(thks, b, speeds, upper_lim, title=None, savepath=None, axis_limits=None, show_plots=True):

    if isinstance(speeds, (jnp.ndarray, np.ndarray)):
        speeds = [np.array(u) for u in speeds]
    if isinstance(thks, (jnp.ndarray, np.ndarray)):
        thks = [np.array(h) for h in thks]

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()

    n = len(thks)

    cmap = cm.rainbow
    cs = cmap(jnp.linspace(0, 1, n))

    for thk, speed, c1 in list(zip(thks, speeds, cs)):
        s_gnd = b + thk
        s_flt = thk*(1-rho/rho_w)
        s = jnp.maximum(s_gnd, s_flt)

        base = s-thk
        ax1.plot(s, c=c1)
        # ax1.plot(base, label="base")
        ax1.plot(base, c=c1)

        ax2.plot(speed*3.15e7, color=c1, marker=".", linewidth=0)

    #add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=upper_lim))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.15)
    cbar.set_label('Timestep')

    ax1.plot(b, label="bed", c="k")
    
    ##legend
    ##ax1.legend(loc='lower left')
    ##slightly lower
    #ax2.legend(loc='center left')
    ##stop legends overlapping

    #axis labels
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("elevation (m)")
    ax2.set_ylabel("speed (m/yr)")

    if axis_limits is not None:
        ax1.set_ylim(axis_limits[0])
        ax2.set_ylim(axis_limits[1])

    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.savefig(savepath)

    if show_plots:
        plt.show()



def plotgeoms(thks, b, upper_lim, title=None, savepath=None, axis_limits=None, show_plots=True):

    if isinstance(thks, (jnp.ndarray, np.ndarray)):
        thks = [np.array(h) for h in thks]

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()

    n = len(thks)

    cmap = cm.rainbow
    cs = cmap(jnp.linspace(0, 1, n))

    for thk,  c1 in list(zip(thks, cs)):
        s_gnd = b + thk
        s_flt = thk*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)

        base = s-thk
        ax1.plot(s, c=c1)
        # ax1.plot(base, label="base")
        ax1.plot(base, c=c1)

    #add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=upper_lim))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.15)
    cbar.set_label('Timestep')

    ax1.plot(b, label="bed", c="k")
    
    #axis labels
    ax1.set_xlabel("x")
    ax1.set_ylabel("elevation")

    if axis_limits is not None:
        ax1.set_ylim(axis_limits[0])

    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.savefig(savepath)

    if show_plots:
        plt.show()


#Thanks MS copilot
def extract_grounding_line(thk, b, x, y):
    """Extract grounding-line points from a 2-D thickness field, by linear
    interpolation of the flotation criterion f = h - h_f between grid
    columns straddling each row's grounded/floating transition(s),
    following the point-data convention of Sect. 2.3 of Asay-Davis et al.
    (2016) (one or more xGL/yGL points per row that actually has a
    grounding line; rows with no transition contribute none).

    thk, b : (ny, nx) arrays (works directly on a loaded thickness .npy
              together with the module-level `b`)
    x, y   : 1-D coordinate vectors of length nx, ny

    Returns (xGL, yGL) as 1-D numpy arrays.
    """
    thk = np.asarray(thk)
    b = np.asarray(b)
    x = np.asarray(x)
    y = np.asarray(y)

    h_f = np.maximum(0.0, -(c.RHO_W / c.RHO_I) * b)
    f = thk - h_f          # > 0 grounded, < 0 floating
    ice = thk > 0

    xGL, yGL = [], []
    for j in range(thk.shape[0]):
        row_f, row_ice = f[j, :], ice[j, :]
        for i in range(len(x) - 1):
            if not (row_ice[i] and row_ice[i + 1]):
                continue
            if row_f[i] == 0.0:
                xGL.append(x[i]); yGL.append(y[j])
                continue
            if (row_f[i] > 0) != (row_f[i + 1] > 0):
                frac = row_f[i] / (row_f[i] - row_f[i + 1])
                xGL.append(x[i] + frac * (x[i + 1] - x[i]))
                yGL.append(y[j])

    return np.array(xGL), np.array(yGL)


#def show_field_scaled(field, x, y, ax=None,
#                      cmap="viridis", vmin=None, vmax=None,
#                      cbar_label=None, title=None, y_exaggeration=4.0,
#                      xlabel="x (km)", ylabel="y (km)", figsize=(10, 4),
#                      reflect=True):
#    
#    x = np.asarray(x)
#    y = np.asarray(y)
#    field = np.asarray(field)
#
#    own_fig = ax is None
#    if own_fig:
#        fig, ax = plt.subplots(figsize=figsize)
#
#    extent = [x[0] / 1e3, x[-1] / 1e3, y[0] / 1e3, y[-1] / 1e3]
#    im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap,
#                    vmin=vmin, vmax=vmax, aspect=y_exaggeration)
#    ax.set_xlabel(xlabel)
#    ax.set_ylabel(ylabel)
#    if title:
#        ax.set_title(title)
#    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
#    if cbar_label:
#        cbar.set_label(cbar_label)
#    if own_fig:
#        plt.tight_layout()
#
#    return ax, im
#
#def make_plot_mismip_field_function(b, x, y, reflect=True):
#    def plot_mismip_field(field, thk, ax=None,
#                          cmap="viridis", vmin=None, vmax=None,
#                          cbar_label=None, title=None, y_exaggeration=3.0,
#                          xlabel="x (km)", ylabel="y (km)", figsize=(10, 4),
#                          gl_color="k", filepath=None):
#    
#        ax, im = show_field_scaled(field[::-1,:], x, y, ax=ax, cmap=cmap,
#                                   vmin=vmin, vmax=vmax, 
#                                   cbar_label=cbar_label,
#                                   title=title, y_exaggeration=y_exaggeration,
#                                   xlabel=xlabel, ylabel=ylabel,
#                                   figsize=figsize, reflect=reflect)
#    
#        xGL, yGL = extract_grounding_line(thk, b, x, y)
#    
#        if len(xGL):
#            order = np.argsort(yGL)
#            ax.plot(
#                np.asarray(xGL)[order] / 1e3,
#                np.asarray(yGL)[order][::-1] / 1e3,
#                gl_color + "--",
#                lw=1.5,
#            )
#        
#        if filepath is not None:
#            fig = ax.figure
#            fig.savefig(filepath, bbox_inches="tight", dpi=300)
#            if ax is None:
#                plt.close(fig)
#        
#        return ax, im
#    return plot_mismip_field


def _reflect_domain(field, y):
    """Mirror a half-domain field/y array across the top boundary (y[-1]),
    extending the y-extent (the y[-1] row itself is not duplicated)."""
    field = np.asarray(field)
    y = np.asarray(y)
    field_full = np.concatenate([field, field[-2::-1, :]], axis=0)
    y_full = np.concatenate([y, 2 * y[-1] - y[-2::-1]])
    return field_full, y_full


def show_field_scaled(field, x, y, ax=None,
                      cmap="viridis", vmin=None, vmax=None,
                      cbar_label=None, title=None, y_exaggeration=4.0,
                      xlabel="x (km)", ylabel="y (km)", figsize=(10, 4),
                      reflect=True):

    x = np.asarray(x)
    y = np.asarray(y)
    field = np.asarray(field)

    if reflect:
        field, y = _reflect_domain(field, y)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)

    extent = [x[0] / 1e3, x[-1] / 1e3, y[0] / 1e3, y[-1] / 1e3]
    im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap,
                    vmin=vmin, vmax=vmax, aspect=y_exaggeration)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    if cbar_label:
        cbar.set_label(cbar_label)
    if own_fig:
        plt.tight_layout()

    return ax, im

def make_plot_mismip_field_function(b, x, y, reflect=True, y_exaggeration=2.0):
    def plot_mismip_field(field, thk, ax=None,
                          cmap="viridis", vmin=None, vmax=None,
                          cbar_label=None, title=None,
                          xlabel="x (km)", ylabel="y (km)", figsize=(10, 4),
                          gl_color="k", filepath=None):

        ax, im = show_field_scaled(field[::-1,:], x, y, ax=ax, cmap=cmap,
                                   vmin=vmin, vmax=vmax,
                                   cbar_label=cbar_label,
                                   title=title, y_exaggeration=y_exaggeration,
                                   xlabel=xlabel, ylabel=ylabel,
                                   figsize=figsize, reflect=reflect)

        xGL, yGL = extract_grounding_line(thk, b, x, y)

        if len(xGL):
            xGL = np.asarray(xGL)
            yGL = np.asarray(yGL)[::-1]

            if reflect:
                # mirror the GL points across the top boundary too
                xGL = np.concatenate([xGL, xGL])
                yGL = np.concatenate([yGL, 2 * y[-1] - yGL])

            order = np.argsort(yGL)
            ax.plot(
                xGL[order] / 1e3,
                yGL[order] / 1e3,
                gl_color + "--",
                lw=1.5,
            )

        if filepath is not None:
            fig = ax.figure
            fig.savefig(filepath, bbox_inches="tight", dpi=300)
            if ax is None:
                plt.close(fig)

        return ax, im
    return plot_mismip_field
