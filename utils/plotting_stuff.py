from pathlib import Path

from PIL import Image, ImageOps
import imageio
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def show_vel_field(u, v, spacing=1, cmap='Spectral_r', vmin=None, vmax=None, showcbar=True, savepath=None, show=True, title=None):
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



def plotboth(thk, b, speed, title=None, savepath=None, axis_limits=None, show_plots=True):
    s_gnd = b + thk
    s_flt = thk*(1-rho/rho_w)
    s = jnp.maximum(s_gnd, s_flt)

    base = s-thk


    fig, ax1 = plt.subplots(figsize=(10,5))
    ax2 = ax1.twinx()

    ax1.plot(s, label="surface")
    # ax1.plot(base, label="base")
    ax1.plot(base, label="base")
    ax1.plot(b, label="bed")

    ax2.plot(speed*3.15e7, color='k', marker=".", linewidth=0, label="speed")

    #legend
    ax1.legend(loc='lower left')
    #slightly lower
    ax2.legend(loc='center left')
    #stop legends overlapping

    #axis labels
    ax1.set_xlabel("x")
    ax1.set_ylabel("elevation")
    ax2.set_ylabel("speed")

    if axis_limits is not None:
        ax1.set_ylim(axis_limits[0])
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
