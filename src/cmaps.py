from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

def get_terrain_cmap():
    c0 = '#2351b7'
    c1 = '#1175db'
    c2 = '#00acc4'
    c3 = '#0fce68'
    c4 = '#fefd98'
    c5 = 'peru'
    c6 = 'sienna'
    c7 = 'maroon'
    c8 = 'snow'

    colors = [
        (0.0, c0),
        (0.05, c1),
        (0.1, c2),
        (0.22, c3),
        (0.45, c4),
        (0.65, c5),
        (0.8, c6),
        (1.0,  c7),
    ]

    cmap = LinearSegmentedColormap.from_list('white_terrain', colors=colors, N=256)
    return cmap
    
def get_wave_cmap(center_color='white'):
    """Create a custom colormap with smooth transitions between the given colors.

    `center_color` sets the near-zero (middle plateau) color: 'white' by default; pass e.g.
    'lightgrey' for a Crameri-vik-like grayish centre that reads better as a 3D slice over a white
    figure background (while the flat 2D curtain/colorbar keeps the exact white)."""
    c0 = 'darkslateblue'
    c1 = 'royalblue'
    c2 = 'cornflowerblue'
    #c2 = 'lightblue'
    c3 = 'lavender'
    c4 = center_color # 'white'/'whitesmoke'/'lightgrey'
    c5 = 'palegoldenrod'
    c55 = '#EEE600'
    c6 = 'goldenrod'
    c7 = 'indianred' # firebrick
    c8 = 'darkred'

    # colors = [(0.0, c0), (0.1, c1), (0.2, c2), (0.4, c3), (0.48, c4), (0.52, c4), (0.55, c5), (0.65, c55), (0.8, c6), (0.9, c7), (1.0, c8)]
    colors = [(0.0, c0), (0.1, c1), (0.2, c2), (0.4, c3), (0.48, c4), (0.52, c4), (0.6, c5), (0.75, c6), (0.85, c7), (1.0, c8)]
    # colors = [(0.0, c0), (0.32, c0), (0.4, c1), (0.6, c1), (0.68, c2), (0.73, c2), (0.78, c3), (1.0, c3)]
    
    cmap = LinearSegmentedColormap.from_list('wave', colors=colors, N=256)
    return cmap

def get_spectral_white_cmap():
    """Return a modified 'Spectral' colormap with white replacing the yellow center."""
    # Sample colors from the original 'Spectral' colormap
    base = plt.get_cmap('Spectral')
    
    # Extract colors and manually replace midrange with white
    n = 256
    colors = base(np.linspace(0, 1, n))
    
    # Define a "white region" around the center (original is bright yellow there)
    mid_low, mid_high = int(n * 0.46), int(n * 0.54)
    for i in range(mid_low, mid_high):
        # Blend softly into white instead of abrupt cutoff
        blend = (i - mid_low) / (mid_high - mid_low)
        colors[i, :3] = [1, 1, 1]  # pure white RGB
        colors[i, 3] = 1.0         # full alpha
    
    # Build new colormap
    cmap = LinearSegmentedColormap.from_list('spectral_white', colors, N=n)
    return cmap
    
def get_vik_white_cmap(white_width=0.14, plateau=0.04, n=256):
    """Crameri's 'vik' with a truly white center.

    A pure-white plateau of half-width `plateau`/2 sits at the middle; outside it the
    colors blend back into the original vik with a smoothstep over `white_width` from
    the center (matching the white-plateau style of get_wave_cmap)."""
    from cmcrameri import cm as crameri_cm

    x = np.linspace(0, 1, n)
    colors = crameri_cm.vik(x)

    distance = np.abs(x - 0.5)
    t = np.clip((distance - plateau / 2) / (white_width - plateau / 2), 0.0, 1.0)
    white_weight = 1.0 - t * t * (3.0 - 2.0 * t)  # smoothstep: 1 at center -> 0 at white_width
    colors[:, :3] = colors[:, :3] * (1.0 - white_weight[:, None]) + white_weight[:, None]

    return LinearSegmentedColormap.from_list('vik_white', colors, N=n)


def get_greenpurple_cmap():
    """Diverging colormap: blue → green → white → yellow/red, keeping positive side identical to get_wave_cmap()."""
    
    # Negative side
    c0 = 'darkslategrey'
    c1 = 'teal'
    c2 = 'seagreen'
    c3 = 'mediumseagreen'
    
    # Center
    c4 = 'white'

    c5 = 'mistyrose'
    c6 = 'peachpuff'
    c7 = 'plum' # firebrick
    c8 = 'rebeccapurple'

    colors = [(0.0, c0),
              (0.15, c2),
              (0.3, c3),
              (0.48, c4),
              (0.52, c4),
              (0.6, c5),
              # (0.7, c6),
              (0.8, c7),
              (1.0, c8)]
    
    cmap = LinearSegmentedColormap.from_list('wave_bluegreen', colors=colors, N=256)
    return cmap
    

def get_coolwarm_soft_cmap():
    """Elegant soft diverging colormap from turquoise to rose with white center."""
    c0 = '#006d77'   # deep teal
    c1 = '#83c5be'   # light teal
    c2 = '#edf6f9'   # pale cyan
    c3 = 'white'
    c4 = '#ffe5ec'   # pale pink
    c5 = '#f4a7b9'   # muted rose
    c6 = '#9b2226'   # deep red-brown

    colors = [
        (0.0, c0), (0.2, c1), (0.35, c2),
        (0.48, c3), (0.52, c3),
        (0.65, c4), (0.8, c5), (1.0, c6)
    ]
    return LinearSegmentedColormap.from_list('coolwarm_soft', colors=colors, N=256)

def get_purplegold_cmap():
    """Diverging colormap with purples and golds centered on white."""
    c0 = '#3b0a45'   # deep violet
    c1 = '#7b3294'   # purple
    c2 = '#c2a5cf'   # lilac
    c3 = 'white'
    c4 = '#fddbc7'   # soft beige
    c5 = '#f4a582'   # salmon-gold
    c6 = '#b2182b'   # dark red

    colors = [
        (0.0, c0), (0.2, c1), (0.35, c2),
        (0.48, c3), (0.52, c3),
        (0.65, c4), (0.8, c5), (1.0, c6)
    ]
    return LinearSegmentedColormap.from_list('purplegold', colors=colors, N=256)