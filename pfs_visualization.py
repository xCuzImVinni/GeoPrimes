
import plotly.io as pio
pio.renderers.default = "browser"
import numpy as np
import plotly.graph_objects as go
from sympy import factorint, primerange

# === SETTINGS ===
nums = list(range(2, 100))  # Zahlen die dargestellt werden

# === PRIMZAHLEN UND WINKEL ===
primes = list(primerange(1, 100))  # z. B. [1, 3, 5, 7, 11, 13, 17, 19, 23, 29]
num_axes = len(primes)
angle_map = {p: 2 * np.pi * i / num_axes for i, p in enumerate(primes)}

# === FUNKTION: Koordinaten berechnen aus PFS
def pfs_coords_fixed(n):
    factors = factorint(n)
    coords = []
    for i, p in enumerate(primes):
        exp = factors.get(p, 0)
        if exp > 0:
            theta = angle_map[p]
            x = exp * np.cos(theta)
            y = exp * np.sin(theta)
            coords.append((x, y))
    return coords

# === PLOT FIGURE ===
fig = go.Figure()

# === Für jede Zahl zeichnen ===
for z_offset, n in enumerate(nums):
    coords = pfs_coords_fixed(n)
    if len(coords) < 2:
        continue  # Nicht darstellbar

    x, y = zip(*coords)
    z = [z_offset] * len(x)

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        line=dict(width=3),
        marker=dict(size=4),
        name=f'n={n}'
    ))

# === PRIMACHEN-ACHSEN ALS RAY-DASHLINES ===
for p, theta in angle_map.items():
    x = [0, np.cos(theta) * 4]
    y = [0, np.sin(theta) * 4]
    z = [0, 0]
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+text',
        line=dict(color='gray', dash='dot'),
        text=[None, str(p)],
        textposition="top center",
        showlegend=False
    ))

# === LAYOUT ===
fig.update_layout(
    title='PFS-Geometrie mit festen Primachsen',
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Zahl (Index)'),
        aspectmode='data'
    ),
    width=900,
    height=700
)

fig.show()
