import plotly.io as pio
pio.renderers.default = "browser"
import numpy as np
import plotly.graph_objects as go
from sympy import factorint, primerange
from scipy.spatial import distance
from sklearn.manifold import MDS

# === KONFIGURATION ===
nums = list(range(2, 50))  # Zu visualisierende Zahlen
max_prime = 50  # Maximale Primzahl für Achsen
axis_length = 8  # Länge der Primachsen
pfs_metric = True  # Spezielle PFS-Metrik aktivieren
color_mode = 'omega'  # 'omega', 'bigomega' oder 'value'

# === PRIMZAHLEN UND WINKEL ===
primes = list(primerange(1, max_prime))
num_axes = len(primes)
angle_step = 2 * np.pi / num_axes
angle_map = {p: i * angle_step for i, p in enumerate(primes)}

# === PFS-METRIK TRANSFORMATION ===
def transform_to_pfs_metric(coords):
    """Transformiert Koordinaten in PFS-adäquate Metrik"""
    transformed = []
    for x, y in coords:
        # Radiale Skalierung basierend auf Primzahlgewichtung
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # Nicht-lineare Skalierung für bessere Trennung
        r_transformed = np.log1p(r) * 1.5

        # Winkelkompression für symmetrischere Darstellung
        theta_transformed = theta * (1 + 0.1 * np.sin(num_axes * theta/2))

        transformed.append((
            r_transformed * np.cos(theta_transformed),
            r_transformed * np.sin(theta_transformed)
        ))
    return transformed

# === GEOMETRISCHE ANALYSE ===
def calculate_geometric_features(coords):
    """Berechnet geometrische Merkmale der PFS-Darstellung"""
    if len(coords) < 2:
        return 0, 0, 0

    # Flächeninhalt des Polygons
    x, y = zip(*coords)
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Symmetrie-Index
    centroid = np.mean(coords, axis=0)
    dists = [distance.euclidean(p, centroid) for p in coords]
    symmetry = np.std(dists)

    # Komplexität (Verhältnis Umfang/Fläche)
    perimeter = sum(distance.euclidean(coords[i], coords[(i+1)%len(coords)])
                  for i in range(len(coords)))
    complexity = perimeter / (area + 1e-9)

    return area, symmetry, complexity

# === KOORDINATENBERECHNUNG ===
def calculate_pfs_coords(n):
    """Berechnet PFS-Koordinaten für eine Zahl"""
    factors = factorint(n)
    coords = []
    prime_factors = []

    # Sortiere Primfaktoren nach ihrer Position im Winkelkreis
    for p, exp in factors.items():
        if p in angle_map:
            prime_factors.append((angle_map[p], p, exp))

    # Sortiere nach Winkel für konsistente Polygonerzeugung
    prime_factors.sort(key=lambda x: x[0])

    for theta, p, exp in prime_factors:
        # Exponenten-skalierte Position
        r = exp
        coords.append((r * np.cos(theta), r * np.sin(theta)))

    # Transformiere Koordinaten bei aktivierter PFS-Metrik
    if pfs_metric and len(coords) > 1:
        coords = transform_to_pfs_metric(coords)

    return coords, factors

# === VISUALISIERUNG ===
fig = go.Figure()

# Farbberechnung basierend auf ausgewähltem Modus
def get_color(n, factors):
    if color_mode == 'omega':
        return len(factors)  # Anzahl verschiedener Primfaktoren
    elif color_mode == 'bigomega':
        return sum(factors.values())  # Gesamtzahl der Primfaktoren
    else:
        return n  # Numerischer Wert

# Sammle geometrische Eigenschaften für Analyse
geometric_data = []

for z_offset, n in enumerate(nums):
    coords, factors = calculate_pfs_coords(n)
    if not coords:
        continue

    # Berechne geometrische Eigenschaften
    area, symmetry, complexity = calculate_geometric_features(coords)
    geometric_data.append((n, area, symmetry, complexity))

    # Erzeuge geschlossenes Polygon
    x, y = zip(*coords)
    x = list(x) + [x[0]]  # Polygon schließen
    y = list(y) + [y[0]]
    z = [z_offset] * len(x)

    # Farbe basierend auf gewähltem Modus
    color_val = get_color(n, factors)

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        line=dict(width=4, color=color_val),
        marker=dict(size=6, color=color_val),
        name=f'n={n}',
        hoverinfo='name+text',
        hovertext=f'Faktoren: {factors}<br>Area: {area:.2f} Sym: {symmetry:.2f}',
        showlegend=False
    ))

    # Projektionslinien zur Basisebene
    for i, (x_val, y_val) in enumerate(coords):
        fig.add_trace(go.Scatter3d(
            x=[x_val, x_val],
            y=[y_val, y_val],
            z=[z_offset, 0],
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='none'
        ))

# Primachsen zeichnen
for p, theta in angle_map.items():
    x_end = axis_length * np.cos(theta)
    y_end = axis_length * np.sin(theta)

    fig.add_trace(go.Scatter3d(
        x=[0, x_end],
        y=[0, y_end],
        z=[0, 0],
        mode='lines+text',
        line=dict(color='black', width=2),
        text=["", f"P{p}"],
        textposition="top center",
        hoverinfo='none',
        showlegend=False
    ))

# Layout mit verbessertem 3D-Blickwinkel
fig.update_layout(
    title=f'PFS-Zahlensystem Visualisierung | Metrik: {pfs_metric} | Farbmodus: {color_mode}',
    scene=dict(
        xaxis_title='X (Realteil)',
        yaxis_title='Y (Imaginärteil)',
        zaxis_title='Zahlenindex',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.8),
            up=dict(x=0, y=0, z=1)
        ),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.5)
    ),
    width=1200,
    height=800,
    hovermode='closest'
)

# === ANALYSE-TEIL ===
print("\nGeometrische Analyse der PFS-Darstellungen:")
print("Zahl\tFläche\tSymmetrie\tKomplexität")
for n, area, symmetry, complexity in geometric_data:
    print(f"{n}\t{area:.2f}\t{symmetry:.2f}\t\t{complexity:.2f}")

fig.show()

# === DIMENSIONALITÄTSREDUKTION FÜR PFS-VEKTOREN ===
print("\nErweiterte Analyse mittels multidimensionaler Skalierung:")

# Erstelle PFS-Vektoren für alle Zahlen
pfs_vectors = []
for n in nums:
    factors = factorint(n)
    vec = [factors.get(p, 0) for p in primes]
    pfs_vectors.append(vec)

# MDS zur Dimensionsreduktion
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
distance_matrix = distance.squareform(distance.pdist(pfs_vectors, 'euclidean'))
embedding = mds.fit_transform(distance_matrix)

# Plotte die MDS-Projektion
fig_mds = go.Figure()
fig_mds.add_trace(go.Scatter(
    x=embedding[:, 0],
    y=embedding[:, 1],
    mode='markers+text',
    text=[str(n) for n in nums],
    marker=dict(size=10, color=nums, colorscale='Rainbow'),
    textposition='top center'
))

fig_mds.update_layout(
    title='MDS-Projektion der PFS-Vektoren',
    xaxis_title='MDS Dimension 1',
    yaxis_title='MDS Dimension 2'
)
fig_mds.show()
