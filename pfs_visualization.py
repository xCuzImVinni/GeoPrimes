import plotly.io as pio
pio.renderers.default = "browser"
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sympy import factorint, isprime
from scipy.spatial import distance
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import pandas as pd
import math

# === KONFIGURATION ===
max_prime = 1000  # Maximale Primzahl für Achsen
nums = [p for p in range(2, max_prime) if isprime(p) and isprime(2 * p + 1)]
axis_length = 15  # Länge der Primachsen
pfs_metric = True  # Spezielle PFS-Metrik aktivieren
color_mode = 'complexity'  # 'omega', 'bigomega', 'value', 'symmetry', 'complexity'
analysis_mode = True  # Geometrische Analyse aktivieren
advanced_metrics = True  # Erweiterte Metriken berechnen

# === DYNAMISCHE PRIMZAHLENBESTIMMUNG ===
def get_required_primes(numbers):
    """Bestimmt benötigte Primzahlen für die Zahlenmenge"""
    all_primes = set()
    for n in numbers:
        factors = factorint(n)
        all_primes |= set(factors.keys())
    return sorted(all_primes, reverse=True)  # Absteigende Sortierung

primes = get_required_primes(nums)
num_axes = len(primes)
print(f"Verwendete Primzahlen ({num_axes}): {primes}")

# === SYMMETRISCHE WINKELVERTEILUNG ===
def symmetric_angle_distribution(primes):
    """Komplett symmetrische Winkelverteilung um den Kreismittelpunkt"""
    n = len(primes)
    angles = {}

    # Gleichmäßige Verteilung im Kreis
    for i, p in enumerate(primes):
        angle = 2 * np.pi * i / n
        angles[p] = angle

    return angles

angle_map = symmetric_angle_distribution(primes)

# === ERWEITERTE PFS-METRIK ===
def advanced_pfs_metric(coords, factors):
    """Transformiert Koordinaten mit primzahlabhängiger Metrik"""
    transformed = []
    prime_weights = {p: 1/np.log(p) for p in factors.keys()}  # Gewichtung nach Primzahlgröße

    for i, (x, y) in enumerate(coords):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # Primzahl-spezifische Transformation
        p = list(factors.keys())[i]
        p_weight = prime_weights[p]

        # Nicht-lineare radiale Transformation
        r_transformed = np.power(r, 0.7) * (1 + p_weight)

        # Winkelmodulation für bessere Trennung
        angle_mod = 0.05 * np.sin(3 * theta) * p_weight
        theta_transformed = theta + angle_mod

        transformed.append((
            r_transformed * np.cos(theta_transformed),
            r_transformed * np.sin(theta_transformed)
        ))
    return transformed

# === GEOMETRISCHE ANALYSE MIT ERWEITERTEN FEATURES ===
def extended_geometric_analysis(coords, factors):
    """Berechnet erweiterte geometrische Merkmale"""
    features = {}
    if len(coords) < 1:
        return features

    centroid = np.mean(coords, axis=0) if len(coords) > 0 else (0, 0)

    # Grundmerkmale
    features['num_factors'] = len(factors)
    features['sum_exponents'] = sum(factors.values())

    # Formmerkmale
    if len(coords) > 2:
        # Flächeninhalt
        x, y = zip(*coords)
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        features['area'] = area

        # Umfang
        perimeter = sum(distance.euclidean(coords[i], coords[(i+1)%len(coords)])
                      for i in range(len(coords)))
        features['perimeter'] = perimeter
        features['compactness'] = (4 * np.pi * area) / (perimeter**2 + 1e-9)
    elif len(coords) == 2:
        features['area'] = 0
        features['perimeter'] = distance.euclidean(coords[0], coords[1])
        features['compactness'] = 0
    else:
        features['area'] = 0
        features['perimeter'] = 0
        features['compactness'] = 1

    # Symmetriemaße
    if len(coords) > 0:
        dists = [distance.euclidean(p, centroid) for p in coords]
        features['symmetry_std'] = np.std(dists) if dists else 0
        features['symmetry_max'] = max(dists) - min(dists) if dists and len(dists) > 1 else 0
    else:
        features['symmetry_std'] = 0
        features['symmetry_max'] = 0

    # Schwerpunktdistanz
    features['centroid_distance'] = distance.euclidean(centroid, (0, 0))

    # Winkelverteilung
    if len(coords) > 1:
        angles = [np.arctan2(y, x) for x, y in coords]
        angle_diffs = [abs(angles[i] - angles[(i+1)%len(angles)]) % (2*np.pi)
                     for i in range(len(angles))]
        features['angle_variance'] = np.var(angle_diffs) if angle_diffs else 0
    else:
        features['angle_variance'] = 0

    return features

# === FARBSCHEMA OPTIMIERUNG ===
def get_color(n, factors, coords, features):
    """Dynamische Farbberechnung basierend auf ausgewähltem Modus"""
    if color_mode == 'omega':
        return len(factors)  # Anzahl verschiedener Primfaktoren
    elif color_mode == 'bigomega':
        return sum(factors.values())  # Gesamtzahl der Primfaktoren
    elif color_mode == 'value':
        return n  # Numerischer Wert
    elif color_mode == 'symmetry' and features:
        return features.get('symmetry_std', 0) * 20
    elif color_mode == 'complexity' and features:
        return features.get('compactness', 0) * 100
    else:
        return np.log(n)  # Logarithmischer Wert als Fallback

# === INTERAKTIVE 3D-VISUALISIERUNG ===
def create_pfs_visualization():
    fig = go.Figure()
    geometric_data = []
    max_radius = 0

    # Primfaktor-Vektoren für spätere Analyse
    pfs_vectors = []

    for z_offset, n in enumerate(nums):
        factors_dict = factorint(n)

        # Primfaktoren in absteigender Reihenfolge sortieren
        sorted_primes = sorted(factors_dict.keys(), reverse=True)
        factors = {p: factors_dict[p] for p in sorted_primes}

        primes_in_n = [p for p in primes if p in factors]

        # Koordinaten berechnen
        coords = []
        for p in primes_in_n:
            theta = angle_map[p]
            r = factors[p]  # Korrekt: Exponent direkt verwenden
            coords.append((r * np.cos(theta), r * np.sin(theta)))

        # Metrik-Transformation
        if pfs_metric and coords:
            if advanced_metrics:
                coords = advanced_pfs_metric(coords, factors)

        # Geometrische Analyse
        features = extended_geometric_analysis(coords, factors) if analysis_mode else {}

        # Farbe berechnen
        color_val = get_color(n, factors, coords, features)

        # Daten für Analyse sammeln
        record = {'n': n, 'factors': factors, 'coords': coords, **features}
        geometric_data.append(record)

        # Maximalen Radius aktualisieren
        for x, y in coords:
            radius = np.sqrt(x**2 + y**2)
            max_radius = max(max_radius, radius)

        # Visualisierung vorbereiten
        if coords:
            # Geschlossenes Polygon erstellen (nach Winkel sortieren)
            angles = [np.arctan2(y, x) for x, y in coords]
            sorted_indices = np.argsort(angles)
            sorted_coords = [coords[i] for i in sorted_indices]
            sorted_primes = [primes_in_n[i] for i in sorted_indices]

            x_vals, y_vals = zip(*sorted_coords)
            x_vals = list(x_vals) + [x_vals[0]]  # Polygon schließen
            y_vals = list(y_vals) + [y_vals[0]]
            z_vals = [z_offset] * len(x_vals)

            # Linienbreite basierend auf Faktorenanzahl
            line_width = 3 + len(factors) * 0.5

            fig.add_trace(go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='lines+markers',
                line=dict(width=line_width, color=color_val),
                marker=dict(size=5 + np.log1p(sum(factors.values()))),
                name=f'n={n}',
                hoverinfo='name+text',
                hovertext=f"n={n}<br>Faktoren: {factors}<br>" +
                         (f"Fläche: {features.get('area', 0):.2f}<br>" if analysis_mode else "") +
                         (f"Symmetrie: {features.get('symmetry_std', 0):.3f}<br>" if analysis_mode else "") +
                         (f"Kompaktheit: {features.get('compactness', 0):.3f}" if analysis_mode else ""),
                showlegend=False
            ))

            # Verbesserte Projektionslinien
            for (x, y) in sorted_coords:
                fig.add_trace(go.Scatter3d(
                    x=[x, x],
                    y=[y, y],
                    z=[0, z_offset],
                    mode='lines',
                    line=dict(color=f'rgba(100,100,100,{0.2 + len(factors)*0.05})', width=1),
                    showlegend=False,
                    hoverinfo='none'
                ))

            # Primzahl-Labels für Faktoren
            for i, (x, y) in enumerate(sorted_coords):
                p = sorted_primes[i]
                fig.add_trace(go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z_offset],
                    mode='text',
                    text=f"{p}^{factors[p]}",
                    textposition="middle center",
                    textfont=dict(size=9, color='black'),
                    showlegend=False,
                    hoverinfo='none'
                ))

        pfs_vectors.append([factors.get(p, 0) for p in primes])

    # Primachsen mit optimierter Darstellung
    for p, theta in angle_map.items():
        x_end = min(axis_length, max_radius * 1.2) * np.cos(theta)
        y_end = min(axis_length, max_radius * 1.2) * np.sin(theta)

        fig.add_trace(go.Scatter3d(
            x=[0, x_end],
            y=[0, y_end],
            z=[0, 0],
            mode='lines+text',
            line=dict(color='black', width=1.5),
            text=["", f"P{p}"],
            textposition="top center",
            textfont=dict(size=10),
            hoverinfo='none',
            showlegend=False
        ))

    # Layout-Optimierungen
    fig.update_layout(
        title=f'<b>PFS-Zahlensystem mit symmetrischer Verteilung</b><br>Zahlen: {len(nums)} | Primachsen: {num_axes}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Zahlenindex',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.8),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.2)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4),
            xaxis=dict(range=[-axis_length*1.1, axis_length*1.1]),
            yaxis=dict(range=[-axis_length*1.1, axis_length*1.1])
        ),
        width=1400,
        height=900,
        hovermode='closest',
        margin=dict(l=50, r=50, b=50, t=100)
    )

    # Farbbalken hinzufügen
    fig.update_layout(coloraxis=dict(
        colorscale='Rainbow',
        colorbar=dict(
            title=dict(
                text='Farbkodierung',
                side='right'
            ),
            thickness=20
        )
    ))

    return fig, geometric_data, pfs_vectors

# === HAUPTPROGRAMM ===
fig, geometric_data, pfs_vectors = create_pfs_visualization()
fig.show()
# === ERWEITERTE ANALYSE ===
if analysis_mode and geometric_data:
    # Erstelle DataFrame für Analyse
    df = pd.DataFrame(geometric_data)
    df.set_index('n', inplace=True)

    # Korrelationsmatrix berechnen
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr(method='pearson')

        # Korrelationsmatrix plotten
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            title="Korrelation geometrischer Merkmale",
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1
        )
        fig_corr.update_layout(width=800, height=700)
        fig_corr.show()

    # Top-10 Analyse
    print("\nTop-10 Zahlen nach geometrischen Merkmalen:")
    print("Fläche:")
    print(df.sort_values('area', ascending=False)[['area']].head(10))

    print("\nSymmetrie (niedrigster Wert ist symmetrischer):")
    print(df.sort_values('symmetry_std')[['symmetry_std']].head(10))

    print("\nKompaktheit (höher = kompakter):")
    print(df.sort_values('compactness', ascending=False)[['compactness']].head(10))

    # Clusteranalyse
    if len(df) > 5:
        X = df[['area', 'symmetry_std', 'compactness', 'centroid_distance']].fillna(0)
        kmeans = KMeans(n_clusters=min(5, len(df)), random_state=42)
        df['cluster'] = kmeans.fit_predict(X)

        # 3D-Clusterplot
        fig_cluster = px.scatter_3d(
            df.reset_index(),
            x='area',
            y='symmetry_std',
            z='compactness',
            color='cluster',
            hover_name='n',
            title='Clusteranalyse der geometrischen Merkmale',
            size='centroid_distance',
            symbol='num_factors',
            opacity=0.8
        )
        fig_cluster.update_layout(width=1000, height=700)
        fig_cluster.show()

# === MULTIDIMENSIONALE SKALIERUNG ===
if len(pfs_vectors) > 2:
    print("\nMultidimensionale Skalierung der PFS-Vektoren...")
    try:
        dist_matrix = distance.squareform(distance.pdist(pfs_vectors, 'minkowski', p=0.7))
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
        embedding = mds.fit_transform(dist_matrix)

        if embedding is not None:
            fig_mds = go.Figure()
            fig_mds.add_trace(go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode='markers+text',
                text=[str(n) for n in nums],
                marker=dict(
                    size=12,
                    color=nums,
                    colorscale='Rainbow',
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                textposition='top center',
                textfont=dict(size=9)
            ))

            # Besondere Zahlen hervorheben
            special_numbers = [n for n in nums if isprime(n) or len(factorint(n)) == 1]
            for n in special_numbers:
                idx = nums.index(n)
                fig_mds.add_trace(go.Scatter(
                    x=[embedding[idx, 0]],
                    y=[embedding[idx, 1]],
                    mode='markers',
                    marker=dict(size=15, color='black', symbol='diamond'),
                    name=f'Spezial: {n}',
                    hoverinfo='text',
                    hovertext=f"{n}: {factorint(n)}"
                ))

            fig_mds.update_layout(
                title='MDS-Projektion der PFS-Vektoren (Minkowski p=0.7)',
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                showlegend=True
            )
            fig_mds.show()
    except Exception as e:
        print(f"MDS fehlgeschlagen: {e}")
