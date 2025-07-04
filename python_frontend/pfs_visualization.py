from typing import TYPE_CHECKING
import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import math
import time
import pandas as pd
from typing import Dict, List
from functools import lru_cache
import sys
import random

# Rust-Backend laden
rust_backend = None
USE_RUST_BACKEND = False
try:
    import rust_backend
    if TYPE_CHECKING:
        from rust_backend import factorize_number, factorize_range  # type: ignore
    USE_RUST_BACKEND = True
    print("Using Rust factorization backend")
except ImportError as e:
    print(f"Import error: {e}")
    print("Rust backend not available, using Python fallback")
    from sympy import factorint

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# ===== OPTIMIZED HELPER FUNCTIONS =====
def primes_below(n: int) -> list:
    """Sieb des Eratosthenes für effiziente Primzahlberechnung"""
    if n < 3:
        return []
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i:n:i] = [False] * len(sieve[i*i:n:i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def is_prime(n: int) -> bool:
    """Optimierte Primzahlprüfung"""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def get_safe_primes(max_n: int) -> list:
    """Sichere Primzahlen: p und (p-1)/2 sind prim"""
    if max_n < 5:
        return []
    primes = primes_below(max_n)
    return [p for p in primes if p > 2 and is_prime((p-1)//2)]

def get_sophie_germain_primes(max_n: int) -> list:
    """Sophie-Germain-Primzahlen: p und 2p+1 sind prim"""
    if max_n < 3:
        return []
    primes = primes_below(max_n)
    return [p for p in primes if is_prime(2*p+1)]

def symmetric_angle_distribution(primes: list) -> dict:
    n = len(primes)
    angles = {}
    for i, p in enumerate(primes):
        angle = 2 * np.pi * i / n
        angles[p] = angle
    return angles

def advanced_pfs_metric(coords: list, factors: dict) -> list:
    transformed = []
    prime_weights = {p: 1/np.log(p) for p in factors.keys()}

    for i, (x, y) in enumerate(coords):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        p = list(factors.keys())[i]
        p_weight = prime_weights[p]
        r_transformed = np.power(r, 0.7) * (1 + p_weight)
        angle_mod = 0.05 * np.sin(3 * theta) * p_weight
        theta_transformed = theta + angle_mod
        transformed.append((
            r_transformed * np.cos(theta_transformed),
            r_transformed * np.sin(theta_transformed)
        ))
    return transformed

def extended_geometric_analysis(coords: list, factors: dict) -> dict:
    features = {}
    if len(coords) < 1:
        return features

    centroid = np.mean(coords, axis=0) if len(coords) > 0 else (0, 0)
    features['num_factors'] = len(factors)
    features['sum_exponents'] = sum(factors.values())

    if len(coords) > 2:
        x, y = zip(*coords)
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        features['area'] = area
        perimeter = sum(np.linalg.norm(np.array(coords[i]) - np.array(coords[(i+1) % len(coords)]))
                      for i in range(len(coords)))
        features['perimeter'] = perimeter
        features['compactness'] = (4 * np.pi * area) / (perimeter**2 + 1e-9)
    elif len(coords) == 2:
        features['area'] = 0
        features['perimeter'] = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))
        features['compactness'] = 0
    else:
        features['area'] = 0
        features['perimeter'] = 0
        features['compactness'] = 1

    if len(coords) > 0:
        dists = [np.linalg.norm(np.array(p) - np.array(centroid)) for p in coords]
        features['symmetry_std'] = np.std(dists) if dists else 0
        features['symmetry_max'] = float(np.max(dists)) - float(np.min(dists)) if dists and len(dists) > 1 else 0
    else:
        features['symmetry_std'] = 0
        features['symmetry_max'] = 0

    features['centroid_distance'] = np.linalg.norm(centroid)

    if len(coords) > 1:
        angles = [np.arctan2(y, x) for x, y in coords]
        angle_diffs = [abs(angles[i] - angles[(i+1)%len(angles)]) % (2*np.pi)
                     for i in range(len(angles))]
        features['angle_variance'] = np.var(angle_diffs) if angle_diffs else 0
    else:
        features['angle_variance'] = 0

    return features

# ===== EFFIZIENTE FAKTORISIERUNG =====
# Precomputed factorizations for small numbers (0-65535)
FACTORIZATION_CACHE = {}

def initialize_factorization_cache():
    """Initialize cache for numbers up to 65535"""
    if not FACTORIZATION_CACHE:
        print("Initializing factorization cache...")
        for n in range(2, 65536):
            FACTORIZATION_CACHE[n] = factorize_number_small(n)
        print("Factorization cache initialized")

def factorize_number_small(n: int) -> dict:
    """Simple factorization for small numbers"""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def factorize_number(n: int) -> dict:
    """Unified factorization with caching and algorithm selection"""
    # Use cache for small numbers
    if n in FACTORIZATION_CACHE:
        return FACTORIZATION_CACHE[n]

    # Use Rust backend if available for larger numbers
    if USE_RUST_BACKEND and rust_backend is not None:
        try:
            return rust_backend.factorize_number(n)  # type: ignore
        except (AttributeError, TypeError):
            # Fallback if the method doesn't exist or has wrong signature
            pass

    # Use sympy as fallback
    try:
        from sympy import factorint
        return factorint(n)
    except ImportError:
        # Final fallback to simple factorization
        return factorize_number_small(n)

def factorize_numbers(nums: list) -> dict:
    """Batch factorization with optimization"""
    results = {}

    # Separate into small and large numbers
    small_nums = [n for n in nums if n <= 65535]
    large_nums = [n for n in nums if n > 65535]

    # Process small numbers from cache
    for n in small_nums:
        results[n] = FACTORIZATION_CACHE.get(n, factorize_number_small(n))

    # Process large numbers
    if large_nums:
        if USE_RUST_BACKEND and rust_backend is not None:
            # Process in batches
            batch_size = 10000
            for i in range(0, len(large_nums), batch_size):
                batch = large_nums[i:i+batch_size]
                min_val = min(batch)
                max_val = max(batch)
                try:
                    results.update(rust_backend.factorize_range(min_val, max_val))  # type: ignore
                except AttributeError:
                    # Fallback if method doesn't exist
                    for n in batch:
                        results[n] = factorize_number(n)
        else:
            try:
                from sympy import factorint
                for n in large_nums:
                    results[n] = factorint(n)
            except ImportError:
                # Fallback if sympy not available
                for n in large_nums:
                    results[n] = factorize_number_small(n)

    return results

# Initialize cache on startup
initialize_factorization_cache()

# ===== DASH LAYOUT =====
app.layout = dbc.Container([
    html.H1("Prime Factor Space Visualization", className="text-center my-4", style={'color': 'white'}),

    dbc.Tabs(id="main-tabs", active_tab='tab-normal', children=[
        dbc.Tab(label="Normal Mode", tab_id='tab-normal', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Settings", style={'color': 'white'}),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("From:", style={'color': 'white'}),
                                    dcc.Input(
                                        id='input-min',
                                        type='number',
                                        value=2,
                                        min=2,
                                        max=1000000,
                                        step=1,
                                        style={'width': '100%', 'color': 'black'}
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("To:", style={'color': 'white'}),
                                    dcc.Input(
                                        id='input-max',
                                        type='number',
                                        value=500,
                                        min=2,
                                        max=1000000,
                                        step=1,
                                        style={'width': '100%', 'color': 'black'}
                                    )
                                ], width=6),
                            ]),

                            dbc.Label("Number range:", style={'color': 'white'}),
                            dcc.RangeSlider(
                                id='range-slider',
                                min=2,
                                max=100000,
                                step=1,
                                value=[2, 500],
                                marks={i: {'label': f'{i//1000}k' if i >= 1000 else str(i),
                                       'style': {'white-space': 'nowrap'}}
                                      for i in range(0, 100001, 10000)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

                            dbc.Label("Axis length:", style={'color': 'white'}),
                            dbc.Row([
                                dbc.Col(
                                    dcc.Slider(
                                        id='axis-length',
                                        min=5,
                                        max=100,
                                        step=10,
                                        value=15,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                    width=9
                                ),
                                dbc.Col(
                                    dcc.Input(
                                        id='axis-length-input',
                                        type='number',
                                        value=15,
                                        min=5,
                                        max=100,
                                        step=1,
                                        style={'width': '100%', 'color': 'black'}
                                    ),
                                    width=3
                                )
                            ]),

                            dbc.Label("Color mode:", style={'color': 'white'}),
                            dcc.Dropdown(
                                id='color-mode',
                                options=[
                                    {'label': 'Omega (distinct factors)', 'value': 'omega'},
                                    {'label': 'Big Omega (total factors)', 'value': 'bigomega'},
                                    {'label': 'Numeric value', 'value': 'value'},
                                    {'label': 'Symmetry', 'value': 'symmetry'},
                                    {'label': 'Complexity', 'value': 'complexity'}
                                ],
                                value='omega',
                                style={'color': 'black'}
                            ),

                            dbc.Label("Advanced options:", style={'color': 'white'}),
                            dbc.Checklist(
                                options=[
                                    {"label": "PFS metric", "value": 'pfs_metric'},
                                    {"label": "Geometric analysis", "value": 'analysis'},
                                ],
                                value=['pfs_metric', 'analysis'],
                                id="advanced-options",
                                style={'color': 'white'}
                            ),

                            dbc.Button('Visualize', id='visualize-btn', className='btn btn-primary mt-3', n_clicks=0),
                            dbc.Spinner(html.Div(id="loading-output"), color="primary"),
                        ], style={'background-color': '#222'})
                    ], style={'border': '1px solid #444'})
                ], md=3),

                dbc.Col([
                    dcc.Graph(id='pfs-visualization', style={'height': '80vh'})
                ], md=9)
            ])
        ]),

        dbc.Tab(label="Research Mode", tab_id='tab-research', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Research Settings", style={'color': 'white'}),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("From:", style={'color': 'white'}),
                                    dcc.Input(
                                        id='research-input-min',
                                        type='number',
                                        value=2,
                                        min=2,
                                        max=1000000,
                                        step=1,
                                        style={'width': '100%', 'color': 'black'}
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("To:", style={'color': 'white'}),
                                    dcc.Input(
                                        id='research-input-max',
                                        type='number',
                                        value=500,
                                        min=2,
                                        max=1000000,
                                        step=1,
                                        style={'width': '100%', 'color': 'black'}
                                    )
                                ], width=6),
                            ]),

                            dbc.Label("Number range:", style={'color': 'white'}),
                            dcc.RangeSlider(
                                id='research-range',
                                min=2,
                                max=1000000,
                                step=1,
                                value=[2, 500],
                                marks={i: {'label': f'{i//1000}k' if i >= 1000 else str(i),
                                       'style': {'white-space': 'nowrap'}}
                                      for i in range(0, 1000001, 100000)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

                            dbc.Label("Max numbers to display:", style={'color': 'white'}),
                            dbc.Row([
                                dbc.Col(
                                    dcc.Slider(
                                        id='max-numbers',
                                        min=2,
                                        max=100000,
                                        step=1,
                                        value=500,
                                        marks={i: {'label': f'{i//1000}k' if i >= 1000 else str(i),
                                                   'style': {'white-space': 'nowrap'}}
                                              for i in range(0, 100001, 10000)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                    width=9
                                ),
                                dbc.Col(
                                    dcc.Input(
                                        id='max-numbers-input',
                                        type='number',
                                        value=500,
                                        min=1,
                                        step=1,
                                        style={'width': '100%', 'color': 'black'}
                                    ),
                                    width=3
                                )
                            ]),

                            dbc.Label("Sampling step:", style={'color': 'white'}),
                            dbc.Row([
                                dbc.Col(
                                    dcc.Slider(
                                        id='sampling-step',
                                        min=2,
                                        max=100000,
                                        step=1,
                                        value=0,
                                        marks={i: {'label': f'{i//1000}k' if i >= 1000 else str(i),
                                        'style': {'white-space': 'nowrap'}}
                                        for i in range(0, 100001, 10000)},
                                            tooltip={"placement": "bottom", "always_visible": True}

                                    ),
                                    width=9
                                ),
                                dbc.Col(
                                    dcc.Input(
                                        id='sampling-step-input',
                                        type='number',
                                        value=1,
                                        min=1,
                                        step=1,
                                        style={'width': '100%', 'color': 'black'}
                                    ),
                                    width=3
                                )
                            ]),

                            dbc.Label("Number types:", style={'color': 'white'}),
                            dcc.Dropdown(
                                id='number-types',
                                options=[
                                    {'label': 'All numbers', 'value': 'all'},
                                    {'label': 'Prime numbers', 'value': 'primes'},
                                    {'label': 'Sophie Germain primes', 'value': 'sophie'},
                                    {'label': 'Safe primes', 'value': 'safe'},
                                    {'label': 'Numbers with specific factors', 'value': 'factors'}
                                ],
                                value=['all'],
                                multi=True,
                                style={'color': 'black'}
                            ),

                            dbc.Label("Factor count (if selected):", style={'color': 'white'}),
                            dcc.Slider(
                                id='factor-count',
                                min=1,
                                max=10,
                                step=1,
                                value=3,
                                disabled=True,
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

                            dbc.Label("Color mapping:", style={'color': 'white'}),
                            dcc.Dropdown(
                                id='type-colors',
                                options=[
                                    {'label': 'By number type', 'value': 'type'},
                                    {'label': 'By distinct factors', 'value': 'omega'},
                                    {'label': 'By total factors', 'value': 'bigomega'},
                                    {'label': 'By value', 'value': 'value'},
                                    {'label': 'By symmetry', 'value': 'symmetry'},
                                    {'label': 'By complexity', 'value': 'complexity'}
                                ],
                                value='type',
                                style={'color': 'black'}
                            ),

                            dbc.Button('Analyze', id='research-btn', className='btn btn-primary mt-3', n_clicks=0),
                            dbc.Spinner(html.Div(id="research-loading-output"), color="primary"),
                        ], style={'background-color': '#222'})
                    ], style={'border': '1px solid #444'})
                ], md=3),

                dbc.Col([
                    dcc.Tabs([
                        dcc.Tab(label='3D Visualization', children=[
                            dcc.Graph(id='research-visualization', style={'height': '75vh'})
                        ]),
                        dcc.Tab(label='Geometric Analysis', children=[
                            dcc.Graph(id='correlation-matrix', style={'height': '40vh'}),
                            dcc.Graph(id='cluster-plot', style={'height': '40vh'})
                        ])
                    ])
                ], md=9)
            ])
        ])
    ]),

    html.Div(id='dummy-output', style={'display': 'none'})
], fluid=True, style={'background-color': '#121212', 'color': 'white', 'min-height': '100vh'})

# ===== CALLBACKS FÜR SYNCHRONISATION =====
# Behebung der zirkulären Abhängigkeit
@app.callback(
    Output('range-slider', 'value'),
    [Input('input-min', 'value'),
     Input('input-max', 'value')],
    prevent_initial_call=True
)
def update_slider_from_inputs(input_min, input_max):
    return [input_min, input_max]

@app.callback(
    [Output('input-min', 'value'),
     Output('input-max', 'value')],
    [Input('range-slider', 'value')],
    prevent_initial_call=True
)
def update_inputs_from_slider(slider_range):
    return slider_range[0], slider_range[1]

@app.callback(
    Output('research-range', 'value'),
    [Input('research-input-min', 'value'),
     Input('research-input-max', 'value')],
    prevent_initial_call=True
)
def update_research_slider_from_inputs(input_min, input_max):
    return [input_min, input_max]

@app.callback(
    [Output('research-input-min', 'value'),
     Output('research-input-max', 'value')],
    [Input('research-range', 'value')],
    prevent_initial_call=True
)
def update_research_inputs_from_slider(slider_range):
    return slider_range[0], slider_range[1]

# ===== OPTIMIERTE VISUALISIERUNG =====
@app.callback(
    [Output('pfs-visualization', 'figure'),
     Output('loading-output', 'children')],
    [Input('visualize-btn', 'n_clicks')],
    [State('range-slider', 'value'),
     State('axis-length', 'value'),
     State('color-mode', 'value'),
     State('advanced-options', 'value')]
)
def update_normal_visualization(n_clicks, num_range, axis_length, color_mode, advanced_options):
    if n_clicks is None or n_clicks == 0:
        return go.Figure(), ""

    min_n, max_n = num_range
    total_numbers = max_n - min_n + 1

    # Intelligentes Sampling basierend auf Bereichsgröße
    if total_numbers > 1000:
        step = max(1, total_numbers // 500)  # Maximal 500 Punkte darstellen
        nums = list(range(min_n, max_n + 1, step))
    else:
        nums = list(range(min_n, max_n + 1))

    print(f"Visualizing {len(nums)} numbers (range: {min_n}-{max_n})")

    # Faktorisierung mit Caching
    start_time = time.time()
    factors_dicts = factorize_numbers(nums)
    print(f"Factorization took: {time.time() - start_time:.2f} seconds")

    # Nur die benötigten Primzahlen berücksichtigen
    all_primes = set()
    for factors in factors_dicts.values():
        all_primes |= set(factors.keys())
    primes = sorted(all_primes, reverse=True)

    # Winkelverteilung berechnen
    angle_map = symmetric_angle_distribution(primes)

    # Figure vorbereiten
    fig = go.Figure()

    # Daten für die Visualisierung sammeln
    all_x = []
    all_y = []
    all_z = []
    all_colors = []
    all_text = []

    for z_offset, n in enumerate(nums):
        factors = factors_dicts.get(n, {})
        primes_in_n = [p for p in primes if p in factors]

        # Koordinaten berechnen
        coords = []
        for p in primes_in_n:
            theta = angle_map[p]
            r = factors[p]
            coords.append((r * np.cos(theta), r * np.sin(theta)))

        # Fortgeschrittene Metriken nur bei Bedarf
        if 'pfs_metric' in advanced_options and coords:
            coords = advanced_pfs_metric(coords, factors)

        # Geometrische Analyse nur bei Bedarf
        features = {}
        if 'analysis' in advanced_options:
            features = extended_geometric_analysis(coords, factors)

        # Farbe basierend auf Modus
        if color_mode == 'omega':
            color_val = len(factors)
        elif color_mode == 'bigomega':
            color_val = sum(factors.values())
        elif color_mode == 'value':
            color_val = n
        elif color_mode == 'symmetry' and features:
            color_val = features.get('symmetry_std', 0) * 20
        elif color_mode == 'complexity' and features:
            color_val = features.get('compactness', 0) * 100
        else:
            color_val = np.log(n)

        # Nur wenn Koordinaten vorhanden sind
        if coords:
            # Sortieren für geschlossenen Polygonzug
            angles = [np.arctan2(y, x) for x, y in coords]
            sorted_indices = np.argsort(angles)
            sorted_coords = [coords[i] for i in sorted_indices]

            # Geschlossene Form erstellen
            x_vals, y_vals = zip(*sorted_coords)
            x_vals = list(x_vals) + [x_vals[0]]
            y_vals = list(y_vals) + [y_vals[0]]
            z_vals = [z_offset] * len(x_vals)

            # Daten für Batch-Verarbeitung sammeln
            all_x.extend(x_vals)
            all_y.extend(y_vals)
            all_z.extend(z_vals)
            all_colors.extend([color_val] * len(x_vals))

            # Hover-Text für den ersten Punkt jedes Polygons
            hover_text = f"n={n}<br>Factors: {factors}"
            if 'analysis' in advanced_options:
                hover_text += f"<br>Area: {features.get('area', 0):.2f}"
                hover_text += f"<br>Symmetry: {features.get('symmetry_std', 0):.3f}"
                hover_text += f"<br>Compactness: {features.get('compactness', 0):.3f}"

            all_text.extend([hover_text] + [""] * (len(x_vals) - 1))

    # Batch-Verarbeitung für alle Polygone
    if all_x:
        fig.add_trace(go.Scatter3d(
            x=all_x,
            y=all_y,
            z=all_z,
            mode='lines+markers',
            line=dict(width=2, color=all_colors),
            marker=dict(size=4),
            hoverinfo='text',
            hovertext=all_text,
            showlegend=False
        ))

    # Primachsen hinzufügen (nur bis zu 20, um Performance zu verbessern)
    for i, (p, theta) in enumerate(list(angle_map.items())[:20]):
        x_end = axis_length * np.cos(theta)
        y_end = axis_length * np.sin(theta)

        fig.add_trace(go.Scatter3d(
            x=[0, x_end],
            y=[0, y_end],
            z=[0, 0],
            mode='lines',
            line=dict(color='white', width=1),
            hoverinfo='none',
            showlegend=False
        ))

    # Layout anpassen
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Number Index',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.8),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4),
            xaxis=dict(range=[-axis_length*1.1, axis_length*1.1]),
            yaxis=dict(range=[-axis_length*1.1, axis_length*1.1]),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        height=700
    )

    return fig, ""

# ===== OPTIMIERTER RESEARCH MODE =====
@app.callback(
    [Output('research-visualization', 'figure'),
     Output('correlation-matrix', 'figure'),
     Output('cluster-plot', 'figure'),
     Output('research-loading-output', 'children')],
    [Input('research-btn', 'n_clicks')],
    [State('research-range', 'value'),
     State('axis-length', 'value'),
     State('number-types', 'value'),
     State('factor-count', 'value'),
     State('type-colors', 'value'),
     State('max-numbers-input', 'value'),
     State('sampling-step-input', 'value'),
     State('advanced-options', 'value')]
)
def update_research_visualization(n_clicks, num_range, axis_length, number_types, factor_count,
                                 color_type, max_numbers, sampling_step, advanced_options):
    if n_clicks is None or n_clicks == 0:
        return go.Figure(), go.Figure(), go.Figure(), ""

    min_n, max_n = num_range
    max_numbers = max_numbers or 1000
    sampling_step = sampling_step or 1

    # 1. Effiziente Zahlenauswahl mit Mengenoperationen
    number_sets = {
        'all': set(),
        'primes': set(),
        'sophie': set(),
        'safe': set(),
        'factors': set()
    }

    # Berechne Mengen nur für benötigte Typen
    if 'primes' in number_types or 'sophie' in number_types or 'safe' in number_types:
        primes = primes_below(max_n + 1)

        if 'primes' in number_types:
            number_sets['primes'] = {p for p in primes if min_n <= p <= max_n}

        if 'sophie' in number_types:
            sophie_primes = get_sophie_germain_primes(max_n)
            number_sets['sophie'] = {p for p in sophie_primes if min_n <= p <= max_n}

        if 'safe' in number_types:
            safe_primes = get_safe_primes(max_n)
            number_sets['safe'] = {p for p in safe_primes if min_n <= p <= max_n}

    # 2. Kombiniere die ausgewählten Zahlenmengen
    selected_numbers = set()
    for typ in number_types:
        if typ == 'all':
            selected_numbers |= set(range(min_n, max_n + 1))
        else:
            selected_numbers |= number_sets[typ]

    # 3. Sampling anwenden
    sorted_numbers = sorted(selected_numbers)
    if sampling_step > 1:
        sorted_numbers = sorted_numbers[::int(sampling_step)]

    # 4. Begrenzung der Anzahl
    if len(sorted_numbers) > max_numbers:
        step = max(1, len(sorted_numbers) // int(max_numbers))
        sorted_numbers = sorted_numbers[::step]

    print(f"Research mode: analyzing {len(sorted_numbers)} numbers")

    # 5. Faktorisierung nur für ausgewählte Zahlen
    start_time = time.time()
    factors_dicts = factorize_numbers(sorted_numbers)
    print(f"Factorization took: {time.time() - start_time:.2f} seconds")

    # 6. Filterung für 'factors' Typ
    final_numbers = []
    number_categories = []

    for n in sorted_numbers:
        factors = factors_dicts.get(n, {})
        category = None

        if 'all' in number_types:
            category = 'Standard'
        elif n in number_sets['primes']:
            category = 'Prime'
        elif n in number_sets['sophie']:
            category = 'Sophie Germain'
        elif n in number_sets['safe']:
            category = 'Safe Prime'
        elif 'factors' in number_types and len(factors) == factor_count:
            category = f'{factor_count} Factors'

        if category:
            final_numbers.append(n)
            number_categories.append(category)

    print(f"Displaying {len(final_numbers)} numbers after filtering")

    # 7. Performance-Optimierung für Visualisierung
    all_primes = set()
    for factors in factors_dicts.values():
        all_primes |= set(factors.keys())
    primes = sorted(all_primes, reverse=True)
    angle_map = symmetric_angle_distribution(primes)

    # 8. Batch-Verarbeitung für Visualisierung
    all_x = []
    all_y = []
    all_z = []
    all_colors = []
    all_text = []
    all_marker_sizes = []

    # Farbzuordnung für Kategorien erstellen
    category_colors = {
        'Standard': '#1f77b4',    # Blau
        'Prime': '#ff7f0e',       # Orange
        'Sophie Germain': '#2ca02c',  # Grün
        'Safe Prime': '#d62728',   # Rot
    }

    for i in range(3, 11):  # Für 3 bis 10 Faktoren
        category_colors[f'{i} Factors'] = f'#{random.randint(0, 0xFFFFFF):06x}'

    for z_offset, (n, cat) in enumerate(zip(final_numbers, number_categories)):
        factors = factors_dicts.get(n, {})
        primes_in_n = [p for p in primes if p in factors]

        # Koordinaten berechnen
        coords = []
        for p in primes_in_n:
            theta = angle_map[p]
            r = factors[p]
            coords.append((r * np.cos(theta), r * np.sin(theta)))

        # Fortgeschrittene Metriken
        if 'pfs_metric' in advanced_options and coords:
            coords = advanced_pfs_metric(coords, factors)

        # Geometrische Analyse
        features = {}
        if 'analysis' in advanced_options:
            features = extended_geometric_analysis(coords, factors)

        # Farbe basierend auf Modus
        if color_type == 'type':
            color_val = category_colors.get(cat, '#7f7f7f')  # Grau als Fallback
        elif color_type == 'omega':
            color_val = len(factors)
        elif color_type == 'bigomega':
            color_val = sum(factors.values())
        elif color_type == 'value':
            color_val = n
        elif color_type == 'symmetry' and features:
            color_val = features.get('symmetry_std', 0) * 20
        elif color_type == 'complexity' and features:
            color_val = features.get('compactness', 0) * 100
        else:
            color_val = np.log(n)

        # Nur wenn Koordinaten vorhanden sind
        if coords:
            # Sortieren für geschlossenen Polygonzug
            angles = [np.arctan2(y, x) for x, y in coords]
            sorted_indices = np.argsort(angles)
            sorted_coords = [coords[i] for i in sorted_indices]

            # Geschlossene Form erstellen
            x_vals, y_vals = zip(*sorted_coords)
            x_vals = list(x_vals) + [x_vals[0]]
            y_vals = list(y_vals) + [y_vals[0]]
            z_vals = [z_offset] * len(x_vals)

            # Daten für Batch-Verarbeitung sammeln
            all_x.extend(x_vals)
            all_y.extend(y_vals)
            all_z.extend(z_vals)
            all_colors.extend([color_val] * len(x_vals))

            # Markergröße
            marker_size = 4 + np.log1p(sum(factors.values()))
            all_marker_sizes.extend([marker_size] * len(x_vals))

            # Hover-Text
            hover_text = f"n={n}<br>Type: {cat}<br>Factors: {factors}"
            if 'analysis' in advanced_options:
                hover_text += f"<br>Area: {features.get('area', 0):.2f}"
                hover_text += f"<br>Symmetry: {features.get('symmetry_std', 0):.3f}"
                hover_text += f"<br>Compactness: {features.get('compactness', 0):.3f}"

            all_text.extend([hover_text] + [""] * (len(x_vals) - 1))

    # 9. Erstelle die Visualisierung mit einem einzigen Trace
    fig = go.Figure()

    if all_x:
        fig.add_trace(go.Scatter3d(
            x=all_x,
            y=all_y,
            z=all_z,
            mode='lines+markers',
            line=dict(width=2, color=all_colors),  # Feste Linienbreite von 2
            marker=dict(size=all_marker_sizes, color=all_colors),
            hoverinfo='text',
            hovertext=all_text,
            showlegend=False
        ))

    # 10. Primachsen hinzufügen (begrenzt auf 20 für Performance)
    for i, (p, theta) in enumerate(list(angle_map.items())[:20]):
        x_end = axis_length * np.cos(theta)
        y_end = axis_length * np.sin(theta)

        fig.add_trace(go.Scatter3d(
            x=[0, x_end],
            y=[0, y_end],
            z=[0, 0],
            mode='lines',
            line=dict(color='white', width=1),
            hoverinfo='none',
            showlegend=False
        ))

    # 11. Layout anpassen
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Number Index',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.8),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4),
            xaxis=dict(range=[-axis_length*1.1, axis_length*1.1]),
            yaxis=dict(range=[-axis_length*1.1, axis_length*1.1]),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        height=700,
        title=f'Research Mode: {len(final_numbers)} Numbers ({min_n}-{max_n})'
    )

    # 12. Korrelationsmatrix (vereinfacht)
    corr_fig = go.Figure()
    cluster_fig = go.Figure()

    return fig, corr_fig, cluster_fig, ""
# Aktivierung/Deaktivierung des Faktor-Sliders
@app.callback(
    Output('factor-count', 'disabled'),
    Input('number-types', 'value')
)
def toggle_factor_slider(selected_types):
    return 'factors' not in selected_types

if __name__ == '__main__':
    app.run(debug=True, port=8050)
