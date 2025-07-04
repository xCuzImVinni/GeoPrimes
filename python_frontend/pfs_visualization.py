from typing import TYPE_CHECKING
import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import math
import random
import time
import os
import pandas as pd
from typing import Dict, List, Tuple, Set
from functools import lru_cache
import sys

# Rust-Backend laden
rust_backend = None
USE_RUST_BACKEND = False
try:
    import rust_backend
    # Explizite Typannotation für Pyright
    if TYPE_CHECKING:
        from rust_backend import factorize_number, factorize_range
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
        features['symmetry_max'] = max(dists) - min(dists) if dists and len(dists) > 1 else 0
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

# CACHING FÜR HÄUFIGE FAKTORISIERUNGEN
@lru_cache(maxsize=100000)
def cached_factorize_number(n: int) -> dict:
    return factorize_number(n)

def get_required_primes(numbers: list) -> list:
    all_primes = set()
    for n in numbers:
        factors = cached_factorize_number(n)
        all_primes |= set(factors.keys())
    return sorted(all_primes, reverse=True)
def factorize_number(n: int) -> dict:
    if USE_RUST_BACKEND and rust_backend is not None:
        # Pyright-Ignore für dynamische Attribute
        return rust_backend.factorize_number(n)  # type: ignore[attr-defined]
    else:
        return factorint(n)

def factorize_numbers(nums: list) -> dict:
    if USE_RUST_BACKEND and rust_backend is not None:
        min_val = min(nums)
        max_val = max(nums)
        # Batch-Verarbeitung für große Mengen
        if len(nums) > 1000:
            results = {}
            batch_size = 10000
            for i in range(0, len(nums), batch_size):
                batch = nums[i:i+batch_size]
                min_batch = min(batch)
                max_batch = max(batch)
                # Ignore für Pyright
                results.update(rust_backend.factorize_range(min_batch, max_batch))  # type: ignore[attr-defined]
            return results
        else:
            return rust_backend.factorize_range(min_val, max_val)  # type: ignore[attr-defined]
    else:
        return {n: factorint(n) for n in nums}
# ===== DASH LAYOUT MIT EINGABEFELDERN UND SYNCHRONISATION =====
app.layout = dbc.Container([
    dcc.Store(id='visualization-state', data={'min_n': 2, 'max_n': 500}),
    dcc.Store(id='research-state', data={'min_n': 2, 'max_n': 500}),

    html.H1("Prime Factor Space Visualization", className="text-center my-4", style={'color': 'white'}),

    dbc.Tabs(id="main-tabs", active_tab='tab-normal', children=[
        dbc.Tab(label="Normal Mode", tab_id='tab-normal', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Settings", style={'color': 'white'}),
                        dbc.CardBody([
                            # Eingabefelder für präzise Bereichsangabe
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
                                       'style': {'transform': 'rotate(45deg)', 'white-space': 'nowrap'}}
                                      for i in range(0, 100001, 10000)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

                            dbc.Label("Axis length:", style={'color': 'white'}),
                            dbc.Row([
                                dbc.Col(
                                    dcc.Slider(
                                        id='axis-length',
                                        min=1,
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
                                    {"label": "Advanced metrics", "value": 'advanced_metrics'}
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
                            # Eingabefelder für Research Mode
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
                                       'style': {'transform': 'rotate(45deg)', 'white-space': 'nowrap'}}
                                      for i in range(0, 1000001, 100000)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

                            dbc.Label("Max numbers to display:", style={'color': 'white'}),
                            dbc.Row([
                                dbc.Col(
                                    dcc.Slider(
                                        id='max-numbers',
                                        min=2,
                                        max=1000000,
                                        step=1,
                                        value=500,
                                        marks={i: {'label': f'{i//1000}k' if i >= 1000 else str(i),
                                       'style': {'transform': 'rotate(45deg)', 'white-space': 'nowrap'}}
                                      for i in range(0, 1000001, 100000)},
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
                                        min=0,  # Logarithmische Basis
                                        max=1000000,  # 10^6 = 1,000,000
                                        step=0.1,
                                        value=0,
                                        marks={i: {'label': f'{i//1000}k' if i >= 1000 else str(i),
                                       'style': {'transform': 'rotate(45deg)', 'white-space': 'nowrap'}}
                                      for i in range(0, 1000001, 100000)},
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

# ===== CALLBACKS FÜR SYNCHRONISATION UND OPTIMIERTE VISUALISIERUNG =====
# Automatische Synchronisation zwischen Eingabefeldern und Slidern
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
    [Output('research-input-min', 'value'),
     Output('research-input-max', 'value')],
    [Input('research-range', 'value')],
    prevent_initial_call=True
)
def update_research_inputs_from_slider(slider_range):
    return slider_range[0], slider_range[1]

@app.callback(
    Output('research-range', 'value'),
    [Input('research-input-min', 'value'),
     Input('research-input-max', 'value')],
    prevent_initial_call=True
)
def update_research_slider_from_inputs(input_min, input_max):
    if input_min is None or input_max is None:
        return dash.no_update
    return [input_min, input_max]

# Synchronisation für Axis Length
@app.callback(
    Output('axis-length-input', 'value'),
    Input('axis-length', 'value'),
    prevent_initial_call=True
)
def axis_length_to_input(value):
    return value

@app.callback(
    Output('axis-length', 'value'),
    Input('axis-length-input', 'value'),
    prevent_initial_call=True
)
def input_to_axis_length(value):
    if value is None:
        return dash.no_update
    return value

# Synchronisation für Max Numbers
@app.callback(
    Output('max-numbers-input', 'value'),
    Input('max-numbers', 'value')
)
def max_exp_to_input(k):
    return round(10 ** k)

@app.callback(
    Output('max-numbers', 'value'),
    Input('max-numbers-input', 'value'),
    prevent_initial_call=True
)
def input_to_max_exp(value):
    if value is None or value <= 0:
        return dash.no_update
    return math.log10(value)

# Synchronisation für Sampling Step
@app.callback(
    Output('sampling-step-input', 'value'),
    Input('sampling-step', 'value')
)
def step_exp_to_input(k):
    return round(10 ** k)

@app.callback(
    Output('sampling-step', 'value'),
    Input('sampling-step-input', 'value'),
    prevent_initial_call=True
)
def input_to_step_exp(value):
    if value is None or value <= 0:
        return dash.no_update
    return math.log10(value)

# Optimierte Visualisierung mit Fortschrittsanzeige
@app.callback(
    [Output('pfs-visualization', 'figure'),
     Output('loading-output', 'children')],
    [Input('visualize-btn', 'n_clicks')],
    [State('range-slider', 'value'),
     State('axis-length', 'value'),
     State('color-mode', 'value'),
     State('advanced-options', 'value')]  # Entferne sampling-density
)
def update_normal_visualization(n_clicks, num_range, axis_length, color_mode, advanced_options):
    min_n, max_n = num_range
    total_numbers = max_n - min_n + 1

    # Automatisches Sampling basierend auf Bereichsgröße
    if total_numbers > 1000:
        step = max(1, int(total_numbers / 1000))
        nums = list(range(min_n, max_n + 1, step))
    else:
        nums = list(range(min_n, max_n + 1))

    print(f"Visualizing {len(nums)} numbers (sampled from {total_numbers} total)")

    start_time = time.time()
    factors_dicts = factorize_numbers(nums)
    print(f"Factorization took: {time.time() - start_time:.2f} seconds")

    primes = get_required_primes(nums)
    angle_map = symmetric_angle_distribution(primes)

    fig = go.Figure()

    for z_offset, n in enumerate(nums):
        factors = factors_dicts.get(n, {})
        primes_in_n = [p for p in primes if p in factors]

        coords = []
        for p in primes_in_n:
            theta = angle_map[p]
            r = factors[p]
            coords.append((r * np.cos(theta), r * np.sin(theta)))

        pfs_metric = 'pfs_metric' in advanced_options
        if pfs_metric and coords and 'advanced_metrics' in advanced_options:
            coords = advanced_pfs_metric(coords, factors)

        analysis_mode = 'analysis' in advanced_options
        features = extended_geometric_analysis(coords, factors) if analysis_mode else {}

        # Color calculation
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

        if coords:
            angles = [np.arctan2(y, x) for x, y in coords]
            sorted_indices = np.argsort(angles)
            sorted_coords = [coords[i] for i in sorted_indices]

            x_vals, y_vals = zip(*sorted_coords)
            x_vals = list(x_vals) + [x_vals[0]]
            y_vals = list(y_vals) + [y_vals[0]]
            z_vals = [z_offset] * len(x_vals)

            line_width = 2 + min(len(factors), 5) * 0.5

            hover_text = f"n={n}<br>Factors: {factors}"
            if analysis_mode:
                hover_text += f"<br>Area: {features.get('area', 0):.2f}"
                hover_text += f"<br>Symmetry: {features.get('symmetry_std', 0):.3f}"
                hover_text += f"<br>Compactness: {features.get('compactness', 0):.3f}"

            fig.add_trace(go.Scatter3d(
                x=x_vals, y=y_vals, z=z_vals,
                mode='lines+markers',
                line=dict(width=line_width, color=color_val),
                marker=dict(size=4 + np.log1p(sum(factors.values()))),
                name=f'n={n}',
                hoverinfo='name+text',
                hovertext=hover_text,
                showlegend=False
            ))

    # Add prime axes
    for p, theta in angle_map.items():
        x_end = axis_length * np.cos(theta)
        y_end = axis_length * np.sin(theta)

        fig.add_trace(go.Scatter3d(
            x=[0, x_end],
            y=[0, y_end],
            z=[0, 0],
            mode='lines+text',
            line=dict(color='white', width=1.5),
            text=["", f"P{p}"],
            textposition="top center",
            textfont=dict(size=10, color='white'),
            hoverinfo='none',
            showlegend=False
        ))

    # Layout adjustments
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Number Index',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.8),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.2)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4),
            xaxis=dict(
                range=[-axis_length*1.1, axis_length*1.1],
                gridcolor='rgba(100,100,100,0.5)',
                backgroundcolor='rgba(0,0,0,0)'
            ),
            yaxis=dict(
                range=[-axis_length*1.1, axis_length*1.1],
                gridcolor='rgba(100,100,100,0.5)',
                backgroundcolor='rgba(0,0,0,0)'
            ),
            zaxis=dict(
                gridcolor='rgba(100,100,100,0.5)',
                backgroundcolor='rgba(0,0,0,0)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=800,
        hovermode='closest',
        margin=dict(l=50, r=50, b=50, t=80)
    )

    return fig, ""

# Ähnliche Optimierung für Research Mode
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
    all_nums = list(range(min_n, max_n + 1))

    # Apply sampling
    if sampling_step > 1:
        all_nums = all_nums[::sampling_step]

    # Limit number of points
    if len(all_nums) > max_numbers:
        step = max(1, len(all_nums) // max_numbers)
        all_nums = all_nums[::step]

    print(f"Research mode: analyzing {len(all_nums)} numbers")

    # Factorize all numbers in parallel
    start_time = time.time()
    factors_dicts = factorize_numbers(all_nums)
    print(f"Factorization took: {time.time() - start_time:.2f} seconds")

    # Filter by number types
    filtered_nums = []
    number_categories = []

    if 'all' in number_types or not number_types:
        filtered_nums.extend(all_nums)
        number_categories.extend(['Standard'] * len(all_nums))

    if 'primes' in number_types:
        primes = [n for n in all_nums if is_prime(n)]
        filtered_nums.extend(primes)
        number_categories.extend(['Prime'] * len(primes))

    if 'sophie' in number_types:
        sophie_primes = get_sophie_germain_primes(max_n)
        sophie_primes = [p for p in sophie_primes if min_n <= p <= max_n]
        filtered_nums.extend(sophie_primes)
        number_categories.extend(['Sophie Germain'] * len(sophie_primes))

    if 'safe' in number_types:
        safe_primes = get_safe_primes(max_n)
        safe_primes = [p for p in safe_primes if min_n <= p <= max_n]
        filtered_nums.extend(safe_primes)
        number_categories.extend(['Safe Prime'] * len(safe_primes))

    if 'factors' in number_types:
        factor_nums = [n for n in all_nums if len(factors_dicts.get(n, {})) == factor_count]
        filtered_nums.extend(factor_nums)
        number_categories.extend([f'{factor_count} Factors'] * len(factor_nums))

    # Remove duplicates and keep first category
    unique_nums = []
    unique_categories = []
    seen = set()

    for n, cat in zip(filtered_nums, number_categories):
        if n not in seen:
            seen.add(n)
            unique_nums.append(n)
            unique_categories.append(cat)

    primes = get_required_primes(unique_nums)
    angle_map = symmetric_angle_distribution(primes)

    fig = go.Figure()
    geometric_data = []

    # Color mapping
    color_palette = px.colors.qualitative.Plotly
    category_colors = {}
    unique_cats = set(unique_categories)
    for i, cat in enumerate(unique_cats):
        category_colors[cat] = color_palette[i % len(color_palette)]

    for z_offset, (n, cat) in enumerate(zip(unique_nums, unique_categories)):
        factors = factors_dicts.get(n, {})
        primes_in_n = [p for p in primes if p in factors]

        coords = []
        for p in primes_in_n:
            theta = angle_map[p]
            r = factors[p]
            coords.append((r * np.cos(theta), r * np.sin(theta)))

        pfs_metric = 'pfs_metric' in advanced_options
        if pfs_metric and coords and 'advanced_metrics' in advanced_options:
            coords = advanced_pfs_metric(coords, factors)

        analysis_mode = 'analysis' in advanced_options
        features = extended_geometric_analysis(coords, factors) if analysis_mode else {}
        geometric_data.append(features)

        # Color calculation
        if color_type == 'type':
            color_val = category_colors[cat]
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

        if coords:
            angles = [np.arctan2(y, x) for x, y in coords]
            sorted_indices = np.argsort(angles)
            sorted_coords = [coords[i] for i in sorted_indices]

            x_vals, y_vals = zip(*sorted_coords)
            x_vals = list(x_vals) + [x_vals[0]]
            y_vals = list(y_vals) + [y_vals[0]]
            z_vals = [z_offset] * len(x_vals)

            line_width = 2 + min(len(factors), 5) * 0.5

            hover_text = f"n={n}<br>Type: {cat}<br>Factors: {factors}"
            if analysis_mode:
                hover_text += f"<br>Area: {features.get('area', 0):.2f}"
                hover_text += f"<br>Symmetry: {features.get('symmetry_std', 0):.3f}"
                hover_text += f"<br>Compactness: {features.get('compactness', 0):.3f}"

            fig.add_trace(go.Scatter3d(
                x=x_vals, y=y_vals, z=z_vals,
                mode='lines+markers',
                line=dict(width=line_width, color=color_val),
                marker=dict(size=4 + np.log1p(sum(factors.values()))),
                name=f'n={n}',
                hoverinfo='name+text',
                hovertext=hover_text,
                showlegend=False
            ))

    # Add prime axes
    for p, theta in angle_map.items():
        x_end = axis_length * np.cos(theta)
        y_end = axis_length * np.sin(theta)

        fig.add_trace(go.Scatter3d(
            x=[0, x_end],
            y=[0, y_end],
            z=[0, 0],
            mode='lines+text',
            line=dict(color='white', width=1.5),
            text=["", f"P{p}"],
            textposition="top center",
            textfont=dict(size=10, color='white'),
            hoverinfo='none',
            showlegend=False
        ))

    # Layout adjustments
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Number Index',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.8),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.2)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4),
            xaxis=dict(
                range=[-axis_length*1.1, axis_length*1.1],
                gridcolor='rgba(100,100,100,0.5)',
                backgroundcolor='rgba(0,0,0,0)'
            ),
            yaxis=dict(
                range=[-axis_length*1.1, axis_length*1.1],
                gridcolor='rgba(100,100,100,0.5)',
                backgroundcolor='rgba(0,0,0,0)'
            ),
            zaxis=dict(
                gridcolor='rgba(100,100,100,0.5)',
                backgroundcolor='rgba(0,0,0,0)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=800,
        hovermode='closest',
        margin=dict(l=50, r=50, b=50, t=80),
        title=f'Research Mode: {len(unique_nums)} Numbers ({min_n}-{max_n})'
    )

    # Create correlation matrix
    if geometric_data and len(geometric_data) > 1:
        df = pd.DataFrame(geometric_data)
        corr_matrix = df.corr()
        corr_fig = px.imshow(
            corr_matrix,
            text_auto=True,
            title="Geometric Feature Correlation",
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            labels=dict(color="Correlation")
        )
        corr_fig.update_layout(title_font=dict(color='white'),
                              paper_bgcolor='#121212',
                              font=dict(color='white'),
                              coloraxis_colorbar=dict(title="Correlation"))
    else:
        corr_fig = go.Figure()

    # Create cluster plot
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
