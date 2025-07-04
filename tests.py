import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sympy import factorint, isprime
from scipy.spatial import distance
import pandas as pd
import dash_bootstrap_components as dbc
import math
import random

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# ===== HELPER FUNCTIONS =====
def get_safe_primes(max_n):
    primes = [p for p in range(2, max_n) if isprime(p)]
    return [p for p in primes if isprime((p-1)//2) and (p-1)//2 > 1]

def get_sophie_germain_primes(max_n):
    primes = [p for p in range(2, max_n) if isprime(p)]
    return [p for p in primes if isprime(2*p+1)]

def get_required_primes(numbers):
    all_primes = set()
    for n in numbers:
        factors = factorint(n)
        all_primes |= set(factors.keys())
    return sorted(all_primes, reverse=True)

def symmetric_angle_distribution(primes):
    n = len(primes)
    angles = {}
    for i, p in enumerate(primes):
        angle = 2 * np.pi * i / n
        angles[p] = angle
    return angles

def advanced_pfs_metric(coords, factors):
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

def extended_geometric_analysis(coords, factors):
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

    if len(coords) > 0:
        dists = [distance.euclidean(p, centroid) for p in coords]
        features['symmetry_std'] = np.std(dists) if dists else 0
        features['symmetry_max'] = max(dists) - min(dists) if dists and len(dists) > 1 else 0
    else:
        features['symmetry_std'] = 0
        features['symmetry_max'] = 0

    features['centroid_distance'] = distance.euclidean(centroid, (0, 0))

    if len(coords) > 1:
        angles = [np.arctan2(y, x) for x, y in coords]
        angle_diffs = [abs(angles[i] - angles[(i+1)%len(angles)]) % (2*np.pi)
                     for i in range(len(angles))]
        features['angle_variance'] = np.var(angle_diffs) if angle_diffs else 0
    else:
        features['angle_variance'] = 0

    return features

def create_pfs_figure(nums, axis_length, color_mode, advanced_options, number_types=None, color_type=None):
    primes = get_required_primes(nums)
    angle_map = symmetric_angle_distribution(primes)

    fig = go.Figure()
    max_radius = 0
    geometric_data = []
    pfs_vectors = []

    # Color mapping for research types
    type_colors = {}
    if number_types and color_type == 'type':
        color_palette = px.colors.qualitative.Plotly
        unique_categories = set(number_types)
        for i, cat in enumerate(unique_categories):
            type_colors[cat] = color_palette[i % len(color_palette)]

    for z_offset, n in enumerate(nums):
        factors_dict = factorint(n)
        sorted_primes = sorted(factors_dict.keys(), reverse=True)
        factors = {p: factors_dict[p] for p in sorted_primes}
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
        if color_type == 'type' and number_types:
            color_val = type_colors.get(number_types[z_offset], 'gray')
        else:
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
            # Update max radius
            for x, y in coords:
                radius = np.sqrt(x**2 + y**2)
                max_radius = max(max_radius, radius)

            angles = [np.arctan2(y, x) for x, y in coords]
            sorted_indices = np.argsort(angles)
            sorted_coords = [coords[i] for i in sorted_indices]
            sorted_primes = [primes_in_n[i] for i in sorted_indices]

            x_vals, y_vals = zip(*sorted_coords)
            x_vals = list(x_vals) + [x_vals[0]]
            y_vals = list(y_vals) + [y_vals[0]]
            z_vals = [z_offset] * len(x_vals)

            line_width = 2 + min(len(factors), 5) * 0.5

            # Hover text with type information
            hover_text = f"n={n}<br>Factors: {factors}"
            if number_types:
                hover_text += f"<br>Type: {number_types[z_offset]}"
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

            # Projection lines
            for (x, y) in sorted_coords:
                fig.add_trace(go.Scatter3d(
                    x=[x, x], y=[y, y], z=[0, z_offset],
                    mode='lines',
                    line=dict(color=f'rgba(150,150,150,{0.2 + len(factors)*0.05})', width=0.8),
                    showlegend=False,
                    hoverinfo='none'
                ))

    # Add prime axes
    for p, theta in angle_map.items():
        x_end = min(axis_length, max_radius * 1.2) * np.cos(theta)
        y_end = min(axis_length, max_radius * 1.2) * np.sin(theta)

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

    return fig, geometric_data, pfs_vectors

# ===== DASH LAYOUT =====
app.layout = dbc.Container([
    html.H1("Prime Factor Space Visualization", className="text-center my-4", style={'color': 'white'}),

    dbc.Tabs([
        dbc.Tab(label="Normal Mode", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Settings", style={'color': 'white'}),
                        dbc.CardBody([
                            dbc.Label("Number range:", style={'color': 'white'}),
                            dcc.RangeSlider(
                                id='range-slider',
                                min=2,
                                max=1000,
                                step=1,
                                value=[2, 100],
                                marks={i: str(i) for i in range(0, 1001, 100)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

                            dbc.Label("Axis length:", style={'color': 'white'}),
                            dcc.Slider(
                                id='axis-length',
                                min=5,
                                max=50,
                                step=1,
                                value=15,
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

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

                            html.Button('Visualize', id='visualize-btn', className='btn btn-primary mt-3')
                        ], style={'background-color': '#222'})
                    ], style={'border': '1px solid #444'})
                ], md=3),

                dbc.Col([
                    dcc.Graph(id='pfs-visualization', style={'height': '80vh'})
                ], md=9)
            ])
        ]),

        dbc.Tab(label="Research Mode", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Research Settings", style={'color': 'white'}),
                        dbc.CardBody([
                            dbc.Label("Number range:", style={'color': 'white'}),
                            dcc.RangeSlider(
                                id='research-range',
                                min=2,
                                max=10**9,
                                step=1,
                                value=[2, 10**9],
                                marks={i: str(i) for i in range(0, 10**9+1,10**1)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

                            dbc.Label("Max numbers to display:", style={'color': 'white'}),
                            dcc.Slider(
                                id='max-numbers',
                                min=10,
                                max=1000,
                                step=10,
                                value=200,
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

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

                            html.Button('Analyze', id='research-btn', className='btn btn-primary mt-3')
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

# ===== CALLBACKS =====
@app.callback(
    Output('factor-count', 'disabled'),
    Input('number-types', 'value')
)
def toggle_factor_slider(selected_types):
    return 'factors' not in selected_types

@app.callback(
    Output('pfs-visualization', 'figure'),
    [Input('visualize-btn', 'n_clicks'),
     Input('range-slider', 'value'),
     Input('axis-length', 'value'),
     Input('color-mode', 'value'),
     Input('advanced-options', 'value')],
    prevent_initial_call=False
)
def update_normal_visualization(n_clicks, num_range, axis_length, color_mode, advanced_options):
    min_n, max_n = num_range
    nums = list(range(min_n, max_n + 1))
    fig, _, _ = create_pfs_figure(nums, axis_length, color_mode, advanced_options)
    fig.update_layout(title=f'<b>Normal Mode: Numbers {min_n}-{max_n}</b>', title_font=dict(color='white'))
    return fig

@app.callback(
    [Output('research-visualization', 'figure'),
     Output('correlation-matrix', 'figure'),
     Output('cluster-plot', 'figure')],
    [Input('research-btn', 'n_clicks'),
     Input('research-range', 'value'),
     Input('axis-length', 'value'),
     Input('number-types', 'value'),
     Input('factor-count', 'value'),
     Input('type-colors', 'value'),
     Input('max-numbers', 'value'),
     Input('advanced-options', 'value')],
    prevent_initial_call=False
)
def update_research_visualization(n_clicks, num_range, axis_length, number_types, factor_count,
                                 color_type, max_numbers, advanced_options):
    min_n, max_n = num_range
    all_nums = list(range(min_n, max_n + 1))

    # Apply sampling if too many numbers
    if len(all_nums) > max_numbers:
        step = max(1, len(all_nums) // max_numbers)
        all_nums = all_nums[::step]

    # Filter by number types
    filtered_nums = []
    number_categories = []

    if 'all' in number_types or not number_types:
        filtered_nums.extend(all_nums)
        number_categories.extend(['Standard'] * len(all_nums))

    if 'primes' in number_types:
        primes = [p for p in all_nums if isprime(p)]
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
        factor_nums = [n for n in all_nums if len(factorint(n)) == factor_count]
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

    # Create visualization
    fig, geometric_data, pfs_vectors = create_pfs_figure(
        unique_nums,
        axis_length,
        color_type,
        advanced_options,
        unique_categories,
        color_type
    )

    title = f'<b>Research Mode: Numbers {min_n}-{max_n} | '
    title += f"{len(unique_nums)} numbers | Types: {', '.join(set(unique_categories))}</b>"
    fig.update_layout(title=title, title_font=dict(color='white'))

    # Create correlation matrix
    if geometric_data:
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
        corr_fig.update_layout(title_font=dict(color='white'), paper_bgcolor='#121212',
                              font=dict(color='white'), coloraxis_colorbar=dict(title="Correlation"))
    else:
        corr_fig = go.Figure()

    # Create cluster plot
    cluster_fig = go.Figure()

    return fig, corr_fig, cluster_fig

@app.callback(
    Output('dummy-output', 'children'),
    Input('research-visualization', 'figure')
)
def dummy_callback(fig):
    return ""

if __name__ == '__main__':
    app.run(debug=True, port=8050)
