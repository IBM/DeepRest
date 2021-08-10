from dash.dependencies import Input, Output, State
from dataloader import DataLoader
from utils import *
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import numpy as np
import dash

########################################################################################################################
# Dash-related setup
########################################################################################################################
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
app.title = "Deep Learning for API-aware Resource Estimation"
server = app.server
app.config.suppress_callback_exceptions = True

########################################################################################################################
# Read data
########################################################################################################################
dataloader = DataLoader(path='assets/results.pkl')
components = ['nginx-thrift', 'compose-post-service', 'post-storage-service', 'post-storage-mongodb',
              'user-timeline-service', 'user-timeline-mongodb', 'media-frontend', 'media-mongodb']

########################################################################################################################
# Prepare for the static figure (application learning traffic)
########################################################################################################################
app_learning_traffic = dataloader.get_learning_traffic()
xs, xs_val, xs_labels = get_timeseries_xaxis()

fig = go.Figure()
fig['layout'].update(margin=dict(l=30, r=10, b=30, t=30))
fig.add_trace(go.Scatter(x=xs, y=app_learning_traffic['ALL'],
                         name='ALL', line=dict(color='royalblue', dash='dot')))
fig.add_trace(go.Scatter(x=xs, y=app_learning_traffic['/composePost'],
                         name='/composePost', line=dict(color='firebrick')))
fig.add_trace(go.Scatter(x=xs, y=app_learning_traffic['/uploadMedia'],
                         name='/uploadMedia', line=dict(color='forestgreen')))
fig.add_trace(go.Scatter(x=xs, y=app_learning_traffic['/readTimeline'],
                         name='/readTimeline', line=dict(color='salmon')))
fig.update_traces(hovertemplate=None)
fig.update_layout(xaxis_title='Timeline', yaxis_title='Requests per Second', height=250, hovermode="x",
                  xaxis=dict(tickmode='array', tickvals=xs_val, ticktext=xs_labels),
                  yaxis=dict(tickmode='array', tickvals=list(range(0, 60, 5)), ticktext=list(range(0, 60, 5))),
                  yaxis_range=[0, 55], xaxis_range=[0, 420])


########################################################################################################################
# Start the UI script and callbacks
########################################################################################################################
app.layout = html.Div(
    id="app-container",
    children=[
        html.Div(
            id="banner", className="banner",
            children=[html.Img(src=app.get_asset_url("ibm.png"), style={'height': '24px'})],
        ),
        html.Div(
            id="left-column", className="three columns",
            children=[description_card(), generate_control_card(dataloader)]
        ),
        dcc.Loading(
            id="loading-0", type="default", fullscreen=True,
            children=html.Div(id="loading-output-0")
        ),
        html.Div(
            id="middle-column", className="five columns",
            children=[
                html.Div(
                    id="api_traffic_learning_card",
                    children=[
                        html.B("API Traffic - Application Learning"),
                        html.Hr(),
                        dcc.Graph(figure=fig),
                    ],
                )
            ],
        ),
        html.Div(
            id="right-column", className="four columns",
            children=[
                html.Div(
                    id="api_traffic_query_card",
                    children=[
                        html.B("Hypothetical API Traffic - Query (From Your Config Input)"),
                        html.Hr(),
                        dcc.Graph(id='linechart-query-traffic')
                    ],
                )
            ],
        ),
        html.Div(
            id="bottom-column", className="nine columns",
            children=[
                html.Div(
                    id="resource_estimation_card",
                    children=[
                        html.B("Resource Estimation"),
                        html.Hr(),
                        html.Div(className='component', children=[
                            html.Div(className='component-logo'),
                            html.Div(className='component-scale'),
                            html.Div(id="metric-selector", className='component-ts', style={'text-align': 'left', 'display': 'none'}, children=[
                                html.Div(style={'display': 'inline-block', 'float': 'left', 'font-weight': 'bold'},
                                         children='Visualize: '),
                                html.Div(style={'display': 'inline-block'},
                                         children=dcc.RadioItems(id='radio-visualize', options=[
                                             {'label': 'CPU', 'value': 'cpu'},
                                             {'label': 'Memory', 'value': 'memory'},
                                             {'label': 'IOps', 'value': 'write-iops'},
                                             {'label': 'Throughput', 'value': 'write-tp'},
                                             {'label': 'Disk Usage', 'value': 'usage'}], value='cpu',
                                                                 labelStyle={'display': 'inline-block'}))])
                        ]),
                        dcc.Loading(id="loading-1", type="default", children=html.Div(id="loading-output-1")),
                        html.Div(id='div-resrc-estimation')
                    ],
                )
            ],
        ),
    ],
)


@app.callback(
    [Output("div-resrc-estimation", "children"), Output("metric-selector", "style"), Output("loading-output-0", "children")],
    [Input("estimate-btn", "n_clicks")],
    [State("dropdown-load-shape", "value"),
     State("dropdown-multiplier", "value"),
     State("dropdown-api-composition", "value"),
     State('radio-visualize', 'value')]
)
def click_estimate(est_click, selected_load_shape, selected_multiplier, selected_composition, selected_metric):
    if selected_load_shape is None or selected_multiplier is None or selected_composition is None:
        return [], {'text-align': 'left', 'display': 'none'}, []
    component2metrics = dataloader.get_component2metrics(selected_load_shape, selected_multiplier, selected_composition)

    children = []
    for component in components:
        metadata = component2metrics[component]

        metric_names = ['cpu', 'memory', 'write-iops', 'write-tp', 'usage']
        metric_names_show = {
            'cpu': 'CPU',
            'memory': 'Memory',
            'write-iops': 'IOps',
            'write-tp': 'Throughput',
            'usage': 'Disk<br>Usage'
        }
        fig_bar = go.Figure(data=[
            go.Bar(name='BL: Resrc-aware ANN', x=[metric_names_show[m] for m in metric_names],
                   y=[metadata['scale'][k][1] for k in metric_names], marker_color="orange"),
            go.Bar(name='BL: Regr w/o Traces', x=[metric_names_show[m] for m in metric_names],
                   y=[metadata['scale'][k][2] for k in metric_names], marker_color="green"),
            go.Bar(name='BL: Regr w/ Traces', x=[metric_names_show[m] for m in metric_names],
                   y=[metadata['scale'][k][3] for k in metric_names], marker_color="blue"),
            go.Bar(name='Ours: QRNN', x=[metric_names_show[m] for m in metric_names],
                   y=[metadata['scale'][k][4] for k in metric_names], marker_color="black"),
        ])
        fig_bar['layout'].update(margin=dict(l=30, r=10, b=30, t=30))
        fig_bar.update_layout(barmode='group',
                              height=200,
                              yaxis_title='Scaling Factor'
                              )
        fig_bar.add_shape(type="line", xref="paper", yref="y", x0=0, y0=metadata['scale']['cpu'][0],
                          x1=0.20, y1=metadata['scale']['cpu'][0],
                          line=dict(color="magenta", width=2, dash="dot"))
        fig_bar.add_shape(type="line", xref="paper", yref="y", x0=0.20, y0=metadata['scale']['memory'][0],
                          x1=0.40, y1=metadata['scale']['memory'][0],
                          line=dict(color="magenta", width=2, dash="dot"))
        if metadata['scale']['write-iops'][0] != 0.0:
            fig_bar.add_shape(type="line", xref="paper", yref="y", x0=0.40, y0=metadata['scale']['write-iops'][0],
                              x1=0.60, y1=metadata['scale']['write-iops'][0],
                              line=dict(color="magenta", width=2, dash="dot"))
            fig_bar.add_shape(type="line", xref="paper", yref="y", x0=0.60, y0=metadata['scale']['write-tp'][0],
                              x1=0.80, y1=metadata['scale']['write-tp'][0],
                              line=dict(color="magenta", width=2, dash="dot"))
            fig_bar.add_shape(type="line", xref="paper", yref="y", x0=0.80, y0=metadata['scale']['usage'][0],
                              x1=1.00, y1=metadata['scale']['usage'][0],
                              line=dict(color="magenta", width=2, dash="dot"))

        fig_line = generate_timeseries_figure(xs, xs_val, xs_labels, metadata, selected_metric)
        children.append(
            html.Div(className='component', children=[
                html.Div(className='component-logo', children=[
                    html.Img(src=metadata['icon_path'], style={'margin-top': '30px', 'width': '80px'}),
                    html.Div(style={'display': 'block'}, children=html.B(metadata['name']))
                ]),
                html.Div(className='component-scale', children=dcc.Graph(figure=fig_bar)),
                html.Div(className='component-ts', children=dcc.Graph(id='graph-%s' % component, figure=fig_line))
            ])
        )
    return children, {'text-align': 'left', 'display': 'block'}, []


@app.callback(
    Output('dropdown-multiplier', 'options'),
    [Input('dropdown-load-shape', 'value')])
def set_load_shape(selected_load_shape):
    minmax = dataloader.get_options_multiplier(selected_load_shape)
    return [{'label': '%dx more users' % d, 'value': d} for d in range(minmax[0], minmax[1]+1)]


@app.callback(
    [Output('graph-%s' % component, 'figure') for component in components] + [Output("loading-output-1", "children")],
    [Input('radio-visualize', 'value')],
    [State("dropdown-load-shape", "value"),
     State("dropdown-multiplier", "value"),
     State("dropdown-api-composition", "value")])
def set_visualize_metric(selected_metric, selected_load_shape, selected_multiplier, selected_composition):
    if selected_load_shape is None or selected_multiplier is None or selected_composition is None:
        return [[] for _ in range(len(components)+1)]
    component2metrics = dataloader.get_component2metrics(selected_load_shape, selected_multiplier, selected_composition)
    figures = []
    for component in components:
        metadata = component2metrics[component]
        fig_line = generate_timeseries_figure(xs, xs_val, xs_labels, metadata, selected_metric)
        figures.append(fig_line)
    return figures + [[]]


@app.callback(
    Output('dropdown-api-composition', 'options'),
    [Input('dropdown-multiplier', 'value')],
    [State("dropdown-load-shape", "value")])
def set_multiplier(selected_multiplier, selected_load_shape):
    compositions = dataloader.get_options_composition(selected_load_shape, selected_multiplier)
    compositions = sorted(compositions, key=lambda x: (-max(x), np.argmax(x)))
    options = [{'label': '/compose: %d%% | /upload: %d%% | /read: %d%%' % composition,
                'value': '_'.join(map(str, composition))} for composition in compositions]

    return options


@app.callback(
    Output('linechart-query-traffic', 'figure'),
    [Input('dropdown-api-composition', 'value')],
    [State("dropdown-multiplier", "value"), State("dropdown-load-shape", "value")])
def set_composition(selected_composition, selected_multiplier, selected_load_shape):
    xs_ = []
    hour = 00
    minute = 00
    for i in range(61):
        xs_.append('%.2d:%.2d' % (hour, minute))
        minute += 24
        if minute >= 60:
            minute %= 60
            hour += 1
    fig = go.Figure()
    fig['layout'].update(margin=dict(l=30, r=10, b=30, t=30))
    fig.update_traces(hovertemplate=None)
    fig.update_layout(xaxis_title='Timeline', yaxis_title='Requests per Second',
                      xaxis=dict(tickmode='array', tickvals=list(range(0, len(xs_), 3)),
                                 ticktext=[xs_[i] for i in range(0, len(xs_), 3)]),
                      yaxis=dict(tickmode='array', tickvals=list(range(0, 60, 5)),
                                 ticktext=list(range(0, 60, 5))),
                      height=250,
                      hovermode="x",
                      yaxis_range=[0, 55],
                      xaxis_range=[0, 59]
                      )
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0., y0=0,
        x1=1.0, y1=1,
        fillcolor='red', opacity=0.3, line={'width': 0}
    )
    if selected_composition is None:
        return fig
    traffic = dataloader.get_query_traffic(selected_load_shape, selected_multiplier, selected_composition)

    fig.add_trace(go.Scatter(x=xs_, y=traffic['ALL'], name='ALL', line=dict(color='royalblue', dash='dot')))
    fig.add_trace(go.Scatter(x=xs_, y=traffic['/composePost'], name='/composePost', line=dict(color='firebrick')))
    fig.add_trace(go.Scatter(x=xs_, y=traffic['/uploadMedia'], name='/uploadMedia', line=dict(color='forestgreen')))
    fig.add_trace(go.Scatter(x=xs_, y=traffic['/readTimeline'], name='/readTimeline', line=dict(color='salmon')))
    return fig


# Run the server
if __name__ == "__main__":
    app.run_server(port=8050, host='0.0.0.0')
