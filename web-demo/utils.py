import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go


def get_timeseries_xaxis():
    xs = []
    day = 6
    hour = 0
    minute = 24
    for i in range(8 * 60):
        minute += 24
        if minute >= 60:
            minute = minute % 60
            hour += 1
        if hour >= 24:
            hour = hour % 24
            day += 1
        xs.append('07/%.2d %.2d:%.2d' % (day, hour % 24, minute % 60))
    xs_val = [0, 60, 120, 180, 240, 300, 360, 420]
    xs_labels = ['07/06<br>(TUE)', '07/07<br>(WED)', '07/08<br>(THU)', '07/09<br>(FRI)', '07/10<br>(SAT)',
                 '07/11<br>(SUN)', '07/12<br>(MON)', 'NOW']
    return xs, xs_val, xs_labels


def description_card():
    return html.Div(
        id="description-card",
        children=[
            html.H5("Machine Learning for Systems"),
            html.H3("Deep Learning for API-Aware Resource Estimation"),
            html.Div(id="intro",
                     children="This project designs a machine learning algorithm, quantile recurrent neural networks, "
                              "with distributed traces to understand how each API in an API-driven system utilizes "
                              "resources in different microservices. The demonstration showcases one of the use cases "
                              "of this project: \"what-if\" analysis given a hypothetical API traffic. We use a social "
                              "network as an example. It supports three API calls: (i) /composePost, (ii) /uploadMedia, "
                              "and (iii) /readTimeline. We can tell the application owner how to allocate minimum "
                              "resources for each component to serve the specified traffic."),
        ],
    )


def generate_control_card(dataloader):
    return html.Div(
        id="control-card", style={'width': '100%'},
        children=[
            html.P("Load Shape"),
            dcc.Dropdown(
                id='dropdown-load-shape',
                options=dataloader.get_options_shape(), searchable=False, clearable=False
            ),
            html.Div(className='desc', children='Description: Select the load shape you want to estimate'),
            html.Br(),
            html.P("Users"),
            dcc.Dropdown(id='dropdown-multiplier', searchable=False, clearable=False),
            html.Div(className='desc', children='Description: Select how many users you want to estimate'),
            html.Br(),
            html.P("API Composition"),
            dcc.Dropdown(id='dropdown-api-composition', searchable=False, clearable=False),
            html.Div(className='desc', children='Description: Select the composition of '
                                                'API calls you want to estimate'),
            html.Br(),
            html.Div(
                id="estimate-btn-outer",
                children=html.Button(id="estimate-btn", children="ESTIMATE", n_clicks=0),
            ),
            html.Br(),
            html.Br(),
            html.Hr(),
            html.Div(className='desc', children=['Research project by Ka-Ho Chow (Georgia Tech) with '
                                                 'Dr. Umesh Deshpande, '
                                                 'Dr. Sangeetha Seshadri, and '
                                                 'Dr. Wil Plouffe (IBM Research - Almaden)']),
        ],
    )


def generate_timeseries_figure(xs, xs_val, xs_labels, metadata, selected_metric):
    y_empty = [0. for _ in range(480)]
    y0 = metadata['utilization'][selected_metric][0] if selected_metric in metadata['utilization'] else y_empty
    y1 = metadata['utilization'][selected_metric][1] if selected_metric in metadata['utilization'] else y_empty
    y2 = metadata['utilization'][selected_metric][2] if selected_metric in metadata['utilization'] else y_empty
    y3 = metadata['utilization'][selected_metric][3] if selected_metric in metadata['utilization'] else y_empty
    y4 = metadata['utilization'][selected_metric][4] if selected_metric in metadata['utilization'] else y_empty
    fig = go.Figure(data=[
        go.Scatter(name='Actual Usage', x=xs, y=y0, line=dict(color="magenta", width=2, dash="dot")),
        go.Scatter(name='BL: Resrc-aware ANN', x=xs[-60:], y=y1, line=dict(color="orange")),
        go.Scatter(name='BL: Simple Scaling', x=xs[-60:], y=y2, line=dict(color="green")),
        go.Scatter(name='BL: API-aware Scaling', x=xs[-60:], y=y3, line=dict(color="blue")),
        go.Scatter(name='Ours: API-aware QRNN', x=xs[-60:], y=y4, line=dict(color="black")),
    ])
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=7. / 8, y0=0,
        x1=1.0, y1=1,
        fillcolor='red', opacity=0.3, line={'width': 0}
    )
    fig['layout'].update(margin=dict(l=30, r=10, b=30, t=30))
    fig.update_layout(height=200, yaxis_title=metadata['unit'][selected_metric],
                      hovermode="x",
                      xaxis=dict(tickmode='array', tickvals=xs_val + [480], ticktext=xs_labels + ['TMR']),
                      xaxis_range=[0, 480],
                      yaxis_range=[max(0, min(min(y0), min(y1), min(y2), min(y3), min(y4)) - 5),
                                   max(10.0, (max(max(y0), max(y1), max(y2), max(y3), max(y4)) +
                                              max(5, max(max(y0), max(y1), max(y2), max(y3), max(y4)) * 0.1)))])
    return fig
