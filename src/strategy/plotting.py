from typing import List
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px


def plot_sweep_results(
        sweep_results: pd.DataFrame,
        parameters: List[str],
        objective: str = 'value',
        top_n: int = 5,
        round: int = 2,
        title: str = "Hyperparameters search results"):
    """Helper function for plotting results of hyperparameter search."""
    data = sweep_results[list(parameters) + [objective]].round(round)

    fig = ff.create_table(
        data.sort_values(
            objective,
            ascending=False).head(top_n),
        height_constant=80)
    fig.layout.yaxis.update({'domain': [0, .4]})

    parcoords = px.parallel_coordinates(
        data,
        color=objective,
        color_continuous_midpoint=1.0,
        color_continuous_scale=px.colors.diverging.Tealrose_r)
    parcoords.data[0].domain.update({'x': [0.05, 0.8], 'y': [0.5, 0.90]})

    fig.add_trace(parcoords.data[0])
    fig.layout.update({'coloraxis': parcoords.layout.coloraxis})
    fig.update_layout(coloraxis_colorbar=dict(
        yanchor="top",
        xanchor='right',
        y=1,
        x=0.95,
        len=0.5,
        thickness=40,
    ))

    fig.layout.margin.update({'l': 20, 'r': 20, 'b': 20, 't': 40})
    fig.update_layout(
        title={
            'text': title,
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 28
            }}
    )

    return fig
