from plotly.subplots import make_subplots
import plotly.graph_objects as go


def fig_tm(t_matrix, ui, ui_freq):
    ''' Returns a figure with pitch probabilities and transition probabilites
        based on acomputed t_matrix, its row labels (ui) and how often the
        pitches occurred in the input data
    '''
    
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]],
                    subplot_titles = ('Pitch probability', 'Transition probability'))

    trace1 = go.Bar(x=ui, y=ui_freq)
    trace2 = go.Heatmap(x=ui, y=ui, z=t_matrix, colorscale='Viridis')
    
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)

    return fig


