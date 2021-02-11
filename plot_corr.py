import pandas as pd
import plotly.graph_objs as go
corr_rand = pd.read_csv('./corr_random.csv', index_col=0)
corr_cons = pd.read_csv('./corr_cons.csv', index_col=0)
corr_clean_rand = pd.read_csv('./corr_clean_random.csv', index_col=0)
corr_clean_cons = pd.read_csv('./corr_clean_cons.csv', index_col=0)


def get_subset(corr, substring):
    """
    param:corr df
    param :substring, can be swvVar/dmed format or swv, subsets the output from hminsc
    """
    df = corr[(corr['column'].str.contains(substring)) & (corr['row'] != 'catRigidity')]
    df = df[~df['row'].str.contains('Var')]
    df = df[~df['row'].str.contains('med')]
    return df


def plot_sig_results(sig_df, substring):
    """
    INPUT: df of results, significant features


    OUTPUT: graphs
    """

    fig = go.Figure()

    fig.add_trace(go.Bar(x=sig_df['row'], y=sig_df['cor'], name=substring,
                         marker=dict(color=sig_df['p'],
                                     colorscale='bluered',  # choose a colorscale
                                     opacity=0.8,
                                     showscale=True
                                     )))

    fig.update_layout(
        title=substring,
        xaxis_title="Baseline Traits",
        yaxis_title="Corr, colored by P-value")
    return fig


def plot_many(corr_df, possible):
    graph_ls = []

    for i in possible:
        new_df = get_subset(corr_df, i)
        new_fig = plot_sig_results(new_df, i)
        graph_ls.append(new_fig)

    return graph_ls


def multiplot(substring):
    """
    INPUT: df of results, significant features


    OUTPUT: graphs
    """

    fig = go.Figure()

    corr_rand_sub = get_subset(corr_rand, substring)
    fig.add_trace(go.Bar(x=corr_rand_sub['row'], y=corr_rand_sub['cor'], name='corr_rand',
                         marker=dict(color='Crimson',
                                     showscale=False
                                     )))

    corr_clean_rand_sub = get_subset(corr_clean_rand, substring)
    fig.add_trace(go.Bar(x=corr_clean_rand_sub['row'], y=corr_clean_rand_sub['cor'], name='corr_clean_rand',
                         marker=dict(color='Pink',
                                     showscale=False
                                     )))

    corr_cons_sub = get_subset(corr_cons, substring)
    fig.add_trace(go.Bar(x=corr_cons_sub['row'], y=corr_cons_sub['cor'], name='corr_cons',
                         marker=dict(color='Blue',
                                     showscale=False
                                     )))

    corr_clean_cons_sub = get_subset(corr_clean_cons, substring)
    fig.add_trace(go.Bar(x=corr_clean_cons_sub['row'], y=corr_clean_cons_sub['cor'], name='corr_clean_cons',
                         marker=dict(color='LightBlue',
                                     showscale=False
                                     )))

    fig.update_layout(
        title="Variation x Baseline Traits" + substring.capitalize(),
        xaxis_title="Baseline Traits",
        yaxis_title="Correlation with Larger Variation")
    return fig

