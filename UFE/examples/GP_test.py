import pandas as pd
import numpy as np
from numpy.random import multivariate_normal
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel, RBF, RationalQuadratic
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set_style(
    style='darkgrid',
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

def simple_plot(data_df: pd.DataFrame, y_lab: str, titleTxt: str):
    fig, ax = plt.subplots()
    sns.lineplot(x='t', y=y_lab, data=data_df, color=sns_c[0], label=y_lab, ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set(title=titleTxt, xlabel='t', ylabel='')
    plt.show()

def update_layout_of_graph(fig: go.Figure, title: str = 'Plot') -> go.Figure:
    fig.update_layout(
        width=800,
        height=600,
        autosize=False,
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,

    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                      xaxis_title='input values',
                      yaxis_title='output values',
                      legend=dict(yanchor="top",
                                  y=0.9,
                                  xanchor="right",
                                  x=0.95),
                      title={
                          'x': 0.5,
                          'xanchor': 'center'
                      })
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    return fig

def dot_scatter(
    visible: bool = True,
    x_dots: np.array = np.array([]),
    y_dots: np.array = np.array([]),
    name_dots: str = 'Observed points',
    showlegend: bool = True
) -> go.Scatter:
    # Adding the dots
    return go.Scatter(
        x=x_dots,
        visible=visible,
        y=y_dots,
        mode="markers",
        name=name_dots,
        marker=dict(color='red', size=8),
        showlegend=showlegend
    )



def uncertainty_area_scatter(
        visible: bool = True,
        x_lines: np.array = np.array([]),
        y_upper: np.array = np.array([]),
        y_lower: np.array = np.array([]),
        name: str = "mean plus/minus standard deviation",
) -> go.Scatter:

    return go.Scatter(
        visible=visible,
        x=np.concatenate((x_lines, x_lines[::-1])),  # x, then x reversed
        # upper, then lower reversed
        y=np.concatenate((y_upper, y_lower[::-1])),
        fill='toself',
        fillcolor='rgba(189,195,199,0.5)',
        line=dict(color='rgba(200,200,200,0)'),
        hoverinfo="skip",
        showlegend=True,
        name= name,
    )

def add_slider_GPR(figure: go.Figure, parameters):
    figure.data[0].visible = True
    figure.data[1].visible = True

    # Create and add slider
    steps = []
    for i in range(int((len(figure.data) - 1) / 2)):
        step = dict(
            method="update",
            label=f'{parameters[i]: .2f}',
            args=[{
                "visible": [False] * (len(figure.data) - 1) + [True]
            }],
        )
        step["args"][0]["visible"][2 *
                                   i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][2 * i + 1] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        pad={"t": 50},
        steps=steps,
    )]
    figure.update_layout(sliders=sliders, )
    return figure

def add_slider_to_function(figure: go.Figure, parameters):
    figure.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(figure.data)):
        step = dict(
            method="update",
            label=f'{parameters[i]: .2f}',
            args=[{
                "visible": [False] * len(figure.data)
            }],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        pad={"t": 50},
        steps=steps,
    )]
    figure.update_layout(sliders=sliders, )
    return figure
########################### import data  ###########################
Y_df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/Y_df.csv', index_col=1) # cols = (index), unique_id, ds,y
n=len(Y_df.loc['onfido'])
t = np.arange(n)
data_df = pd.DataFrame({'t' : t})
data_df['y1'] = Y_df.loc['onfido/0010800003Hy4TeAAJ/facial_similarity_report_motion']['y'].values
# scale and normalise Y
data_df['y1'] = data_df['y1']/10

simple_plot(data_df, y_lab ='y1', titleTxt='Onfido Agg')

########################### Define the Kernels  ###########################
k0 = WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))
k1 = ConstantKernel(constant_value=3)+ConstantKernel(constant_value=2) * ExpSineSquared(length_scale=1.0, periodicity=7, periodicity_bounds=(5, 10))
k2 = ConstantKernel(constant_value=1, constant_value_bounds=(1, 5)) * RationalQuadratic(length_scale=50, length_scale_bounds=(1, 1e2), alpha= 50.0, alpha_bounds=(1, 1e3))
kernel_2  = k0 + k1 + k2

gp1 = GaussianProcessRegressor(
    kernel=kernel_2,
    n_restarts_optimizer=10,
    normalize_y=True,
    alpha=0.0
)

########################### Define Data  ###########################
X = data_df['t'].values.reshape(n, 1)
y = data_df['y1'].values.reshape(n, 1)

prop_train = 0.7
n_train = round(prop_train * n)

X_train = X[:n_train]
y_train = y[:n_train]

X_test = X[n_train:]
y_test = y[n_train:]

gp1_prior_samples = gp1.sample_y(X=X_train, n_samples=100)

fig, ax = plt.subplots()
for i in range(100):
    sns.lineplot(x=X_train[...,0], y = gp1_prior_samples[:, i], color=sns_c[1], alpha=0.2, ax=ax)
sns.lineplot(x=X_train[...,0], y=y_train[..., 0], color=sns_c[0], label='y1', ax=ax)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set(title='GP1 Prior Samples', xlabel='t')
plt.show()


########################### Fit and Predict  ###########################
gp1.fit(X_train, y_train)

# Generate predictions.
y_pred, y_std = gp1.predict(X, return_std=True)

data_df['y_pred'] = y_pred
data_df['y_std'] = y_std
data_df['y_pred_lwr'] = data_df['y_pred'] - 2*data_df['y_std']
data_df['y_pred_upr'] = data_df['y_pred'] + 2*data_df['y_std']

fig, ax = plt.subplots()

ax.fill_between(
    x=data_df['t'],
    y1=data_df['y_pred_lwr'],
    y2=data_df['y_pred_upr'],
    color=sns_c[2],
    alpha=0.15,
    label='credible_interval'
)

sns.lineplot(x='t', y='y1', data=data_df, color=sns_c[0], label = 'y1', ax=ax)
sns.lineplot(x='t', y='y_pred', data=data_df, color=sns_c[2], label='y_pred', ax=ax)

ax.axvline(n_train, color=sns_c[3], linestyle='--', label='train-test split')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set(title='Prediction Sample 1', xlabel='t', ylabel='')
plt.show()

print(f'R2 Score Train = {gp1.score(X=X_train, y=y_train): 0.3f}')
print(f'R2 Score Test = {gp1.score(X=X_test, y=y_test): 0.3f}')
print(f'MAE Train = {mean_absolute_error(y_true=y_train, y_pred=gp1.predict(X_train)): 0.3f}')
print(f'MAE Test = {mean_absolute_error(y_true=y_test, y_pred=gp1.predict(X_test)): 0.3f}')

errors = gp1.predict(X_test) - y_test
errors = errors.flatten()
errors_mean = errors.mean()
errors_std = errors.std()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.regplot(x=y_test.flatten(), y=gp1.predict(X_test).flatten(), ax=ax[0])
sns.histplot(x=errors, ax=ax[1])
ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--', label=f'$\mu$')
ax[1].axvline(x=errors_mean + 2*errors_std, color=sns_c[4], linestyle='--', label=f'$\mu \pm 2\sigma$')
ax[1].axvline(x=errors_mean - 2*errors_std, color=sns_c[4], linestyle='--')
ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--')
ax[1].legend()
ax[0].set(title='Model 1 - Test vs Predictions (Test Set)', xlabel='y_test', ylabel='y_pred');
ax[1].set(title='Model 1  - Errors', xlabel='error', ylabel=None)
plt.show()