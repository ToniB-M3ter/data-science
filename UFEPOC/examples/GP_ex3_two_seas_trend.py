import pandas as pd
import numpy as np
from numpy.random import multivariate_normal
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel, RBF
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

# Number of samples.
n = 1000
# Generate "time" variable.
t = np.arange(n)

data_df = pd.DataFrame({'t' : t})

# Generate seasonal variables.
def seasonal(t, amplitude, period):
    """Generate a sinusoidal curve."""
    y1 = amplitude * np.sin((2*np.pi)*t/period)
    return y1

# Add two seasonal components.
data_df['s1'] = data_df['t'].apply(lambda t : seasonal(t, amplitude=2, period=40))

# Define target variable.
data_df['y1'] = data_df['s1']

# Set noise standard deviation.
sigma_n = 0.3

data_df['epsilon'] = np.random.normal(loc=0, scale=sigma_n, size=n)
# Add noise to target variable.
data_df ['y1'] = data_df ['y1'] + data_df ['epsilon']

simple_plot(data_df, y_lab ='s1', titleTxt='Seasonal Component')
simple_plot(data_df, y_lab ='y1', titleTxt='Sample Data (1): Noise & Season')

# Generate trend component.
def linear_trend(beta, x):
    """Scale vector by a scalar."""
    trend_comp = beta * x
    return trend_comp

data_df['tr1'] = data_df['t'].apply(lambda x : linear_trend(0.01, x))

# Add trend to target variable y_1.
data_df['y2'] = data_df['y1'] + data_df['tr1']

simple_plot(data_df, y_lab ='y2', titleTxt='Sample Data (2): Noise, Season & Trend')

# Create other seasonal component.
data_df['s2'] = data_df['t'].apply(lambda t : seasonal(t, amplitude=1, period=13.3))
# Add to y_2.
data_df['y3'] = data_df['y2'] + data_df['s2']

simple_plot(data_df, y_lab ='y3', titleTxt='Sample Data (3) Noise, 2 Seasons & Trend')

########################### Define the Kernels  ###########################
k0 = WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))
k1 = ConstantKernel(constant_value=2) * \
  ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45))
k2 = ConstantKernel(constant_value=25, constant_value_bounds=(1e-2, 1e3)) * \
  RBF(length_scale=100.0, length_scale_bounds=(1, 1e4))
k3 = ConstantKernel(constant_value=1) * \
  ExpSineSquared(length_scale=1.0, periodicity=12, periodicity_bounds=(10, 15))
kernel_3  = k0 + k1 + k2 + k3

# Define GaussianProcessRegressor object.
gp3 = GaussianProcessRegressor(
    kernel=kernel_3,
    n_restarts_optimizer=10,
    normalize_y=True,
    alpha=0.0
)

########################### Define Data  ###########################
X = data_df['t'].values.reshape(n, 1)
y = data_df['y1'].values.reshape(n, 1)

prop_train = 0.7
n_train = round(prop_train * n)
y = data_df['y3'].values.reshape(n, 1)

X_train = X[:n_train]
X_test = X[:n_train]
y_train = y[:n_train]
y_test = y[n_train:]

gp3_prior_samples = gp3.sample_y(X=X_train, n_samples=100)

fig, ax = plt.subplots()
for i in range(100):
    sns.lineplot(x=X_train[...,0], y = gp3_prior_samples[:, i], color=sns_c[1], alpha=0.2, ax=ax)
sns.lineplot(x=X_train[...,0], y=y_train[..., 0], color=sns_c[0], label='y3', ax=ax)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set(title='GP3 Prior Samples', xlabel='t')
plt.show()


########################### Fit and Predict  ###########################
gp3.fit(X_train, y_train)

GaussianProcessRegressor(alpha=0.0,
                         kernel=WhiteKernel(noise_level=0.09) + 1.41**2 * ExpSineSquared(length_scale=1, periodicity=40) + 5**2 * RBF(length_scale=100) + 1**2 * ExpSineSquared(length_scale=1, periodicity=12),
                         n_restarts_optimizer=10, normalize_y=True)

y_pred, y_std = gp3.predict(X, return_std=True)

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

sns.lineplot(x='t', y='y3', data=data_df, color=sns_c[0], label = 'y3', ax=ax)
sns.lineplot(x='t', y='y_pred', data=data_df, color=sns_c[2], label='y_pred', ax=ax)

ax.axvline(n_train, color=sns_c[3], linestyle='--', label='train-test split')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set(title='Prediction Sample Data 3', xlabel='t', ylabel='')
plt.show()

print(f'R2 Score Train = {gp3.score(X=X_train, y=y_train): 0.3f}')
print(f'R2 Score Test = {gp3.score(X=X_test, y=y_test): 0.3f}')
print(f'MAE Train = {mean_absolute_error(y_true=y_train, y_pred=gp3.predict(X_train)): 0.3f}')
print(f'MAE Test = {mean_absolute_error(y_true=y_test, y_pred=gp3.predict(X_test)): 0.3f}')

errors = gp3.predict(X_test) - y_test
errors = errors.flatten()
errors_mean = errors.mean()
errors_std = errors.std()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.regplot(x=y_test.flatten(), y=gp3.predict(X_test).flatten(), ax=ax[0])
sns.histplot(x=errors, ax=ax[1])
ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--', label=f'$\mu$')
ax[1].axvline(x=errors_mean + 2*errors_std, color=sns_c[4], linestyle='--', label=f'$\mu \pm 2\sigma$')
ax[1].axvline(x=errors_mean - 2*errors_std, color=sns_c[4], linestyle='--')
ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--')
ax[1].legend()
ax[0].set(title='Model 3 - Test vs Predictions (Test Set)', xlabel='y_test', ylabel='y_pred');
ax[1].set(title='Model 3  - Errors', xlabel='error', ylabel=None)
plt.show()
