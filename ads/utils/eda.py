import base64
from io import BytesIO
from typing import Iterable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from IPython.display import display, HTML
from matplotlib.offsetbox import AnchoredText
from scipy.stats import probplot
from statsmodels.graphics.gofplots import qqplot


def _num_col_describe(data, col):
    df = pd.DataFrame()
    df.loc["dtype", col] = data[col].dtype
    df.loc["cnt", col] = data[col].count()
    df.loc["ucnt", col] = data[col].nunique(dropna=False)
    df.loc["nans", col] = data[col].isna().sum()
    df.loc["mean", col] = data[col].mean()
    df.loc["std", col] = data[col].std()
    df.loc["min", col] = data[col].min()
    df.loc["25%", col] = data[col].quantile(q=0.25, interpolation='linear')
    df.loc["50%", col] = data[col].quantile(q=0.5, interpolation='linear')
    df.loc["75%", col] = data[col].quantile(q=0.75, interpolation='linear')
    df.loc["max", col] = data[col].max()
    vals = str(data[col].value_counts(dropna=False).sort_values(ascending=False).index.tolist())
    df.loc["vals", col] = vals
    return df


def _num_col_plots_regression(data, col, y):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 5))
    sns.scatterplot(x=data.iloc[:len(y)][col], y=y, ax=ax[0, 0])
    sns.regplot(x=data.iloc[:len(y)][col], y=y, ax=ax[0, 0], scatter=False)
    # ax[0, 0].set_title(scipy.stats.pearsonr(data.iloc[:len(y)][col], y))  # TODO add correlation coefficient
    sns.histplot(data[col].dropna(), ax=ax[0, 1])
    text = f'Skewness: {data[col].dropna().skew():.2f}, Kurtosis: {data[col].dropna().kurt():.2f}'
    ax[0, 1].set_title(text)
    scipy.stats.probplot(data[col].dropna(), dist="norm", plot=ax[1, 0], fit=True, rvalue=True)
    sns.boxplot(data[col].dropna(), ax=ax[1, 1], orient='h')
    return fig


def _num_col_plots_classification(data, col, y):  # TODO complete the function
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 5))
    sns.scatterplot(x=data.iloc[:len(y)][col], y=y, ax=ax[0, 0])
    sns.regplot(x=col, y=y.name, ax=ax[0, 0], data=data.iloc[:len(y)], scatter=False)
    cats = y.unique()
    for cat in cats:
        sns.histplot(data.iloc[:len(y)].loc[y == cat][col].dropna(), ax=ax[0, 1], label=cat)
    ax[0, 1].legend()
    text = f'Skewness: {data[col].dropna().skew():.2f}, Kurtosis: {data[col].dropna().kurt():.2f}'
    ax[0, 1].set_title(text)
    scipy.stats.probplot(data[col].dropna(), dist="norm", plot=ax[1, 0], fit=True, rvalue=True)
    sns.boxplot(data[col].dropna(), ax=ax[1, 1], orient='h')
    return fig


def display_num_col_info(data, col, y, problem, n=1):
    current_backend = plt.get_backend()
    df = _num_col_describe(data, col)
    # pd.set_option('max_colwidth', 40)
    if problem == 'regression':
        fig = _num_col_plots_regression(data, col, y)
    else:
        fig = _num_col_plots_classification(data, col, y)
    plt.tight_layout()

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = f"""<h3>{n} - num - {col}</h3>
        <table>
            <tr>
                <td><img src='data:image/png;base64,{encoded}'></td>
                <td width="60%">{df.to_html()}<td/>
            </tr>
        """
    plt.switch_backend('Agg')
    display(HTML(html))
    plt.switch_backend(current_backend)


def display_cat_col_info(data, col, y, problem, n=1):
    current_backend = plt.get_backend()
    df = pd.DataFrame()
    df.loc["dtype", col] = data[col].dtype
    df.loc["cnt", col] = data[col].count()
    df.loc["ucnt", col] = data[col].nunique(dropna=False)
    df.loc["nans", col] = data[col].isna().sum()
    vals = str(data[col].value_counts(dropna=False).sort_values(ascending=False).index.tolist())
    df.loc["vals", col] = vals

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 5))
    medians = y.groupby(data[col].iloc[:len(y)]).median().sort_values(ascending=False)
    counts = y.groupby(data[col].iloc[:len(y)]).count().sort_values(ascending=False)
    ax[0, 0].set_ylabel('median')
    sns.barplot(x=medians.index, y=medians.values, ax=ax[0, 0], order=medians.index)
    sns.countplot(x=col, data=data, ax=ax[0, 1], order=counts.index)

    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = f"""<h3>{n} - cat - {col}</h3>
        <table>
            <tr>
                <td><img src='data:image/png;base64,{encoded}'></td>
                <td>{df.to_html()}<td/>
            </tr>
        """
    plt.switch_backend('Agg')
    display(HTML(html))
    plt.switch_backend(current_backend)


def custom_describe(df: pd.DataFrame, y: pd.Series, problem, skip_cols: Iterable = None, var_types: dict = None):
    """
    :param df:
    :param y:
    :param skip_cols: list of col names
    :param var_types: {'col_name1': 'num', 'col_name2': 'cat'}
    :return:
    """
    col2type = {col: 'num' if df[col].dtype != object else 'cat' for col in df.columns}
    if var_types is not None:
        col2type.update(var_types)

    n = 0
    for col in [col for col, tp in col2type.items() if tp == 'num']:
        if skip_cols and col in skip_cols:
            continue
        n += 1
        try:
            display_num_col_info(df, col, y, problem, n)
        except:
            print(f'Failed to render {col}')

    for col in [col for col, tp in col2type.items() if tp == 'cat']:
        if skip_cols and col in skip_cols:
            continue
        n += 1
        try:
            display_cat_col_info(df, col, y, problem, n)
        except:
            print(f'Failed to render {col}')


def unique_columns(data, dropna=False):
    counts = data.nunique(dropna=dropna)
    return counts.loc[counts == data.shape[0]].index.tolist()


def constant_columns(data, dropna=False):
    counts = data.nunique(dropna=dropna)
    return counts.loc[counts == 1].index.tolist()


def with_nans_columns(data):
    nan_count = data.isna().sum(axis=0)
    return nan_count.loc[nan_count > 0].index.tolist()


def plot_continuous_distribution(a):
    print("Skewness:", a.skew())
    print("Kurtosis", a.kurt())
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    sns.histplot(a, kde=False)


def nans_plot(data, nan_column=None):
    df = pd.DataFrame(columns=["row", "col", "val"])
    if nan_column:
        cols_to_show = data.columns[data.isna().sum(axis=0) > 0]
        rows_to_show = data[nan_column].isna()
        cases = {
            "NA Values": data.loc[rows_to_show, cols_to_show].isin([None, np.nan]).values,
            "Empty String": (data == "").values,
            "-1": (data == -1).values
        }
        for case_name, case_vals in cases.items():
            tmp_df = pd.DataFrame()
            tmp_df["row"] = np.nonzero(case_vals)[0]
            tmp_df["col"] = cols_to_show[np.nonzero(case_vals)[1]]
            tmp_df["val"] = case_name
            df = pd.concat([df, tmp_df])

        df["cnt"] = df.col.map(df.col.value_counts())
        df = df.sort_values(by=["cnt", "col"], ascending=False)
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.scatterplot(x="col", y="row", data=df, ax=ax)
        new_labels = ["%s: %d" % (col, cnt) for col, cnt in df.col.value_counts().sort_values(ascending=False).iteritems()]
        ax.set_xticklabels(new_labels, rotation=60, fontsize=12)
    else:
        df['row'] = np.repeat(range(data.shape[0]), data.shape[1])
        df['col'] = np.tile(range(data.shape[1]), data.shape[0])
        df['val'] = pd.isna(data.values.ravel())
        new_labels = []
        for col in data.columns:
            nan_cnt = pd.isna(data[col]).sum()
            nan_rate = round(nan_cnt * 100/ data.shape[0], 1)
            label = f'{col}: {nan_cnt} / {nan_rate}%'
            new_labels.append(label)
        fig, ax = plt.subplots(figsize=(25, 20))
        sns.scatterplot(x="row", y="col", data=df, ax=ax, hue='val')
        ax.set_yticks(range(data.shape[1]))
        ax.set_yticklabels(new_labels, rotation=0, fontsize=10)


def nan_stat(data):
    return f'{100 * data.isna().values.sum() / data.size:.2f}% of the data is missing'


def correlation_heatmap(data: pd.DataFrame, target: pd.Series, lower=0.1, upper=1.0):
    df = pd.concat([data, target], axis=1)
    ordered_cols = df.corr().abs()[target.name].sort_values(ascending=False).index
    # ordered_cols = df.corr().abs().mean(axis=0).sort_values(ascending=False).index
    corr = df.loc[:, ordered_cols].corr()
    # fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    # sns.heatmap(corr, annot=True, annot_kws={"size": 25}, fmt='.1f', cmap='PiYG', lidths=.5)
    # corr = data.corr()
    sns.set(font_scale=2)
    plt.figure(figsize=(50, 35))
    sns.heatmap(corr, annot=True, annot_kws={"size": 25}, fmt='.1f', cmap='PiYG', linewidths=.5)
    important_cols = df.corr().abs()[target.name]
    important_cols = set(important_cols.loc[(important_cols >= lower) & (important_cols <= upper)].index)
    important_cols -= set(target.name)
    return plt, important_cols


def cols_info_df(data):
    df = pd.DataFrame(index=data.columns)
    for col in data.columns:
        df.loc[col, 'type'] = data[col].dtype
        df.loc[col, 'len'] = data.shape[0]
        df.loc[col, '#unique'] = data[col].nunique()
        df.loc[col, '%unique'] = round(df.loc[col, '#unique'] / df.loc[col, 'len'], 2)
        df.loc[col, '#nans'] = pd.isna(data[col]).sum()
        df.loc[col, '%nans'] = round(df.loc[col, '#nans'] * 100 / df.loc[col, 'len'], 1)
        df.loc[col, '%values'] = str(data[col].value_counts(normalize=True, dropna=False).round(2).to_dict())
        df.loc[col, 'used'] = False
        df.loc[col, 'reason'] = None
    return df


def is_121_mapped(data, col1, col2):
    unique = data[[col1, col2]].drop_duplicates()
    col1_to_col2 = dict(zip(unique[col1].values, unique[col2].values))
    ret = data[col1].map(col1_to_col2) == data[col2]
    return ret.sum() == data.shape[0]



def correlation_matrix(df: pd.DataFrame):
    """
    A function to calculate and plot
    correlation matrix of a DataFrame.
    """
    # Create the matrix
    matrix = df.corr()

    # Create cmap
    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                                 n=9, center="light", as_cmap=True)
    # Create a mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    # Make figsize bigger
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot the matrix
    _ = sns.heatmap(matrix, mask=mask, center=0, annot=True,
                    fmt='.2f', square=True, cmap=cmap, ax=ax)