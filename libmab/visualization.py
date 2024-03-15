import numpy as np
from matplotlib import pyplot as plt

markers = ["o", "v", "*", "s", "h"]
# red = "#ef476f"
# blue = "#118ab2"


class Colors:
    red = "#C1121F"
    blue = "#3A86FF"
    green = "#76c893"  # "#06d6a0"
    orange = "#ffd166"
    dblue = "#073b4c"


def plotci(ax, x, E, exps, color, label, marker="*"):
    y = np.mean(np.cumsum(exps, axis=1), axis=0)
    ax.plot(
        x,
        y,
        color=color,
        label=label,
        marker=marker,
        markevery=len(x) // len(ax.get_xticks()),
        rasterized=True,
    )
    ci = 1.96 * np.std(np.cumsum(exps, axis=1), axis=0) / np.sqrt(E)
    ax.fill_between(x, (y - ci), (y + ci), color=color, alpha=0.1)


def plotconfint(ax, x, y, n_exps, color, label, marker="*"):
    ax.plot(
        x,
        y,
        color=color,
        label=label,
        marker=marker,
        markevery=len(x) // len(ax.get_xticks()),
        rasterized=True,
    )
    ci = 1.96 * np.std(y, axis=0) / np.sqrt(n_exps)
    ax.fill_between(x, (y - ci), (y + ci), color=color, alpha=0.1)


def get_ci(exps, n_exps):
    ci = 1.96 * np.std(exps) / np.sqrt(n_exps)
    return ci


def getconfint(exps, n_exps):
    ci = 1.96 * np.std(exps, axis=0) / np.sqrt(n_exps)
    y = np.mean(exps, axis=0)
    return y, (y - ci), (y + ci)


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(
                x + x_offset,
                y,
                width=bar_width * single_width,
                color=colors[i % len(colors)],
            )

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())
