import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colour import Color
from sklearn.metrics import r2_score

LABELSIZE = 8
LEGENDSIZE = 8
TICKSIZE = 8


def get_markers(n=None):
    # gives a list of markers
    markers = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "s",
        "p",
        "P",
        "*",
        "X",
        "D",
        "d",
        "8",
        "H",
        "h",
        "o",
        "v",
        "^",
        "<",
        ">",
    ]
    if n is None:
        return markers
    else:
        return markers[0:n]


def get_gradient_colors(n, start, end):
    # gives you a color range between start and end
    if isinstance(start, str):
        first_color = Color(start)
    elif isinstance(start, tuple):
        first_color = Color(rgb=start)

    if isinstance(end, str):
        final_color = Color(end)
    elif isinstance(end, tuple):
        final_color = Color(rgb=end)

    colors = list(first_color.range_to(final_color, n))
    colors = [color.rgb for color in colors]

    return colors


def plot_array_data(
    exp_set,
    input_data,
    xlim,
    ylim=None,
    color_dict=None,
    legend=True,
    ax=None,
    linewidth=None,
    linestyle="-",
    label=True,
):
    if ax == None:
        _, ax = plt.subplots()

    for chem in exp_set:
        nums = exp_set[chem]
        data = pd.DataFrame([input_data[n] for n in nums])

        # calculate average
        avg = data.mean()
        std = data.std()

        try:
            ax.plot(
                np.linspace(xlim[0], xlim[1], len(avg)),
                avg,
                label=chem,
                linestyle=linestyle,
                linewidth=linewidth,
                color=color_dict[chem],
            )
            ax.fill_between(
                np.linspace(xlim[0], xlim[1], len(avg)),
                np.array(avg) - np.array(std),
                np.array(avg) + np.array(std),
                color=color_dict[chem],
                alpha=0.1,
            )
        except:
            ax.plot(
                np.linspace(xlim[0], xlim[1], len(avg)),
                avg,
                label=chem,
                color=None,
            )
            ax.fill_between(
                np.linspace(xlim[0], xlim[1], len(avg)),
                np.array(avg) - np.array(std),
                np.array(avg) + np.array(std),
                alpha=0.1,
            )
        if legend:
            if len(exp_set) > 8:
                ax.legend(
                    loc="upper right",
                    bbox_to_anchor=(1.015, 1.02),
                    ncol=2,
                    frameon=False,
                    shadow=False,
                    fontsize=LEGENDSIZE,
                )
            else:
                ax.legend(
                    loc="upper right",
                    bbox_to_anchor=(1.015, 1.02),
                    ncol=1,
                    frameon=False,
                    shadow=False,
                    fontsize=LEGENDSIZE,
                )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    if label == True:
        if xlim[0] == 0:
            ax.set_xlabel("Time (s)", fontsize=LABELSIZE)
            ax.set_ylabel("Phase derivative (a.u.)", fontsize=LABELSIZE)
        else:
            ax.set_xlabel("Wavelength (nm)", fontsize=LABELSIZE)
            ax.set_ylabel("$\Delta I$ (a.u.)", fontsize=LABELSIZE)
        ax.tick_params(labelsize=TICKSIZE)
    else:
        ax.set_xticks = []


def plot_regression(
    actual,
    predicted,
    c="k",
    model=None,
    initial_colorID=0,
    ax=None,
    labelsize: int = 12,
    ticksize: int = 12,
):
    # plots the real vs predicted concentrations
    if ax == None:
        plt.figure(figsize=(3, 5))
        ax = plt.gca()
    ax.set_xlabel("Actual Pentane Concentration", fontsize=labelsize)
    ax.set_ylabel("Predicted Pentane Concentration", fontsize=labelsize)
    ax.tick_params(
        direction="out",
        top=False,
        bottom=True,
        left=True,
        right=False,
        labelsize=ticksize,
    )

    # plot line representing 1:1 correspondence
    ax.plot(
        np.linspace(0, 1, 100), np.linspace(0, 1, 100), "--", c="k", linewidth=1
    )

    # scatter predictions
    ax.scatter(np.array(actual), np.array(predicted), c=c, s=15)

    # Label performance
    score = r2_score(actual, predicted)
    ax.text(
        0.065,
        0.97,
        "$R^2$ = " + str(round(score, 3)),
        fontsize=labelsize,
        verticalalignment="top",
        transform=ax.transAxes,
    )

    # add model regions
    if model:
        for n, segment in enumerate(model["segments"]):
            ax.axvspan(
                segment.start_t,
                segment.inclusive_end_t,
                facecolor=segmentcolors[
                    np.mod(n + initial_colorID, len(segmentcolors))
                ],
                alpha=0.1,
            )

    # clean up plot
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
