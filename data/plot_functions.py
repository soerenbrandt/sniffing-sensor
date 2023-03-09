import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colour import Color
from sklearn.metrics import confusion_matrix, r2_score

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
                linestyle=linestyle.get(chem, "-")
                if isinstance(linestyle, dict)
                else linestyle,
                linewidth=linewidth.get(chem, None)
                if isinstance(linewidth, dict)
                else linewidth,
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

    # clean up plot
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])


def plot_river(data: np.ndarray, wavelengths: np.ndarray, ax=None):
    # Plot river plot
    if ax is None:
        _, ax = plt.subplots()
    extent = [wavelengths[0], wavelengths[-1], 0, 600]

    vmin = np.amin(data)
    vmax = np.amax(data)
    im = ax.imshow(
        data,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        extent=extent,
        cmap="jet",
    )

    # Define image labels
    ax.set_ylabel("Time (s)", fontsize=16)
    ax.set_xlabel("Wavelength (nm)", fontsize=16)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")

    # Image aspect ratio
    xext, yext = ax.axes.get_xlim(), ax.axes.get_ylim()
    xrange = xext[1] - xext[0]
    yrange = yext[1] - yext[0]
    ax.set_aspect(
        1 * abs(xrange / yrange)
    )  # This is the line that causes the warnings about unicode

    # Flip y-ticklabels
    times = range(0, int(np.ceil(yext[1])) + 1, 100)
    ax.set_yticks(times)
    ax.set_yticklabels(times[::-1])

    plt.tight_layout()
    return im


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes=None,
    normalize=False,
    title=None,
    cmap=plt.cm.Blues,
    ax=None,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if ax == None:
        fig, ax = plt.subplots()
    if title == True:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
    )
    ax.tick_params(labelsize=12)
    ax.set_ylabel("True label", fontsize=13)
    ax.set_xlabel("Predicted label", fontsize=13)

    # Rotate the tick labels and set their alignment.
    ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
