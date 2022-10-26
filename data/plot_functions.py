import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELSIZE = 8
LEGENDSIZE = 8
TICKSIZE = 8


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
