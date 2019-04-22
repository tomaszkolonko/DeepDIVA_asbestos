import matplotlib as mpl
# To facilitate plotting on a headless server
mpl.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def make_heatmap(confusion_matrix, class_names):
    """
    This function prints and plots the confusion matrix.

    Adapted from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        Array containing the confusion matrix to be plotted
    class_names: list of strings
        Names of the different classes

    Returns
    -------
    data : numpy.ndarray
        Contains an RGB image of the plotted confusion matrix
    """

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )


    plt.style.use(['seaborn-white', 'seaborn-paper'])
    sns.set(font_scale=2.5)

    fig = plt.figure(figsize=(14, 14))
    ax1 = fig.add_axes([0.3, 0.05, 0.7, 0.7])
    plt.tight_layout()

    heatmap = sns.heatmap(df_cm, ax=ax1, annot=True, fmt="d", cmap=plt.get_cmap('Blues'), annot_kws={"size": 32})

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=28)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='center', fontsize=28)
    heatmap.xaxis.set_ticks_position('top')
    heatmap.xaxis.set_label_position('top')

    xlabel_t = plt.xlabel("PREDICTED CLASS", fontsize=32)
    xlabel_t.set_position([0.5, 12])

    plt.ylabel("ACTUAL CLASS", fontsize=32)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.clf()
    plt.close()
    return data
