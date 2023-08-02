import itertools
import os

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import figure
from torch import Tensor


def plot_confusion_matrix(confusion_matrix: Tensor, class_names: list[str]) -> Any:
    confusion_matrix = confusion_matrix.cpu().detach().numpy()

    figure(num=None, figsize=(16, 12), dpi=60, facecolor="w", edgecolor="k")
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Purples)  # type: ignore
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=20)
    plt.yticks(tick_marks, class_names, fontsize=20)

    fmt = "d"
    thresh = confusion_matrix.max() / 2.0
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(
            j,
            i,
            format(confusion_matrix[i, j], fmt),
            horizontalalignment="center",
            color="white" if confusion_matrix[i, j] > thresh else "black",
            fontsize=20,
        )

    plt.title("Confusion matrix")
    plt.ylabel("Actual label", fontsize=20)
    plt.xlabel("Predicted label", fontsize=20)
    plt.tight_layout()

    return plt.gcf()


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", -1))
