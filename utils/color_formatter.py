from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
import numpy as np


class color_formatter:

    def __init__(self, max_val, min_val, cmap_color="coolwarm"):
        self.min_val = min_val
        self.max_val = max_val

        self.norm = Normalize(min_val, max_val)
        self.cmap = plt.get_cmap(cmap_color)
        self.mappable = ScalarMappable(norm=self.norm, cmap=self.cmap)

        self.out_of_bounds_color = "purple"

    def plotColorbar(self, ax, fig, label):
        colorbar = fig.colorbar(self.mappable, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(label)

    def getColor(self, val):
        if val < self.min_val or val > self.max_val:
            return self.out_of_bounds_color

        return self.cmap(self.norm(val))
