'''
Legend stuff.
'''

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

class Legend():
    '''
    Add customized legend to axes.

    Example:

        lgd1 = Legend(ax)
        lgd1.add_line(label, color, linestyle, linewidth)
        lgd1.add_patch(label, facecolor, edgecolor)
        lgd1.loc = "upper right"
        lgd1.fontsize = 12
        lgd1.draw()

    '''
    
    def __init__(self, ax=None, loc='best', outside=False, fontsize=12):
        self.ax = ax
        self._lgds = []
        self.loc = loc
        self.fontsize = fontsize
        self.borderaxespad = 0.5
        self.bbox_to_anchor = None
        self.outside = outside

    def set_axes(self, ax):
        self.ax = ax

    def add_patch(self, label="", fc="blue", ec="black"):
        self._lgds.append(Patch(label=label, facecolor=fc, edgecolor=ec))

    def add_line(self, label="", color="black", linestyle="-", linewidth=1):
        self._lgds.append(Line2D([0], [0], label=label, color=color,
                                 linestyle=linestyle, linewidth=linewidth))

    def draw(self):
        if(self.outside):
            if(self.loc == "upper right"):
                self.bbox_to_anchor = (1.05, 1.0)
                loc = "upper left"
            elif(self.loc == "lower right"):
                self.bbox_to_anchor = (1.05, 0.0)                
                loc = "lower left"
            elif(self.loc == "upper left"):
                self.bbox_to_anchor = (-0.1, 1.0)                
                loc = "upper right"
            elif(self.loc == "lower left"):
                self.bbox_to_anchor = (-0.1, 0.0)                
                loc = "lower right"
            else:
                raise ValueError("when outside is True, loc must be one of the following: ['upper right', 'lower right', 'upper left', 'lower left']")
            borderaxespad = 0.0
        else:
            loc = self.loc
            borderaxespad = self.borderaxespad
        
        if(self.ax != None):
            legend_panel = self.ax.legend(
                handles=self._lgds,
                loc=loc,
                fontsize=self.fontsize,
                bbox_to_anchor=self.bbox_to_anchor,
                borderaxespad=borderaxespad
            )
            lgd = self.ax.add_artist(legend_panel)
        else:
            legend_panel = plt.legend(
                handles=self._lgds,
                loc=loc,
                fontsize=self.fontsize,
                bbox_to_anchor=self.bbox_to_anchor,
                borderaxespad=borderaxespad
            )
            lgd = plt.gca().add_artist(legend_panel)
        return lgd

    def set_labels(self, labels):
        if(len(labels) != len(self._lgds)):
            raise ValueError("legend.set_labels(): number of legend txts must match the number of legends!")
        else:
            for i in range(len(labels)):
                self._lgds[i].set_label(labels[i])
                
    def set_colors(self, colors):
        if(len(colors) != len(self._lgds)):
            raise ValueError("legend.set_colors(): number of colors must match the number of legends!")
        else:
            for i in range(len(colors)):
                self._lgds[i].set_color(colors[i])

    @staticmethod
    def demo():
        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        lgd1 = Legend(None)
        lgd1.loc = "lower left"
        lgd1.add_line("Solid", "black", "-", 1)
        lgd1.add_line("Dashed", "black", "--", 1)
        lgd1.add_line(lgd1.loc, "blue", ":", 1) 
        lgd1.draw()
        lgd2 = Legend(ax)
        lgd2.loc = "upper left"
        lgd2.add_line("Thick Blue Line", "blue", "-", 2)
        lgd2.add_patch("Red Patch", "red", "black")
        lgd2.add_patch(lgd2.loc, "green", "blue")
        lbls = ["Thick Blue Line", "Red Patch", lgd2.loc]
        lgd2.set_labels(lbls)
        lgd2.draw()
        lgd3 = Legend(ax)
        lgd3.loc = "upper right"
        lgd3.add_line("outside", "blue", "-", 2)
        lgd3.outside = True
        lgd = lgd3.draw()
        lgd4 = Legend(ax)
        lgd4.loc = "lower left"
        lgd4.add_line("outside", "red", "-", 2)
        lgd4.outside = True
        lgd = lgd4.draw()
        fig.subplots_adjust(right=0.72)
        plt.show()

