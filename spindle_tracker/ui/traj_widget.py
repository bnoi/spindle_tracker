import matplotlib.pyplot as plt

import pyqtgraph as pg
import numpy as np

from pyqtgraph.Qt import QtGui


class TrajectoriesWidget(QtGui.QWidget):
    """
    """

    def __init__(self, trajs, xaxis='t', yaxis='x', parent=None):
        """
        """
        super().__init__(parent=parent)

        self.trajs = trajs
        self.xaxis = xaxis
        self.yaxis = yaxis

        self.l = QtGui.QVBoxLayout()
        self.setLayout(self.l)

        self.pw = pg.PlotWidget()
        self.l.addWidget(self.pw)

        self.l_status = QtGui.QHBoxLayout()
        self.l.addLayout(self.l_status)
        self.l_status.addStretch(1)

        self.status = QtGui.QLabel(self)
        self.l_status.addWidget(self.status)

        self.pw.showGrid(x=True, y=True)
        self.pw.setLabel('bottom', self.xaxis)
        self.pw.setLabel('left', self.yaxis)

        self.create_items()

    def create_items(self):
        """
        """

        # Setup color gradient for segment labels
        labels = self.trajs.index.get_level_values('label').unique().tolist()
        id_labels = np.arange(len(labels))
        id_labels = id_labels / id_labels.max() * 255
        cmap = plt.get_cmap('hsv')
        colors = [cmap(int(l), alpha=1, bytes=True) for l in id_labels]

        for color, (label, peaks) in zip(colors, self.trajs.groupby(level='label')):

            coords = peaks.loc[:, [self.xaxis, self.yaxis]].values

            curve = pg.PlotCurveItem(x=coords[:, 0], y=coords[:, 1],
                                     pen={'color': color, 'width': 1})
            self.pw.addItem(curve)

            point = pg.ScatterPlotItem(x=coords[:, 0], y=coords[:, 1],
                                       symbol='o', pen={'color': (0, 0, 0, 0)},
                                       brush=pg.mkBrush(color), size=8)
            self.pw.addItem(point)
