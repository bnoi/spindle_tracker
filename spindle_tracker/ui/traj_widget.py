import logging
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt

import pyqtgraph as pg
import numpy as np

from pyqtgraph.Qt import QtGui


class TrajectoriesWidget(QtGui.QWidget):
    """
    """

    def __init__(self, trajs, xaxis='t', yaxis='x',
                 scale_x=1, scale_y=1, parent=None):
        """
        """
        super().__init__(parent=parent)

        self.trajs = trajs
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.scale_x = scale_x
        self.scale_y = scale_y

        self.curve_width = 1
        self.scatter_size = 8

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

        self.traj_items = []
        self.selected_items = []
        self._colors = []

        self.update_items()
        self.install_clicked_hooks()

    def update_items(self):
        """
        """

        self.remove_items()
        self.setup_color_list()

        for label, peaks in self.trajs.groupby(level='label'):

            color = self.colors(label)

            coords = peaks.loc[:, [self.xaxis, self.yaxis]].values
            x = coords[:, 0] * self.scale_x
            y = coords[:, 1] * self.scale_y

            index_list = peaks.index.tolist()

            curve = pg.PlotCurveItem(x=x, y=y,
                                     pen={'color': color, 'width': self.curve_width},
                                     clickable=True)
            curve.label = label
            self.pw.addItem(curve)
            self.traj_items.append(curve)

            points_items = pg.ScatterPlotItem(symbol='o',
                                              pen={'color': (0, 0, 0, 0)},
                                              brush=pg.mkBrush(color=color),
                                              size=self.scatter_size)

            points = [{'x': xx, 'y': yy, 'data': idx} for idx, xx, yy in zip(index_list, x, y)]
            points_items.addPoints(points)
            self.pw.addItem(points_items)
            self.traj_items.append(points_items)

    def install_clicked_hooks(self):
        """
        """

        for item in self.traj_items:
            if isinstance(item, pg.PlotCurveItem):
                item.sigClicked.connect(self.item_selected)
            elif isinstance(item, pg.ScatterPlotItem):
                item.sigClicked.connect(self.points_clicked)
            else:
                log.warning("Item {} not handled".format(item))

    def points_clicked(self, plot, points):
        """
        """
        for point in points:
            self.item_selected(point)

    def item_selected(self, item):
        """
        """
        if item not in self.selected_items:
            self.selected_items.append(item)
            self.select_item(item)
        else:
            self.selected_items.remove(item)
            self.unselect_item(item)

    def select_item(self, item):
        """
        """
        if isinstance(item, pg.SpotItem):
            item.setPen(width=2, color='r')
            item.setSize(self.scatter_size * 1.5)
        elif isinstance(item, pg.PlotCurveItem):
            color = self.item_colors(item)
            item.setPen(width=self.curve_width * 2, color=color)
        else:
            log.warning("Item {} not handled".format(item))

    def unselect_item(self, item):
        """
        """
        if isinstance(item, pg.SpotItem):
            item.setPen(None)
            item.setSize(self.scatter_size)
        elif isinstance(item, pg.PlotCurveItem):
            color = self.item_colors(item)
            item.setPen(width=self.curve_width, color=color)
        else:
            log.warning("Item {} not handled".format(item))

    def remove_item(self, item):
        """
        """
        self.pw.removeItem(item)
        del item

    def remove_items(self):
        """
        """
        for item in self.pw.items():
            self.remove_item(item)

    def setup_color_list(self):
        """Setup color gradient for segment labels
        """
        labels = self.trajs.index.get_level_values('label').unique().tolist()
        id_labels = np.arange(len(labels))
        id_labels = id_labels / id_labels.max() * 255
        cmap = plt.get_cmap('hsv')
        cols = [cmap(int(l), alpha=1, bytes=True) for l in id_labels]

        self._colors = dict(zip(labels, cols))

    def colors(self, label):
        """
        """
        return self._colors[label]

    def item_colors(self, item):
        """
        """
        label = None
        if isinstance(item, pg.SpotItem):
            label = item.data()[1]
        elif isinstance(item, pg.PlotCurveItem):
            label = item.label
        else:
            log.warning("Item {} not handled".format(item))
            return False
        return self.colors(label)
