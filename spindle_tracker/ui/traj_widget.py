import logging
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt

import pyqtgraph as pg
import numpy as np

from pyqtgraph.Qt import QtGui
from pyqtgraph.Qt import QtCore
from pyqtgraph import dockarea

from .viewbox import DataSelectorViewBox


class TrajectoriesWidget(QtGui.QWidget):
    """
    """

    def __init__(self, trajs, xaxis='t', yaxis='x',
                 scale_x=1, scale_y=1, parent=None):
        """
        """
        super().__init__(parent=parent)

        if parent is None:
            self.resize(1000, 500)

        self.trajs = trajs
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.scale_x = scale_x
        self.scale_y = scale_y

        self.curve_width = 1
        self.scatter_size = 8

        self.setLayout(QtGui.QVBoxLayout())

        self.area = dockarea.DockArea()
        self.layout().addWidget(self.area)

        self.dock_traj = dockarea.Dock("Trajectories Plot", size=(3, 12))
        self.dock_info = dockarea.Dock("Info Panel", size=(1, 12))
        self.dock_buttons = dockarea.Dock("Buttons", size=(3, 1), hideTitle=True)
        self.area.addDock(self.dock_traj, 'left')
        self.area.addDock(self.dock_info, 'right', self.dock_traj)
        self.area.addDock(self.dock_buttons, 'bottom')

        # Trajectory Plot Dock
        self.vb = DataSelectorViewBox()
        self.pw = pg.PlotWidget(viewBox=self.vb)
        self.vb.traj_widget = self
        self.dock_traj.addWidget(self.pw)
        self.dock_traj.layout.setContentsMargins(5, 5, 5, 5)

        self.status = QtGui.QLabel(self)
        self.dock_traj.addWidget(self.status)

        # Buttons Dock
        self.dock_buttons.layout.setContentsMargins(5, 5, 5, 5)
        self.but_select_all = QtGui.QPushButton("Select All")
        self.but_unselect_all = QtGui.QPushButton("Unselect All")
        self.dock_buttons.addWidget(self.but_select_all, row=0, col=0)
        self.dock_buttons.addWidget(self.but_unselect_all, row=0, col=1)
        self.dock_buttons.layout.setColumnStretch(10, 10)
        self.but_select_all.clicked.connect(self.select_all_items)
        self.but_unselect_all.clicked.connect(self.unselect_all_items)

        # Setup trajectories and plot logic
        self._colors = []
        self.traj_items = []

        self.pw.showGrid(x=True, y=True)
        self.pw.setLabel('bottom', self.xaxis)
        self.pw.setLabel('left', self.yaxis)

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
            curve.is_selected = False
            self.pw.addItem(curve)
            self.traj_items.append(curve)

            points_item = pg.ScatterPlotItem(symbol='o',
                                             pen={'color': (0, 0, 0, 0)},
                                             brush=pg.mkBrush(color=color),
                                             size=self.scatter_size)

            points = [{'x': xx, 'y': yy, 'data': idx} for idx, xx, yy in zip(index_list, x, y)]
            points_item.addPoints(points)

            for point in points_item.points():
                point.is_selected = False
                point.parent = points_item
                self.traj_items.append(point)

            self.pw.addItem(points_item)

    def install_clicked_hooks(self):
        """
        """
        for item in self.pw.items():
            if isinstance(item, pg.PlotCurveItem):
                item.sigClicked.connect(self.item_clicked)
            elif isinstance(item, pg.ScatterPlotItem):
                item.sigClicked.connect(self.points_clicked)

    def points_clicked(self, plot, points):
        """
        """
        for point in points:
            self.item_clicked(point)

    def item_clicked(self, item):
        """
        """
        self.check_control_key()
        if not item.is_selected:
            self.select_item(item)
        else:
            self.unselect_item(item)

    def select_item(self, item):
        """
        """
        if not item.is_selected:
            item.is_selected = True

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
        if item.is_selected:
            item.is_selected = False

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

    def unselect_all_items(self):
        """
        """
        for item in self.traj_items:
            self.unselect_item(item)

    def select_all_items(self):
        """
        """
        for item in self.traj_items:
            self.select_item(item)

    def check_control_key(self):
        """Unselect all previously selected items if CTRL key is not pressed.
        """
        modifiers = QtGui.QApplication.keyboardModifiers()
        if modifiers != QtCore.Qt.ControlModifier:
            self.unselect_all_items()

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
