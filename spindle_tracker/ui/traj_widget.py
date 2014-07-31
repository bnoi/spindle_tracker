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
                 scale_x=1, scale_y=1,
                 column_to_display=['t', 'x', 'y', 'I', 'w'],
                 parent=None):
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
        self.column_to_display = column_to_display

        self.curve_width = 1
        self.scatter_size = 8

        self.setup_ui()

        # Setup trajectories and plot logic
        self._colors = []
        self.traj_items = []

        self.pw.showGrid(x=True, y=True)
        self.pw.setLabel('bottom', self.xaxis)
        self.pw.setLabel('left', self.yaxis)

        self.update_items()
        self.install_clicked_hooks()

    def setup_ui(self):
        """
        """

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

        self.pw.scene().sigMouseMoved.connect(self.update_mouse_infos)

        # Buttons Dock
        self.dock_buttons.layout.setContentsMargins(5, 5, 5, 5)
        self.but_select_all = QtGui.QPushButton("Select All")
        self.but_unselect_all = QtGui.QPushButton("Unselect All")
        self.dock_buttons.addWidget(self.but_select_all, row=0, col=0)
        self.dock_buttons.addWidget(self.but_unselect_all, row=0, col=1)
        self.dock_buttons.layout.setColumnStretch(10, 10)
        self.but_select_all.clicked.connect(self.select_all_items)
        self.but_unselect_all.clicked.connect(self.unselect_all_items)

        # Info Panel Dock
        self.dock_info.setContentsMargins(5, 5, 5, 5)
        self.mouse_text = self.build_text_groupbox('Under Mouse', self.dock_info)
        self.selection_box = QtGui.QGroupBox('Selected Items')
        self.dock_info.addWidget(self.selection_box)
        self.selection_tree = pg.TreeWidget()
        self.selection_tree.setColumnCount(1)
        self.selection_tree.setHeaderLabels(["t_stamp, label"])
        self.selection_box.setLayout(QtGui.QVBoxLayout())
        self.selection_box.layout().addWidget(self.selection_tree)
        self.status_text = self.build_text_groupbox('Message', self.dock_info)

    # Message management

    def update_mouse_infos(self, pos):
        """
        """
        pos = self.pw.plotItem.vb.mapDeviceToView(pos)

        mess = ""
        mess += "x = {x}\ny = {y}\n"
        # mess += "label = {label}\ntime = {time}\n"
        # mess += "w = {w}\nI = {I}\n"

        x = np.round(pos.x(), 2)
        y = np.round(pos.y(), 2)

        args = dict(x=x, y=y, label=None, time=None, w=None, I=None)
        mess = mess.format(**args)

        self.mouse_text.setText(mess)

    def update_selection_infos(self):
        """
        """

        for item in self.selection_tree.listAllItems():
            try:
                self.selection_tree.removeTopLevelItem(item)
            except:
                pass

        for item in self.traj_items:
            if item.is_selected and isinstance(item, pg.SpotItem):
                t_stamp, label = item.data()
                title = "{}, {}".format(t_stamp, label)
                twi = QtGui.QTreeWidgetItem([title])

                peak = self.trajs.loc[t_stamp, label]
                for l in self.column_to_display:
                    ctwi = QtGui.QTreeWidgetItem(["{} : {}".format(l, peak[l])])
                    twi.addChild(ctwi)

                self.selection_tree.addTopLevelItem(twi)

    # Items management

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
        self.check_control_key(ignore_items=[item])
        if not item.is_selected:
            self.select_item(item)
        else:
            self.unselect_item(item)

        self.update_selection_infos()

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

    def unselect_all_items(self, event=None, ignore_items=[]):
        """
        """
        for item in self.traj_items:
            if item not in ignore_items:
                self.unselect_item(item)
        self.update_selection_infos()

    def select_all_items(self):
        """
        """
        for item in self.traj_items:
            self.select_item(item)
        self.update_selection_infos()

    def check_control_key(self, ignore_items=[]):
        """Unselect all previously selected items if CTRL key is not pressed.
        """
        modifiers = QtGui.QApplication.keyboardModifiers()
        if modifiers != QtCore.Qt.ControlModifier:
            self.unselect_all_items(ignore_items=ignore_items)

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

    # Colors management

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

    # Factories

    def build_text_groupbox(self, title="", parent_widget=None):
        """
        """
        gbox = QtGui.QGroupBox(title)
        text = QtGui.QTextEdit()
        text.setReadOnly(True)
        l = QtGui.QVBoxLayout()
        l.addWidget(text)
        l.addStretch(1)
        gbox.setLayout(l)

        if parent_widget:
            parent_widget.addWidget(gbox)

        return text
