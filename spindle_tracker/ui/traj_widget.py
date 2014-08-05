import logging
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt

import pyqtgraph as pg
import numpy as np

from pyqtgraph.Qt import QtGui
from pyqtgraph.Qt import QtCore
from pyqtgraph import dockarea
import pyqtgraph.exporters as pgexporters

from .viewbox import DataSelectorViewBox


class TrajectoriesWidget(QtGui.QWidget):
    """
    """

    def __init__(self, trajs, xaxis='t', yaxis='x',
                 scale_x=1, scale_y=1,
                 column_to_display=['t', 'x', 'y', 'I', 'w'],
                 add_draggable_line=False,
                 parent=None):
        """
        """
        super().__init__(parent=parent)

        self.setWindowTitle("Trajectories plot")

        if parent is None:
            self.resize(1000, 500)

        if isinstance(trajs, list):
            self.current_traj_id = 0
            self.trajs = trajs[self.current_traj_id]
            self.all_trajs = trajs
            self.len_trajs = len(trajs)
        else:
            self.current_traj_id = None
            self.trajs = trajs
            self.all_trajs = None
            self.len_trajs = len(trajs)

        self.historic_trajs = []
        self.historic_trajs.append(self.trajs)

        self.xaxis = xaxis
        self.yaxis = yaxis
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.column_to_display = column_to_display
        self.add_draggable_line = add_draggable_line
        self.draggable_line = None

        self.curve_width = 1
        self.scatter_size = 8

        self.setup_ui()
        self.setup_menus()
        self.update_historic_buttons()

        # Setup trajectories and plot logic
        self._colors = []
        self.traj_items = []

        self.update_trajectory()

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
        self.dock_buttons_parent = QtGui.QWidget()
        self.dock_buttons_parent.setLayout(QtGui.QHBoxLayout())
        self.dock_buttons.addWidget(self.dock_buttons_parent)

        # Build axis buttons

        self.axis_container = QtGui.QWidget()
        self.axis_container.setLayout(QtGui.QGridLayout())

        self.cb_xaxis_label = QtGui.QLabel('X axis : ')
        self.axis_container.layout().addWidget(self.cb_xaxis_label, 0, 0)
        self.cb_xaxis = QtGui.QComboBox()
        for label in self.trajs.columns:
            self.cb_xaxis.addItem(label)
        self.axis_container.layout().addWidget(self.cb_xaxis, 0, 1)
        self.cb_xaxis.currentIndexChanged.connect(self.set_xaxis)

        self.cb_yaxis_label = QtGui.QLabel('Y axis : ')
        self.axis_container.layout().addWidget(self.cb_yaxis_label, 1, 0)
        self.cb_yaxis = QtGui.QComboBox()
        for label in self.trajs.columns:
            self.cb_yaxis.addItem(label)
        self.axis_container.layout().addWidget(self.cb_yaxis, 1, 1)
        self.cb_yaxis.currentIndexChanged.connect(self.set_yaxis)

        self.dock_buttons_parent.layout().addWidget(self.axis_container)

        # Build undo / redo buttons

        self.history_container = QtGui.QWidget()
        self.history_container.setLayout(QtGui.QGridLayout())

        self.but_undo = QtGui.QPushButton("Undo < (0)")
        self.history_container.layout().addWidget(self.but_undo, 0, 0)
        self.but_undo.clicked.connect(self.undo)

        self.but_redo = QtGui.QPushButton("Redo > (0)")
        self.history_container.layout().addWidget(self.but_redo, 1, 0)
        self.but_redo.clicked.connect(self.redo)

        self.dock_buttons_parent.layout().addWidget(self.history_container)

        # Build select / unselect buttons

        self.selection_container = QtGui.QWidget()
        self.selection_container.setLayout(QtGui.QGridLayout())

        self.but_select_all = QtGui.QPushButton("Select All")
        self.selection_container.layout().addWidget(self.but_select_all, 0, 0)
        self.but_select_all.clicked.connect(self.select_all_items)

        self.but_unselect_all = QtGui.QPushButton("Unselect All")
        self.selection_container.layout().addWidget(self.but_unselect_all, 1, 0)
        self.but_unselect_all.clicked.connect(self.unselect_all_items)

        self.dock_buttons_parent.layout().addWidget(self.selection_container)

        # Build trajs selector

        if self.all_trajs:
            self.all_trajs_container = QtGui.QWidget()
            self.all_trajs_container.setLayout(QtGui.QGridLayout())

            self.all_trajs_label = QtGui.QLabel()
            self.set_all_trajs_label()
            self.all_trajs_container.layout().addWidget(self.all_trajs_label, 0, 0)

            self.all_trajs_previous = QtGui.QPushButton("Next >")
            self.all_trajs_container.layout().addWidget(self.all_trajs_previous, 1, 0)
            self.all_trajs_previous.clicked.connect(self.next_traj)

            self.all_trajs_next = QtGui.QPushButton("Previous <")
            self.all_trajs_container.layout().addWidget(self.all_trajs_next, 2, 0)
            self.all_trajs_next.clicked.connect(self.previous_trajs)

            self.dock_buttons_parent.layout().addWidget(self.all_trajs_container)

        # Build Quit button

        if not self.parent():
            self.dock_buttons_parent.layout().addStretch(1)
            self.but_quit = QtGui.QPushButton("Quit")
            self.dock_buttons_parent.layout().addWidget(self.but_quit)
            self.but_quit.clicked.connect(self.close)

        # Build info Panel Dock
        self.dock_info.setContentsMargins(5, 5, 5, 5)
        self.mouse_text = self.build_text_groupbox('Under Mouse', self.dock_info)
        self.selection_box = QtGui.QGroupBox('Selected Items (0)')
        self.dock_info.addWidget(self.selection_box)
        self.selection_tree = pg.TreeWidget()
        self.selection_tree.setColumnCount(1)
        self.selection_tree.setHeaderLabels(["t_stamp, label"])
        self.selection_box.setLayout(QtGui.QVBoxLayout())
        self.selection_box.layout().addWidget(self.selection_tree)
        self.status_text = self.build_text_groupbox('Message', self.dock_info)

    # Menus

    def setup_menus(self):
        """
        """
        self.menu_spots = QtGui.QMenu("Spots")
        action_add_spot = QtGui.QAction("Add spot", self.menu_spots)
        action_remove_spot = QtGui.QAction("Remove spot", self.menu_spots)
        self.menu_spots.addAction(action_add_spot)
        self.menu_spots.addAction(action_remove_spot)
        action_add_spot.triggered.connect(lambda x: x)
        action_remove_spot.triggered.connect(lambda x: x)

        self.menu_trajs = QtGui.QMenu("Trajectories")
        action_merge_trajs = QtGui.QAction("Merge two trajectories", self.menu_trajs)
        action_remove_traj = QtGui.QAction("Remove trajectory", self.menu_trajs)
        action_cut_traj = QtGui.QAction("Cut trajectory", self.menu_trajs)
        action_duplicate_traj = QtGui.QAction("Duplicate trajectory", self.menu_trajs)
        self.menu_trajs.addAction(action_merge_trajs)
        self.menu_trajs.addAction(action_remove_traj)
        self.menu_trajs.addAction(action_cut_traj)
        self.menu_trajs.addAction(action_duplicate_traj)
        action_merge_trajs.triggered.connect(lambda x: x)
        action_remove_traj.triggered.connect(lambda x: x)
        action_cut_traj.triggered.connect(lambda x: x)
        action_duplicate_traj.triggered.connect(lambda x: x)

        self.vb.menu.addSeparator()
        self.vb.menu.addMenu(self.menu_spots)
        self.vb.menu.addMenu(self.menu_trajs)

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

        self.clear_selection_infos()

        i = 0
        for item in self.traj_items:
            if item.is_selected and isinstance(item, pg.SpotItem):
                i += 1
                t_stamp, label = item.data()
                title = "{}, {}".format(t_stamp, label)
                twi = QtGui.QTreeWidgetItem([title])

                peak = self.trajs.loc[t_stamp, label]
                for l in self.column_to_display:
                    ctwi = QtGui.QTreeWidgetItem(["{} : {}".format(l, peak[l])])
                    twi.addChild(ctwi)

                self.selection_tree.addTopLevelItem(twi)

        self.selection_box.setTitle('Selected Items ({})'.format(i))

    def clear_selection_infos(self):
        """
        """
        self.selection_tree.clear()

    # Items management

    def update_trajectory(self):
        """
        """

        self.remove_items()
        self.setup_color_list()

        if self.add_draggable_line:
            self.draggable_line = pg.InfiniteLine(angle=90, movable=True)
            self.pw.addItem(self.draggable_line)

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
            curve.sigClicked.connect(self.item_clicked)
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

            points_item.sigClicked.connect(self.points_clicked)
            self.pw.addItem(points_item)

        self.pw.showGrid(x=True, y=True)
        self.pw.setLabel('bottom', self.xaxis)
        self.pw.setLabel('left', self.yaxis)

        self.cb_xaxis.setCurrentIndex(self.trajs.columns.tolist().index(self.xaxis))
        self.cb_yaxis.setCurrentIndex(self.trajs.columns.tolist().index(self.yaxis))

        self.clear_selection_infos()

    def points_clicked(self, plot, points):
        """
        """
        for point in points:
            self.item_clicked(point)

    def item_clicked(self, item):
        """
        """
        self.check_control_key(ignore_items=[item])
        if item.is_selected is False:
            self.select_item(item)
        else:
            self.unselect_item(item)

        self.update_selection_infos()

    def select_item(self, item):
        """
        """
        if item.is_selected is False:
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
        self.traj_items = []
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

    # Trajectories selector

    def set_traj(self, i):
        """
        """

        if i < 0 or i >= self.len_trajs:
            return

        self.current_traj_id = i
        self.trajs = self.all_trajs[self.current_traj_id]

        self.set_all_trajs_label()

        self.historic_trajs = []
        self.historic_trajs.append(self.trajs)
        self.update_historic_buttons()

        self._colors = []
        self.traj_items = []

        self.update_trajectory()

    def next_traj(self):
        """
        """
        self.set_traj(self.current_traj_id + 1)

    def previous_trajs(self):
        """
        """
        self.set_traj(self.current_traj_id - 1)

    def set_all_trajs_label(self):
        """
        """
        m = "Trajs selector : {}/{}"
        self.all_trajs_label.setText(m.format(self.current_traj_id + 1, self.len_trajs))

    # Historic management

    def undo(self):
        """
        """

        i = self.current_traj_index()
        self.trajs = self.historic_trajs[i - 1]

        self.update_trajectory()
        self.update_historic_buttons()

    def redo(self):
        """
        """
        i = self.current_traj_index()
        self.trajs = self.historic_trajs[i + 1]

        self.update_trajectory()
        self.update_historic_buttons()

    def update_historic_buttons(self):
        """
        """

        i = self.current_traj_index()
        self.but_undo.setText('Undo < ({})'.format(i))
        self.but_redo.setText('Redo > ({})'.format(len(self.historic_trajs) - i - 1))

        if self.historic_trajs[0] is self.trajs:
            self.but_undo.setDisabled(True)
        else:
            self.but_undo.setDisabled(False)

        if self.historic_trajs[-1] is self.trajs:
            self.but_redo.setDisabled(True)
        else:
            self.but_redo.setDisabled(False)

    def current_traj_index(self):
        """
        """

        for i, trajs in enumerate(self.historic_trajs):
            if trajs is self.trajs:
                return i

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
        text.setReadOnly(False)
        l = QtGui.QVBoxLayout()
        l.addWidget(text)
        l.addStretch(1)
        gbox.setLayout(l)

        if parent_widget:
            parent_widget.addWidget(gbox)

        return text

    # Exporters

    def save(self, fname):
        """
        """
        if fname.endswith('.svg'):
            exporter = pgexporters.SVGExporter(self.pw.plotItem)
        elif fname.endswith('.png'):
            exporter = pgexporters.ImageExporter(self.pw.plotItem)
        elif fname.endswith('.jpg'):
            exporter = pgexporters.ImageExporter(self.pw.plotItem)
        elif fname.endswith('.tif'):
            exporter = pgexporters.ImageExporter(self.pw.plotItem)
        else:
            log.error('Wrong filename extension')
        exporter.export(fname)

    # Axes setter

    def set_xaxis(self, ax_name, scale=None):
        """
        """
        if isinstance(ax_name, int):
            ax_name = self.cb_xaxis.itemText(ax_name)

        if ax_name not in self.trajs.columns:
            log.error('"{}" is not in Trajectories columns'.format(ax_name))
            return

        self.xaxis = ax_name
        if scale:
            self.scale_x = scale
        self.update_trajectory()

    def set_yaxis(self, ax_name, scale=None):
        """
        """
        if isinstance(ax_name, int):
            ax_name = self.cb_yaxis.itemText(ax_name)

        if ax_name not in self.trajs.columns:
            log.error('"{}" is not in Trajectories columns'.format(ax_name))
            return

        self.yaxis = ax_name
        if scale:
            self.scale_y = scale

        self.update_trajectory()
