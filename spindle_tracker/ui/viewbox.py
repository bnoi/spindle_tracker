import logging
log = logging.getLogger(__name__)

import pyqtgraph as pg
from pyqtgraph.Point import Point
from pyqtgraph.Qt import QtCore
from pyqtgraph.Qt import QtGui


class DataSelectorViewBox(pg.ViewBox):
    """
    """

    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)
        self.old_selection = []

    def mouseDragEvent(self, ev):
        """
        """

        ev.accept()
        pos = ev.pos()

        modifiers = QtGui.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier and ev.button() == QtCore.Qt.LeftButton:

            if ev.isFinish():
                self.rbScaleBox.hide()
                self.traj_widget.update_selection_infos()
            else:
                rect_box = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
                rect_box = self.childGroup.mapRectFromParent(rect_box)
                self.update_selection(rect_box)

                self.updateScaleBox(ev.buttonDownPos(), ev.pos())
        else:
            pg.ViewBox.mouseDragEvent(self, ev)

    def update_selection(self, rect_box):
        """
        """

        items = self.items_inside_rectangle(rect_box)

        for item in self.old_selection:
            if item not in items:
                self.traj_widget.unselect_item(item)

        for item in items:
            if item not in self.old_selection:
                self.traj_widget.select_item(item)

        self.old_selection = items

    def items_inside_rectangle(self, rect):
        """
        """

        x1 = rect.x()
        x2 = rect.x() + rect.width()
        y1 = rect.y()
        y2 = rect.y() + rect.height()

        items_inside = []
        for point in self.traj_widget.traj_items:
            if isinstance(point, pg.SpotItem):
                x = point.pos().x()
                y = point.pos().y()

                if x > x1 and x < x2 and y > y1 and y < y2:
                    items_inside.append(point)
        return items_inside