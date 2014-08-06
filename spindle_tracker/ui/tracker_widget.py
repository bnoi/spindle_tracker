import logging
log = logging.getLogger(__name__)

from pyqtgraph.Qt import QtGui
from pyqtgraph.Qt import QtCore


class TrackersWidget(QtGui.QWidget):
    """
    """

    def __init__(self, trackers, traj_name="trajs",
                 xaxis='t', yaxis='x',
                 scale_x=1, scale_y=1,
                 parent=None):
        """
        """
        super().__init__(parent=parent)

        self.setWindowTitle("Trackers plot")

        if parent is None:
            self.resize(1000, 700)

        if isinstance(trackers, list):
            self.trackers = tracker
        else:
            self.trackers = [tracker]


        self.tracker = self.trackers
