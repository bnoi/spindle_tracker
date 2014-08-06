import logging
log = logging.getLogger(__name__)

from pyqtgraph.Qt import QtGui
from pyqtgraph.Qt import QtCore

from . import TrajectoriesWidget


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
            self.trackers = trackers
        else:
            self.trackers = [trackers]

        self.traj_name = traj_name

        self.xaxis = xaxis
        self.yaxis = yaxis
        self.scale_x = scale_x
        self.scale_y = scale_y

        self.setLayout(QtGui.QVBoxLayout())

        self.main = QtGui.QWidget()
        self.main.setLayout(QtGui.QHBoxLayout())
        self.layout().addWidget(self.main)

        self.cb_infos = QtGui.QGroupBox("Informations")
        self.cb_infos.setLayout(QtGui.QVBoxLayout())
        self.main.layout().addWidget(self.cb_infos)

        self.infos = QtGui.QTextEdit()
        self.cb_infos.layout().addWidget(self.infos)
        self.infos.setReadOnly(True)

        self.cb_annot = QtGui.QGroupBox("Annotations")
        self.cb_annot.setLayout(QtGui.QHBoxLayout())
        self.main.layout().addWidget(self.cb_annot)

        if not self.parent():
            self.main.layout().addStretch(1)
            self.but_quit = QtGui.QPushButton("Quit")
            self.main.layout().addWidget(self.but_quit)
            self.but_quit.clicked.connect(self.close)

        self.setup_traj_widget()

    def setup_traj_widget(self):
        """
        """

        all_trajs = [getattr(tracker, self.traj_name) for tracker in self.trackers]

        self.tw = TrajectoriesWidget(all_trajs,
                                     xaxis=self.xaxis,
                                     yaxis=self.yaxis,
                                     scale_x=self.scale_x,
                                     scale_y=self.scale_y,
                                     parent=self)

        self.layout().insertWidget(0, self.tw)

        self.tw.sig_traj_change.connect(self.set_tracker)
        self.tw.sig_traj_change.emit(0)

    def set_tracker(self, i):
        """
        """
        self.tracker = self.trackers[i]
        self.update_infos()
        self.update_annotations()

    def update_infos(self):
        """
        """
        m = ""
        m += "Name : {}\n".format(str(self.tracker))

        self.infos.setText(m)

    def update_annotations(self):
        """
        """

        # Clear previous widgets
        for i in range(self.cb_annot.layout().count()):
            item = self.cb_annot.layout().itemAt(i)
            item.widget().deleteLater()

        template = self.tracker.__class__.ANNOTATIONS
        for key, (default, choices, value_type) in template.items():
            current_value = self.tracker.annotations[key]

            self.build_widget(key, default, choices, value_type, current_value)

    def build_widget(self, key, default, choices, value_type, current_value):
        """
        """

        self.w = QtGui.QWidget()
        self.w.setLayout(QtGui.QVBoxLayout())

        title = QtGui.QLabel(key)

        if isinstance(choices, list):
            box = QtGui.QComboBox()
            for choice in choices:
                box.addItem(str(choice))
            box.setCurrentIndex(choices.index(current_value))
            box.currentIndexChanged.connect(lambda args: self.update_annotation(key, title, args))
        elif choices is None:
            if value_type == int:
                box = QtGui.QSpinBox()
                box.setRange(-1e10, 1e10)
                box.setValue(current_value)
                box.valueChanged.connect(lambda args: self.update_annotation(key, title, args))
            elif value_type == float:
                box = QtGui.QDoubleSpinBox()
                box.setDecimals(3)
                box.setSingleStep(1)
                box.setRange(-1e10, 1e10)
                box.setValue(current_value)
                box.valueChanged.connect(lambda args: self.update_annotation(key, title, args))
            elif value_type == str:
                box = QtGui.QlineEdit()
                box.setValue(current_value)
                box.editingFinished.connect(lambda args: self.update_annotation(key, title, args))

        self.w.layout().addWidget(title)
        self.w.layout().addWidget(box)

        self.cb_annot.layout().addWidget(self.w)

    def update_annotation(self, key, widget, args):
        """
        """
        template = self.tracker.__class__.ANNOTATIONS
        default, choices, value_type = template[key]

        if isinstance(choices, list):
            self.tracker.annotations[key] = args
        elif choices is None:
            self.tracker.annotations[key] = value_type(args)

        self.animate_annotation_label(widget)
        self.tracker.save_oio()

    def animate_annotation_label(self, widget, start=True):
        """
        """
        if start:
            widget.setStyleSheet("font-weight:bold;")

            self.timeoutTimer = QtCore.QTimer()
            self.timeoutTimer.setInterval(400)
            self.timeoutTimer.setSingleShot(True)
            self.timeoutTimer.timeout.connect(lambda: self.animate_annotation_label(widget,
                                              start=False))
            self.timeoutTimer.start()
        else:
            widget.setStyleSheet("font-weight:normal;")
