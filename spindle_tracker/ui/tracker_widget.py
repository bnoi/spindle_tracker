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

        self.splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.layout().addWidget(self.splitter)

        self.main = QtGui.QWidget()
        self.main.setLayout(QtGui.QHBoxLayout())

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
        self.splitter.addWidget(self.main)

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

        self.splitter.addWidget(self.tw)

        self.tw.sig_traj_change.connect(self.set_tracker)
        self.tw.sig_traj_change.emit(0)

        self.tw.sig_update_trajectories.connect(self.connect_draggable_line)
        self.connect_draggable_line()

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
        m += "<p><b>Name</b> : {}</p>\n".format(str(self.tracker))

        for k, v in self.tracker.metadata.items():
            m += "<p><b>{}</b> : {}</p>\n".format(k, v)

        self.infos.setHtml(m)

    def update_annotations(self):
        """
        """

        # Clear previous widgets
        for i in range(self.cb_annot.layout().count()):
            item = self.cb_annot.layout().itemAt(i)
            item.widget().deleteLater()
            del item

        self.widgets_annot = {}

        template = self.tracker.__class__.ANNOTATIONS
        for key, (default, choices, value_type) in template.items():
            current_value = self.tracker.annotations[key]

            w = self.build_widget(key, default, choices, value_type, current_value)

            self.widgets_annot[key] = w

    def build_widget(self, key, default, choices, value_type, current_value):
        """
        """

        w = QtGui.QWidget()
        w.setLayout(QtGui.QVBoxLayout())

        title = QtGui.QLabel(key)

        if isinstance(choices, list):
            box = QtGui.QComboBox()
            for choice in choices:
                box.addItem(str(choice))
            box.setCurrentIndex(choices.index(current_value))
            box.currentIndexChanged.connect(lambda args: self.update_annotation(key, w, args))
        elif choices is None:
            if value_type == int:
                box = QtGui.QSpinBox()
                box.setRange(-1e10, 1e10)
                box.setValue(current_value)
                box.valueChanged.connect(lambda args: self.update_annotation(key, w, args))
            elif value_type == float:
                box = QtGui.QDoubleSpinBox()
                box.setDecimals(3)
                box.setSingleStep(1)
                box.setRange(-1e10, 1e10)
                box.setValue(current_value)
                box.valueChanged.connect(lambda args: self.update_annotation(key, w, args))
            elif value_type == str:
                box = QtGui.QlineEdit()
                box.setValue(current_value)
                box.editingFinished.connect(lambda args: self.update_annotation(key, w, args))

        w.layout().addWidget(title)
        w.layout().addWidget(box)

        self.cb_annot.layout().addWidget(w)

        self.update_annotation(key, w, current_value, anim=False)

        return w

    def update_annotation(self, key, widgets, args, anim=True):
        """
        """
        title = widgets.children()[1]
        box = widgets.children()[2]

        template = self.tracker.__class__.ANNOTATIONS
        default, choices, value_type = template[key]

        if isinstance(choices, list):
            self.tracker.annotations[key] = args
        elif choices is None:
            self.tracker.annotations[key] = value_type(args)

        if anim:
            self.animate_annotation_label(title)
        self.tracker.save_oio()

        if key == 'anaphase_start':
            self.tw.draggable_line.setValue(args)

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

    def update_anaphase_start(self, line):
        """
        """
        t = self.tw.draggable_line.value()
        w = self.widgets_annot['anaphase_start']
        self.update_annotation('anaphase_start', w, t)
        w.children()[2].setValue(t)

    def connect_draggable_line(self):
        """
        """
        self.tw.draggable_line.sigPositionChangeFinished.connect(self.update_anaphase_start)
        self.tw.draggable_line.setValue(self.tracker.annotations['anaphase_start'])
