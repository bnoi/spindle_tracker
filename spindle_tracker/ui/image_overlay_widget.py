import matplotlib.pyplot as plt

import pyqtgraph as pg
import numpy as np


class ImageOverlayWidget(pg.ImageView):
    """
    """

    def __init__(self, image, trajs, alpha=0.5, scale_factor=1,
                 cmap='hsv', parent=None):
        """
        """

        super().__init__(parent=parent, view=pg.PlotItem())

        self.setWindowTitle("Spots overlay on image")

        self.cmap = cmap
        self.alpha = alpha
        self.scale_factor = scale_factor

        self.current_rois = []
        self.overlay_rois = {}

        self.trajs = trajs
        self.create_rois()

        self.im = image
        self.setImage(self.im)

        self.display_rois(0, 0)
        self.sigTimeChanged.connect(self.display_rois)

    def create_rois(self):
        """
        """

        xy_pixels = self.trajs.loc[:, ['x', 'y', 'w']] / self.scale_factor
        labels = xy_pixels.index.get_level_values('label').unique().tolist()

        # Setup color gradient for segment labels
        id_labels = np.arange(len(labels))
        id_labels = id_labels / id_labels.max() * 255
        cmap = plt.get_cmap(self.cmap)
        colors = [cmap(int(l), alpha=self.alpha, bytes=True) for l in id_labels]

        for t_stamp, peaks in xy_pixels.groupby(level='t_stamp'):
            for (t_stamp, label), peak in peaks.iterrows():

                color = colors[labels.index(label)]
                pen = pg.mkPen(color=color)

                roi = pg.CircleROI((peak['y'] - peak['w'] / 2, peak['x'] - peak['w'] / 2),
                                   peak['w'], pen=pen,  movable=True, scaleSnap=False)
                roi.label = label
                roi.t_stamp = t_stamp

                if t_stamp not in self.overlay_rois.keys():
                    self.overlay_rois[t_stamp] = []

                self.overlay_rois[t_stamp].append(roi)

    def display_rois(self, ind, time):
        """
        """

        for roi in self.current_rois:
            self.removeItem(roi)

        self.current_rois = []
        for roi in self.overlay_rois[int(time)]:
            self.addItem(roi)
            self.current_rois.append(roi)
