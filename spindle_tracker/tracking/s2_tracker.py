import logging

log = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from ..tracking import Tracker


class S2Tracker(Tracker):

    MINIMUM_METADATA = ['SizeX', 'SizeY',
                        'PhysicalSizeX', 'PhysicalSizeY',
                        'TimeIncrement']

    # ANNOTATIONS = {'state': (0, [0, 1, 2], None),}

    def __init__(self, *args, **kwargs):
        """
        """

        super().__init__(*args, **kwargs)


    def show(self, traj):
        """
        """

        fig = plt.figure(figsize=(15, 12))

        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        ax4 = plt.subplot2grid((2, 2), (1, 1), projection='3d')

        traj.show(xaxis='t', yaxis='x', ax=ax1)
        traj.show(xaxis='t', yaxis='y', ax=ax2)
        traj.show(xaxis='t', yaxis='z', ax=ax3)

        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
