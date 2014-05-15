import logging

log = logging.getLogger(__name__)

from sktracker.trajectories import Trajectories

from ..tracking import Tracker


class Ndc80Tracker(Tracker):

    MINIMUM_METADATA = ['SizeX', 'SizeY', 'SizeZ',
                        'PhysicalSizeX', 'PhysicalSizeY',
                        'TimeIncrement']

    def __init__(self, *args, **kwargs):
        """
        """

        super().__init__(*args, **kwargs)

    def show(self, var_name='trajs'):
        """
        """

        import matplotlib.pyplot as plt

        trajs = Trajectories(getattr(self, var_name))

        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1 = trajs.show(xaxis='t', yaxis='x',
                         groupby_args={'level': "label"},
                         ax=ax1, ls='-', marker='')

        ax2 = trajs.show(xaxis='t', yaxis='y',
                         groupby_args={'level': "label"},
                         ax=ax2, ls='-', marker='')

        return fig

    def kymo(self, var_name='trajs'):
        """
        """

        import matplotlib.pyplot as plt

        trajs = Trajectories(getattr(self, var_name))

        fig, ax = plt.subplots(nrows=1)
        ax = trajs.show(xaxis='t', yaxis='x_proj',
                        groupby_args={'level': "label"},
                        ax=ax, ls='-', marker='')

        return fig
