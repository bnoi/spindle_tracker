import logging

log = logging.getLogger(__name__)

from ..tracking import Tracker


class SPBTracker(Tracker):

    MINIMUM_METADATA = ['SizeX', 'SizeY',
                        'PhysicalSizeX', 'PhysicalSizeY',
                        'TimeIncrement']
