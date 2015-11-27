import xml.etree.cElementTree as et

import pandas as pd
import numpy as np


def trackmate_peak_import(trackmate_xml_path, get_tracks=False):
    """Import detected peaks with TrackMate Fiji plugin.

    Parameters
    ----------
    trackmate_xml_path : str
        TrackMate XML file path.
    get_tracks : boolean
        Add tracks to label
    """

    root = et.fromstring(open(trackmate_xml_path).read())

    objects = []
    object_labels = {'FRAME': 't_stamp',
                     'POSITION_T': 't',
                     'POSITION_X': 'x',
                     'POSITION_Y': 'y',
                     'POSITION_Z': 'z',
                     'MEAN_INTENSITY': 'I',
                     'ESTIMATED_DIAMETER': 'w',
                     'QUALITY': 'q',
                     'ID': 'spot_id',
                     # 'MEAN_INTENSITY': 'mean_intensity',
                     'MEDIAN_INTENSITY': 'median_intensity',
                     'MIN_INTENSITY': 'min_intensity',
                     'MAX_INTENSITY': 'max_intensity',
                     'TOTAL_INTENSITY': 'total_intensity',
                     'STANDARD_DEVIATION': 'std_intensity',
                     'CONTRAST': 'contrast',
                     'SNR': 'snr',}


    features = root.find('Model').find('FeatureDeclarations').find('SpotFeatures')
    features = [c.get('feature') for c in features.getchildren()] + ['ID']

    spots = root.find('Model').find('AllSpots')
    trajs = pd.DataFrame([])
    objects = []
    for frame in spots.findall('SpotsInFrame'):
        for spot in frame.findall('Spot'):

            single_object = []
            for label in features:
                single_object.append(spot.get(label))

            objects.append(single_object)
    trajs = pd.DataFrame(objects, columns=features)
    trajs = trajs.astype(np.float)

    # Apply initial filtering
    initial_filter = root.find("Settings").find("InitialSpotFilter")

    trajs = filter_spots(trajs,
                         name=initial_filter.get('feature'),
                         value=float(initial_filter.get('value')),
                         isabove=True if initial_filter.get('isabove') == 'true' else False)

    # Apply filters
    spot_filters = root.find("Settings").find("SpotFilterCollection")

    for spot_filter in spot_filters.findall('Filter'):

        trajs = filter_spots(trajs,
                             name=spot_filter.get('feature'),
                             value=float(spot_filter.get('value')),
                             isabove=True if spot_filter.get('isabove') == 'true' else False)

    trajs = trajs.loc[:, object_labels.keys()]
    trajs.columns = [object_labels[k] for k in object_labels.keys()]
    trajs['label'] = np.arange(trajs.shape[0])

    # Get tracks
    if get_tracks:
        filtered_track_ids = [int(track.get('TRACK_ID')) for track in root.find('Model').find('FilteredTracks').findall('TrackID')]

        trajs['label'] = np.nan
        trajs = trajs.set_index('spot_id')

        tracks = root.find('Model').find('AllTracks')
        for track in tracks.findall('Track'):
            track_id = int(track.get("TRACK_ID"))

            if track_id in filtered_track_ids:

                spot_ids = []
                for edge in track.findall('Edge'):
                    spot_ids.append(int(edge.get('SPOT_SOURCE_ID')))
                    spot_ids.append(int(edge.get('SPOT_TARGET_ID')))

                spot_ids = np.unique(spot_ids)
                trajs.loc[spot_ids, 'label'] = track_id

        trajs = trajs.reset_index()

        # Remove spot without labels
        trajs = trajs.dropna(subset=['label'])

    trajs.set_index(['t_stamp', 'label'], inplace=True)
    trajs = trajs.sort_index()

    return trajs


def filter_spots(spots, name, value, isabove):
    if isabove:
        spots = spots[spots[name] > value]
    else:
        spots = spots[spots[name] < value]

    return spots
