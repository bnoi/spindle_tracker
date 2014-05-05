from IPython.html import widgets
from IPython.display import display
from IPython.display import clear_output

import matplotlib.pyplot as plt

from mpld3 import plugins


class KymoAnnotator:

    def __init__(self, i, tracker):
        """
        """
        self.tracker = tracker
        self.widgets = []
        self.i = i

    def run(self):
        """
        """

        display("File : {}".format(self.tracker))

        self.build_widgets()
        display(self.kymo_state_w)
        display(self.anaphase_w)
        display(self.button)

        self.fig = self.tracker.kymo()
        plugins.clear(self.fig)
        plugins.connect(self.fig,
                        plugins.Reset(),
                        plugins.BoxZoom(),
                        plugins.Zoom(),
                        plugins.MousePosition(fontsize=14))
        self.fig.show()

    def save_tracker(self, button=None):
        """
        """

        to_save = False

        kymo_state = self.kymo_state_w.value
        anaphase = self.anaphase_w.value

        if int(kymo_state) != self.tracker.annotations['kymo']:
            display("Kymo state updated : {}".format(kymo_state))
            self.tracker.annotations['kymo'] = int(kymo_state)
            to_save = True

        if int(anaphase) != self.tracker.annotations['anaphase']:
            display("Anaphase updated : {}".format(anaphase))
            self.tracker.annotations['anaphase'] = int(anaphase)
            to_save = True

        if to_save:
            display('Saving {}'.format(self.tracker))
            self.tracker.save_oio()
        else:
            display("Nothing to save")

    def build_widgets(self):
        """
        """

        self.kymo_state_w = widgets.ToggleButtonsWidget(values={'no annotated': 0, 'good': 1, 'bad': 2},
                                                        value=int(self.tracker.annotations['kymo']),
                                                        description='Kymo state : ')

        self.anaphase_w = widgets.BoundedIntTextWidget(value=int(self.tracker.annotations['anaphase']),
                                                       min=0, max=1000000,
                                                       description='Anaphase onset : ')

        self.button = widgets.ButtonWidget(description="Save")
        self.button.on_click(self.save_tracker)

    def clear(self):
        """
        """
        self.save_tracker()
        del self.fig
        plt.clf()
        plt.close()
        self.kymo_state_w.close()
        self.anaphase_w.close()
        self.button.close()
        clear_output()
        del self.tracker


class DatasetKymoAnnotator:

    def __init__(self, dataset, start=0):
        """
        """
        self.dataset = dataset
        self.i = start
        self.n = len(self.dataset)
        self.ka = None

    def run(self):
        """
        """

        self.button = widgets.ButtonWidget(description="Next")
        display(self.button)
        self.button.on_click(self.next_tracker)
        self.next_tracker(self.button)

    def next_tracker(self, button):

        i = self.i

        if self.ka:
            self.ka.clear()

        display("n = {}/{}".format(i+1, self.n))

        tracker = self.dataset[i]
        self.ka = KymoAnnotator(i, tracker)
        self.ka.run()

        self.i += 1
