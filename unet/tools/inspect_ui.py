import tkinter as tk
from tkinter import ttk
import tensorflow as tf

from unet.tools.image_widget import ImageWidget


class InspectUI:
    """
    Given a tkinter window, places image widgets to display original and segmentation images.
    """
    def __init__(self, main_window, model, data_loader: callable):
        self._root = main_window
        self._model = model
        self._data_loader = data_loader

        def _genwidget(col, row):
            source_image = ttk.Label(main_window, text="")
            source_image.grid(column=col, row=row)
            return ImageWidget(source_image, 256, 256)

        # these are always available
        self._source_image = _genwidget(0, 0)
        self._result_image = _genwidget(1, 0)

        # these are only available if we have ground truth
        self._gt_image = _genwidget(0, 1)
        self._error_image = _genwidget(1, 1)

    def load_image(self, path):
        self._source_image.update(path)
        image = self._data_loader(path)
        segmentation = tf.image.convert_image_dtype(self._model(image[None, ...])[0, ..., 0], dtype=tf.uint8).numpy()
        self._result_image.update(segmentation)

    @staticmethod
    def ui_main_loop(model, data_loader):
        win = tk.Tk()
        win.title("Dataset Tool")
        app = InspectUI(win, model, data_loader)
        app.load_image("/home/erik/PycharmProjects/UNet/data/train/original/000.png")
        win.mainloop()
