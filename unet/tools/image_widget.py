from pathlib import Path

from PIL import ImageTk, Image


def reshape_image(image, width, height=None):
    if height is None:
        height = width
    old_size = image.size  # old_size[0] is in (width, height) format
    ratio = min(float(width) / old_size[0], float(height)/old_size[1])
    new_size = tuple([int(x * ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image
    # thumbnail is a in-place operation
    # im.thumbnail(new_size, Image.ANTIALIAS)
    im = image.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it
    new_im = Image.new("RGBA", (width, height), color=0)
    new_im.paste(im, ((width - new_size[0]) // 2,
                      (height - new_size[1]) // 2))
    return new_im


class ImageWidget:
    def __init__(self, target, width, height):
        self._target = target
        self._photo_image = None
        self._width = width
        self._height = height
        self.on_click = None

        def on_click(event):
            x = event.x
            y = event.y
            if self.on_click is not None:
                self.on_click(x, y)

        target.bind("<Button 1>", on_click)

    def update(self, path_or_image):
        if isinstance(path_or_image, (str, Path)):
            img_data = Image.open(path_or_image)
        else:
            img_data = Image.fromarray(path_or_image)

        img_data = reshape_image(img_data, self._width, self._height)
        img_data = img_data.resize((self._width, self._height))

        img = ImageTk.PhotoImage(img_data)
        self._target.configure(image=img)
        self._photo_image = img
