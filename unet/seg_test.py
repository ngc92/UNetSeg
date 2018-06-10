from collections import namedtuple
import numpy as np
import tensorflow as tf


FillRegion = namedtuple("FillRegion", ("x", "y", "area"))


def floodfill(image, xy, get_mean=False):
    image = np.copy(image)
    target = np.zeros_like(image, dtype=np.bool)
    x, y = xy
    edge = [(x, y)]
    mx, my = x, y
    target[x, y] = True
    total = 1.0
    while edge:
        new_edges = []
        for (x, y) in edge:
            for (s, t) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                try:
                    p = image[s, t]
                except IndexError:
                    pass
                else:
                    if p < 1:
                        target[s, t] = True
                        mx += s
                        my += t
                        total += 1.0
                        image[s, t] = 1
                        new_edges.append((s, t))
        edge = new_edges
    if get_mean:
        return target, FillRegion(mx/total, my/total, total)
    else:
        return target


def iterate_segmentation(image):
    # select a random position in the images
    width = image.shape[0]
    height = image.shape[1]
    border = floodfill(image, (0, 0))
    border |= (image != 0)
    rx, ry = 0, 0
    while True:
        while border[rx, ry]:
            rx += 1
            if rx >= width:
                rx = 0
                ry += 1
                if ry >= height:
                    return

        filled, region = floodfill(image, (rx, ry), True)
        border |= filled

        yield filled, region


def segmentation_iou(reference, image_b):
    print("iou")
    intersections = []
    unions = []
    for a, meta in iterate_segmentation(reference):
        cx, cy = int(meta.x), int(meta.y)
        b = floodfill(image_b, (cx, cy))
        intersection = np.sum(a & b)
        union = np.sum(a | b)
        if union < 1e-5:
            union = 1e-5

        intersections.append(intersection)
        unions.append(union)

    return sum(map(lambda x: x[0]/x[1], zip(intersections, unions))) / len(intersections)


def segment_with_reference(reference, image_b):
    result_s = np.zeros_like(image_b, dtype=np.uint8)
    result_r = np.zeros_like(image_b, dtype=np.uint8)
    counter = 1
    for a, meta in iterate_segmentation(reference):
        cx, cy = int(meta.x), int(meta.y)
        b = floodfill(image_b, (cx, cy))
        result_s += (b * counter).astype(np.uint8)
        result_r += (a * counter).astype(np.uint8)
        counter += 1

    return result_r, result_s


def segmentation_iou_op(reference, image_b):
    import tensorflow as tf

    def iou_tf(reference, image_b):
        ious = []
        for i in range(reference.shape[0]):
            ious += [segmentation_iou(reference[i], image_b[i])]
        return np.mean(ious).astype(np.float32)

    iou_t = tf.py_func(iou_tf, [reference, image_b], tf.float32, stateful=False)
    iou_t.set_shape(())
    return iou_t


def segment_op(reference, image_b, num_images=1):
    import tensorflow as tf

    def segment_tf(reference, image_b):
        seg_b = []
        seg_a = []
        for i in range(min(num_images, reference.shape[0])):
            a, b = segment_with_reference(reference[i], image_b[i])
            seg_a += [a]
            seg_b += [b]
        return np.concatenate(seg_a).reshape((-1,)+image_b.shape[1:]), np.concatenate(seg_b).reshape((-1,)+image_b.shape[1:])

    sega_t, segb_t = tf.py_func(segment_tf, [reference, image_b], (tf.uint8, tf.uint8), stateful=False)
    sega_t.set_shape((None,)+tuple(image_b.shape[1:].as_list()))
    segb_t.set_shape((None,)+tuple(image_b.shape[1:].as_list()))
    return sega_t, segb_t


def iou_from_segmentation_batched(seg_a, seg_b):
    ious = []
    for i in range(seg_a.shape[0]):
        ious.append(iou_from_segmentation(seg_a[i], seg_b[i]))
    return np.array(ious)


def iou_from_segmentation(seg_a, seg_b):
    mxa = np.max(seg_a)
    seg_a[seg_a == 0] = mxa + 1
    mxb = np.max(seg_b)
    seg_b[seg_b == 0] = mxb + 1

    mean_iou = 0
    valid_count = 0

    # iterate over all clusters in a
    for segment in range(np.min(seg_a), mxa):
        segment_a = seg_a == segment
        overlap = (segment_a * seg_b)
        counts = np.bincount(np.reshape(overlap, -1))[1:]  # first entry will be the number of zeros (values outside of segment_a)
        best = np.argmax(counts)
        segment_b = seg_b == best + 1

        union = np.count_nonzero(segment_a | segment_b)
        intersection = np.count_nonzero(segment_a & segment_b)

        if union != 0:
            mean_iou += intersection / union
            valid_count += 1

    if valid_count != 0:
        return mean_iou / valid_count
    return 0.0


def iou_from_intersection_op(reference, image):
    import tensorflow as tf

    def iou_tf(reference, image):
        return iou_from_segmentation_batched(reference, image).astype(np.float32)

    iou = tf.py_func(iou_tf, [reference, image], tf.float32, stateful=False)
    return iou
