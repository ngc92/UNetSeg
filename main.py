import os, json
import numpy as np
from unet.cell_segmentation import CellSegmentationConfig, CellSegmentationModel, InputConfig
from collections import namedtuple

from unet.summary import get_scalars_from_event

ExpConfig = namedtuple("ExpConfig", ["name", "input", "model"])

# GLOBAL PARAMS
BATCH_SIZE = 8
SCALE_FACTOR = 2
CROP_SIZE = 256
NUM_STEPS = 10000


# define a set of configurations that will be run
def input_cfg(local_aug, blurring):
    return InputConfig(BATCH_SIZE, local_aug, SCALE_FACTOR, CROP_SIZE, blurring)


def seg_cfg(resize=False, num_layers=4, xent_w=1.0, mse_w=0.0, multiscale=1, disc_w=0.0,
            disc_cap=1.0):
    return CellSegmentationConfig(num_layers, 32, resize, xent_w, mse_w, multiscale, disc_w, disc_cap, NUM_STEPS)


def make_jsonable(data):
    jsonable = {}
    for key in data:
        try:
            jsonable[key] = data[key].tolist()
        except AttributeError:
            if not isinstance(data[key], bytes):
                jsonable[key] = data[key]
    return jsonable


good_input = input_cfg(True, True)
configurations = [
    # use a standard UNet with fixed parameters and vary only the input preprocessing
    #ExpConfig("simple_no_extra_aug", input_cfg(False, False), seg_cfg()),
    #ExpConfig("simple_with_blur", input_cfg(False, True), seg_cfg()),
    #ExpConfig("simple_with_local_aug", input_cfg(True, False), seg_cfg()),
    ExpConfig("simple", input_cfg(True, True), seg_cfg()),
    # the upscaling network
    ExpConfig("upscaling", input_cfg(True, True), seg_cfg(resize=True)),
    # now start with cooler loss functions
    ExpConfig("mse", good_input, seg_cfg(xent_w=0.0, mse_w=1.0)),
    ExpConfig("depth3", good_input, seg_cfg(num_layers=3)),
    ExpConfig("disc_uncapped", good_input, seg_cfg(disc_w=0.2, disc_cap=100)),
    ExpConfig("disc_0.2", good_input, seg_cfg(disc_w=0.2)),
    ExpConfig("disc_0.1", good_input, seg_cfg(disc_w=0.1)),
    ExpConfig("disc_0.5", good_input, seg_cfg(disc_w=0.5)),
]

results = {}

for config in configurations:
    model = CellSegmentationModel(os.path.join("ckp", config.name), config.input, config.model)

    # if not yet trained: train model
    while model.global_step < NUM_STEPS:
        model.train("train.tfrecords", reps=50)
        model.eval("eval.tfrecords")

    # if not yet existing: make predictions
    result_dir = os.path.join("result", config.name)
    if not os.path.exists(result_dir):
        pred = model.predict("eval.tfrecords")
        os.makedirs(result_dir)
        for p in pred:
            import scipy.misc

            name = p["key"].decode("utf-8")
            scipy.misc.imsave(os.path.join(result_dir, name + ".png"),
                              p["generated_soft"][:, :, 0])
            scipy.misc.imsave(os.path.join(result_dir, name + "_seg.png"),
                              p["connected_components"])

        # and also do evaluation on eval and training set and save results
        test_result = model.eval("eval.tfrecords")
        train_result = model.eval("train.tfrecords", name="train")
        results[config.name] = {"test": make_jsonable(test_result), "train": make_jsonable(train_result)}
        print(test_result)
        with open(os.path.join(result_dir, "eval.json"), "w") as file_:
            json.dump(results[config.name], file_, indent=2)

    else:
        with open(os.path.join(result_dir, "eval.json"), "r") as file_:
            results[config.name] = json.load(file_)

print(results)

for config in configurations:
    result_dir = os.path.join("ckp", config.name, "eval")
    steps, values = get_scalars_from_event(result_dir, "iou")
    np.savetxt(os.path.join("report", "data", config.name+"_iou.txt"), np.transpose([steps, values]))

    steps, values = get_scalars_from_event(result_dir, "xent")
    np.savetxt(os.path.join("report", "data", config.name + "_xent.txt"), np.transpose([steps, values]))