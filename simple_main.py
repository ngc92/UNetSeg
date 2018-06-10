import os, json
from unet.cell_segmentation import CellSegmentationConfig, CellSegmentationModel, InputConfig
from collections import namedtuple

ExpConfig = namedtuple("ExpConfig", ["name", "input", "model"])

# GLOBAL PARAMS
BATCH_SIZE = 8
SCALE_FACTOR = 2
CROP_SIZE = 256
NUM_STEPS = 10000


# define a set of configurations that will be run
def input_cfg(local_aug, blurring):
    return InputConfig(BATCH_SIZE, local_aug, SCALE_FACTOR, CROP_SIZE, blurring)


def seg_cfg(resize=False, xent_w=1.0, mse_w=0.0, low_res_w=0.0, low_res_xent_w=0.0, multiscale=1, disc_w=0.0):
    return CellSegmentationConfig(4, 32, resize, xent_w, mse_w, low_res_w, low_res_xent_w, multiscale, disc_w, NUM_STEPS)


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
    ExpConfig("simple_no_extra_aug", input_cfg(False, False), seg_cfg()),
    ExpConfig("simple_with_blur", input_cfg(False, True), seg_cfg()),
    ExpConfig("simple_with_local_aug", input_cfg(True, False), seg_cfg()),
    ExpConfig("simple", input_cfg(True, True), seg_cfg()),
    # the upscaling network
    ExpConfig("upscaling", input_cfg(True, True), seg_cfg(resize=True)),
    # now start with cooler loss functions
    ExpConfig("with_lr_xent", good_input, seg_cfg(low_res_xent_w=1.0)),
    ExpConfig("with_full_mse", good_input, seg_cfg(mse_w=1.0, low_res_w=1.0, multiscale=3)),
    ExpConfig("disc_0.2", good_input, seg_cfg(disc_w=0.2)),
    ExpConfig("disc_0.1", good_input, seg_cfg(disc_w=0.1)),
]

results = {}

for config in configurations:
    model = CellSegmentationModel(os.path.join("ckp", config.name), config.input, config.model)

    # if not yet trained: train model
    while model.global_step < NUM_STEPS:
        model.train("records.tfrecords", reps=50)
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
        train_result = model.eval("records.tfrecords")
        results[config.name] = {"test": make_jsonable(test_result), "train": make_jsonable(train_result)}
        print(test_result)
        with open(os.path.join(result_dir, "eval.json"), "w") as file_:
            json.dump(results[config.name], file_, indent=2)

    else:
        with open(os.path.join(result_dir, "eval.json"), "r") as file_:
            results[config.name] = json.load(file_)

print(results)
exit(0)


icfg = InputConfig(BATCH_SIZE, True, SCALE_FACTOR, CROP_SIZE, True)
ccfg = CellSegmentationConfig(4, 32, True, 0.1, 1.0, 1.0, 3, 0.0, 10000)

model = CellSegmentationModel("ckp/simple", icfg, ccfg)

for i in range(10):
    model.train("/home/erik/PycharmProjects/UNet/records.tfrecords", reps=50)
    model.eval("eval.tfrecords")
    pred = model.predict("eval.tfrecords")
    os.makedirs("result/" + str(i), exist_ok=True)

    for p in pred:
        import scipy.misc
        name = p["key"].decode("utf-8")
        scipy.misc.imsave(os.path.join("result", str(i), name+".png"),
                          p["generated_soft"][:, :, 0])
        scipy.misc.imsave(os.path.join("result", str(i), name + "_seg.png"),
                          p["connected_components"][:, :, 0])


