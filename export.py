import os
from collections import defaultdict

import numpy as np

from unet.summary import get_scalars_from_event


def all_runs_of_iter(name, use_eval=True):
    i = 0
    while True:
        path = os.path.join("ckp", name, str(i))
        if use_eval:
            path = os.path.join(path, "eval")
        if os.path.exists(path):
            yield path
        else:
            return
        i += 1


def all_evals_of(name):
    return list(all_runs_of_iter(name))


def all_runs_of(name):
    return list(all_runs_of_iter(name, False))


def extract_tags(sources, tags):
    result = defaultdict(list)
    for source in sources:
        for tag in tags:
            steps, values = get_scalars_from_event(source, tag)
            result[tag+"_steps"] += steps
            result[tag] += values

    return {k: np.array(result[k]) for k in result}


if __name__ == "__main__":
    EVAL_TAGS = [
        "iou",
        "xent",
        ]

    TRAIN_TAGS = [
        "DiscriminatorLoss/total",
        "GeneratorLoss/total",
    ]

    RUNS = [
        "simple_no_extra_aug",
        "simple",
        "upscaling",
        "depth3",
        "depth3-ups",
        "mse",
        "gan_3_w_0.25",
        "gan_4_w_0.25",
        "gan_3_w_0.15",
        "gan_4_w_0.15",
        "gan_3_w_0.33",
        "gan_4_w_0.33",
        "gan_3_slow",
        "gan_3_sgd",
        "gan_3_fast",
        "gan_3_no_fm",
        "gan_4_cd",
        "less_simple",
        "less_gan"
    ]

    for run in RUNS:
        eval_data = extract_tags(all_evals_of(run), EVAL_TAGS)
        train_data = extract_tags(all_runs_of(run), TRAIN_TAGS)
        np.savez(run+".npz", **train_data, **eval_data)
