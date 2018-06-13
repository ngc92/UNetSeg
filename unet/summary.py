import tensorflow as tf
import glob
import os


def iterate_summaries(directory, tags):
    pattern = os.path.join(directory, "events.out.tfevents.*")
    for fn in glob.iglob(pattern):
        try:
            for e in tf.train.summary_iterator(fn):
                for v in e.summary.value:
                    if v.tag in tags:
                        yield e.step, v
        except tf.errors.DataLossError:
            pass


def get_scalars_from_event(dir, tag):
    steps = []
    values = []
    for step, summary in iterate_summaries(dir, [tag]):
        steps.append(step)
        values.append(summary.simple_value)

    return steps, values



