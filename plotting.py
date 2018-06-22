from unet.summary import get_scalars_from_event
import numpy as np
import os
from contextlib import contextmanager
from statsmodels.nonparametric.api import KernelReg
import matplotlib.pyplot as plt


def plot_mean_and_CI(steps, mean, lb, ub, color_mean=None, color_shading=None, label=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(steps, ub, lb,
                     color=color_shading, alpha=.33)
    # plot the mean on top
    plt.plot(steps, mean, color=color_mean, label=label)


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


def graph_with_error_bars(sources, tag, color, label=None):
    all_steps = []
    all_vals = []
    for source in sources:
        steps, values = get_scalars_from_event(source, tag)
        all_steps += steps
        all_vals += values

    # now fit a line and error bars
    model = KernelReg(endog=[all_vals], exog=[all_steps], var_type='c')
    steps = np.linspace(0, 20000, 100)
    mean, mfx = model.fit(steps)
    mean_at, _ = model.fit(all_steps)
    errors_sq = (all_vals - mean_at)**2
    error_model = KernelReg(endog=[errors_sq], exog=[all_steps], var_type='c', bw=[500])
    estim_err, _ = error_model.fit(steps)
    estim_err = np.sqrt(np.maximum(0, estim_err))

    plot_mean_and_CI(steps, mean, mean - estim_err, mean + estim_err, color, color, label=label)
    plt.scatter(all_steps, all_vals, s=1, c=color)


@contextmanager
def make_plot(name, xlabel="steps", ylabel="IoU"):
    save_path = os.path.join("report", "figures", name+".pdf")
    fig = plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0, 20000])
    yield
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close(fig)


def plot_multiple_evals(tag, data):
    return plot_multiple(tag, data, all_evals_of)


def plot_multiple_runs(tag, data):
    return plot_multiple(tag, data, all_runs_of)


def plot_multiple(tag, data, log_source_fn):
    # data should be pairs source, label
    colors = ["blue", "orange", "red", "black"]
    for i, dat in enumerate(data):
        color = colors[i]
        source, label = dat
        graph_with_error_bars(log_source_fn(source), tag, color, label)


os.makedirs("report/figures", exist_ok=True)

if False:
    with make_plot("augment"):
        plot_multiple_evals("iou", [("simple_no_extra_aug", "Basic Augmentation"), ("simple", "Full Augmentation")])
        plt.ylim([0.8, 0.95])

    with make_plot("mse"):
        plot_multiple_evals("iou", [("mse", "Mean Squared Error"), ("simple", "Cross Entropy")])
        plt.ylim([0.8, 0.95])

    with make_plot("arch"):
        plot_multiple_evals("iou", [("simple", "Depth 4"), ("depth3", "Depth 3"), ("upscaling", "Upscaling")])
        plt.ylim([0.8, 0.95])

    with make_plot("arch_xent"):
        plt.ylabel('cross entropy')
        plot_multiple_evals("xent", [("simple", "Depth 4"), ("depth3", "Depth 3"), ("upscaling", "Upscaling")])

    with make_plot("disc_sgd_adam"):
        plot_multiple_evals("iou", [("upscaling", "No GAN"), ("dlr_1e-3", "SGD"), ("dlr_5e-5_ADAM", "ADAM")])
        plt.ylim([0.8, 0.95])

    with make_plot("disc_lr_sgd"):
        plot_multiple_evals("iou", [("upscaling", "No GAN"), ("dlr_1e-3", "1e-3"), ("dlr_1e-4", "1e-4")])
        plt.ylim([0.85, 0.92])

    with make_plot("disc_clip"):
        plot_multiple_evals("iou", [("dlr_5e-5_ADAM", "ADAM"), ("dlr_5e-5_ADAM_clip", "clip"), ("dlr_5e-5_ADAM_noise", "noise")])
        plt.ylim([0.80, 0.92])

    with make_plot("dloss_sgd_adam", ylabel="Discriminator loss"):
        graph_with_error_bars(all_runs_of("dlr_1e-3"), "DiscriminatorLoss/total", "red", "SGD 1e-3")
        graph_with_error_bars(all_runs_of("dlr_1e-4"), "DiscriminatorLoss/total", "green", "SGD 1e-4")
        graph_with_error_bars(all_runs_of("dlr_5e-5_ADAM"), "DiscriminatorLoss/total", "black", "ADAM")
        graph_with_error_bars(all_runs_of("dlr_5e-5_ADAM_clip"), "DiscriminatorLoss/total", "orange", "clip")
        graph_with_error_bars(all_runs_of("dlr_5e-5_ADAM_noise"), "DiscriminatorLoss/total", "blue", "noise")

    with make_plot("gloss_sgd_adam", ylabel="Generator loss"):
        plt.xlim([500, 20000])
        graph_with_error_bars(all_runs_of("dlr_1e-3"), "GeneratorLoss/total", "red", "SGD 1e-3")
        graph_with_error_bars(all_runs_of("dlr_5e-5_ADAM"), "GeneratorLoss/total", "black", "ADAM")
        graph_with_error_bars(all_runs_of("dlr_5e-5_ADAM_clip"), "GeneratorLoss/total", "orange", "clip")
        graph_with_error_bars(all_runs_of("dlr_5e-5_ADAM_noise"), "GeneratorLoss/total", "blue", "noise")

    with make_plot("disc_more"):
        plot_multiple_evals("iou",
                            [("upscaling", "No GAN"),
                             ("dlr_5e-5_ADAM_clip_0.25", "lambda 0.25"),
                             ("dlr_5e-5_ADAM_clip_0.5", "lambda 0.5"),
                             ("dlr_only_0.1", "stop d")])
        plt.ylim([0.85, 0.95])

    with make_plot("dloss_more"):
        plot_multiple_runs("DiscriminatorLoss/total", [
            ("dlr_1e-3", "SGD 1e-3"), ("dlr_5e-5_ADAM_clip_0.25", "lambda 0.25"), ("dlr_5e-5_ADAM_clip_0.5", "lambda 0.5"),
            ("dlr_only_0.1", "stop d")
        ])

    with make_plot("gloss_more", ylabel="Generator loss"):
        plt.xlim([500, 20000])
        plot_multiple_runs("GeneratorLoss/total", [
            ("dlr_1e-3", "SGD 1e-3"), ("dlr_5e-5_ADAM_clip_0.25", "lambda 0.25"), ("dlr_5e-5_ADAM_clip_0.5", "lambda 0.5"),
            ("dlr_only_0.1", "stop d")
        ])

with make_plot("less_data"):
    plot_multiple_evals("iou",
                        [("less_simple", "U-Net"),
                         ("less_gan", "U-Net+Gan")])
