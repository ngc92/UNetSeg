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
    assert len(all_vals) > 0

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

#with make_plot("augment"):
#    plot_multiple_evals("iou", [("simple_no_extra_aug", "Basic Augmentation"), ("simple", "Full Augmentation")])
#    plt.ylim([0.8, 0.95])
#
#with make_plot("mse"):
#    plot_multiple_evals("iou", [("mse", "Mean Squared Error"), ("simple", "Cross Entropy")])
#    plt.ylim([0.8, 0.95])
#
#with make_plot("arch"):
#    plot_multiple_evals("iou", [("simple", "Deconv 4"),
#                                ("depth3", "Deconv 3"),
#                                ("upscaling", "Upscaling 4"),
#                                ("depth3-ups", "Upscaling 3")])
#    plt.ylim([0.8, 0.95])
#
#with make_plot("arch_xent"):
#    plt.ylabel('cross entropy')
#    plot_multiple_evals("xent", [("simple", "Deconv 4"),
#                                 ("depth3", "Deconv 3"),
#                                 ("upscaling", "Upscaling 4"),
#                                 ("depth3-ups", "Upscaling 3")])
#
#with make_plot("gan_3_opt"):
#    plot_multiple_evals("iou", [("upscaling", "No GAN"),
#                                ("gan_3_sgd", "SGD"),
#                                ("gan_3_slow", "ADAM(1e-5)"),
#                                ("gan_3_fast", "ADAM(1e-4)")])
#    plt.ylim([0.8, 0.95])
#
#with make_plot("gan_3_w"):
#    plot_multiple_evals("iou", [("upscaling", "0.0"),
#                                ("gan_3_w_0.15", "0.15"),
#                                ("gan_3_w_0.25", "0.25"),
#                                ("gan_3_w_0.33", "0.33")])
#    plt.ylim([0.86, 0.94])
#
#with make_plot("gan_3_no_fm"):
#    plot_multiple_evals("iou", [("gan_3_w_0.25", "feature matching"),
#                                ("gan_3_no_fm", "no feature matching")])
#    plt.ylim([0.86, 0.94])
#
#with make_plot("gan_4_w"):
#    plot_multiple_evals("iou", [("upscaling", "0.0"),
#                                ("gan_4_w_0.15", "0.15"),
#                                ("gan_4_w_0.25", "0.25"),
#                                ("gan_4_w_0.33", "0.33")])
#    plt.ylim([0.86, 0.94])
#
#with make_plot("dloss_gan4", ylabel="Discriminator loss"):
#    graph_with_error_bars(all_runs_of("gan_4_w_0.15"), "DiscriminatorLoss/total", "red", "0.15")
#    graph_with_error_bars(all_runs_of("gan_4_w_0.25"), "DiscriminatorLoss/total", "blue", "0.25")
#    graph_with_error_bars(all_runs_of("gan_4_w_0.33"), "DiscriminatorLoss/total", "black", "0.33")
#
#with make_plot("gloss_gan4", ylabel="Generator loss"):
#    plt.xlim([500, 20000])
#    graph_with_error_bars(all_runs_of("gan_3_w_0.15"), "GeneratorLoss/total", "red", "0.15")
#    graph_with_error_bars(all_runs_of("gan_3_w_0.25"), "GeneratorLoss/total", "blue", "0.25")
#    graph_with_error_bars(all_runs_of("gan_3_w_0.33"), "GeneratorLoss/total", "black", "0.33")
#
with make_plot("dloss_opt", ylabel="Discriminator loss"):
    plt.xlim([500, 20000])
    graph_with_error_bars(all_runs_of("gan_3_sgd"), "DiscriminatorLoss/total", "blue", "SGD")
    graph_with_error_bars(all_runs_of("gan_3_slow"), "DiscriminatorLoss/total", "black", "ADAM(1e-5)")
    graph_with_error_bars(all_runs_of("gan_3_fast"), "DiscriminatorLoss/total", "orange", "ADAM(1e-4)")

with make_plot("gloss_opt", ylabel="Generator loss"):
    plt.xlim([500, 20000])
    graph_with_error_bars(all_runs_of("gan_3_sgd"), "GeneratorLoss/total", "blue", "SGD")
    graph_with_error_bars(all_runs_of("gan_3_slow"), "GeneratorLoss/total", "black", "ADAM(1e-5)")
    graph_with_error_bars(all_runs_of("gan_3_fast"), "GeneratorLoss/total", "orange", "ADAM(1e-4)")



with make_plot("less_data"):
    plot_multiple_evals("iou",
                        [("less_simple", "U-Net"),
                         ("less_gan", "U-Net+Gan")])
