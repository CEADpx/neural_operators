"""Post-process saved MCMC tracer results: curves and posterior field plots."""
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_plotting = _root / "plotting"
if str(_plotting) not in sys.path:
    sys.path.insert(0, str(_plotting))

from plot_curve import plot_curve
from mcmc_plot_fields import mcmc_plot_fields_base


def plot_mcmc_tracer_results(
    savepath,
    mcmc,
    pp_params,
    field_suptitle,
    sample_i=-1,
    tag="Final",
    plot_surrogate_forward=True,
    save_figures=True,
    tracer_filename="tracer.pkl",
):
    """Load tracer from savepath and plot cost, acceptance rate, and field panels.

    Parameters
    ----------
    savepath : str
        MCMC results directory containing the tracer pickle.
    mcmc : MCMC
        MCMC object used for the run (model, surrogate, ground truth data).
    pp_params : dict
        Plot styling with keys ``curve_plot`` and ``field_plot``.
    field_suptitle : str
        Figure suptitle for the field plot panels.
    sample_i : int
        Posterior sample index to plot. Default -1 uses the last sample.
    tag : str
        Suffix in saved figure filenames (e.g. ``Final``).
    plot_surrogate_forward : bool
        If True and a surrogate is active, also save the surrogate-forward panel.
    save_figures : bool
        If True, save PNGs to ``savepath``; if False, display only.
    tracer_filename : str
        Tracer pickle filename inside ``savepath``.

    Returns
    -------
    tracer
        Loaded tracer object.
    """
    saved_results = savepath
    tracer_path = os.path.join(saved_results, tracer_filename)

    with open(tracer_path, "rb") as f:
        tracer = pickle.load(f)

    curve_params = pp_params["curve_plot"]
    cost_savefile = (
        os.path.join(saved_results, f"cost_{tag}.png") if save_figures else None
    )
    acceptance_savefile = (
        os.path.join(saved_results, f"acceptance_rate_{tag}.png")
        if save_figures
        else None
    )

    plot_curve(
        tracer.accepted_samples_cost,
        xl=r"Samples",
        yl=r"Cost = $-\log(\pi_{like}(u_{obs} | w))$",
        fs=curve_params["fs"],
        lw=curve_params["lw"],
        figsize=curve_params["figsize"],
        savefile=cost_savefile,
    )
    plot_curve(
        tracer.acceptance_rate,
        xl=r"Samples",
        yl=r"Acceptance rate",
        fs=curve_params["fs"],
        lw=curve_params["lw"],
        figsize=curve_params["figsize"],
        savefile=acceptance_savefile,
    )

    w_mean = tracer.accepted_samples_mean_m
    w_sample_i = sample_i if sample_i >= 0 else len(tracer.accepted_samples_m) - 1
    w_sample = tracer.accepted_samples_m[w_sample_i]

    field_params = deepcopy(pp_params["field_plot"])
    field_params["sup_title"] = field_suptitle

    surrogate_to_use = mcmc.surrogate_to_use
    base = os.path.join(saved_results, "true_and_posterior_mean_w_m_u")
    field_savefile = f"{base}_true_F_for_u_{tag}.png" if save_figures else None

    mcmc_plot_fields_base(
        w_mean,
        w_sample,
        w_sample_i,
        mcmc,
        savefilename=field_savefile,
        params=field_params,
        surrogate_to_use=surrogate_to_use,
        use_surrogate_F_for_u=False,
    )

    if plot_surrogate_forward:
        surr_savefile = (
            f"{base}_surrogate_F_for_u_{tag}.png" if save_figures else None
        )
        mcmc_plot_fields_base(
            w_mean,
            w_sample,
            w_sample_i,
            mcmc,
            savefilename=surr_savefile,
            params=field_params,
            surrogate_to_use=surrogate_to_use,
            use_surrogate_F_for_u=True,
        )

    return tracer
