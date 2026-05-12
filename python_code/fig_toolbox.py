from logging import config
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.io
from matplotlib.ticker import MaxNLocator, FixedLocator
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from collections import OrderedDict
import re
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

def serr(x, dim=0):
    """Calculate the standard error of the mean."""
    s = np.std(x, axis=dim, ddof=1)  # Standard deviation
    n = x.shape[dim]  # Number of samples
    return s / np.sqrt(n)

def preprocess_effect_data(lr):
    """
        Compute effect sizes for learning-rate data based on a 2×2 factorial design.

        This function takes an array of learning-rate values (`lr`) organized
        by experimental condition (small/large volatility × small/large stochasticity),
        applies a predefined contrast transformation to extract:
          1. The average learning-rate across all conditions (intercept),
          2. The main effect of stochasticity (small vs. large),
          3. The main effect of volatility (small vs. large).

        It then returns:
          • m_eff:  Array of mean values for each of the three contrasts.
          • e_eff:  Array of standard errors (SE) for each contrast column.
          • dis_avglr:  Flattened array of the “average learning-rate” contrast per subject.
          • dis_sto:    Flattened array of the “stochasticity” contrast per subject.
          • dis_vol:    Flattened array of the “volatility” contrast per subject.

        Parameters
        ----------
        lr : np.ndarray, shape (n_subjects, 4)
            Learning-rate data for each subject in the following order of columns:
            [small volatility & small stochasticity,
             large volatility & small stochasticity,
             small volatility & large stochasticity,
             large volatility & large stochasticity]

        Returns
        -------
        m_eff : np.ndarray, shape (3,)
            Mean effect sizes across subjects for:
              [average learning-rate, stochasticity effect, volatility effect].

        e_eff : np.ndarray, shape (3,)
            Standard errors of those three effect‐size estimates.

        dis_avglr : np.ndarray, shape (n_subjects,)
            Per‐subject contrast values corresponding to the average learning‐rate effect.

        dis_sto : np.ndarray, shape (n_subjects,)
            Per‐subject contrast values corresponding to the main effect of stochasticity
            (small vs. large).

        dis_vol : np.ndarray, shape (n_subjects,)
            Per‐subject contrast values corresponding to the main effect of volatility
            (small vs. large).
    """
    # Define the 4×3 contrast matrix for:
    #   - Column 0: average over all four cells
    #   - Column 1: small vs. large stochasticity
    #   - Column 2: small vs. large volatility
    transformation_matrix = np.array([[.25, .25, .25, .25],[-1, -1, 1, 1], [-1, 1, -1, 1]]).T # shape: (4 conditions, 3 contrasts)
    # Apply contrast transformation: (n_subjects × 4) dot (4 × 3) → (n_subjects × 3)
    eff = np.dot(lr, transformation_matrix)
    # Compute means and standard errors across subjects for each of the 3 contrasts
    m_eff = np.mean(eff, axis=0)
    sem_eff = serr(eff)
    # Extract per-subject contrast columns:
    #   * dis_avglr: column 0 → “average” effect (intercept)
    #   * dis_sto:   column 1 → main effect of stochasticity
    #   * dis_vol:   column 2 → main effect of volatility
    dis_avglr = eff[:, 0] # shape: (n_subjects,)
    dis_sto = eff[:, 1] # shape: (n_subjects,)
    dis_vol = eff[:, 2] # shape: (n_subjects,)

    # Return values into dict
    # m_eff: mean effect sizes for each contrast
    # e_eff: standard errors for each contrast
    # dis_avglr: per-subject contrast values for average learning-rate
    # dis_sto: per-subject contrast values for stochasticity effect
    # dis_vol: per-subject contrast values for volatility effect
    results = {'m_eff': m_eff, 'sem_eff': sem_eff,
               'Baseline Learning-Rate': dis_avglr,
               'Stochasticity Effect': dis_sto,
               'Volatility Effect': dis_vol,
               'eff': eff}

    return results

def plot_bars(betas, ses, config):
    """
    Plot grouped bar charts on a single axis.

    Parameters
    ----------
    betas : array-like, shape (n_groups, n_effects)
        Rows are groups, columns are effects.
    ses : array-like, shape (n_groups, n_effects)
        Standard errors matching betas.
    config : dict
        Plot settings.

    Optional config keys
    --------------------
    """

    betas = np.asarray(betas, dtype=float)
    ses = np.asarray(ses, dtype=float)

    if betas.ndim != 2:
        raise ValueError("betas must be a 2D array of shape (n_groups, n_effects).")
    if ses.shape != betas.shape:
        raise ValueError("ses must have the same shape as betas.")

    n_groups, n_effects = betas.shape

    figsize = config.get('FIGSIZE', (4, 3.5))
    pairwise_comps = config.get('PAIRWISE_COMPARISONS', None)
    ps = config.get('P_VALUES', None)
    p_strict = config.get('P_STRICT', 0.001)
    p_lenient = config.get('P_LENIENT', 0.05)
    star_offset = config.get('STAR_OFFSET', 0.01)

    if config['X_TICK_LABEL'] is not None and len(config['X_TICK_LABEL']) != n_effects:
        raise ValueError(f"X_TICK_LABEL must have length {n_effects}, but got {len(config['X_TICK_LABEL'])}.")
    if config['GROUP_LABEL'] is not None and len(config['GROUP_LABEL']) != n_groups:
        raise ValueError(f"GROUP_LABEL must have length {n_groups}, but got {len(config['GROUP_LABEL'])}.")
    if ps is not None:
        ps = np.asarray(ps, dtype=float)
        if ps.shape != betas.shape:
            raise ValueError("P_VALUES must have the same shape as betas.")
    if len(config['GROUP_COLOR']) < n_groups:
        raise ValueError(f"Need at least {n_groups} GROUP_COLOR, but got {len(config['GROUP_COLOR'])}.")

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_effects)
    # Fluid, symmetric offsets around each x position
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * config['BAR_SPACING']

    for g in range(n_groups):
        b = betas[g]
        se = ses[g]
        xpos = x + offsets[g]

        ax.bar(
            xpos,
            b,
            yerr=se,
            capsize=config['CAPSIZE'],
            alpha=config['BAR_ALPHA'],
            width=config['BAR_WIDTH'],
            color=config['GROUP_COLOR'][g],
            label=config['GROUP_LABEL'][g] if config['GROUP_LABEL'] is not None else None,
            edgecolor='white',
        )

        if ps is not None and pairwise_comps is None:
            p = ps[g]
            sig_strict = p < p_strict
            sig_lenient = (p < p_lenient) & (~sig_strict)
            # strict (two asterisks **)
            if np.any(sig_strict):
                y_strict = b[sig_strict] + np.sign(np.where(b[sig_strict] == 0, 1, b[sig_strict])) * (
                    se[sig_strict] + star_offset)
                for x_pos, y_pos in zip(xpos[sig_strict], y_strict):
                    ax.text(x_pos, y_pos, '**', ha='center', va='bottom' if y_pos > 0 else 'top', 
                           fontsize=config['TICK_FONTSIZE'], fontweight='bold', color='black', zorder=5)
            # lenient (one asterisk *)
            if np.any(sig_lenient):
                y_lenient = b[sig_lenient] + np.sign(np.where(b[sig_lenient] == 0, 1, b[sig_lenient])) * (
                    se[sig_lenient] + star_offset)
                for x_pos, y_pos in zip(xpos[sig_lenient], y_lenient):
                    ax.text(x_pos, y_pos, '*', ha='center', va='bottom' if y_pos > 0 else 'top',
                           fontsize=config['TICK_FONTSIZE'], fontweight='bold', color='black', zorder=5)

    # Reporting
    if ps is not None:
        print("\nSignificance report:")
        for g in range(n_groups):
            if config['GROUP_LABEL'] is not None:
                print(f"\n{config['GROUP_LABEL'][g]}:")
            for j in range(n_effects):
                p = ps[g, j]
                effect_name = config['X_TICK_LABEL'][j] if config['X_TICK_LABEL'] is not None else f"Effect {j+1}"
                if p < p_strict:
                    print(f"  Effect {effect_name}: p={p:.3g}  (**)")
                elif p < p_lenient:
                    print(f"  Effect {effect_name}: p={p:.3g}  (*)")
                else:
                    print(f"  Effect {effect_name}: p={p:.3g}")

    # Draw pairwise comparison significance lines (optional)
    if pairwise_comps is not None:
        line_y_offset = config.get('PAIRWISE_LINE_Y_OFFSET', 0.05)  # fraction of y-range for line spacing
        
        y_min, y_max = config['YLIM']
        y_range = y_max - y_min
        
        for effect_idx, group1_idx, group2_idx, p_val in pairwise_comps:
            if effect_idx >= n_effects or group1_idx >= n_groups or group2_idx >= n_groups:
                continue
            
            # Get x positions of both bars
            x1 = x[effect_idx] + offsets[group1_idx]
            x2 = x[effect_idx] + offsets[group2_idx]
            
            # Get bar heights
            y1 = betas[group1_idx, effect_idx] + ses[group1_idx, effect_idx]
            y2 = betas[group2_idx, effect_idx] + ses[group2_idx, effect_idx]
            
            # Draw line above both bars
            max_y = max(y1, y2)
            line_y = max_y + line_y_offset * y_range
            
            # Draw connecting line
            ax.plot([x1, x2], [line_y, line_y], 'k-', linewidth=1.5, zorder=5)
            
            # Add significance marker
            mid_x = (x1 + x2) / 2
            marker_y = line_y + 0.01 * y_range
            
            if p_val < p_strict:
                ax.text(mid_x, marker_y, '**', ha='center', va='bottom', 
                       fontsize=config['TICK_FONTSIZE'], fontweight='bold', color='black', zorder=5)
            elif p_val < p_lenient:
                ax.text(mid_x, marker_y, '*', ha='center', va='bottom', 
                       fontsize=config['TICK_FONTSIZE'], fontweight='bold', color='black', zorder=5)

    if config['X_TICK_LABEL'] is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(config['X_TICK_LABEL'], ha='center', fontsize=config['TICK_FONTSIZE'])
    else:
        # Hide x-axis tick labels (and tick marks) when no labels are provided
        ax.set_xticks([])
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.set_ylabel(config['YLABEL'], fontsize=config['LABEL_FONTSIZE'])
    ax.set_xlabel(config['XLABEL'], fontsize=config['LABEL_FONTSIZE'])
    ax.set_title(config['TITLE'], fontsize=config['TITLE_FONTSIZE'])
    ax.set_ylim(config['YLIM'])
    ax.tick_params(labelsize=config['TICK_FONTSIZE'])
    if config['SHOW_LEGEND']:
        ax.legend(frameon=False, fontsize=config['LEGEND_FONTSIZE'], title=config['LEGEND_TITLE'], loc=config['LEGEND_LOC'])
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    if config['SAVE_PATH']:
        os.makedirs(os.path.dirname(config['SAVE_PATH']), exist_ok=True)
        fig.savefig(config['SAVE_PATH'], dpi=300, bbox_inches='tight')

def plot_effect_size_with_distribution(eff_results, config):
    """
        Creates three vertically stacked subplots; in each subplot:
          • A horizontal bar (mean ± SE) for that effect type.
          • A vertical histogram of individual effect‐size samples beside the bar.

        Inputs:
        - eff_results: dict with keys:
            'm_eff'                   : [mean_avg_lr, mean_sto, mean_vol]
            'se_eff'                  : [se_avg_lr, se_sto,  se_vol]
            'Average Learning-Rate'   : array of individual avg_lr effect sizes
            'Stochasticity Effect'    : array of individual dis_sto effect sizes
            'Volatility Effect'       : array of individual dis_vol effect sizes
        - config: dict of plotting parameters (all optional; defaults shown):
    """
    # Unpack effect‐size data
    m_eff   = eff_results['m_eff']
    se_eff  = eff_results['sem_eff']
    avg_lr  = eff_results['Baseline Learning-Rate']
    dis_sto = eff_results['Stochasticity Effect']
    dis_vol = eff_results['Volatility Effect']

    # Configuration with defaults
    figsize     = config.get('FIGSIZE', (11, 3.5))
    xlim2       = config.get('XLIM2', (0, 0.75))  # for horizontal bar + histogram plot
    ylim        = config.get('YLIM', (0, 1.5))  # for horizontal bar + histogram plot
    hist_width  = config.get('HIST_WIDTH', 0.05)   # for scaling histograms
    hist_height = config.get('HIST_HEIGHT', 0.3) # for scaling histograms
    hist_color  = config.get('HIST_COLOR', 'gray')
    bar_colors  = config.get('GROUP_COLOR', ['green','red','cyan'])
    vline_style = config.get('VLINE_STYLE', {'color':'k','linestyle':'--','linewidth':1})

    # Prepare figure with 1 rows, 3 column
    gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[.5, .15, 1])  # Adjust width ratios if needed
    fig= plt.figure(figsize=figsize)
    ax0 = plt.subplot(gs[0])  # Main plot for average learning-rate
    ax1 = plt.subplot(gs[2])  # Secondary plot for horizontal bars and histograms

    def plot_bar_hist(ax, data, mean_val, se_val, bar_colors):
        """
        On ax:
        1) Plot a single vertical bar at x=0, height = mean_val, with y‐error = se_val
        2) Plot a horizontal histogram of the individual data points to the right of the bar
        """
        # 1) Vertical bar
        bar_loc = config['BAR_WIDTH']/2 + 0.1
        ax.bar(x=bar_loc, height=mean_val, yerr=se_val, width=config['BAR_WIDTH'], color=bar_colors[0], capsize=config['CAPSIZE'], alpha=config['BAR_ALPHA'])

        # 2) Horizontal histogram to the right of the bar
        counts, bin_edges = np.histogram(data, bins=np.arange(config['YLIM2'][0], config['YLIM2'][1], hist_width))
        # Plot vertical bars
        left_edge = bar_loc*2
        ax.barh(y= bin_edges[:-1], width=counts/counts.max() * hist_height, height=hist_width,
                left=left_edge, color=hist_color, alpha=config['BAR_ALPHA'], align='edge', edgecolor='white')
        # Axis limits
        ax.set_xlim(xlim2)
        ax.set_xticks([])
        ax.set_xlabel(config['YLABEL'][0], fontsize=config['LABEL_FONTSIZE'])
        ax.set_ylim(config['YLIM2'])
        ax.set_ylabel(config['XLABEL'], fontsize=config['LABEL_FONTSIZE'])
        # Clean up spines and grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    # Plot 1: Baseline Learning-Rate
    plot_bar_hist(ax0, avg_lr, m_eff[0], se_eff[0], bar_colors)
    bar_center_stoch = 0.5  # Center for red bar
    bar_center_vol = 1  # Center for blue bar
    
    # Plot 2: Stochasticity and Volatility Effects
    # Add horizontal bars for means with error bars
    ax1.errorbar(m_eff[1], bar_center_stoch, xerr=se_eff[1], fmt='none', color='k', capsize=config['CAPSIZE'], elinewidth=config['LINE_WIDTH'])
    ax1.barh(bar_center_stoch, m_eff[1], color=bar_colors[1], alpha=config['BAR_ALPHA'], height=config['BAR_WIDTH'])
    ax1.errorbar(m_eff[2], bar_center_vol, xerr=se_eff[2], fmt='none', color='k', capsize=config['CAPSIZE'], elinewidth=config['LINE_WIDTH'])
    ax1.barh(bar_center_vol, m_eff[2], color=bar_colors[2], alpha=config['BAR_ALPHA'], height=config['BAR_WIDTH'])
    # Histogram for True Stochasticity (Flipped upside down)
    bins = np.round(np.arange(config['XLIM'][0], config['XLIM'][1], hist_width), 2)
    hist_bs, bin_edges_bs = np.histogram(dis_sto, bins=bins)
    ax1.bar(bin_edges_bs[:-1], -hist_bs / np.max(hist_bs) * hist_height,  # Flip the red histogram
            width=hist_width, color=hist_color, alpha=0.5, bottom=bar_center_stoch - 0.15,
            align='edge', edgecolor='white')
    # Histogram for True Volatility (Placed above the blue bar)
    hist_bv, bin_edges_bv = np.histogram(dis_vol, bins=bins)
    ax1.bar(bin_edges_bv[:-1], hist_bv / np.max(hist_bv) * hist_height,  # Normal histogram
            width=hist_width, color=hist_color, alpha=0.5, bottom=bar_center_vol + 0.15,
            align='edge', edgecolor='white')
    # Add vertical line at 0
    ax1.axvline(0, color=vline_style['color'], linestyle=vline_style['linestyle'],
                    linewidth=vline_style['linewidth'], zorder=0)

    # Customize the plot
    ax1.set_xlim(config['XLIM'])
    ax1.set_xlabel(config['XLABEL'], fontsize=config['LABEL_FONTSIZE'])
    ax1.set_ylim(ylim)  # Adjusted limits to bring bars to the center
    ax1.set_yticks([bar_center_stoch, bar_center_vol])
    ax1.set_yticklabels([config['YLABEL'][1], config['YLABEL'][2]], va='center', fontsize=config['LABEL_FONTSIZE'])
    ax1.tick_params(axis='y', which='both', length=0)  # Removes the tick marks
    # Clean up spines and grid
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(False)
    plt.show()

    if config['SAVE_PATH']:
        os.makedirs(os.path.dirname(config['SAVE_PATH']), exist_ok=True)
        fig.savefig(config['SAVE_PATH'], dpi=300, bbox_inches='tight')

def survey_field_preprocess(fields):
    """
        Map each field to its survey prefix, build a unique survey list (with labels
        upper‐cased and numbers stripped, replacing 'yboc'→'yboc-cb10'), and assign
        each survey a distinct color.
    """
    survey_groups = {f: f.split('_')[0] for f in fields}
    surveys = list(OrderedDict.fromkeys(survey_groups.values()))
    surveys_legend = [s.upper().replace('YBOC', 'YBOC-CB') for s in surveys]
    surveys_legend = [re.sub(r'\d+', '', s) for s in surveys_legend]
    palette = sns.color_palette("tab20", len(surveys))
    survey_colors = dict(zip(surveys, palette))
    return survey_groups, surveys_legend, survey_colors

def summarize_survey_primary_factors(
    loadings: np.ndarray, var_names: List[str], survey_groups: Dict[str, str],
    aggfunc: Callable[[np.ndarray], float] = np.mean, absolute: bool = False) \
        -> Tuple[Dict[str, List[float]], Dict[int, List[str]]]:
    """
        Group rows by survey, compute each survey’s per‐factor aggregate loading,
        then identify which factor(s) each survey loads on most strongly.
    """
    n_vars, n_factors = loadings.shape

    # assign each var index to its survey
    survey_to_idxs = defaultdict(list)
    for i, vn in enumerate(var_names):
        survey_to_idxs[survey_groups[vn]].append(i)
    data = np.abs(loadings) if absolute else loadings

    # compute raw means & SDs
    fa1, fa2 = {}, {}
    for survey, idxs in survey_to_idxs.items():
        # factor 1
        vals = data[idxs, 0]
        m, s = float(aggfunc(vals)), float(np.std(vals, ddof=1))
        fa1[survey] = {'mean': float(f"{m:.3g}"), 'sd': float(f"{s:.3g}")}
        # factor 2
        vals = data[idxs, 1]
        m, s = float(aggfunc(vals)), float(np.std(vals, ddof=1))
        fa2[survey] = {'mean': float(f"{m:.3g}"), 'sd': float(f"{s:.3g}")}

    # now sort each by descending mean
    fa1_loadings = OrderedDict(
        sorted(fa1.items(), key=lambda kv: kv[1]['mean'], reverse=True)
    )
    fa2_loadings = OrderedDict(
        sorted(fa2.items(), key=lambda kv: kv[1]['mean'], reverse=True)
    )

    # select primary by comparing survey's mean1 vs mean2
    primary_factors = {
        'Factor 1': [s for s in fa1_loadings if fa1_loadings[s]['mean'] > fa2_loadings[s]['mean']],
        'Factor 2': [s for s in fa2_loadings if fa2_loadings[s]['mean'] > fa1_loadings[s]['mean']],
    }

    return fa1_loadings, fa2_loadings, primary_factors

def plot_scree(eigvals, config):
    """
        Plot a scree plot of eigenvalues, highlighting the significant `num_factors`.
    """
    figsize = config.get('FIGSIZE', (4,3.5))
    n_points = config.get('N_POINTS', 10)
    num_factors = config.get('MARK_FACTOR', 2)
    line_kwargs = config.get('LINE_KWARGS', {'linewidth': 1.5})
    highlight_kwargs = config.get('HIGHLIGHT_KWARGS', {'color': 'red', 's': 75, 'zorder': 10, 'marker': '*'})
    factor_titles = config.get('FACTOR_TITLES', None)

    eig = np.array(eigvals).flatten()
    if n_points is not None:
        eig = eig[:n_points]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(1, len(eig) + 1)
    ax.plot(x, eig, **line_kwargs)
    ax.scatter(x[num_factors-1], eig[num_factors-1], **highlight_kwargs)

    # annotate if component title provided
    if factor_titles is not None:
        for i, label in enumerate(factor_titles[:num_factors]):
            # compute horizontal offset in *points* based on string length
            dx = len(label)/2 * 6
            ax.annotate(
                label,
                xy=(x[i], eig[i]),  # point to annotate
                xytext=(dx, 0),  # offset in points: (→, no up/down)
                textcoords='offset points',
                ha='center',  # text to the right of the point
                va='center',
                fontsize=10
            )

    if config.get('XLIM') is not None:
        ax.set_xlim(config['XLIM'])
    # integer xticks
    ticks = list(x[:num_factors]) + [x[-1]]
    ax.set_xticks(ticks, minor=False)
    ax.tick_params(
        axis='both',  # apply to the x-axis
        which='major',  # major ticks only
        bottom=True, top=False,  # x‐axis
        left=True, right=False,  # y‐axis
        length=5,  # length of the tick in points
        width=1,  # thickness of the tick
        direction='out'  # tick direction (in, out, or inout)
    )
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xlabel(config['XLABEL'], fontsize=config['LABEL_FONTSIZE'])
    ax.set_ylabel(config['YLABEL'], fontsize=config['LABEL_FONTSIZE'])
    # Clean up spines and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.show()

    if config.get('SAVE_PATH') is not None:
        os.makedirs(os.path.dirname(config['SAVE_PATH']), exist_ok=True)
        fig.savefig(config['SAVE_PATH'], dpi=300, bbox_inches='tight')

def plot_grouped_loadings(loadings: np.ndarray, survey_groups: dict, surveys_legend: list, survey_colors: dict, config: dict):
    # Unpack configuration parameters
    figsize = config.get('FIGSIZE', (8, 6))
    factor_titles = config.get('FACTOR_TITLES', None)
    x_tick_interval = config.get('X_TICK_INTERVAL', 1)
    xlabel = config.get('XLABEL', 'Variables')
    ylabel = config.get('YLABEL', 'Loadings')

    n_vars, n_factors = loadings.shape

    # default factor titles
    if factor_titles is None or len(factor_titles) != n_factors:
        if factor_titles is None:
            factor_titles = []
        else:
            factor_titles = list(factor_titles)
        if len(factor_titles) < n_factors:
            factor_titles += [f'Factor {i+1}' for i in range(len(factor_titles), n_factors)]

    # x positions and xticks
    x = np.arange(1, n_vars + 1)
    x_ticks = np.arange(0, n_vars + 1, x_tick_interval)
    x_ticks[0] = 1

    # create figure + axes
    fig, axes = plt.subplots(n_factors, 1, figsize=figsize)
    if n_factors == 1:
        axes = [axes]
    plt.subplots_adjust(hspace=0.4)

    # prepare bar colors
    bar_colors = [survey_colors[survey_groups[q]] for q in survey_groups.keys()]

    for i, ax in enumerate(axes):
        vals = loadings[:, i]

        # --- median ---
        median_val = np.median(vals)

        # draw bars
        ax.bar(x, vals, config['BAR_WIDTH'], color=bar_colors, edgecolor='white', alpha=0.7)

        # --- add median horizontal line ---
        ax.axhline(median_val, color='red', linestyle='--', linewidth=1.5)

        # # --- annotate median value ---
        # ax.text(
        #     0.98, 0.90,
        #     f'Median = {median_val:.2f}',
        #     transform=ax.transAxes,
        #     ha='right',
        #     va='top',
        #     fontsize=config['TICK_FONTSIZE'],
        #     color='red'
        # )

        # titles & limits
        ax.set_title(factor_titles[i], fontsize=config['TITLE_FONTSIZE'])
        ax.set_ylim(config['YLIM'])
        ax.set_yticks(np.linspace(config['YLIM'][0], config['YLIM'][1], 6))
        ax.tick_params(axis='y', labelsize=config['TICK_FONTSIZE'])

        # x-axis only on bottom plot
        if i == n_factors - 1:
            ax.set_xlabel(xlabel, fontsize=config['LABEL_FONTSIZE'])
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks, fontsize=config['TICK_FONTSIZE'])
        else:
            ax.set_xticks([])

        ax.set_ylabel(ylabel, fontsize=config['LABEL_FONTSIZE'])

        # grid & style
        ax.grid(True, axis='y')
        ax.grid(False, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in survey_colors.values()]
    fig.legend(
        handles, surveys_legend,
        title='Surveys',
        loc='center left',
        bbox_to_anchor=(0.9, 0.5),
        fontsize=config['LEGEND_FONTSIZE'],
        title_fontsize=config['LEGEND_FONTSIZE']
    )
    
    plt.show()

    if config.get('SAVE_PATH') is not None:
        os.makedirs(os.path.dirname(config['SAVE_PATH']), exist_ok=True)
        fig.savefig(config['SAVE_PATH'], dpi=300, bbox_inches='tight')

def load_glm_results(path):
    """
        Generic loader for MATLAB .mat containing `result` struct.
        Returns a dict with:
          - num_factors: int
          - fa_scores: np.ndarray (N x F)
          - effects:   np.ndarray (N x P)
          - glm_p:     list of p-value arrays for each factor
          - glm_beta:  list of beta arrays for each factor
          - glm_se:    list of standard error arrays for each factor
    """
    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    res = data['result']
    num_f = int(res.num_factors)

    # Extract glm fields
    glm_p = []
    glm_beta = []
    glm_se = []
    for f in range(num_f):
        glm_p.append(np.array(res.glm[f].p).flatten())
        glm_beta.append(np.array(res.glm[f].coeff).flatten())
        glm_se.append(np.array(res.glm[f].se).flatten())

    # Build output
    out = {
        'num_factors': num_f,
        'fa_scores': np.array(res.fa_scores),
        'effects': np.array(getattr(res, 'effects_lr')),
        'glm_p': glm_p,
        'glm_beta': glm_beta,
        'glm_se': glm_se
    }
    return out

def plot_regression_density(result, condition, factor_index, config):
    """
        Create a scatter+regression plot showing FA scores versus specified effects.

        Parameters
        ----------
        result : dict
            Must contain 'effects' (N×3 array of [b, sto, vol]) and 'fa_scores' (N×F array).
        condition : str
            Which effect ('b','sto','vol') to plot on x.
        factor_index : int
            Which FA factor to plot on y.
        config : dict
            Plot settings including:
                - FIGSIZE, XLABEL, YLABEL, XLIM, YLIM
                - LABEL_FONTSIZE, TICK_FONTSIZE
                - DENSITY_BAR (bool), HEXBIN_GRIDSIZE, HEXBIN_CMAP, HEXBIN_MINCNT
                - SAVE_PATH
    """
    # Extract configuration with defaults
    figsize = config.get('FIGSIZE', (4, 3.5))
    density_bar = config.get('DENSITY_BAR', True)
    hexbin_gridsize = config.get('HEXBIN_GRIDSIZE', 30)
    hexbin_cmap = config.get('HEXBIN_CMAP', 'Blues')
    hexbin_mincnt = config.get('HEXBIN_MINCNT', 1)
    hexbin_vmin = config.get('HEXBIN_VMIN', 0)
    hexbin_vmax = config.get('HEXBIN_VMAX', None)
    cbar_label = config.get('CBAR_LABEL', 'Density')
    
    # Use GridSpec to reserve space for colorbar consistently
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[20, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    # Extract effect data
    eff = result['effects']
    if   condition == 'b':  x = eff[:,0]
    elif condition == 'sto': x = eff[:,1]
    elif condition == 'vol': x = eff[:,2]
    elif condition == 'valence_b': x = eff[:, 4]
    elif condition == 'valence_s': x = eff[:, 5]
    elif condition == 'valence_v': x = eff[:, 6]
    else:
        raise ValueError("condition must be 'lr','sto' or 'vol' or 'valence_b' or 'valence_s' or 'valence_v'")
    y = result['fa_scores'][:,factor_index-1]

    # Fit OLS model
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    # Print regression statistics
    print(f"\nCondition={condition}, factor={factor_index}")
    print(f"  OLS R² = {model.rsquared:.3f}, p = {model.pvalues[1]:.3e}")

    # Robustness checks
    try:
        rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
        print(f"  Robust slope = {rlm.params[1]:.4f}, p ≈ {rlm.pvalues[1]:.3e}")
    except Exception as e:
        print("  [Robust regression failed]", e)

    from scipy.stats import spearmanr
    rho, pval = spearmanr(x, y)
    print(f"  Spearman ρ = {rho:.3f}, p = {pval:.3e}")

    # Generate prediction line and confidence band
    xg = np.linspace(x.min(), x.max(), 100)
    preds = model.get_prediction(sm.add_constant(xg)).summary_frame(alpha=0.05)

    # Plot hexbin density and regression
    hb = ax.hexbin(x, y, gridsize=hexbin_gridsize, cmap=hexbin_cmap, mincnt=hexbin_mincnt,
                    vmin=hexbin_vmin, vmax=hexbin_vmax)
    ax.plot(xg, preds['mean'], 'r', lw=2)
    ax.fill_between(xg,
                    preds['mean_ci_lower'],
                    preds['mean_ci_upper'],
                    color='r', alpha=0.2)

    # Set axis limits
    ax.set_xlim(config['XLIM'])
    ax.set_ylim(config['YLIM'])

    # assign the i-th label
    ax.set_xlabel(config['XLABEL'], fontsize=config['LABEL_FONTSIZE'])
    ax.set_ylabel(config['YLABEL'], fontsize=config['LABEL_FONTSIZE'])

    ax.tick_params(labelsize=config['TICK_FONTSIZE'])
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Colorbar for density (space reserved via GridSpec)
    if density_bar:
        cbar = fig.colorbar(hb, cax=cax, orientation='vertical')
        cbar.set_label(cbar_label, size=config['LABEL_FONTSIZE'])
        cbar.ax.tick_params(labelsize=config['TICK_FONTSIZE'])
    else:
        cax.set_visible(False)

    plt.show()
    
    if config.get('SAVE_PATH') is not None:
        os.makedirs(os.path.dirname(config['SAVE_PATH']), exist_ok=True)
        fig.savefig(config['SAVE_PATH'], dpi=300, bbox_inches='tight')

def plot_half_violin(groups, config):
    """
    Plot horizontal half-violin distributions for group-wise values.

    Parameters
    ----------
    groups : list of 1D array-like
        One vector per group.
    config : dict
        Uses your existing config style. Relevant keys:

        Existing-style keys
        -------------------
        FIGSIZE : tuple, default (7, 4)
        TITLE : str or None
        XLIM : tuple or None
        XLABEL : str or None
        YLIM : tuple or None
        YLABEL : str or None
        GROUP_LABEL : list[str] or None
        GROUP_COLOR : list[color]
        LINE_ALPHA : float
        LINE_WIDTH : float
        TITLE_FONTSIZE : int
        LABEL_FONTSIZE : int
        TICK_FONTSIZE : int
        SAVE_PATH : str or None

        New keys
        --------
        DISCARD_OUTLIERS : bool, default False
        OUTLIER_METHOD : {'iqr', 'zscore'}, default 'iqr'
        IQR_K : float, default 1.5
        Z_THRESH : float, default 3.0
        PRINT_OUTLIERS : bool, default True
        SHOW_POINTS : bool, default True
        POINT_ALPHA : float, default 0.18
        POINT_SIZE : float, default 14
        POINT_JITTER : float, default 0.08
        VIOLIN_SCALE : float, default 0.35
        KDE_GRIDSIZE : int, default 300
        KDE_PAD : float, default 0.15
        SUMMARY_MARKER : {'median', 'mean', None}, default 'median'
        SUMMARY_LINE_WIDTH : float, default 2.2
        SUMMARY_LINE_HEIGHT : float, default 0.26
        ZERO_LINE : bool, default True
        ZERO_LINE_COLOR : color, default 'gray'
        ZERO_LINE_STYLE : str, default '--'
        ZERO_LINE_WIDTH : float, default 1
        SHOW_N_IN_LABEL : bool, default False
        FILL_ALPHA : float, default 0.65
        EDGE_COLOR : color or None, default None
    """

    def _remove_outliers(x, method='iqr', iqr_k=1.5, z_thresh=3.0):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]

        if len(x) == 0:
            return x, np.zeros(0, dtype=bool)

        if method is None:
            return x, np.ones(len(x), dtype=bool)

        if method == 'iqr':
            if len(x) < 4:
                return x, np.ones(len(x), dtype=bool)
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            lower = q1 - iqr_k * iqr
            upper = q3 + iqr_k * iqr
            mask = (x >= lower) & (x <= upper)
            return x[mask], mask

        if method == 'zscore':
            if len(x) < 3:
                return x, np.ones(len(x), dtype=bool)
            mu = np.mean(x)
            sd = np.std(x, ddof=1)
            if sd == 0:
                return x, np.ones(len(x), dtype=bool)
            z = np.abs((x - mu) / sd)
            mask = z <= z_thresh
            return x[mask], mask

        raise ValueError("OUTLIER_METHOD must be one of {'iqr', 'zscore', None}")

    # -------------------------
    # config
    # -------------------------
    figsize = config.get('FIGSIZE', (7, 4))
    title = config.get('TITLE', None)
    xlim = config.get('XLIM', None)
    ylim = config.get('YLIM', None)
    xlabel = config.get('XLABEL', None)
    ylabel = config.get('YLABEL', None)

    group_labels = config.get('GROUP_LABEL', None)
    group_colors = config.get('GROUP_COLOR', None)

    line_alpha = config.get('LINE_ALPHA', 0.3)
    line_width = config.get('LINE_WIDTH', 2)

    title_fs = config.get('TITLE_FONTSIZE', 14)
    label_fs = config.get('LABEL_FONTSIZE', 12)
    tick_fs = config.get('TICK_FONTSIZE', 10)

    save_path = config.get('SAVE_PATH', None)

    discard_outliers = config.get('DISCARD_OUTLIERS', False)
    outlier_method = config.get('OUTLIER_METHOD', 'iqr')
    iqr_k = config.get('IQR_K', 1.5)
    z_thresh = config.get('Z_THRESH', 3.0)
    print_outliers = config.get('PRINT_OUTLIERS', True)

    show_points = config.get('SHOW_POINTS', True)
    point_alpha = config.get('POINT_ALPHA', 0.3)
    point_size = config.get('POINT_SIZE', 15)
    point_jitter = config.get('POINT_JITTER', 0.2)

    violin_scale = config.get('VIOLIN_SCALE', 0.5)
    kde_gridsize = config.get('KDE_GRIDSIZE', 300)
    kde_pad = config.get('KDE_PAD', 0.15)

    summary_marker = config.get('SUMMARY_MARKER', 'median')  # 'median', 'mean', None
    summary_line_width = config.get('SUMMARY_LINE_WIDTH', 1)
    summary_line_height = config.get('SUMMARY_LINE_HEIGHT', 0.26)

    zero_line = config.get('ZERO_LINE', True)
    zero_line_color = config.get('ZERO_LINE_COLOR', 'gray')
    zero_line_style = config.get('ZERO_LINE_STYLE', '--')
    zero_line_width = config.get('ZERO_LINE_WIDTH', 1)

    show_n_in_label = config.get('SHOW_N_IN_LABEL', False)
    fill_alpha = config.get('FILL_ALPHA', 0.7)
    edge_color = config.get('EDGE_COLOR', None)

    n_groups = len(groups)

    if group_labels is None:
        group_labels = [f'Group {i+1}' for i in range(n_groups)]
    if len(group_labels) != n_groups:
        raise ValueError(f"GROUP_LABEL must have length {n_groups}, but got {len(group_labels)}.")

    if group_colors is None:
        # fallback distinct muted colors
        group_colors = ['#8FBFA5', '#BFD8DD', '#DBB6A2'][:n_groups]
    if len(group_colors) < n_groups:
        raise ValueError(f"Need at least {n_groups} GROUP_COLOR, but got {len(group_colors)}.")

    # -------------------------
    # prep data
    # -------------------------
    processed = []
    outlier_counts = []
    kept_counts = []

    for vals in groups:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        original_n = len(vals)

        if discard_outliers:
            vals_kept, mask = _remove_outliers(vals, method=outlier_method, iqr_k=iqr_k, z_thresh=z_thresh)
            n_out = original_n - len(vals_kept)
        else:
            vals_kept = vals
            n_out = 0

        processed.append(vals_kept)
        outlier_counts.append(n_out)
        kept_counts.append(len(vals_kept))

    if print_outliers:
        print("\nOutlier report:")
        for name, n_keep, n_out in zip(group_labels, kept_counts, outlier_counts):
            print(f"  {name}: discarded {n_out} outlier(s), kept {n_keep}")

    # -------------------------
    # plot
    # -------------------------
    fig, ax = plt.subplots(figsize=figsize)
    y_positions = np.arange(n_groups)[::-1] + 1  # top to bottom

    for y, vals, color, label, n_keep in zip(y_positions, processed, group_colors, group_labels, kept_counts):
        if len(vals) == 0:
            continue

        vals = np.asarray(vals, dtype=float)
        xmin, xmax = np.min(vals), np.max(vals)
        xr = xmax - xmin
        if xr == 0:
            xr = 1e-6
        x_grid = np.linspace(xmin - kde_pad * xr, xmax + kde_pad * xr, kde_gridsize)

        if len(vals) >= 2 and np.std(vals) > 0:
            kde = gaussian_kde(vals)
            dens = kde(x_grid)
            if np.max(dens) > 0:
                dens = dens / np.max(dens) * violin_scale
            else:
                dens = np.zeros_like(x_grid)
        else:
            dens = np.zeros_like(x_grid)

        # half violin: fill above the centerline only
        ax.fill_between(
            x_grid,
            y,
            y + dens,
            color=color,
            alpha=fill_alpha,
            linewidth=0,
            zorder=2
        )

        # outline
        ax.plot(
            x_grid,
            y + dens,
            color=edge_color if edge_color is not None else color,
            alpha=line_alpha,
            linewidth=line_width,
            zorder=3
        )

        # baseline line at group center
        ax.plot([x_grid[0], x_grid[-1]], [y, y], color='none')

        # raw points
        if show_points:
            jitter = np.random.uniform(-point_jitter, point_jitter, size=len(vals))
            ax.scatter(
                vals,
                y + jitter,
                s=point_size,
                color=color,
                alpha=point_alpha,
                edgecolor='none',
                zorder=1
            )

        # summary marker as vertical line
        if summary_marker is not None:
            if summary_marker == 'median':
                x0 = np.median(vals)
            elif summary_marker == 'mean':
                x0 = np.mean(vals)
            else:
                raise ValueError("SUMMARY_MARKER must be one of {'median', 'mean', None}")

            ax.plot(
                [x0, x0],
                [y, y + summary_line_height],
                color='black',
                linewidth=summary_line_width,
                solid_capstyle='round',
                zorder=5
            )

    # zero line
    if zero_line:
        ax.axvline(
            0,
            color=zero_line_color,
            linestyle=zero_line_style,
            linewidth=zero_line_width,
            zorder=0
        )

    # y labels
    if show_n_in_label:
        labels = [f"{lab} (n={n})" for lab, n in zip(group_labels, kept_counts)]
    else:
        labels = group_labels

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=tick_fs)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fs)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fs)
    if title is not None:
        ax.set_title(title, fontsize=title_fs)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0.5, n_groups + 0.5 + violin_scale + 0.1)

    ax.tick_params(labelsize=tick_fs, left=False)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
  