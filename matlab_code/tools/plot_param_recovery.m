function stats = plot_param_recovery(sim_params, recovered_params, varargin)
% plot_param_recovery  True vs Recovered scatter with fit + diagnostics.
%
% Works for:
%   - 4 parameters: [u_vol, u_sto, v_vol, v_sto]
%   - 5 parameters: [u_vol, u_sto, v_vol, v_sto, rho]

% ---------- parse ----------
p = inputParser;
p.addParameter('Nsim', size(sim_params,1), @(x) isnumeric(x) && isscalar(x));
p.addParameter('Ntrials', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
p.addParameter('Title', 'Correlation Between True and Recovered Parameters', ...
    @(x) ischar(x) || isstring(x));
p.addParameter('SavePath', [], @(x) isempty(x) || ischar(x) || isstring(x));
p.addParameter('FilePrefix', 'plot_param_recovery', @(x) ischar(x) || isstring(x));

p.addParameter('PanelTitles', [], @(x) isempty(x) || iscell(x) || isstring(x));
p.addParameter('XLabels', [], @(x) isempty(x) || iscell(x) || isstring(x));
p.addParameter('YLabels', [], @(x) isempty(x) || iscell(x) || isstring(x));

p.parse(varargin{:});

savePath    = p.Results.SavePath;
filePref    = char(p.Results.FilePrefix);
panelCustom = p.Results.PanelTitles;
xlabCustom  = p.Results.XLabels;
ylabCustom  = p.Results.YLabels;

% ---------- checks ----------
if size(sim_params, 2) ~= size(recovered_params, 2)
    error('sim_params and recovered_params must have the same number of columns.');
end

P = size(sim_params, 2);

% ---------- default labels ----------
switch P
    case 4
        defaultPanelTitles = { ...
            'Mean Volatility: True vs Recovered', ...
            'Mean Stochasticity: True vs Recovered', ...
            'Variance Volatility: True vs Recovered', ...
            'Variance Stochasticity: True vs Recovered'};

        defaultXLabels = { ...
            'True Mean Volatility', ...
            'True Mean Stochasticity', ...
            'True Variance Volatility', ...
            'True Variance Stochasticity'};

        defaultYLabels = { ...
            'Recovered Mean Volatility', ...
            'Recovered Mean Stochasticity', ...
            'Recovered Variance Volatility', ...
            'Recovered Variance Stochasticity'};

    case 5
        defaultPanelTitles = { ...
            'Mean Volatility: True vs Recovered', ...
            'Mean Stochasticity: True vs Recovered', ...
            'Variance Volatility: True vs Recovered', ...
            'Variance Stochasticity: True vs Recovered', ...
            'Rho: True vs Recovered'};

        defaultXLabels = { ...
            'True Mean Volatility', ...
            'True Mean Stochasticity', ...
            'True Variance Volatility', ...
            'True Variance Stochasticity', ...
            'True Rho'};

        defaultYLabels = { ...
            'Recovered Mean Volatility', ...
            'Recovered Mean Stochasticity', ...
            'Recovered Variance Volatility', ...
            'Recovered Variance Stochasticity', ...
            'Recovered Rho'};

    otherwise
        defaultPanelTitles = arrayfun(@(k) sprintf('Parameter %d: True vs Recovered', k), ...
            1:P, 'UniformOutput', false);
        defaultXLabels = arrayfun(@(k) sprintf('True Parameter %d', k), ...
            1:P, 'UniformOutput', false);
        defaultYLabels = arrayfun(@(k) sprintf('Recovered Parameter %d', k), ...
            1:P, 'UniformOutput', false);
end

% ---------- apply overrides ----------
if isempty(panelCustom)
    panelTitles = defaultPanelTitles;
else
    panelTitles = cellstr(panelCustom);
    if numel(panelTitles) ~= P
        error('PanelTitles must contain exactly %d labels.', P);
    end
end

if isempty(xlabCustom)
    xlabels = defaultXLabels;
else
    xlabels = cellstr(xlabCustom);
    if numel(xlabels) ~= P
        error('XLabels must contain exactly %d labels.', P);
    end
end

if isempty(ylabCustom)
    ylabels = defaultYLabels;
else
    ylabels = cellstr(ylabCustom);
    if numel(ylabels) ~= P
        error('YLabels must contain exactly %d labels.', P);
    end
end

% ---------- figure layout ----------
fh = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 4*P 4]);

tiledlayout(1, P, 'Padding', 'compact', 'TileSpacing', 'compact');

stats = struct([]);

for c = 1:P
    x = sim_params(:, c);
    y = recovered_params(:, c);

    valid = ~isnan(x) & ~isnan(y);
    x = x(valid);
    y = y(valid);

    r = corr(x, y, 'rows', 'pairwise', 'type','Spearman');

    if numel(x) >= 2
        Pfit = polyfit(x, y, 1);
        yhat = polyval(Pfit, x);
    else
        Pfit = [NaN NaN];
        yhat = y;
    end

    rmse = sqrt(mean((y - yhat).^2, 'omitnan'));
    mae  = mean(abs(y - yhat), 'omitnan');

    mn = min([x; y]);
    mx = max([x; y]);
    pad = 0.08 * max(mx - mn, eps);
    lims = [mn - pad, mx + pad];

    nexttile;
    scatter(x, y, 28, 'filled'); hold on; box on;
    plot(lims, lims, 'k--', 'LineWidth', 1);
    if all(isfinite(Pfit))
        plot(lims, polyval(Pfit, lims), 'LineWidth', 1.5);
    end
    xlim(lims);
    ylim(lims);
    axis square;

    title(panelTitles{c}, 'FontSize', 11);
    xlabel(xlabels{c}, 'FontSize', 11);
    ylabel(ylabels{c}, 'FontSize', 11);
    set(gca, 'FontSize', 11);
    grid off;

    stats(c).name      = panelTitles{c};
    stats(c).r         = r;
    stats(c).slope     = Pfit(1);
    stats(c).intercept = Pfit(2);
    stats(c).rmse      = rmse;
    stats(c).mae       = mae;
    stats(c).n         = numel(x);
end

maybe_save_figure(fh, savePath, filePref);

end