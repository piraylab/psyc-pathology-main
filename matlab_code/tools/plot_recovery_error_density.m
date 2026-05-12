function plot_recovery_error_density(sim_params, recovered_params, varargin)
% plot_recovery_error_density  KDE of fitting errors with ±1σ shading.
%
% Works for:
%   - 4 parameters: [u_vol, u_sto, v_vol, v_sto]
%   - 5 parameters: [u_vol, u_sto, v_vol, v_sto, rho]
%
% Error is defined as (True - Recovered).
%
% OPTIONAL NAME/VALUE ARGUMENTS
%   'BandwidthScale' scalar > 0
%   'XLim'           [1x2]
%   'Title'          char/string   (overall title; optional)
%   'SavePath'       char/string
%   'FilePrefix'     char/string
%   'PanelTitles'    cellstr/string array of length P
%   'XLabels'        cellstr/string array of length P OR char/string scalar
%   'YLabels'        cellstr/string array of length P OR char/string scalar
%   'PanelSize'      [1x2] inches per panel, default [4 4]
%   'LabelFontSize'  scalar, default 11
%   'TitleFontSize'  scalar, default 11
%   'TickFontSize'   scalar, default 11
%
% Notes
%   - Panels are arranged in one row.
%   - Each panel is square.

% ---------- parse inputs ----------
p = inputParser;
p.addParameter('BandwidthScale', 1.5, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('XLim', [], @(x) isempty(x) || (isnumeric(x) && numel(x) == 2 && x(1) < x(2)));
p.addParameter('Title', '', @(x) ischar(x) || isstring(x));
p.addParameter('SavePath', [], @(x) isempty(x) || ischar(x) || isstring(x));
p.addParameter('FilePrefix', 'plot_recovery_error_density', @(x) ischar(x) || isstring(x));

p.addParameter('PanelTitles', [], @(x) isempty(x) || iscell(x) || isstring(x));
p.addParameter('XLabels', 'Fitting error (True - Recovered)', @(x) iscell(x) || isstring(x) || ischar(x));
p.addParameter('YLabels', 'Empirical density', @(x) iscell(x) || isstring(x) || ischar(x));

p.addParameter('PanelSize', [4 4], @(x) isnumeric(x) && numel(x) == 2 && all(x > 0));
p.addParameter('LabelFontSize', 11, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('TitleFontSize', 11, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('TickFontSize', 11, @(x) isnumeric(x) && isscalar(x) && x > 0);

p.parse(varargin{:});

bwScale    = p.Results.BandwidthScale;
xlim_user  = p.Results.XLim;
savePath   = p.Results.SavePath;
filePref   = char(p.Results.FilePrefix);

panelCustom = p.Results.PanelTitles;
xlabCustom  = p.Results.XLabels;
ylabCustom  = p.Results.YLabels;

panelSize = p.Results.PanelSize;
labelFS   = p.Results.LabelFontSize;
titleFS   = p.Results.TitleFontSize;
tickFS    = p.Results.TickFontSize;

% ---------- checks ----------
if size(sim_params, 2) ~= size(recovered_params, 2)
    error('sim_params and recovered_params must have the same number of columns.');
end

npar = size(sim_params, 2);

% ---------- default panel titles ----------
switch npar
    case 4
        defaultPanelTitles = {'u_{vol}', 'u_{sto}', 'v_{vol}', 'v_{sto}'};
    case 5
        defaultPanelTitles = {'u_{vol}', 'u_{sto}', 'v_{vol}', 'v_{sto}', 'rho'};
    otherwise
        defaultPanelTitles = arrayfun(@(k) sprintf('param_{%d}', k), 1:npar, 'UniformOutput', false);
end

% ---------- apply title overrides ----------
if isempty(panelCustom)
    panelTitles = defaultPanelTitles;
else
    panelTitles = cellstr(panelCustom);
    if numel(panelTitles) ~= npar
        error('PanelTitles must contain exactly %d labels.', npar);
    end
end

% X labels
if ischar(xlabCustom) || (isstring(xlabCustom) && isscalar(xlabCustom))
    xlabels = repmat({char(xlabCustom)}, 1, npar);
else
    xlabels = cellstr(xlabCustom);
    if numel(xlabels) ~= npar
        error('XLabels must contain exactly %d labels or be a single label.', npar);
    end
end

% Y labels
if ischar(ylabCustom) || (isstring(ylabCustom) && isscalar(ylabCustom))
    ylabels = repmat({char(ylabCustom)}, 1, npar);
else
    ylabels = cellstr(ylabCustom);
    if numel(ylabels) ~= npar
        error('YLabels must contain exactly %d labels or be a single label.', npar);
    end
end

% ---------- compute errors ----------
err = sim_params - recovered_params;  % (True - Recovered)

% ---------- layout ----------
nr = 1;
nc = npar;

% ---------- determine x-limits ----------
if isempty(xlim_user)
    all_err = err(:);
    all_err = all_err(~isnan(all_err));
    pad = 0.1 * range(all_err);
    if isempty(pad) || isnan(pad)
        pad = 1;
    end
    xlim_auto = [min(all_err) - pad, max(all_err) + pad];
else
    xlim_auto = xlim_user;
end

% ---------- figure size ----------
figW = nc * panelSize(1);
figH = panelSize(2);

% ---------- figure ----------
fh = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 figW figH]);
tiledlayout(nr, nc, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:npar
    e = err(:, i);
    e = e(~isnan(e));

    nexttile;

    if isempty(e)
        title(panelTitles{i}, 'Interpreter', 'tex', 'FontSize', titleFS);
        xlabel(xlabels{i}, 'FontSize', labelFS);
        ylabel(ylabels{i}, 'FontSize', labelFS);
        set(gca, 'FontSize', tickFS);
        xlim(xlim_auto);
        axis square;
        grid off;
        continue;
    end

    % KDE bandwidth
    [~, ~, bw] = ksdensity(e);
    smoothBW = bw * bwScale;
    [fq, xq] = ksdensity(e, 'Bandwidth', smoothBW);

    plot(xq, fq, '-', 'LineWidth', 1.5, 'Color', [0.1 0.4 0.8]);
    hold on;
    box on;

    % mean and SD
    mu = mean(e);
    sd = std(e, 0);

    % highlight ±1 SD region
    [~, t1] = min(abs(xq - (mu - sd)));
    [~, t2] = min(abs(xq - (mu + sd)));
    if t1 > t2
        tmp = t1; t1 = t2; t2 = tmp;
    end

    xBand = xq(t1:t2);
    yBand = fq(t1:t2);

    fill([xBand, fliplr(xBand)], [zeros(size(yBand)), fliplr(yBand)], ...
        [0.2 0.6 0.3], 'FaceAlpha', 0.20, 'EdgeColor', 'none');

    yl = ylim;
    plot([mu mu], yl, 'r-', 'LineWidth', 1.2);
    plot([0 0], yl, 'k-', 'LineWidth', 1);
    ylim(yl);
    xlim(xlim_auto);

    xlabel(xlabels{i}, 'FontSize', labelFS);
    ylabel(ylabels{i}, 'FontSize', labelFS);
    title(panelTitles{i}, 'Interpreter', 'tex', 'FontSize', titleFS);
    set(gca, 'FontSize', tickFS);
    axis square;
    grid off;
end

maybe_save_figure(fh, savePath, filePref);

end