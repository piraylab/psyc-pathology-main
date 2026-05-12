function stats = distrHMM_recovery_lr(sim_lr, rec_lr, savePath, fname)
% distrHMM_recovery_lr  Correlation between simulated and recovered learning rates.
%
% INPUTS
%   sim_lr   [N×4] simulated learning rates
%   rec_lr   [N×4] recovered learning rates
%   savePath (optional) folder to save figures
%   fname    (optional) file prefix for saved figures
%
% OUTPUT
%   stats struct with correlations for raw LR and LR effects

    % ------------ sanity checks ------------
    if ~isequal(size(sim_lr), size(rec_lr))
        error('sim_lr and rec_lr must have the same size (N×4).');
    end
    if size(sim_lr, 2) ~= 4
        error('Expected sim_lr and rec_lr to have 4 columns (one per block).');
    end

    % ------------ (1) raw LR correlations (per block) ------------
    blockLabels = arrayfun(@(k) sprintf('Block %d', k), 1:4, 'UniformOutput', false);
    r_lr = nan(1,4);
    p_lr = nan(1,4);

    for j = 1:4
        x = sim_lr(:, j);
        y = rec_lr(:, j);
        [r_lr(j), p_lr(j)] = corr(x, y, 'rows', 'pairwise', 'type', 'Spearman');
    end

    % ------------ (2) LR effect correlations ------------
    [sim_lr_eff, effLabels] = calc_lr_effects(sim_lr);
    [rec_lr_eff, ~]         = calc_lr_effects(rec_lr);

    r_eff = nan(1,4);
    p_eff = nan(1,4);

    for j = 1:4
        x = sim_lr_eff(:, j);
        y = rec_lr_eff(:, j);
        [r_eff(j), p_eff(j)] = corr(x, y, 'rows', 'pairwise', 'type', 'Spearman');
    end

    % ------------ pack output struct ------------
    stats.lr.labels = blockLabels;
    stats.lr.r      = r_lr;
    stats.lr.p      = p_lr;

    stats.effects.labels = effLabels;
    stats.effects.r      = r_eff;
    stats.effects.p      = p_eff;

    % % ------------ pretty print ------------
    % fprintf('\n=== Correlation: Raw Learning Rates (sim vs recovered) ===\n');
    % for j = 1:4
    %     fprintf('  %-8s: r = %6.3f, p = %.3g\n', blockLabels{j}, r_lr(j), p_lr(j));
    % end
    % 
    % fprintf('\n=== Correlation: Learning-Rate Effects (sim vs recovered) ===\n');
    % for j = 1:4
    %     fprintf('  %-24s: r = %6.3f, p = %.3g\n', effLabels{j}, r_eff(j), p_eff(j));
    % end
    % fprintf('\n');

    % ------------ optional save args ------------
    doSave = (nargin >= 3) && ~isempty(savePath) && (nargin >= 4) && ~isempty(fname);

    % ===== FIGURE 1: Raw learning rates =====
    if doSave
        % plot_param_recovery already handles saving if SAVE_PATH/FilePrefix are used
        % If your plot_param_recovery only saves through config, keep this section minimal.
        % Here we assume it accepts these name/value args:
        plot_param_recovery(sim_lr, rec_lr, ...
            'PanelTitles', blockLabels, ...
            'XLabels', repmat({'True LR'}, 1, 4), ...
            'YLabels', repmat({'Recovered LR'}, 1, 4), ...
            'Title', 'Recovery of Learning Rates', ...
            'SavePath', savePath, ...
            'FilePrefix', [fname, '_distr_lr_recovery']);

         plot_recovery_error_density(sim_lr, rec_lr, ...
            'PanelTitles', blockLabels, ...
            'XLabels', repmat({'True LR'}, 1, 4), ...
            'XLim', [-0.3, 0.3], ...
            'YLabels', repmat({'Recovered LR'}, 1, 4), ...
            'Title', 'Recovery of Learning Rates', ...
            'SavePath', savePath, ...
            'FilePrefix', [fname, '_distr_lr_recovery_density']);
    else
        plot_param_recovery(sim_lr, rec_lr, ...
        'PanelTitles', blockLabels, ...
        'XLabels', repmat({'True LR'}, 1, 4), ...
        'YLabels', repmat({'Recovered LR'}, 1, 4), ...
        'Title', 'Recovery of Learning Rates');

        plot_recovery_error_density(sim_lr, rec_lr, ...
            'PanelTitles', blockLabels, ...
            'XLabels', repmat({'True LR'}, 1, 4), ...
            'XLim', [-0.3, 0.3], ...
            'YLabels', repmat({'Recovered LR'}, 1, 4), ...
            'Title', 'Recovery of Learning Rates');
    end

    % ===== FIGURE 2: LR Effects =====
    % if doSave
    %     plot_param_recovery(sim_lr_eff, rec_lr_eff, ...
    %         'PanelTitles', effLabels, ...
    %         'XLabels', repmat({'True LR'}, 1, 4), ...
    %         'YLabels', repmat({'Recovered LR'}, 1, 4), ...
    %         'Title', 'Recovery of Learning-Rate Effects', ...
    %         'SavePath', savePath, ...
    %         'FilePrefix', [fname '_distr_lr_effects_recovery']);
    % else
        % plot_param_recovery(sim_lr_eff, rec_lr_eff, ...
        %     'PanelTitles', effLabels, ...
        %     'XLabels', repmat({'True LR'}, 1, 4), ...
        %     'YLabels', repmat({'Recovered LR'}, 1, 4), ...
        %     'Title', 'Recovery of Learning-Rate Effects');
    % end

end


function [lr_eff, labels] = calc_lr_effects(lr)
% calc_lr_effects  Compute baseline and contrast-coded LR effects.

    C = [ 0.25  0.25  0.25  0.25;   % baseline
         -1.00 -1.00  1.00  1.00;   % stochasticity effect
         -1.00  1.00 -1.00  1.00;   % volatility effect
          1.00 -1.00 -1.00  1.00];  % interaction effect

    lr_eff = lr * C.' / 2;

    labels = { ...
        'Baseline LR', ...
        'Stochasticity LR Effect', ...
        'Volatility LR Effect', ...
        'Interaction Effect'};
end
