function out = critical_value_grouping_fascore(experiment)
% critical_value_grouping
% Theory-guided grouping based on pilot-derived critical values of
% stochasticity and volatility LR effects.
%
% Outputs:
%   out.grouping
%   out.lr
%   out.factors
%   out.covariates
%
% No results are saved to disk.

    if nargin < 1
        experiment = 'bird';
    end

    filter = true;
    exclude_criteria = 2;
    num_factors = 2;

    %% Paths
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);
    addpath(fullfile('..', 'tools'));

    fdir = fullfile('..', '..', 'mat_data', 'experiment_1');

    %% Load factor scores
    facFile = fullfile(fdir, sprintf('fa_score_ex%g_factors%d_%s.mat', exclude_criteria, num_factors, experiment));
    fprintf('Loading factor scores from: %s\n', facFile);
    F = load(facFile);
    fa_scores = F.F;
    workerIds = F.workerIds;

    %% Load pilot LR
    fname1 = fullfile(fdir, 'model_neutral_pilot1.mat');
    fname2 = fullfile(fdir, 'model_neutral_pilot2.mat');
    p1 = load(fname1);
    p2 = load(fname2);
    pilot_lr = [p1.lr; p2.lr];
    fprintf('Using %s as LR critical value source.\n', fname1);
    fprintf('Using %s as LR critical value source.\n', fname2);

    pilot_lr_eff = pilot_lr * [-1 1 -1 1; -1 -1 1 1]';
    pilot_m_vol_lr_eff = mean(pilot_lr_eff(:,1));
    pilot_m_sto_lr_eff = mean(pilot_lr_eff(:,2));

    fprintf('Mean pilot_m_vol_lr_eff: %.4f\n', pilot_m_vol_lr_eff);
    fprintf('Mean pilot_m_sto_lr_eff: %.4f\n', pilot_m_sto_lr_eff);
    fprintf('Pilot subjects: %d\n', size(pilot_lr,1));

    %% Load main LR
    lrFile = fullfile(fdir, 'model_neutral_bird.mat');
    L = load(lrFile);
    lr = L.lr;
    lr_workerIds = L.workerIds;
    fprintf('Using %s as LR source.\n', lrFile);
    
    %% Match LR to factor-score subjects
    lr_matched = match_effects(workerIds, lr_workerIds, lr);

    lr_eff = lr_matched * [-1 1 -1 1; -1 -1 1 1]';
    vol_lr_eff = lr_eff(:,1);
    sto_lr_eff = lr_eff(:,2);

    fprintf('Mean vol_lr_eff: %.4f\n', mean(vol_lr_eff, 'omitnan'));
    fprintf('Mean sto_lr_eff: %.4f\n', mean(sto_lr_eff, 'omitnan'));

    %% Load AGE
    [data, ~, ~] = get_data(experiment, filter, exclude_criteria);
    d_workerIds = string(cellfun(@(x) x.workerId, data, 'UniformOutput', false));
    d_age = cellfun(@(x) x.age, data);
    d_age = match_effects(workerIds, d_workerIds, d_age(:));

    %% Valid subjects
    valid = ~isnan(d_age) & all(~isnan(lr_eff), 2) & all(~isnan(fa_scores), 2);

    workerIds_valid = workerIds(valid);
    fa_scores_valid = fa_scores(valid, :);
    lr_valid = lr_matched(valid, :);
    lr_eff_valid = lr_eff(valid, :);
    vol_lr_eff_valid = vol_lr_eff(valid);
    sto_lr_eff_valid = sto_lr_eff(valid);
    d_age_valid = d_age(valid);

    %% Grouping
    idx_intact = (sto_lr_eff_valid < pilot_m_sto_lr_eff) & (vol_lr_eff_valid > pilot_m_vol_lr_eff);
    idx_vol_blind = (sto_lr_eff_valid < pilot_m_sto_lr_eff) & (vol_lr_eff_valid < pilot_m_sto_lr_eff);
    idx_sto_blind = (sto_lr_eff_valid > pilot_m_vol_lr_eff) & (vol_lr_eff_valid > pilot_m_vol_lr_eff);

    group_names = {'Stochasticity-blind', 'Intact', 'Volatility-blind'};
    group_fields = {'sto_blind', 'intact', 'vol_blind'};
    group_masks = {idx_sto_blind, idx_intact, idx_vol_blind};

    %% LR summaries by group
    out = struct();
    out.grouping.experiment = experiment;
    out.grouping.workerIds = workerIds_valid;
    out.grouping.group_names = group_names;
    out.grouping.group_fields = group_fields;
    out.grouping.group_masks = group_masks;
    out.grouping.pilot_m_vol_lr_eff = pilot_m_vol_lr_eff;
    out.grouping.pilot_m_sto_lr_eff = pilot_m_sto_lr_eff;

    out.lr.raw = lr_valid;
    out.lr.effects = lr_eff_valid;
    out.lr.vol_lr_eff = vol_lr_eff_valid;
    out.lr.sto_lr_eff = sto_lr_eff_valid;

    for g = 1:3
        field = group_fields{g};
        mask = group_masks{g};
        vals = lr_valid(mask, :);

        out.lr.groups.(field).lr_raw = vals;
        out.lr.groups.(field).mean_lr = mean(vals, 1, 'omitnan');
        out.lr.groups.(field).sem_lr = serr(vals);
        out.lr.groups.(field).n = sum(mask);
    end

    %% Factor variables
    q_f1 = fa_scores_valid(:,1);
    q_f2 = fa_scores_valid(:,2);
    q_diff = q_f1 - q_f2;

    factor_vars = {q_f1, q_f2, q_diff};
    factor_names = {'Factor 1', 'Factor 2', 'F1 - F2'};

    fprintf('\n==============================\n');
    fprintf('BETWEEN-GROUP ANALYSIS: FACTORS\n');
    fprintf('==============================\n\n');

    out.factors.names = factor_names;
    out.factors.values = factor_vars;
    out.factors.group_names = group_names;

    out.factors.between = run_group_anova_set(factor_vars, factor_names, group_masks, group_names);

    %% Covariates: age
    fprintf('\n==============================\n');
    fprintf('BETWEEN-GROUP ANALYSIS: COVARIATES\n');
    fprintf('==============================\n\n');

    covar_vars = {d_age_valid};
    covar_names = {'Age'};

    out.covariates.names = covar_names;
    out.covariates.values = covar_vars;
    out.covariates.group_names = group_names;
    out.covariates.between = run_group_anova_set(covar_vars, covar_names, group_masks, group_names);

    savefile = fullfile(fdir, sprintf('%s_%s.mat', mfilename, experiment));
    save(savefile, 'out');
    
end


function results = run_group_anova_set(var_list, var_names, group_masks, group_names)
% Run:
% - 3-group ANOVA
% - pairwise 2-group ANOVAs:
%   1) Sto-blind vs Intact
%   2) Vol-blind vs Intact
%   3) Sto-blind vs Vol-blind

    n_vars = numel(var_list);
    n_groups = numel(group_masks);

    results.values = cell(n_vars, n_groups);
    results.mean   = nan(n_vars, n_groups);
    results.sd     = nan(n_vars, n_groups);
    results.se     = nan(n_vars, n_groups);
    results.n      = nan(n_vars, n_groups);

    results.anova3.F = nan(n_vars, 1);
    results.anova3.df1 = nan(n_vars, 1);
    results.anova3.df2 = nan(n_vars, 1);
    results.anova3.p = nan(n_vars, 1);

    pair_names = {'sto_vs_intact', 'vol_vs_intact', 'sto_vs_vol'};
    pair_idx = {[1 2], [3 2], [1 3]};

    for p = 1:numel(pair_names)
        results.(pair_names{p}).F = nan(n_vars, 1);
        results.(pair_names{p}).df1 = nan(n_vars, 1);
        results.(pair_names{p}).df2 = nan(n_vars, 1);
        results.(pair_names{p}).p = nan(n_vars, 1);
    end

    for i = 1:n_vars
        q = var_list{i};

        fprintf('----- %s -----\n', var_names{i});

        x_all = [];
        g_all = [];

        for g = 1:3
            vals = q(group_masks{g});
            vals = vals(~isnan(vals));

            results.values{i,g} = vals;
            results.mean(i,g) = mean(vals, 'omitnan');
            results.sd(i,g)   = std(vals, 'omitnan');
            results.se(i,g)   = serr(vals);
            results.n(i,g)    = numel(vals);

            fprintf('%s (n=%d): mean = %.3f ± %.3f\n', ...
                group_names{g}, results.n(i,g), results.mean(i,g), results.se(i,g));

            x_all = [x_all; vals];
            g_all = [g_all; g * ones(numel(vals),1)];
        end

        % 3-group ANOVA
        [p3, tbl3] = anova1(x_all, g_all, 'off');
        results.anova3.F(i) = tbl3{2,5};
        results.anova3.df1(i) = tbl3{2,3};
        results.anova3.df2(i) = tbl3{3,3};
        results.anova3.p(i) = p3;

        fprintf('3-group ANOVA: F(%d, %d) = %.3f, p = %.4g\n', ...
            results.anova3.df1(i), results.anova3.df2(i), results.anova3.F(i), results.anova3.p(i));

        % pairwise 2-group ANOVAs
        for p = 1:numel(pair_names)
            idx = pair_idx{p};

            vals1 = q(group_masks{idx(1)});
            vals2 = q(group_masks{idx(2)});
            vals1 = vals1(~isnan(vals1));
            vals2 = vals2(~isnan(vals2));

            x2 = [vals1; vals2];
            g2 = [ones(numel(vals1),1); 2*ones(numel(vals2),1)];

            [p2, tbl2] = anova1(x2, g2, 'off');
            results.(pair_names{p}).F(i) = tbl2{2,5};
            results.(pair_names{p}).df1(i) = tbl2{2,3};
            results.(pair_names{p}).df2(i) = tbl2{3,3};
            results.(pair_names{p}).p(i) = p2;

            fprintf('%s vs %s ANOVA: F(%d, %d) = %.3f, p = %.4g\n', ...
                group_names{idx(1)}, group_names{idx(2)}, ...
                results.(pair_names{p}).df1(i), results.(pair_names{p}).df2(i), ...
                results.(pair_names{p}).F(i), results.(pair_names{p}).p(i));
        end

        fprintf('--------------------------------------------\n\n');
    end
end


function mat_out = match_effects(target_ids, source_ids, source_mat)
    target_ids = cellstr(target_ids);
    source_ids = cellstr(source_ids);

    N = numel(target_ids);
    C = size(source_mat,2);
    mat_out = nan(N,C);

    for i = 1:N
        idx = strcmp(source_ids, target_ids{i});
        if any(idx)
            mat_out(i,:) = source_mat(idx,:);
        end
    end
end
