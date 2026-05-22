function [st] = factor_glm_mn_lr(num_factors, print)
% factor_glm_mn_lr
% Regress factor scores on model-neutral learning-rate effects.
%
% Outputs
% -------
% st
%   .factor(f).name
%   .factor(f).table.data
%   .factor(f).table.rows
%   .factor(f).table.columns
%   .factor(f).coeff
%   .factor(f).se
%   .factor(f).t
%   .factor(f).p
%
% result
%   Backward-compatible raw output structure

if nargin < 2
    num_factors = 2;
    print = true;
end

%% Parameters
experiment = 'bird';
filter_command_q = true; %#ok<NASGU>
exclude_criteria = 2;
postfix = '';

%% Paths and setup
baseDir = fileparts(mfilename('fullpath'));
cd(baseDir);

resultsDir = fullfile('..', '..', 'mat_data', 'experiment_1');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

%% Load factor scores
facFile = fullfile(resultsDir, sprintf('fa_score_ex%d_factors%d_%s.mat', exclude_criteria, num_factors, experiment));

if ~exist(facFile, 'file')
    error('Factor file not found: %s', facFile);
end

F = load(facFile);
fa_scores = F.F;
workerIds = F.workerIds;

%% Load model-neutral learning rates
mnFile = fullfile(resultsDir, sprintf('model_neutral_%s.mat', experiment));
if ~exist(mnFile, 'file')
    error('Model-neutral file not found: %s', mnFile);
end

H = load(mnFile);
lr_full = H.lr;
lr_ids = H.workerIds;

%% Match learning rates to factor subjects
lr_matched = match_effects(workerIds, lr_ids, lr_full);

%% Compute LR effects
% columns:
% 1 = baseline
% 2 = stochasticity effect
% 3 = volatility effect
% 4 = interaction
eff_lr = lr_matched * [.25 .25 .25 .25; -1 -1 1 1; -1 1 -1 1; 1 -1 -1 1]';
fprintf('Mean lr_eff: %.4f, %.4f, %.4f, %.4f\n', mean(eff_lr, 1));

%% Standardize
eff_lr = zscore(eff_lr);
fa_scores = zscore(fa_scores);

%% Add factor difference
fa_scores = [fa_scores(:,1), fa_scores(:,2), fa_scores(:,1) - fa_scores(:,2)];
factor_names = {'Factor 1', 'Factor 2', 'F1 - F2'};
num_factors = 3;

predictor_names = {'Baseline LR', 'Sto Effect', 'Vol Effect', 'Interaction', 'Intercept'};
row_names = {'Coeff.', 'S.E.M.', 't-value', 'p-value'};

%% Initialize outputs
result = struct();
result.num_factors = num_factors;
result.workerIds = workerIds;
result.fa_scores = fa_scores;
result.effects_lr = eff_lr;

st = struct();
st.num_factors = num_factors;
st.workerIds = workerIds;
st.factor_names = factor_names;
st.predictor_names = predictor_names;
st.row_names = row_names;
st.effects_lr = eff_lr;
st.fa_scores = fa_scores;

%% GLM fits across all factors
fprintf('\n===== GLM: Regression of Factor Scores on MN Effects =====\n');

for f = 1:num_factors
    X = eff_lr;
    y = fa_scores(:, f);

    [b, ~, stats] = glmfit(X, y, 'normal');
    tvals = b ./ stats.se;

    % glmfit returns [Intercept; predictors]
    % reorder to: predictors first, intercept last
    coeff_out = [b(2:end); b(1)]';
    se_out    = [stats.se(2:end); stats.se(1)]';
    t_out     = [tvals(2:end); tvals(1)]';
    p_out     = [stats.p(2:end); stats.p(1)]';

    % ---------- result (raw compatibility) ----------
    result.glm(f).factor = factor_names{f};
    result.glm(f).coeff = b;
    result.glm(f).p = stats.p;
    result.glm(f).se = stats.se;
    result.glm(f).t = tvals;

    % ---------- st ----------
    st.factor(f).name = factor_names{f};
    st.factor(f).coeff = coeff_out;
    st.factor(f).se = se_out;
    st.factor(f).t = t_out;
    st.factor(f).p = p_out;

    st.factor(f).table.rows = row_names;
    st.factor(f).table.columns = predictor_names;
    st.factor(f).table.data = [
        coeff_out;
        se_out;
        t_out;
        p_out
    ];
    
    if print
        fprintf('\n%s GLM Results: (n = %d)\n', factor_names{f}, length(workerIds));
        for k = 1:numel(predictor_names)
            fprintf('  %s: beta = %.3f (t = %.3f, p = %.3g, SE = %.3f)\n', ...
                predictor_names{k}, coeff_out(k), t_out(k), p_out(k), se_out(k));
        end
    end
end

%% Save
outFile = fullfile(resultsDir, sprintf('%s_%s%s.mat', mfilename, experiment, postfix));
save(outFile, 'st', 'result');
if print, fprintf('\nResults saved to %s\n', outFile); end

end

%% Helper function to match rows by ID
function mat_out = match_effects(target_ids, source_ids, source_mat)
    N = numel(target_ids); C = size(source_mat,2);
    mat_out = nan(N,C);
    for i=1:N
        idx = strcmp(source_ids, target_ids{i});
        if any(idx)
            mat_out(i,:) = source_mat(idx,:);
        end
    end
end


