function [st] = mn_lr_glm_factor(num_factors, do_print)
% factor_predict_mn_effects
% Regress model-neutral learning-rate effects on factor scores.
%
% Runs two GLM families for each dependent effect:
%
% Model A:
%   Effect ~ Factor 1 + Factor 2
%
% Model B:
%   Effect ~ (Factor 1 - Factor 2) + 0.5*(Factor 1 + Factor 2)
%
% Dependent variables:
%   1) Baseline LR
%   2) Sto Effect
%   3) Vol Effect
%
% Outputs
% -------
% st.modelA.effect(e) : regression on [Factor 1, Factor 2]
% st.modelB.effect(e) : regression on [F1-F2, 0.5*(F1+F2)]
%
% Each contains:
%   .name
%   .coeff
%   .se
%   .t
%   .p
%   .table.data
%   .table.rows
%   .table.columns

if nargin < 1
    num_factors = 2;
end
if nargin < 2
    do_print = true;
end

%% Parameters
experiment = 'bird';
exclude_criteria = 2;
postfix = '';

%% Paths and setup
baseDir = fileparts(mfilename('fullpath'));
cd(baseDir);
addpath(fullfile('..', 'tools'));

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
mnFile = fullfile(resultsDir, sprintf('model_neutral_%s%s.mat', experiment, postfix));
if ~exist(mnFile, 'file')
    error('Model-neutral file not found: %s', mnFile);
end

H = load(mnFile);
lr_full = H.lr;
lr_ids = H.workerIds;

%% Match learning rates to factor subjects
lr_matched = match_effects(workerIds, lr_ids, lr_full);

%% Compute MN LR effects
% columns:
% 1 = baseline
% 2 = stochasticity effect
% 3 = volatility effect
% 4 = interaction
eff_lr = lr_matched * [.25 .25 .25 .25; -1 -1 1 1; -1 1 -1 1; 1 -1 -1 1]';

%% Standardize
fa_scores = zscore(fa_scores);
eff_lr = zscore(eff_lr);

%% Build predictors
if size(fa_scores,2) < 2
    error('Expected at least 2 factor columns.');
end

f1 = fa_scores(:,1);
f2 = fa_scores(:,2);

X_A = [f1, f2];
X_B = [f1 - f2, 0.5 * (f1 + f2)];

predictor_names_A = {'Factor 1', 'Factor 2', 'Intercept'};
predictor_names_B = {'F1 - F2', '0.5*(F1 + F2)', 'Intercept'};
row_names = {'Coeff.', 'S.E.M.', 't-value', 'p-value'};

%% Dependent variables: 3 separate GLMs
effect_cols = [1 2 3];
effect_names = {'Baseline LR', 'Sto Effect', 'Vol Effect'};

%% Remove rows with NaNs
valid = all(~isnan(X_A), 2) & all(~isnan(X_B), 2) & all(~isnan(eff_lr(:, effect_cols)), 2);

workerIds_valid = workerIds(valid);
X_A = X_A(valid, :);
X_B = X_B(valid, :);
Y_effects = eff_lr(valid, effect_cols);

%% Initialize output
st = struct();
st.num_effects = numel(effect_cols);
st.workerIds = workerIds_valid;
st.effect_names = effect_names;
st.row_names = row_names;

st.modelA.name = 'Effect ~ Factor 1 + Factor 2';
st.modelA.predictor_names = predictor_names_A;
st.modelA.factor_scores = X_A;

st.modelB.name = 'Effect ~ (F1 - F2) + 0.5*(F1 + F2)';
st.modelB.predictor_names = predictor_names_B;
st.modelB.factor_scores = X_B;

st.effects = Y_effects;

fprintf('\n===== GLM: Regression of MN Effects on Factor Scores =====\n');

for e = 1:numel(effect_cols)
    y = Y_effects(:, e);

    % -------------------------
    % Model A: Factor 1 + Factor 2
    % -------------------------
    [bA, ~, statsA] = glmfit(X_A, y, 'normal');
    tA = bA ./ statsA.se;

    coeff_A = [bA(2:end); bA(1)]';
    se_A    = [statsA.se(2:end); statsA.se(1)]';
    t_out_A = [tA(2:end); tA(1)]';
    p_A     = [statsA.p(2:end); statsA.p(1)]';

    st.modelA.effect(e).name = effect_names{e};
    st.modelA.effect(e).coeff = coeff_A;
    st.modelA.effect(e).se = se_A;
    st.modelA.effect(e).t = t_out_A;
    st.modelA.effect(e).p = p_A;
    st.modelA.effect(e).table.rows = row_names;
    st.modelA.effect(e).table.columns = predictor_names_A;
    st.modelA.effect(e).table.data = [
        coeff_A;
        se_A;
        t_out_A;
        p_A
    ];

    % -------------------------
    % Model B: F1-F2 + Average
    % -------------------------
    [bB, ~, statsB] = glmfit(X_B, y, 'normal');
    tB = bB ./ statsB.se;

    coeff_B = [bB(2:end); bB(1)]';
    se_B    = [statsB.se(2:end); statsB.se(1)]';
    t_out_B = [tB(2:end); tB(1)]';
    p_B     = [statsB.p(2:end); statsB.p(1)]';

    st.modelB.effect(e).name = effect_names{e};
    st.modelB.effect(e).coeff = coeff_B;
    st.modelB.effect(e).se = se_B;
    st.modelB.effect(e).t = t_out_B;
    st.modelB.effect(e).p = p_B;
    st.modelB.effect(e).table.rows = row_names;
    st.modelB.effect(e).table.columns = predictor_names_B;
    st.modelB.effect(e).table.data = [
        coeff_B;
        se_B;
        t_out_B;
        p_B
    ];

    if do_print
        fprintf('\n----------------------------------------\n');
        fprintf('%s\n', effect_names{e});
        fprintf('----------------------------------------\n');

        fprintf('\nModel A: %s\n', st.modelA.name);
        for k = 1:numel(predictor_names_A)
            fprintf('  %s: beta = %.3f (t = %.3f, p = %.3g, SE = %.3f)\n', ...
                predictor_names_A{k}, coeff_A(k), t_out_A(k), p_A(k), se_A(k));
        end

        fprintf('\nModel B: %s\n', st.modelB.name);
        for k = 1:numel(predictor_names_B)
            fprintf('  %s: beta = %.3f (t = %.3f, p = %.3g, SE = %.3f)\n', ...
                predictor_names_B{k}, coeff_B(k), t_out_B(k), p_B(k), se_B(k));
        end
    end
end

%% Save
outFile = fullfile(resultsDir, sprintf('%s_%s%s.mat', mfilename, experiment, postfix));
save(outFile, 'st');
if do_print
    fprintf('\nResults saved to %s\n', outFile);
end

end

%% Helper function to match rows by ID
function mat_out = match_effects(target_ids, source_ids, source_mat)
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