function [st] = factor_glm_lr_valence(num_factors, print)
% factor_glm_lr_valence
% Analyze pooled binary-task learning effects (shared + valence contrasts)
% against pooled psychiatric factor scores using GLM.
%
% Saves:
%   - coefficients
%   - standard errors
%   - Wald t-statistics
%   - p-values
%   - residual degrees of freedom
%   - per-factor summary table
%
% Outputs
% -------
% result : raw output struct
% st     : table-oriented struct for manuscript reporting

if nargin < 1 || isempty(num_factors)
    num_factors = 2;
end
if nargin < 2 || isempty(print)
    print = true;
end

postfix = '';
m = [.25 .25 .25 .25; ...
     -1 -1  1  1; ...
     -1  1 -1  1; ...
      1 -1 -1  1]';

%% Paths and setup
baseDir = fileparts(mfilename('fullpath'));
cd(baseDir);
resultsDir = fullfile('..', '..', 'mat_data', 'experiment_2');

%% Parameters and load aligned results
lr_file1 = 'distrHMM_rho_fit_params_sealion_aligned.mat';
lr_file2 = 'distrHMM_rho_fit_params_turtle_aligned.mat';

F1 = load(fullfile(resultsDir, lr_file1));
lr1 = F1.lr;
lr1_ids = F1.workerIds;

F2 = load(fullfile(resultsDir, lr_file2));
lr2 = F2.lr;
lr2_ids = F2.workerIds;

%% Load pooled factor analysis
fa_file = fullfile(resultsDir, 'fa_score_pooled_ex2_factors2.mat');
fprintf('Loading pooled factor scores from: %s\n', fa_file);
F = load(fa_file);

fa_scores = F.F;
workerIds = F.commonIds;

%% Match learning rates to factor subjects
lr1_matched = match_effects(workerIds, lr1_ids, lr1);
lr2_matched = match_effects(workerIds, lr2_ids, lr2);

%% Compute shared and valence effects
% lr*_eff columns:
%   1 = baseline (B)
%   2 = stochasticity effect (S)
%   3 = volatility effect (V)
%   4 = interaction (S_x_V)
lr1_eff = lr1_matched * m;
lr2_eff = lr2_matched * m;

% pooled predictors:
%   1: B
%   2: S
%   3: V
%   4: S_x_V
%   5: Valence_x_B
%   6: Valence_x_S
%   7: Valence_x_V
%   8: Valence_S_x_V
eff_lr = [lr1_eff + lr2_eff, lr1_eff - lr2_eff];

fprintf('Mean lr1_eff: %.4f, %.4f, %.4f, %.4f\n', mean(lr1_eff, 1));
fprintf('Mean lr2_eff: %.4f, %.4f, %.4f, %.4f\n', mean(lr2_eff, 1));

%% Standardize regressors and dependent variables
eff_lr = zscore(eff_lr);
fa_scores = zscore(fa_scores);

%% Add factor difference
fa_scores = [fa_scores(:,1), fa_scores(:,2), fa_scores(:,1) - fa_scores(:,2)];
num_factors = 3;

%% Initialize result struct
result = struct();
result.num_factors = num_factors;
result.workerIds   = workerIds;
result.fa_scores   = fa_scores;
result.effects_lr  = eff_lr;

result.predictor_names = { ...
    'Intercept', ...
    'B', ...
    'S', ...
    'V', ...
    'S_x_V', ...
    'Valence_x_B', ...
    'Valence_x_S', ...
    'Valence_x_V', ...
    'Valence_S_x_V'};

result.factor_names = {'Factor1','Factor2','Factor1_minus_Factor2'};

%% Initialize st struct
st = struct();
st.num_factors = num_factors;
st.workerIds = workerIds;
st.factor_names = result.factor_names;
st.row_names = {'Coeff.','S.E.M.','t-value','p-value'};
st.predictor_names = {'B','S','V','S_x_V','Valence_x_B','Valence_x_S','Valence_x_V','Valence_S_x_V','I'};
st.effects_lr = eff_lr;
st.fa_scores = fa_scores;

%% GLM fits across all factors
fprintf('\n===== GLM: Regression of Factor Scores on distr Effects =====\n');

predictor_names = result.predictor_names;

for f = 1:num_factors
    X = eff_lr;          % N x 8 predictors
    y = fa_scores(:, f); % N x 1

    [b, ~, stats] = glmfit(X, y, 'normal');

    % Wald t-statistics
    tvals = b ./ stats.se;

    % residual degrees of freedom
    df = length(y) - size(X,2) - 1;

    % save raw outputs
    result.glm(f).coeff = b;
    result.glm(f).p     = stats.p;
    result.glm(f).se    = stats.se;
    result.glm(f).t     = tvals;
    result.glm(f).df    = df;

    % reorder for table display: predictors first, intercept last
    coeff_out = [b(2:end); b(1)]';
    se_out    = [stats.se(2:end); stats.se(1)]';
    t_out     = [tvals(2:end); tvals(1)]';
    p_out     = [stats.p(2:end); stats.p(1)]';

    st.factor(f).name = result.factor_names{f};
    st.factor(f).df = df;
    st.factor(f).coeff = coeff_out;
    st.factor(f).se = se_out;
    st.factor(f).t = t_out;
    st.factor(f).p = p_out;

    st.factor(f).table.rows = st.row_names;
    st.factor(f).table.columns = st.predictor_names;
    st.factor(f).table.data = [
        coeff_out;
        se_out;
        t_out;
        p_out
    ];

    if print
        fprintf('\n%s GLM Results: (n = %d, dof = %d)\n', ...
            result.factor_names{f}, length(workerIds), df);
        for k = 1:length(b)
            fprintf('  %s: beta = %.3f, SE = %.3f, t(%d) = %.3f, p = %.3f\n', ...
                predictor_names{k}, b(k), stats.se(k), df, tvals(k), stats.p(k));
        end
    end
end

%% Save for Python / manuscript tables
outFile = fullfile(resultsDir, sprintf('%s.mat', mfilename));
save(outFile, 'result', 'st');
if print
    fprintf('\nResults saved to %s\n', outFile);
end

end

%% Helper function to match rows by ID
function mat_out = match_effects(target_ids, source_ids, source_mat)
    N = numel(target_ids);
    C = size(source_mat, 2);
    mat_out = nan(N, C);

    for i = 1:N
        idx = strcmp(source_ids, target_ids{i});
        if any(idx)
            mat_out(i, :) = source_mat(idx, :);
        end
    end
end
