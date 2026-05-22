function [st] = factor_glm_distr_lr_supp(experiment, num_factors, do_print)
% factor_glm_analysis  Analyze HMM effects and factor scores via GLM and t-tests
%   Usage: result = factor_glm_analysis(num_factors)
%   Loads factor analysis results for the 'turtle' experiment,
%   matches HMM-derived learning rates, computes maladaptive effect masks,
%   runs t-tests and GLMs across all factors, prints summaries, and saves output.

if nargin<1 || isempty(experiment)
    experiment = 'turtle_aligned';
end
if nargin < 2 || isempty(num_factors)
    num_factors = 2;
end
if nargin < 3 || isempty(do_print)
    do_print = true;
end

%% Paths and setup
baseDir = fileparts(mfilename('fullpath'));
cd(baseDir);
resultsDir = fullfile('..', '..', 'mat_data', 'experiment_2');
if ~exist(resultsDir,'dir'), mkdir(resultsDir); end

%% Load pooled factor analysis
fa_file = fullfile(resultsDir, 'fa_score_pooled_ex2_factors2.mat');
fprintf('Loading factor scores from: %s\n', fa_file);
F = load(fa_file);

fa_scores = F.F;
workerIds = F.commonIds;

%% Load distr-HMM learning rates
hmmFile = fullfile(resultsDir, sprintf('distrHMM_rho_fit_params_%s.mat', experiment));
fprintf("Loading distrHMM lr from: %s  \n", hmmFile);
H = load(hmmFile);

lr_full = H.lr; 
lr_ids = H.workerIds;

eff_lr = lr_full * [.25 .25 .25 .25; 1 1 -1 -1; -1 1 -1 1; 1 -1 -1 1]'/2;

% Match learning rates to factor subjects
eff_lr = match_effects(workerIds, lr_ids, eff_lr);

%% Load AGE
[data, ~, ~] = get_data(experiment, true, 2);
d_workerIds = string(cellfun(@(x) x.workerId, data, 'UniformOutput', false));
d_age = cellfun(@(x) x.age, data);
% Match age to factor subjects
d_age = match_effects(workerIds, d_workerIds, d_age(:));

%% Remove NaNs first (for ALL regressors)
valid = ~isnan(d_age) & all(~isnan(eff_lr), 2);

% Apply filtering consistently
eff_lr_valid = eff_lr(valid,:);
fa_scores_valid = fa_scores(valid,:);
workerIds_valid = workerIds(valid);
d_age_valid = d_age(valid);

%% Standardize ALL regressors and dependent variable
eff_lr_valid = zscore(eff_lr_valid);       % N x 4
d_age_valid = zscore(d_age_valid); % N x 1
fa_scores_valid = zscore(fa_scores_valid);

%% Initialize result struct
result.num_factors = num_factors;
result.workerIds   = workerIds_valid;
result.fa_scores   = fa_scores_valid;
result.effects_lr  = eff_lr_valid;
result.age   = d_age_valid;

predictor_names_raw = {'Intercept','B','S','V','S_x_V', 'Age'};
factor_names = {'Factor1','Factor2','Factor1_minus_Factor2'};
result.predictor_names = predictor_names_raw;
result.factor_names = factor_names;

%% Initialize st struct for display only
predictor_names_table = {'B','S','V','S_x_V','Age','I'};

st = struct();
st.num_factors = num_factors;
st.workerIds = workerIds;
st.factor_names = factor_names;
st.predictor_names = predictor_names_table;
st.row_names = {'Coeff.','S.E.M.','t-value','p-value'};

%% GLM fits across all factors
% Observed learning behavior to predict latent psychological factor
fprintf('\n===== GLM: Regression of Factor Scores on distr Effects + (Age) =====\n'); % RPM
for f = 1:num_factors
    % X = [eff_lr_valid, d_rpm_valid, d_age_valid];
    X = [eff_lr_valid, d_age_valid];
    y = fa_scores_valid(:, f);
    [b, ~, stats] = glmfit(X, y, 'normal'); 

    % Wald t-statistics
    tvals = b ./ stats.se;
    % Residual degrees of freedom
    df = length(y) - size(X,2) - 1;
    % Preserve original result structure exactly
    result.glm(f).coeff = b;
    result.glm(f).p     = stats.p;
    result.glm(f).se    = stats.se;
    result.glm(f).t     = tvals;
    result.glm(f).df    = df;

    % Build separate st for display: reorder to put intercept last
    coeff_out = [b(2:end); b(1)]';
    se_out    = [stats.se(2:end); stats.se(1)]';
    t_out     = [tvals(2:end); tvals(1)]';
    p_out     = [stats.p(2:end); stats.p(1)]';

    st.factor(f).name = factor_names{f};
    st.factor(f).df = df;
    st.factor(f).table.rows = st.row_names;
    st.factor(f).table.columns = predictor_names_table;
    st.factor(f).table.data = [
        coeff_out;
        se_out;
        t_out;
        p_out
    ];

    if do_print
        fprintf('\n%s GLM Results: (n = %d, dof = %d)\n', ...
            factor_names{f}, length(workerIds), df);

        for k = 1:length(b)
            fprintf('  %s: beta = %.3f, SE = %.3f, t(%d) = %.3f, p = %.3f\n', ...
                predictor_names_raw{k}, b(k), stats.se(k), df, tvals(k), stats.p(k));
        end
    end
end

%% Save for Python
outFile = fullfile(resultsDir, sprintf('%s_%s.mat', mfilename, experiment));
save(outFile, 'result');
if do_print, fprintf('\nResults saved to %s\n', outFile); end
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