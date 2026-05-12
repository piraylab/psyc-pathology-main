function result = factor_glm_distr_lr_supp(experiment, num_factors)
% factor_glm_analysis  Analyze HMM effects and factor scores via GLM and t-tests
%   Usage: result = factor_glm_analysis(num_factors)
%   Loads factor analysis results for the 'turtle' experiment,
%   matches HMM-derived learning rates, computes maladaptive effect masks,
%   runs t-tests and GLMs across all factors, prints summaries, and saves output.

if nargin<1
    experiment = 'sealion_aligned';
    num_factors = 2;  % default number of factors to analyze
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

%% Load RPM and AGE
bird_dir = fullfile('..', 'code_exp1');
cd(bird_dir);
d_experiment = 'bird';
[data, ~, ~] = get_data(d_experiment, true, 2);
d_workerIds = string(cellfun(@(x) x.workerId, data, 'UniformOutput', false));
d_rpm       = cellfun(@(x) x.rpm_score, data);
d_age_month = cellfun(@(x) x.age_month, data);
% Match rpm and age to factor subjects
d_rpm       = match_effects(workerIds, d_workerIds, d_rpm(:));
d_age_month = match_effects(workerIds, d_workerIds, d_age_month(:));

%% Remove NaNs first (for ALL regressors)
valid = ~isnan(d_age_month) & ~isnan(d_rpm) & all(~isnan(eff_lr), 2);

% Apply filtering consistently
eff_lr_valid = eff_lr(valid,:);
fa_scores_valid = fa_scores(valid,:);
workerIds_valid = workerIds(valid);
d_rpm_valid = d_rpm(valid);
d_age_month_valid = d_age_month(valid);

%% Standardize ALL regressors and dependent variable
eff_lr_valid = zscore(eff_lr_valid);       % N x 4
d_rpm_valid = zscore(d_rpm_valid);         % N x 1
d_age_month_valid = zscore(d_age_month_valid); % N x 1
fa_scores_valid = zscore(fa_scores_valid);

%% Initialize result struct
result.num_factors = num_factors;
result.workerIds   = workerIds_valid;
result.fa_scores   = fa_scores_valid;
result.effects_lr  = eff_lr_valid;
result.rpm         = d_rpm_valid;
result.age_month   = d_age_month_valid;

%% GLM fits across all factors
% Observed learning behavior to predict latent psychological factor
fprintf('\n===== GLM: Regression of Factor Scores on distr Effects + (RPM + Age) =====\n');
for f = 1:num_factors
    X = [eff_lr_valid, d_rpm_valid, d_age_month_valid];
    y = fa_scores_valid(:, f);
    [b, ~, stats] = glmfit(X, y, 'normal'); 

    result.glm(f).coeff = b;
    result.glm(f).p     = stats.p;
    result.glm(f).se    = stats.se;
    fprintf('\nFactor %d GLM Results: (n = %d) \n', f, length(workerIds_valid));
    for k = 1:length(b)
        fprintf('  beta%g = %.3f (p=%.3f, SE=%.3f)\n', k-1, b(k), stats.p(k), stats.se(k));
    end
end

%% Save for Python
outFile = fullfile(resultsDir, sprintf('%s_%s.mat', mfilename, experiment));
save(outFile, 'result');
fprintf('\nResults saved to %s\n', outFile);
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