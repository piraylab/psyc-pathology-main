function st = factor_score_pooled(experiment1, experiment2, filter, exclude_criteria, num_factors)
% FACTOR_SCORE_POOLED
% Fit one EFA model on pooled survey responses from two aligned tasks/sessions,
% using only workerIds common to both sessions.
%
% Example:
%   factor_score_pooled('sealion_aligned', 'turtle_aligned', true, 2, 2)
%
% Output:
%   - one shared factor solution
%   - pooled factor scores for both sessions
%   - subject-level averaged scores across sessions
%
% Requirements:
%   - survey item names must match exactly across sessions
%   - workerId must be present in data{i}.workerId

close all;

if nargin < 1 || isempty(experiment1), experiment1 = 'sealion_aligned'; end
if nargin < 2 || isempty(experiment2), experiment2 = 'turtle_aligned'; end
if nargin < 3 || isempty(filter), filter = true; end
if nargin < 4 || isempty(exclude_criteria), exclude_criteria = 2; end
if nargin < 5 || isempty(num_factors), num_factors = 2; end

%% Working directory / paths
currentFolder = fileparts(mfilename('fullpath'));
cd(currentFolder);

%% Load both datasets
[data1, surveys1, ~] = get_data(experiment1, filter, exclude_criteria);
[data2, surveys2, ~] = get_data(experiment2, filter, exclude_criteria);

if isempty(surveys1) || isempty(surveys2)
    error('One of the survey sets is empty.');
end

%% Check item names
field_names1 = fieldnames(surveys1{1});
field_names2 = fieldnames(surveys2{1});

if ~isequal(field_names1, field_names2)
    error('Survey item fields differ between %s and %s. Cannot pool directly.', experiment1, experiment2);
end

field_names = field_names1;
num_fields = numel(field_names);

%% Build survey matrices and workerId vectors
N1 = numel(surveys1);
N2 = numel(surveys2);

X1 = nan(N1, num_fields);
X2 = nan(N2, num_fields);

workerIds1 = strings(N1, 1);
workerIds2 = strings(N2, 1);

for i = 1:N1
    for j = 1:num_fields
        X1(i, j) = surveys1{i}.(field_names{j});
    end
    if isfield(data1{i}, 'workerId')
        workerIds1(i) = string(data1{i}.workerId);
    else
        workerIds1(i) = "task1_sub" + sprintf('%04d', i);
    end
end

for i = 1:N2
    for j = 1:num_fields
        X2(i, j) = surveys2{i}.(field_names{j});
    end
    if isfield(data2{i}, 'workerId')
        workerIds2(i) = string(data2{i}.workerId);
    else
        workerIds2(i) = "task2_sub" + sprintf('%04d', i);
    end
end

%% Keep only subjects common to both sessions
[commonIds, idx1, idx2] = intersect(workerIds1, workerIds2, 'stable');

if isempty(commonIds)
    error('No common workerIds found between %s and %s.', experiment1, experiment2);
end

fprintf('Found %d common subjects across sessions.\n', numel(commonIds));

% Subset to matched subjects only
X1 = X1(idx1, :);
X2 = X2(idx2, :);
workerIds1 = workerIds1(idx1);
workerIds2 = workerIds2(idx2);

%% Drop any paired subject with missing survey values in either session
valid1 = all(~isnan(X1), 2);
valid2 = all(~isnan(X2), 2);
validPair = valid1 & valid2;

X1 = X1(validPair, :);
X2 = X2(validPair, :);
workerIds1 = workerIds1(validPair);
workerIds2 = workerIds2(validPair);
commonIds = commonIds(validPair);

%% Pool matched rows
% data_matrix = [X1; X2];
% session_label = [repmat({experiment1}, size(X1,1), 1); repmat({experiment2}, size(X2,1), 1)];
% workerIds = [workerIds1; workerIds2];
data_matrix = [X1, X2];
workerIds = commonIds;

%% Output directory and file names
outdir = fullfile('..', '..', 'mat_data', 'experiment_2');
if ~exist(outdir, 'dir'), mkdir(outdir); end

fname = fullfile(outdir, sprintf('fa_score_pooled_ex%d_factors%d.mat', ...
                 exclude_criteria, num_factors));

%% Run factor analysis
if ~exist(fname, 'file')
    fprintf('Running pooled factor analysis on %d rows x %d items...\n', size(data_matrix,1), size(data_matrix,2));

    % Factor analysis
    [loadings, psi, T0, stats, F0] = factoran(data_matrix, num_factors, 'maxit', 1000, 'delta', 0);

    % Oblique rotation
    [loadings_rot, rotMat] = rotatefactors(loadings, 'Method', 'promax');

    % Rotate factor scores to match rotated loadings
    F = F0 * rotMat;
    loadings = loadings_rot;
    T = rotMat;

    % Scree eigenvalues
    eigvals = sort(eig(corr(data_matrix)), 'descend');

    save(fname, 'loadings', 'psi', 'T', 'stats', 'F', ...
        'data_matrix', 'workerIds', 'commonIds', ...
        'field_names', 'eigvals', 'num_factors', ...
        'experiment1', 'experiment2', 'exclude_criteria');

    fprintf('Pooled factor analysis completed and saved to %s\n', fname);
else
    fprintf('Pooled factor analysis results already exist: %s\n', fname);
end

%% Load results
Fa = load(fname);
eigvals = Fa.eigvals;
loadings = Fa.loadings;
field_names = Fa.field_names;
num_fields = numel(field_names);

%% Explained variance table
p = num_fields;
SS_loadings = sum(loadings.^2, 1);
percent_variance = SS_loadings / p * 100;
cum_variance = cumsum(percent_variance);

factorNames = arrayfun(@(k) sprintf('Factor%d', k), 1:num_factors, 'UniformOutput', false).';
Tvar = table( ...
    SS_loadings.', ...
    percent_variance.', ...
    cum_variance.', ...
    'RowNames', factorNames, ...
    'VariableNames', {'SSLoadings', 'PercentVariance', 'CumulativeVariance'} ...
);

fprintf('\n=== Explained Variance by Factor (Pooled) ===\n');
disp(Tvar);


end