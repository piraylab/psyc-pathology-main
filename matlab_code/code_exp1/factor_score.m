function factor_score(experiment, filter, exclude_criteria)
% factor_score Perform EFA and save outputs with dynamic file naming and plots
%   Saves all variables needed for Python replication of plots.
close all;

if nargin<1
    experiment = 'bird';
    filter = true;
    exclude_criteria = 2;
end

% Set paths
% Determine the current folder (location of this file) and set the working directory
currentFolder = fileparts(mfilename('fullpath'));
cd(currentFolder);

% Define directory and load data
fdir = fullfile('..', '..', 'mat_data', 'experiment_1');
[data, surveys, ~] = get_data(experiment, filter, exclude_criteria);

% Specify number of factors
num_factors = 2;  % <-- change as needed or pass as an argument

fname = fullfile(fdir, sprintf('fa_score_ex%d_factors%d_%s.mat', exclude_criteria, num_factors, experiment));

if ~exist(fdir, 'dir')
    mkdir(fdir);
end

if ~exist(fname, 'file')
    % Build data matrix
    field_names = fieldnames(surveys{1});
    % field_names = field_names(1:end-3);  % drop text fields
    % Remove any field that contains 'whodas' (case-insensitive), not a diagnostic scale survey
    field_names = field_names(~contains(field_names, 'whodas', 'IgnoreCase', true));
    num_fields = numel(field_names);
    N = numel(surveys);
    data_matrix = zeros(N, num_fields);
    workerIds = cell(N,1);
    for i = 1:N
        for j = 1:num_fields
            data_matrix(i,j) = surveys{i}.(field_names{j});
        end
        workerIds{i} = data{i}.workerId;
    end
    
    % Standardize items: zero mean, unit variance
    data_matrix = (data_matrix - mean(data_matrix)) ./ std(data_matrix);
    
    % Perform factor analysis
    [loadings, psi, T, stats, F] = factoran(data_matrix, num_factors, 'maxit',1000,'delta',0);
    % apply a promax (oblique) rotation
    [loadings_rot, rotMat] = rotatefactors(loadings, 'Method','promax');
    % replace the originals
    loadings = loadings_rot;
    F        = F * rotMat;        % rotated factor scores
    T        = rotMat;            % transform matrix for consistency
    
    % Calculate eigenvalues for scree
    eigvals = sort(eig(corr(data_matrix)), 'descend');
    % Save results and plotting variables
    save(fname, 'loadings', 'psi', 'T', 'stats', 'F', ...
          'data_matrix', 'workerIds', 'field_names', 'eigvals', 'num_factors');
    fprintf('Factor analysis completed and saved to %s\n', fname);
else
    fprintf("Factor analysis results already exists\n")
end
Fa = load(fname);
factor_sc = Fa.F;
eigvals = Fa.eigvals;
loadings = Fa.loadings;
field_names = Fa.field_names;
num_fields = numel(field_names);

%% === Cattell–Nelson–Gorsuch (CNG) elbow test ===
% (Gorsuch and Nelson, 1981)
% We take the first Kmax eigenvalues and compute:
%   s(i)  = λ(i+1) – λ(i)
%   ds(i) = |s(i+1) – s(i)|
% The elbow is at k = idx+1 where ds(idx) is maximal.

Kmax   = min(10, numel(eigvals));    % limit to first 10 (or fewer) factors
lambda = eigvals(1:Kmax);

% 1) slopes between adjacent eigenvalues
s  = diff(lambda);                   % length Kmax–1

% 2) slope‐differences between adjacent slopes
ds = abs(diff(s));                   % length Kmax–2

% 3) find elbow position
[~, idx]   = max(ds);                % idx in 1:(Kmax–2)
optimal_k  = idx + 1;                % elbow at factor number

% 4) plot annotated scree
figure;
plot(1:Kmax, lambda, '-o','LineWidth',1.5); hold on;
plot(optimal_k, lambda(optimal_k), 'rp','MarkerSize',12,'MarkerFaceColor','r');
xlabel('Factor Number'); ylabel('Eigenvalue');
title('Scree Plot with CNG Elbow');
grid on;
legend('Eigenvalues','CNG elbow','Location','best');
hold off;

fprintf('CNG test suggests retaining %d factors.\n', optimal_k);

%% Factor loadings plot
figure;
bar(loadings(:,1:num_factors));
title(sprintf('Factor Loadings (%d factors)', num_factors));
xlabel('Variables'); ylabel('Loading');
xticks(1:num_fields);
xticklabels(field_names);
legend(arrayfun(@(x) sprintf('Factor %d', x), 1:num_factors, 'UniformOutput', false), 'Location', 'best');
grid on;

%% Explained variance table
p = num_fields;                             % number of variables
% Sum of squared loadings for each factor
SS_loadings        = sum(loadings.^2, 1);
% Percent of total variance explained by each factor
percent_variance   = SS_loadings / p * 100;
% Cumulative percent variance
cum_variance       = cumsum(percent_variance);

% Build a MATLAB table
factorNames = arrayfun(@(k) sprintf('Factor%d', k), 1:num_factors, 'UniformOutput', false).';
Tvar = table( ...
    SS_loadings.', ...
    percent_variance.', ...
    cum_variance.', ...
    'RowNames', factorNames, ...
    'VariableNames', {'SSLoadings','PercentVariance','CumulativeVariance'} ...
);

% Display it
fprintf('\n=== Explained Variance by Factor ===\n');
disp(Tvar);

end

