function cbm = distrHMM_rho_fit(experiment, postfix)

    if nargin < 1 
        experiment = 'turtle_aligned'; 
        postfix = "";
    end
    filter = true;
    exclude_criteria = 2;

    addpath('cbm'); addpath(fullfile('..', 'tools'));   
    %% Set up paths and directories
    % Determine the current folder (location of this file) and set the working directory
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);
    % Add folder 'cbm' to the MATLAB path (assumes helper functions are in this folder)
    fdir = fullfile('..', '..', 'mat_data', 'experiment_2');

    %% Load experimental data using the get_data function
    if strcmpi(experiment, 'sim')
        f = load(fullfile(fdir, sprintf('data_sim_binary_distrHMM_rho%s.mat', postfix)));
        data = f.sim_data;
    else
        [data, ~, ~] = get_data(experiment, filter, exclude_criteria);
    end
    N = length(data);  % number of subjects (data entries)

    % Define the filename to save CBM fitting results
    fname = fullfile(fdir, sprintf('%s_%s%s.mat', mfilename, experiment, postfix)); 
    
    if ~exist(fname, 'file')
        subdir = fullfile(fdir, sprintf('%s_%s%s', mfilename, experiment, postfix));
        fprintf("Fitting per subject data, save in folder %s \n", subdir);
        if ~exist(subdir,'dir'), mkdir(subdir); end
        
        number_parameters = 5; % including rho
        
        for n=1:length(data)
            fname_n = fullfile(subdir, sprintf('sub%04d.mat', n));
            if ~exist(fname_n, 'file')
                fprintf("Fitting subject %d: \n", n);
                prior = struct('mean', zeros(number_parameters,1), ...
                               'variance', 6.25);
                config = struct('numinit', 10, 'verbose', 0);
                cbm_lap(data(n), @model_4fit, prior, fname_n, config);
            end
        end
        [fnames] = getfileordered(subdir,sprintf('sub%s.mat','%04d'),1:length(data));
        cbm = cbm_lap_aggregate(fnames,fname);
        fprintf("Saved cbm output to: %s \n", fname);
        save(fname, 'cbm');
    end
    f = load(fname);
    cbm = f.cbm;
    parameters = cbm.output.parameters;
    
    % Define the filename for the model parameters and associated data.
    if strcmpi(experiment, 'sim')
        filename = sprintf('%s_sim_params%s', mfilename, postfix);
    else
        filename = sprintf('%s_params_%s%s', mfilename, experiment, postfix);
    end
    fname = fullfile(fdir, [filename '.mat']); 

    if ~exist(fname, 'file')
        tx = nan(N, 5);
        lr = nan(N, 4);
        block_effect = nan(N, 4);
        signals = cell(N, 1);
        workerIds = cell(1, N);

        fprintf("Save fitted model parameters as %s \n", fname);
        for n=1:length(data)
            [~, tx(n, :), lr(n, :), signals{n}, block_effect(n, :), accuracy{n}]  = model_4fit(parameters(n, :), data{n});
            if strcmpi(experiment,'sim')
                workerIds{n} = n;
            else
                workerIds{n} = data{n}.workerId;
            end
        end
        save(fname, 'tx', 'lr', 'signals', 'block_effect', 'accuracy', 'workerIds');
    end
    f = load(fname);
    tx = f.tx;
    lr = f.lr;

    fprintf("Fitted params: %.3f %.3f %.3f %.3f\n", median(tx,1));
    fprintf("Fitted lr: %.3f %.3f %.3f %.3f\n", mean(lr,1));

    close all;
    end
    
    function [loglik, transformed_parameters, lr, signals, block_effect, accuracy] = model_4fit(parameters, data)
        outcome = data.outcome;
        choices = data.choice;
        outcome(isnan(choices)) = NaN;    
  
        params = parameters(1:4);
        rho = parameters(5);

        [predictions] = distrHMM_model(params, outcome, 1); % predictions=val=weights*r
        % Response model with shared rho
        resp_params = struct('rho', rho);
        out = response_model(predictions, choices, resp_params);        
        loglik = out.loglik;
    
        
        [~, tx, lr, signals, block_effect] = distrHMM_model(parameters, outcome, 1);

        % Return transformed parameters
        transformed_parameters = [tx rho];
        
        % Calculate the number of time steps from the outcome data.
        stats = size(outcome, 1);
        % Define time indices for regression:
        % t1: indices for predictors (all but the last time point)
        % t: indices for response variable (all but the first time point)
        t1 = 1:(stats-1);

        % Compute accuracy: a binary indicator where choice equals outcome.
        accuracy = (choices == outcome);
        % Use predictors corresponding to time indices t1.
        accuracy = accuracy(t1, :);
    
    end
