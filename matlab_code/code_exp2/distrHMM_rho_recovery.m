function st = distrHMM_rho_recovery(doPlot)
% DISTRHMM_RHO_RECOVERY  Parameter recovery for distributed HMM + response model (rho)
%
% Recovers:
%   - u_vol, u_sto, v_vol, v_sto, rho
%
% Simulates choices using distrHMM_model + response_model.
    if nargin<1, doPlot = true; end
    close all;

    %% Setup working environment and paths
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);

    %% Load real task outcome sequence
    experimentData = 'sealion_aligned';
    [data, ~, ~] = get_data(experimentData);

    %% Simulation file setup
    simExperiment = 'sim';
    fdir = fullfile('..', '..', 'mat_data', sprintf('experiment_%s', simExperiment));
    if ~exist(fdir, 'dir')
        mkdir(fdir);
    end

    N = 1000;
    sim_data = cell(N,1);
    fname = fullfile(fdir, sprintf('data_sim_binary_distrHMM_rho.mat'));
    fprintf("Using simulated data file: %s\n", fname);

    % Canonical order of parameters
    paramFields = {'u_vol','u_sto','v_vol','v_sto','rho'};

    %% Simulate if needed
    if ~exist(fname, 'file')
        fprintf('Simulating distr-HMM recovery data...\n');
        % Use fitted CBM parameters from real sealion task
        real_experiment = 'sealion';
        real_postfix = '';
        real_fdir = fullfile('..', '..', 'mat_data', 'experiment_2');
        real_fname = fullfile(real_fdir, sprintf('distrHMM_rho_fit%s.mat', real_postfix));

        if ~exist(real_fname, 'file')
            error('Cannot find real fit file: %s', real_fname);
        end

        f_real = load(real_fname);
        cbm_real = f_real.cbm.output.parameters;

        % Each subject has 5 raw parameters: u_vol, u_sto, v_vol, v_sto, rho
        Nreal = length(cbm_real);
        raw_params = zeros(Nreal, 5);
        for i = 1:Nreal
            raw_params(i,:) = cbm_real(i,:);
        end

        % Empirical mean and SD in raw space
        mu_params = mean(raw_params, 1);
        sd_params = std(raw_params, 0, 1);

        % Simulate subjects from empirical parameter distribution
        for n = 1:N
            rng(n);

            theta_raw = mu_params + sd_params .* randn(1,5);

            % Same constraint used in your earlier script
            half_sigmoid = @(x) 0.5 ./ (1 + exp(-x));

            sim_params = struct( ...
                'u_vol', half_sigmoid(theta_raw(1)), ...
                'u_sto', half_sigmoid(theta_raw(2)), ...
                'v_vol', half_sigmoid(theta_raw(3)), ...
                'v_sto', half_sigmoid(theta_raw(4)), ...
                'rho',   theta_raw(5) ...
            );

            config = struct();
            config.seed = n;
            config.outcome = data{1}.outcome;
            config.sim_params = sim_params;
            config.transform = 0;   % parameters already constrained here

            sim_data{n} = sim_outcome_choice(config);
        end

        save(fname, 'sim_data', 'paramFields');
        fprintf('Saved simulated data to %s\n', fname);
    else
        fprintf("Simulation data already exists: %s\n", fname);
    end

    %% Reload simulated data
    S = load(fname);
    sim_data = S.sim_data;
    nSim = numel(sim_data);
    P = numel(paramFields);
    sim_params = nan(nSim, P);
    sim_lr = nan(nSim, 4);
    for i = 1:nSim
        % get fields in the canonical order
        sim_lr(i, :) = sim_data{i}.lr;
        for j = 1:P
            sim_params(i,j) = sim_data{i}.sim_params.(paramFields{j});
        end
    end

    %% Fit distributed HMM to simulated data if needed
    fitfile = fullfile(fdir, sprintf('distrHMM_rho_fit_sim_params.mat'));
    fprintf("Loading recovered parameters file: %s\n", fitfile);
    if ~exist(fitfile,'file')
        distrHMM_rho_fit('sim', postfix);  % assumes it reads data_sim*.mat internally
    end
    F = load(fitfile);

    %% Extract recovered parameters robustly
    recovered_params = F.tx; 
    rec_lr = F.lr;
    % revert sigmoid transformation back to Gaussian using logit function
    logit = @(x, k)log((k*x)./(1-(k*x)));
    transform_func = @(x)[logit(x(:, 1:2), 1), logit(x(:, 3:4), 2)];
    sim_params = [transform_func(sim_params), sim_params(:,5)];
    recovered_params = [transform_func(recovered_params), recovered_params(:,5)];

    fprintf("Mean simulated params: %.3f %.3f %.3f %.3f %.3f\n", mean(sim_params,1));
    fprintf("Mean recovered params: %.3f %.3f %.3f %.3f %.3f\n", mean(recovered_params,1));
    
    
    %% Plots / stats
    save_path = fullfile('..', '..', 'saved_figures');
    fprintf('Saved figure to: %s\n', save_path);
    % Compare recovered lr and lr main effects as that is major results
    stats = distrHMM_recovery_lr(sim_lr, rec_lr, save_path, 'FigureSupp');
    if doPlot
        if exist('plot_param_recovery','file')
            fname = 'FigureSupp_distr_param_recovery';
            plot_param_recovery(sim_params, recovered_params); %'SavePath', save_path, 'FilePrefix', fname);
        end
        % if exist('plot_recovery_errors','file')
        %     fname = 'FigureSupp_distr_param_recovery_errors';
        %     plot_recovery_errors(sim_params, recovered_params); %'SavePath', save_path, 'FilePrefix', fname);
        % end
        if exist('plot_recovery_error_density', 'file')
            fname = 'FigureSupp_distr_param_recovery_errors_density';
            plot_recovery_error_density(sim_params, recovered_params); % 'SavePath', save_path, 'FilePrefix', fname);
        end
    end
    % % Save to ./figs
    % plot_recovery_error_density(sim_params, recovered_params, ...
    %     'XLim', [-3 3], 'SavePath','./figs');

     %% Collect output
    % 1) Parameter recovery error summary table
    % rows = parameters
    % cols = error quantiles
    st.table.data = [ ...
        quantile(sim_params - recovered_params, 0.25); ...
        quantile(sim_params - recovered_params, 0.50); ...
        quantile(sim_params - recovered_params, 0.75)]';

    st.table.rows = {'u\_vol','u\_sto','v\_vol','v\_sto','rho'};
    st.table.columns = {'25% quantile','Median','75% quantile'};

    % T = array2table(st.table.data, ...
    %     'RowNames', st.table.rows, ...
    %     'VariableNames', st.table.columns);
    % fprintf('\n=== Parameter recovery error summary ===\n');
    % disp(T);

    % 2) Learning-rate recovery correlation table
    % rows = blocks
    % cols = r and p
    st.lr_tbl.data = [stats.lr.r; stats.lr.p];
    st.lr_tbl.rows = {'r','p'};
    st.lr_tbl.columns = stats.lr.labels;

    % Tlr = array2table(st.lr_tbl.data, ...
    %     'RowNames', st.lr_tbl.rows, ...
    %     'VariableNames', st.lr_tbl.columns);
    % fprintf('\n=== Learning-rate recovery correlations ===\n');
    % disp(Tlr);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper: simulate outcome and choice data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function data = sim_outcome_choice(config)
    sim_params = config.sim_params;
    outcome = config.outcome;
    rng(config.seed);

    % Transform flag for distrHMM_model
    if isfield(config, 'transform')
        do_transform = config.transform;
    else
        do_transform = 0;
    end

    % Model parameters in the order expected by distrHMM_model
    parameters = [sim_params.u_vol, sim_params.u_sto, sim_params.v_vol, sim_params.v_sto];

    % HMM belief update / prediction
    [predictions, ~, lr, ~] = distrHMM_model(parameters, outcome, do_transform);

    % Generate choices sequentially so rho is active
    [T, B] = size(predictions);
    choice = nan(T, B);

    for t = 1:T
        if t == 1
            prev = nan(1, B);
        else
            prev = choice(t-1,:);
        end

        resp = response_model( ...
            predictions(t,:), ...
            prev, ...
            struct('rho', sim_params.rho) ...
        );

        choice(t,:) = binornd(1, resp.p1);
    end

    data = struct( ...
        'outcome', outcome, ...
        'choice', choice, ...
        'sim_params', sim_params, ...
        'lr', lr);
end