function stats_table
    baseDir = fileparts(mfilename('fullpath'));
    cd(baseDir);
    addpath('code_exp1');
    addpath('code_exp2');
    addpath('tools');
%% -------BIRD-------
    %% Supplementary Table: Demographics
    tbl_num = '1';
    experiment = 'bird';
    table_name = sprintf('%s: Demographic summary (bird)', tbl_num);
    [st_gender, st_age, st_rxec_count, st_rxec_pct] = demo2tbl(experiment);

    copy_table(st_gender.table.data, 0);
    T = array2table(st_gender.table.data, ...
        'VariableNames', st_gender.table.columns, ...
        'RowNames', st_gender.table.rows);
    fprintf('\n=== Supplementary Table %s ===\n', table_name);
    disp(T);

    copy_table(st_age.table.data, 0);
    T = array2table(st_age.table.data, ...
        'VariableNames', st_age.table.columns, ...
        'RowNames', st_age.table.rows);
    disp(T);

    copy_table(st_rxec_count.table.data, 0);
    T = array2table(st_rxec_count.table.data, ...
        'VariableNames', st_rxec_count.table.columns, ...
        'RowNames', st_rxec_count.table.rows);
    disp(T);

    %% Supplementary Table: MN LR Regression Effect (Figure 2 stats in-text)
    tbl_num = '2';
    fdir = fullfile('..', '..', 'mat_data', 'experiment_1');
    fname = fullfile(fdir, 'model_neutral_bird.mat'); 
    f = load(fname);
    lr = f.lr;
    block_effect = f.block_effect;
    st = compute_effect(lr, block_effect);
    table_name = sprintf('%s: Model Neutral Regression Effect (%s)(dof: %d)', tbl_num, experiment, length(lr)-1);

    copy_table(st.table.data, 3);

    T = array2table(st.table.data, 'VariableNames', st.table.columns, ...
            'RowNames', st.table.rows);
    % Display the table of parameter recovery in the Command Window
    fprintf('\n=== Supplementary Table %s ===\n', table_name);
    disp(T);

    %% Supplementary Table: Factor with MN LR Effect GLM (Figure 4 stats in-text)
    tbl_num = '3';
    st = factor_glm_mn_lr(2, false);

    for f = 1:st.num_factors
        table_name = sprintf('%s: Factor with MN LR effect GLM (%s, %s)', ...
            tbl_num, experiment, st.factor(f).name);

        fprintf('\n=== Supplementary Table %s ===\n', table_name);

        copy_table(st.factor(f).table.data, 3);

        T = array2table(st.factor(f).table.data, ...
            'VariableNames', matlab.lang.makeValidName(st.factor(f).table.columns), ...
            'RowNames', st.factor(f).table.rows);

        disp(T);
    end
    
%% -------Binary-------
    %% Supplementary Table: Fitted distr-HMM LR Regression Effect (sea lion)
    tbl_num = '4';
    experiment = 'sealion_aligned'; 
    fdir = fullfile('..', '..', 'mat_data', 'experiment_2');
    fname = fullfile(fdir, 'distrHMM_rho_fit_params_sealion_aligned.mat'); 
    f = load(fname);
    sl_lr = f.lr;
    sl_block_effect = f.block_effect;
    st = compute_effect(sl_lr, sl_block_effect);

    copy_table(st.table.data, 3);
    table_name = sprintf('%s: Fitted distr-HMM Regression Effect (%s)(dof: %d)', tbl_num, experiment, length(sl_lr)-1);

    T = array2table(st.table.data, 'VariableNames', st.table.columns, ...
            'RowNames', st.table.rows);
    % Display the table of parameter recovery in the Command Window
    fprintf('\n=== Supplementary Table %s ===\n', table_name);
    disp(T);

    %% Supplementary Table: Fitted distr-HMM LR Regression Effect (turtle)
    tbl_num = '5';
    experiment = 'turtle_aligned'; 
    fname = fullfile(fdir, 'distrHMM_rho_fit_params_turtle_aligned.mat'); 
    f = load(fname);
    t_lr = f.lr;
    t_block_effect = f.block_effect;
    st = compute_effect(t_lr, t_block_effect);

    copy_table(st.table.data, 3);
    table_name = sprintf('%s: Fitted distr-HMM Regression Effect (%s)(dof: %d)', tbl_num, experiment, length(t_lr)-1);

    T = array2table(st.table.data, 'VariableNames', st.table.columns, ...
            'RowNames', st.table.rows);
    % Display the table of parameter recovery in the Command Window
    fprintf('\n=== Supplementary Table %s ===\n', table_name);
    disp(T);

    %% Supplementary Table: Fitted distr-HMM LR Regression Effect (sea lion - turtle)
    tbl_num = '6';
    experiment = 'valence'; 
    st = compute_effect(sl_lr-t_lr, sl_block_effect-t_block_effect);

    copy_table(st.table.data, 3);
    table_name = sprintf('%s: Fitted distr-HMM Regression Effect (%s)(dof: %d)', tbl_num, experiment, length(t_lr)-1);

    T = array2table(st.table.data, 'VariableNames', st.table.columns, ...
            'RowNames', st.table.rows);
    % Display the table of parameter recovery in the Command Window
    fprintf('\n=== Supplementary Table %s ===\n', table_name);
    disp(T);

    %% Supplementary Table: Pooled Factor with distr-HMM Valence LR Effect GLM (Figure 6 stats in-text)
    tbl_num = '7';
    experiment = 'valence';
    [st] = factor_glm_lr_valence(2, false); 

    for f = 1:st.num_factors
        table_name = sprintf('%s: Pooled Binary-task Factor Valence LR Effect GLM (%s)(dof: %d)', ...
            tbl_num, st.factor(f).name, st.factor(f).df);
    
        fprintf('\n=== Supplementary Table %s ===\n', table_name);
    
        copy_table(st.factor(f).table.data, 3);
    
        T = array2table(st.factor(f).table.data, ...
            'VariableNames', matlab.lang.makeValidName(st.factor(f).table.columns), ...
            'RowNames', st.factor(f).table.rows);
    
        disp(T);
    end

    %% Supplementary Table: Factor with distr-HMM LR Effect GLM (Figure 6 stats in-text; sea lion)
    tbl_num = '8';
    experiment = 'sealion_aligned';
    [st] = factor_glm_distr_lr(experiment, 2, false); 
    
    for f = 1:st.num_factors
        table_name = sprintf('%s: Factor with distr-HMM LR Effects (%s, %s)(dof: %d)', ...
                tbl_num, experiment, st.factor(f).name, st.factor(f).df);
        
        fprintf('\n=== Supplementary Table %s ===\n', table_name);
        copy_table(st.factor(f).table.data, 3);
    
        T = array2table(st.factor(f).table.data, ...
            'VariableNames', matlab.lang.makeValidName(st.factor(f).table.columns), ...
            'RowNames', st.factor(f).table.rows);
    
        disp(T);
    end

    %% Supplementary Table: Factor with distr-HMM LR Effect GLM (Figure 6 stats in-text; turtle)
    tbl_num = '9';
    experiment = 'turtle_aligned';
    [st] = factor_glm_distr_lr(experiment, 2, false); 
    
    for f = 1:st.num_factors
        table_name = sprintf('%s: Factor with distr-HMM LR Effects (%s, %s)(dof: %d)', ...
                tbl_num, experiment, st.factor(f).name, st.factor(f).df);
        
        fprintf('\n=== Supplementary Table %s ===\n', table_name);
        copy_table(st.factor(f).table.data, 3);
    
        T = array2table(st.factor(f).table.data, ...
            'VariableNames', matlab.lang.makeValidName(st.factor(f).table.columns), ...
            'RowNames', st.factor(f).table.rows);
    
        disp(T);
    end

    %% Supplementary Table: Learning-rate recovery correlations by block
    tbl_num = '10';
    st = distrHMM_rho_recovery(false);
    copy_table(st.lr_tbl.data, 3);

    T = array2table(st.lr_tbl.data, ...
        'VariableNames', st.lr_tbl.columns, ...
        'RowNames', st.lr_tbl.rows);
    
    fprintf('\n=== Supplementary Table %s: Learning-rate recovery correlations by block ===\n', tbl_num);
    disp(T);
end

function str = copy_table(x, n)
% copy_table
% Copy numeric matrix or numeric columns of a table to clipboard.
%
% Examples
%   copy_table(Tgender, 0)      % copies numeric columns only (e.g., N)
%   copy_table(TwidePct, 3)     % copies numeric part of table rounded to 3 dp
%   copy_table([1.234 5.678],2)

    if nargin < 2
        n = 2;
    end

    % If x is a table, extract numeric columns only
    if istable(x)
        isNum = varfun(@isnumeric, x, 'OutputFormat', 'uniform');
        y = table2array(x(:, isNum));
    else
        y = x;
    end

    if ~isnumeric(y)
        error('Input must be numeric or a table containing numeric columns.');
    end

    % If n is scalar, repeat for each row
    if isscalar(n)
        n = repmat(n, size(y,1), 1);
    end

    z = nan(size(y));
    for i = 1:size(y,1)
        z(i,:) = round(y(i,:) * 10^n(i)) / 10^n(i);
    end

    str = num2clip(z);
end
