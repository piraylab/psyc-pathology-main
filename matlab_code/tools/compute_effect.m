function st = compute_effect(lr, b)
% COMPUTE_EFFECT Computes and displays regression effects using learning rate 
%             and block effect data.
%
%   compute_effect(lr, b)
%
%   Inputs:
%       lr - A matrix containing learning rates (e.g., subjects x 4 blocks).
%       b  - A matrix of block effects (e.g., subjects x 4 blocks).
%
%   This function computes two sets of effects:
%       1. Effects derived from the learning rates (eff_lr) using a specific 
%          contrast matrix.
%       2. Simple effects from the block data (eff_simple) using a different 
%          contrast matrix.
%
%   The function then:
%       - Combines these effects.
%       - Computes summary statistics such as mean, standard error, and t-test 
%         statistics for each effect.
%       - Calculates the correlation between two of the learning rate effects.
%       - Organizes these statistical outputs into a table and displays it.
%

    %% Compute combined effects using linear contrasts
    % Calculate effects based on the learning rates (lr). The contrast matrix is
    % designed so that each column represents a different combination of effects.
    eff_lr = lr * [1 1 1 1; -1 -1 1 1; -1 1 -1 1; 1 -1 -1 1]' / 2;
    
    % Compute the "simple" block effects using a separate contrast matrix.
    eff_simple = b * [-1 -1 1 1; -1 1 -1 1; 1 -1 -1 1; 1 1 1 1]' / 2;
    
    % Concatenate both sets of effects (learning rate and simple effects) into one matrix.
    eff = [eff_lr, eff_simple];
    
    %% Compute summary statistics for the combined effects
    % Calculate the mean of each effect column.
    mean_eff = mean(eff, 'omitmissing');
    
    % Compute the standard error (S.E.M.) for each effect. The function 'serr'
    % is assumed to compute the standard error across subjects.
    serr_eff = serr(eff, 1);
    
    % Perform a one-sample t-test on each column of 'eff'. The t-test returns:
    %   - p_eff: p-values,
    %   - ci_eff: confidence intervals,
    %   - st: a structure containing the t-statistics.
    [~, p_eff, ci_eff, st] = ttest(eff);
    
    %% Organize statistical outputs into a table structure
    % Combine mean effect, standard error, t-statistics, and p-values into one table.
    tbl_data = [mean_eff; serr_eff; st.tstat; p_eff];
    
    % Define row labels for the table:
    %   'Mean Effect': Mean values for each effect.
    %   'S.E.M.': Standard error of the mean.
    %   't-statistics': t-test statistic values.
    %   'P-value': p-values from the t-tests.
    st.table.rows = {'Mean Effect', 'S.E.M.', 't-statistics', 'P-value'};
    
    % Define column labels for each computed effect. These labels indicate the 
    % nature of each effect (e.g., main effect of prediction error 'PE', interaction
    % effects involving volatility and stochasticity, and an intercept term).
    st.table.columns = {'PE','PE x Sto', 'PE x Vol', 'PE x Sto x Vol', ...
                        'Sto', 'Vol', 'Sto x Vol', 'Intercept'};
    
    % Store the table data in the structure.
    st.table.data = tbl_data;
    
    % Save additional computed statistics for later use or display.
    st.p      = p_eff;                    % p-values from the t-test.
    st.ci     = ci_eff;                   % Confidence intervals.
    st.labels = {'PE','PE x Sto', 'PE x Vol', 'PE x Sto x Vol', ...
                 'Sto', 'Vol', 'Sto x Vol', 'Intercept'}; % Effect labels.
    st.mean   = mean_eff;                 % Mean effects.
    
    % Compute the percent of subjects showing a negative effect for each effect.
    st.percent_neg = round(100 * mean(eff < 0));
    
    %% Compute correlation between specific learning rate effects
    % Calculate the Pearson correlation between columns 2 and 3 of eff_lr, which
    % represent specific contrasts (e.g., interaction effects).
    [cr, cp] = corr(eff_lr(:, 2:3));
    st.corr_stovol_r = cr(:)';    % Flatten the correlation coefficients into a row vector.
    st.corr_stovol_p = cp(:)';    % Flatten the p-values of the correlation.

end


