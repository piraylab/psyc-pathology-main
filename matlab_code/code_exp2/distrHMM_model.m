function [val, tx, lr, signals, block_effect] = distrHMM_model(parameters, observations, do_transform)
    if nargin<3, do_transform = 1; end
    if nargin<2
        [data, ~, ~] = get_data('sealion_aligned', true, 1);
        observations = data{1}.outcome;
    end
    if nargin<1, parameters = zeros(1, 4); end
    
    if do_transform == 1
        % transform parameters
        % eps_sigmoid = 1e-6;
        % sigmoid = @(x)(1-eps_sigmoid)./(1+exp(-x)) + eps_sigmoid;
        sigmoid = @(x)(1)./(1+exp(-x)); 

        umax = 1; % max of beta distribution (between 0 and 1)
        v_max = 0.5; % not actual variance of beta distribution (different way of calculating variance)
        u_vol = umax*sigmoid(parameters(1));
        u_sto = umax*sigmoid(parameters(2));
        v_vol = v_max*sigmoid(parameters(3));
        v_sto = v_max*sigmoid(parameters(4));
        
        tx = [u_vol, u_sto, v_vol, v_sto];

    % elseif do_transform == 2 % beta
    %     eps_sigmoid = eps;
    %     sigmoid = @(x)(1-eps_sigmoid)./(1+exp(-x)) + eps_sigmoid;    
    %     a_vol = 1+9*sigmoid(parameters(1));
    %     a_sto = 1+9*sigmoid(parameters(2));
    %     b_vol = 1+9*sigmoid(parameters(3));
    %     b_sto = 1+9*sigmoid(parameters(4));
    %     u_vol = a_vol./(a_vol + b_vol);    
    %     v_vol = u_vol / a_vol;
    %     u_sto = a_sto./(a_sto + b_sto);
    %     v_sto = u_sto./a_sto;
    %     tx = [u_vol, u_sto, v_vol, v_sto];
    else
        tx = parameters;
    end
    
    config_model = struct('num_quantiles', 300);
    
    [N, num_blocks] = size(observations);
    val = nan(N, num_blocks);
    vol = nan(N, num_blocks);
    sto = nan(N, num_blocks);
    for i=1:size(observations,2)
        [val(:, i), vol(:, i), sto(:, i), y_std(:, i), y_mean(:, i)] = model_per_block(tx, observations(:,i), config_model);
    end
    signals = struct('val', val, 'vol', vol, 'sto', sto, 'y_std', y_std, 'y_mean', y_mean);    
    
    % if nargout > 3
        delta  = observations - val; delta(end, :) = [];
        update = val(2:end, :) - val(1:end-1, :);
        signals.delta = delta;
        signals.update = update;
    
        % Initialize empty matrices for accumulating GLM data.
        y = [];        % Dependent variable: concatenated belief updates (learning signal).
        x = [];        % Independent variable: block-diagonal matrix of prediction errors.
        s_const = [];  % Constant term (ones) for each trial (used as an intercept in GLM).
        
        % Loop through each dimension to compile the data for GLM.
        for i = 1:size(delta, 2)
            % Concatenate the update values for dimension i into the response vector.
            y = [y; update(:, i)];
            
            % Build a block-diagonal matrix for delta (prediction error) for each dimension.
            x = blkdiag(x, delta(:, i));
            
            % Build a block-diagonal matrix of ones (acting as the constant term)
            % for each trial within the current dimension.
            s_const = blkdiag(s_const, ones(size(delta, 1), 1));
        end
        
        % Fit a Generalized Linear Model (GLM) to relate the prediction error (delta) to
        % the update in belief. Here, the combined matrix [x s_const] is used as the design
        % matrix, 'y' as the response variable, and a normal distribution is assumed.
        % The 'constant','off' option is set because the constant is already included in s_const.
        b = glmfit([x s_const], y, 'normal', 'constant', 'off');
        lr = b(1:4, :);
        block_effect = b(5:end, :);
    % end    

end

function [val, vol, sto, y_std, y_mean] = model_per_block(parameters, observations, config)

    N = length(observations); % number of time steps
    M = config.num_quantiles;  % number of quantiles

    u_vol = parameters(1); % mean
    u_sto = parameters(2);
    v_vol = parameters(3); % variance
    v_sto = parameters(4);

    a_vol = eps + u_vol / v_vol;
    b_vol = eps + (1 - u_vol) / v_vol;
    a_sto = eps + u_sto / v_sto;
    b_sto = eps + (1 - u_sto) / v_sto;
 
    quantiles = linspace(0.01, 0.99, M);

    v = eps + 0.5 * betainv(quantiles, a_vol, b_vol); 
    s = eps + 0.5 * betainv(quantiles, a_sto, b_sto);

    [v_, s_] = meshgrid(v, s);
    v = v_(:);
    s = s_(:);    

    weights = ones(M * M, 1) / (M * M);
    r = .5 * ones(M * M, 1);
    
    val = nan(N, 1);
    vol = nan(N, 1);
    sto = nan(N, 1);
    y_mean = nan(N, 1);
    y_std = nan(N, 1);

    for t=1:size(observations,1)
        val(t) = sum(r.*weights);           
    
        vol(t) = sum(v.*weights);
        sto(t) = sum(s.*weights); 
    
        if ~isnan(observations(t)) 
            y = bernoulli_predictive(observations(t),r, v, s);

            % Store mean and standard deviation of the predictive likelihood.
            y_mean(t) = mean(y, 'omitnan');
            y_std(t) = std(y, 'omitnan');          

            % numNaN = sum(isnan(y));
            % fprintf("Mean = %.3f, SD = %.3f (ignored %d NaNs)\n", y_mean(t), y_std(t), numNaN);
            
            % update weights
            weights = weights.*(y)+eps; % weights updated based on bernoulli likelihood
            weights = weights/sum(weights);
    
            [r] = hmm(observations(t),r, v, s);
        end        
    end    
end

function y = bernoulli_predictive(o, r, v, s)
    q = r .* (1 - v) + (1 - r) .* v;

    % Calculate the effective probability of observing 1.
    p = (1 - s) .* q + s .* (1 - q);
    
    % Compute the Bernoulli likelihood:
    %   If o == 1, likelihood is p; if o == 0, likelihood is (1-p).
    y = p.*o + (1 - p).*(1 - o);
end

function [r_new, delta, update]=hmm(o, r, v, s)
    q = r .* (1 - v) + (1 - r) .* v;
    
    % Extract the outcome for the current trial.
    
    % Assign the updated belief to the prior q.
    r_new = q;
    
    % If the outcome is valid (i.e., not NaN), update the belief.
    if ~isnan(o)
        % Calculate the prediction error: difference between the observed outcome and
        % the current prediction (belief).
        delta = o - r;
        
        % Compute the likelihood ('ell') of observing the outcome given the noise.
        % When o equals 1, ell is (1-s); when o equals 0, ell is s.
        ell = o .* (1 - s) + (1 - o) .* s;
        
        % Apply a Bayesian update rule to compute the new belief.
        % The numerator weights the prior q by the likelihood of the outcome,
        % while the denominator normalizes the value.
        r_new = ell .* q ./ (ell .* q + (1 - ell) .* (1 - q));
        
        % Record the magnitude of the update (i.e., the change in belief).
        update = r_new - r;
    end
end
