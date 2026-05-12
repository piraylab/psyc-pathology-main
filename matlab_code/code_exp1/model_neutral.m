function model_neutral(experiment)
    if nargin<1, experiment = 'bird'; end
    filter = true;
    exclude_criteria = 2;
    
    % set path
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);
    % define directory
    fdir = fullfile('..', '..', 'mat_data', 'experiment_1');
    % load data 
    [data, ~, ~] = get_data(experiment, filter, exclude_criteria);
    
    fname = fullfile(fdir, sprintf('%s_%s.mat', mfilename, experiment));
    if ~exist(fname,'file')
        [lr, block_effect, workerIds] = glm_analysis(data);
        save(fname, 'lr', 'block_effect', 'workerIds'); 
    end

end

function [lr_coeff, block_effect, workerIds] = glm_analysis(data)
workerIds = cell(length(data), 1);
b = nan(length(data), 8);
for n=1:length(data)
    delta_all = []; update_all = [];block_all = [];
    for j=1:4
        bucket = data{n}.bucket(:,j);
        bag = data{n}.bag(:,j);
        
        update = bucket(2:end)-bucket(1:end-1); % bucket(t) is an index of m(t-1) after the update
        delta = bag - bucket;
        delta = delta(1:end-1);

        update_all = [update_all; update]; %#ok<AGROW> 
        delta_all = blkdiag(delta_all, delta);        
        block_all = blkdiag(block_all,ones(size(update)));
    end
    b(n,:) = glmfit([delta_all block_all],update_all,'normal', 'constant', 'off');
    workerIds{n} = data{n}.workerId;
end
lr_coeff = b(:, 1:4);
block_effect = b(:, 5:end);
end