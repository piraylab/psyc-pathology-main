function [data, surveys, metadata, demo_data] = get_data(experiment, filter, exclude_criteria)
    % get_data(experiment, [exclude_criteria], [filter])
    %
    % Mandatory:
    %   experiment       - a string (or char array) identifying the experiment.
    %
    % Optional (as parameter/value pairs or positional):
    %   exclude_criteria - numeric value (default: 2)
    %   filter           - logical flag (default: true)
    if nargin<1, experiment='sealion'; filter=true; exclude_criteria=2; end
    if nargin<2, filter=true; exclude_criteria=2; end
    if ~filter, exclude_criteria=Inf; end
    % (Optional) Display the resolved inputs for debugging.
    fprintf('Experiment: %s\n', experiment);
    fprintf('Exclude criteria: %g\n', exclude_criteria);
    fprintf('Filter: %s\n', string(filter));

%% Set up paths and directories
    % Determine the current folder (location of this file) and set the working directory
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);
    
    % Construct the filename for the experiment data
    fname = fullfile('..', '..', 'mat_data', 'experiment_2', sprintf('data_%s.mat', experiment));
    f = load(fname);
    N = length(f.trials_data);

    % Check if 'inattentive_data' exists in the loaded structure
    if isfield(f, 'inattentive_data')
        inattentive_data = f.inattentive_data;  % expected to be a cell array of structures
        % Assume 's' is your structure (for example, inattentive_data{i})
        s = inattentive_data{1};
        % Get all field names of the structure
        allFields = fieldnames(s);
        % Find field names that contain the substring 'command'
        commandFields = allFields(contains(allFields, 'command'));
        % Update valid_idx based on the 'total_inattentive' field and summed 'command' field
        valid_idx = false(N, 1);
        total_inatt = zeros(N, 1);
        total_command = zeros(N, 1);
        for i = 1:N
            % Check for the 'total_inattentive' field; use a default if not present.
            if isfield(inattentive_data{i}, 'total_inattentive')
                total_inatt(i) = inattentive_data{i}.total_inattentive;
            else
                error('Trial %d is missing the "total_inattentive" field.', i);
            end
            % Initialize the total command sum
            total_command(i) = 0;
            % Loop through the matching fields and sum their values
            for j = 1:length(commandFields)
                fieldValue = inattentive_data{i}.(commandFields{j});
                % Check if the field value is numeric before summing
                if isnumeric(fieldValue)
                    % Sum all elements in the field (in case it's an array)
                    total_command(i) = total_command(i) + fieldValue;
                else
                    warning('Field "%s" is not numeric and will be skipped.', commandFields{j});
                end
            end
            % Mark this trial as valid if total_inattent is below the criteria and if no command issues.
            valid_idx(i) = (total_inatt(i) <= exclude_criteria) && (total_command(i) == 0);        
            % valid_idx(i) = (total_inatt <= exclude_criteria);
        end
    else
        % If inattentive_data does not exist, consider all trials valid.
        valid_idx = true(N, 1);
        fprintf('This dataset is not filtered by survey inattentiveness. \n');
    end
    
    if filter==true
        % Calculate the number of inattentive subjects.
        numInattentive = sum(~valid_idx);
        % Print the number using fprintf.
        fprintf('Total inattentive subjects: %d \n', numInattentive);   
    else
        valid_idx = true(N, 1);
        fprintf('Subjects are not filtered\n');
    end

    % filter trial and survey data accordingly
    filtered_trials_data = f.trials_data(valid_idx);
    filtered_demo = f.demo_data(valid_idx);
    if isfield(f, 'survey_data') 
        filtered_survey_data = f.survey_data(valid_idx);
    else
        filtered_survey_data = {};
    end
    filtered_N = length(filtered_trials_data);
    
    % Define data output by iterating through filtered_trials_data
    data = cell(filtered_N,1);
    for i = 1:filtered_N
        if any(filtered_trials_data{i}.choice(:) == -1)
            % fprintf("choice contains -1 at index %d\n", i);
            % Replace -1 with NaN
            filtered_trials_data{i}.choice(filtered_trials_data{i}.choice == -1) = NaN;
        end
        data{i}.choice = filtered_trials_data{i}.choice;
        data{i}.outcome = filtered_trials_data{i}.outcome;
        data{i}.rand_order = filtered_trials_data{i}.randomization_order;
        data{i}.response_time = filtered_trials_data{i}.response_time;
        data{i}.workerId = filtered_trials_data{i}.workerId;
        data{i}.age = filtered_demo{i}.demo_age;
        data{i}.sex = filtered_demo{i}.demo_sex;
        data{i}.age_month = filtered_demo{i}.demo_age_month;
    end
    % Define surveys and metadata output
    surveys = filtered_survey_data;
    metadata = f.meta_data;
    timeseries_fname = fullfile('..', '..', 'mat_data', 'experiment_2', 'hidden_state.mat');
    timeseries_f = load(timeseries_fname);
    metadata.hidden_state = timeseries_f.timeseries.hidden_state;
    demo_data = filtered_demo;
    
    if isfield(metadata, 'total_command')
        metadata.total_command = total_command;
        metadata.total_inatt = total_inatt;
    end
end
