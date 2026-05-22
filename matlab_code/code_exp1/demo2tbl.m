function [st_gender, st_age, st_rxec_count, st_rxec_pct] = demo2tbl(experiment)
% demo2tbl
% Returns 4 demographic summary tables:
%   1) gender counts
%   2) age-bin counts
%   3) race x ethnicity counts
%   4) race x ethnicity percentages within ethnicity
%
% Final conventions:
%   - blank / missing / "Rather not say" -> "Unknown"
%   - age <= 18 -> "Unknown"
currentFolder = fileparts(mfilename('fullpath'));
cd(currentFolder)

if nargin < 1 || isempty(experiment)
    experiment = 'binary';
end


if contains(experiment, 'bird')
    [data_bird, ~, ~, demo_data] = get_data('bird', true, 2);
else contains(experiment, 'binary')
    % Binary aligned sample: use aligned data only for workerIds
    currentFolder = fileparts(mfilename('fullpath'));

    % Load aligned dataset from code_binary to get workerIds
    cd(fullfile(currentFolder, '..', 'code_exp2'));
    [data1_aligned, ~, ~, demo_data1] = get_data('sealion_aligned', true, 2);
    [data2_aligned, ~, ~, demo_data2] = get_data('turtle_aligned', true, 2);
    ids1 = string(cellfun(@(x) x.workerId, data1_aligned, 'UniformOutput', false));
    ids2 = string(cellfun(@(x) x.workerId, data2_aligned, 'UniformOutput', false));
     
    % common IDs, in the order they appear in sealion
    [common_ids, idx1, idx2] = intersect(ids1, ids2, 'stable');
    % matched subsets
    demo_data = demo_data1(idx1);
end

%% Sex
sex_levels = ["Female","Male","Other","Unknown"];
sex_counts = zeros(numel(sex_levels),1);

for i = 1:numel(demo_data)
    x = norm_text(get_field(demo_data{i}, 'demo_gender-categorical'));

    if contains(x, "female")
        lab = "Female";
    elseif contains(x, "male")
        lab = "Male";
    elseif contains(x, "other")
        lab = "Other";
    else
        lab = "Unknown";
    end

    sex_counts(sex_levels == lab) = sex_counts(sex_levels == lab) + 1;
end

st_gender.table.data = sex_counts;
st_gender.table.rows = cellstr(sex_levels');
st_gender.table.columns = {'N'};

%% Age
age_levels = ["19-35","36-50","51-64","65+","Unknown"];
age_counts = zeros(numel(age_levels),1);

for i = 1:numel(demo_data)
    a = get_num(demo_data{i}, 'demo_age');

    if isnan(a) || a <= 18
        k = 5;
    elseif a <= 35
        k = 1;
    elseif a <= 50
        k = 2;
    elseif a <= 64
        k = 3;
    else
        k = 4;
    end

    age_counts(k) = age_counts(k) + 1;
end

st_age.table.data = age_counts;
st_age.table.rows = cellstr(age_levels');
st_age.table.columns = {'N'};

%% Ethnicity x Race
eth_levels = ["Not Hispanic or Latino","Hispanic or Latino","Unknown"];
race_levels = [ ...
    "White", ...
    "Black or African American", ...
    "Asian", ...
    "American Indian or Alaska Native", ...
    "Native Hawaiian or Other Pacific Islander", ...
    "Other", ...
    "Unknown"];

C = zeros(numel(eth_levels), numel(race_levels));

for i = 1:numel(demo_data)
    eth = norm_text(get_field(demo_data{i}, 'demo_ethnicity'));
    race = norm_text(get_field(demo_data{i}, 'demo_race'));

    % ethnicity
    if contains(eth, "not hispanic")
        eth_lab = "Not Hispanic or Latino";
    elseif contains(eth, "hispanic")
        eth_lab = "Hispanic or Latino";
    else
        eth_lab = "Unknown";
    end

    % race
    if contains(race, "white")
        race_lab = "White";
    elseif contains(race, "black") || contains(race, "african")
        race_lab = "Black or African American";
    elseif contains(race, "asian")
        race_lab = "Asian";
    elseif contains(race, "american indian") || contains(race, "alaska native")
        race_lab = "American Indian or Alaska Native";
    elseif contains(race, "native hawaiian") || contains(race, "pacific islander")
        race_lab = "Native Hawaiian or Other Pacific Islander";
    elseif contains(race, "other")
        race_lab = "Other";
    else
        race_lab = "Unknown";
    end

    r = find(eth_levels == eth_lab, 1);
    c = find(race_levels == race_lab, 1);
    C(r,c) = C(r,c) + 1;
end

C_pct = C ./ sum(C,2);

st_rxec_count.table.data = C;
st_rxec_count.table.rows = cellstr(eth_levels');
st_rxec_count.table.columns = matlab.lang.makeValidName(cellstr(race_levels));

st_rxec_pct.table.data = C_pct;
st_rxec_pct.table.rows = cellstr(eth_levels');
st_rxec_pct.table.columns = matlab.lang.makeValidName(cellstr(race_levels));
end

function x = get_field(s, fname)
if isfield(s, fname)
    x = s.(fname);
else
    x = "";
end
end

function x = get_num(s, fname)
if isfield(s, fname) && isnumeric(s.(fname)) && isscalar(s.(fname))
    x = s.(fname);
else
    x = NaN;
end
end

function x = norm_text(x)
if isstring(x)
    x = strjoin(x, ' ');
elseif ischar(x)
    x = string(x);
elseif iscell(x)
    x = strjoin(string(x), ' ');
else
    x = "";
end

x = lower(strtrim(x));

if strlength(x) == 0 || contains(x, "rather not say") || contains(x, "unknown")
    x = "unknown";
end
end