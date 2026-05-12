function [st_gender, st_age, st_rxec_count, st_rxec_pct] = demo2tbl(experiment)
% demo2tbl
% Return 4 demographic summary tables:
%   1) gender counts
%   2) age-bin counts
%   3) race x ethnicity counts
%   4) race x ethnicity percentages within ethnicity
%
% Outputs are structs with:
%   st.table.data
%   st.table.rows
%   st.table.columns

    if nargin < 1
        experiment = 'bird';
    end

    filter = true;
    exclude_criteria = 2;

    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);

    [~, ~, ~, demo_data] = get_data(experiment, filter, exclude_criteria);

    % ---------- Gender ----------
    gender = strings(numel(demo_data),1);
    for i = 1:numel(demo_data)
        gender(i) = string(demo_data{i}.('demo_gender-categorical'));
    end
    gender = strtrim(gender);
    gender = gender(strlength(gender) > 0);

    [g_labels, ~, gi] = unique(gender, 'stable');
    g_counts = accumarray(gi, 1);

    st_gender.table.data = g_counts;
    st_gender.table.rows = cellstr(g_labels);
    st_gender.table.columns = {'N'};

    % ---------- Age bins ----------
    ages = nan(numel(demo_data),1);
    for i = 1:numel(demo_data)
        ages(i) = demo_data{i}.demo_age;
    end
    ages = ages(~isnan(ages) & ages > 0);

    age_labels = ["1-18","19-35","36-50","51-64","65+"];
    age_counts = zeros(numel(age_labels),1);

    age_counts(1) = sum(ages >= 1  & ages <= 18);
    age_counts(2) = sum(ages >= 19 & ages <= 35);
    age_counts(3) = sum(ages >= 36 & ages <= 50);
    age_counts(4) = sum(ages >= 51 & ages <= 64);
    age_counts(5) = sum(ages >= 65);

    st_age.table.data = age_counts;
    st_age.table.rows = cellstr(age_labels');
    st_age.table.columns = {'N'};

    % ---------- Race x Ethnicity ----------
    ethnicity = strings(numel(demo_data),1);
    race = strings(numel(demo_data),1);

    for i = 1:numel(demo_data)
        ethnicity(i) = string(demo_data{i}.demo_ethnicity);
        race(i) = string(demo_data{i}.demo_race);
    end

    ethnicity = strtrim(ethnicity);
    race = strtrim(race);

    keep = strlength(ethnicity) > 0 & strlength(race) > 0;
    ethnicity = ethnicity(keep);
    race = race(keep);

    [eth_labels, ~, ei] = unique(ethnicity, 'stable');
    [race_labels, ~, ri] = unique(race, 'stable');

    C = accumarray([ei, ri], 1, [numel(eth_labels), numel(race_labels)]);
    C_pct = C ./ sum(C, 2);

    st_rxec_count.table.data = C;
    st_rxec_count.table.rows = cellstr(eth_labels);
    st_rxec_count.table.columns = matlab.lang.makeValidName(cellstr(race_labels));

    st_rxec_pct.table.data = C_pct;
    st_rxec_pct.table.rows = cellstr(eth_labels);
    st_rxec_pct.table.columns = matlab.lang.makeValidName(cellstr(race_labels));
end