% --- Helper: save current figure if SavePath is provided ------------------
function maybe_save_figure(figHandle, savePath, baseName)
% maybe_save_figure  Save a figure as PNG (300 dpi) if a path is specified.
%
%   maybe_save_figure(gcf, './figs', 'my_plot');
%
% Creates folder if needed. If savePath is empty ([] or ""), does nothing.

if isempty(savePath)
    return;
end
savePath = char(savePath);     % ensure char for fullfile
if ~exist(savePath, 'dir')
    mkdir(savePath);
end
fname = fullfile(savePath, [baseName, '.png']);
exportgraphics(figHandle, fname, 'Resolution', 300);
end
