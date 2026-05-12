function fnames = getfileordered(subdir, pattern, indices)
% getfileordered  Generate a cell‐array of existing files matching a sprintf pattern.
%
%   fnames = GET_FILE_LIST(subdir, pattern, indices)
%     subdir   – path to the folder containing your files
%     pattern  – e.g. 'sub%04d.mat'
%     indices  – vector of integers, e.g. 1:N
%
%   fnames is an M×1 cell of full paths that exist, ordered by indices.

    fnames = {};
    for k = indices(:).'
        fname = sprintf(pattern, k);
        fullpath = fullfile(subdir, fname);
        if exist(fullpath, 'file') == 2
            fnames{end+1,1} = fullpath;
        else
            warning('File not found: %s', fullpath);
        end
    end
end