function v = getfield_def(s, name, default)
% Helper: get struct field with default
    if isfield(s, name)
        v = s.(name);
    else
        v = default;
    end
end
