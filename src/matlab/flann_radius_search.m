%Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
%Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
%
%THE BSD LICENSE
%
%Redistribution and use in source and binary forms, with or without
%modification, are permitted provided that the following conditions
%are met:
%
%1. Redistributions of source code must retain the above copyright
%   notice, this list of conditions and the following disclaimer.
%2. Redistributions in binary form must reproduce the above copyright
%   notice, this list of conditions and the following disclaimer in the
%   documentation and/or other materials provided with the distribution.
%
%THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
%IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
%OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
%IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
%INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
%NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
%DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
%THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
%THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function [indices, dists] = flann_radius_search(data, testset, radius, search_params)
%RN_SEARCH  Perform radius search
%
% Performs a radius search using an
% index constructed using flann_build_index or directly a 
% dataset.

% Marius Muja, January 2008
% March 2013, modified from flann_search.m


    algos = struct( 'linear', 0, 'kdtree', 1, 'kmeans', 2, 'composite', 3, 'kdtree_single', 4, 'hierarchical', 5, 'lsh', 6, 'saved', 254, 'autotuned', 255 );
    center_algos = struct('random', 0, 'gonzales', 1, 'kmeanspp', 2 );
    log_levels = struct('none', 0, 'fatal', 1, 'error', 2, 'warning', 3, 'info', 4);
    function value = id2value(map, id)
        fields = fieldnames(map);
        for i = 1:length(fields),
            val = cell2mat(fields(i));
            if map.(val) == id
                value = val;
                break;
            end
        end
    end
    function id = value2id(map,value)
        id = map.(value);
    end

    default_params = struct('algorithm', 'kdtree' ,'checks', 32, 'eps', 0.0, 'sorted', 1, 'max_neighbors', -1, 'cores', 1, 'trees', 4, 'leaf_max_size', 15, 'branching', 32, 'iterations', 5, 'centers_init', 'random', 'cb_index', 0.4, 'target_precision', 0.9,'build_weight', 0.01, 'memory_weight', 0, 'sample_fraction', 0.1, 'log_level', 'warning', 'random_seed', 0);

    if ~isstruct(search_params)
        error('The "search_params" argument must be a structure');
    end

    params = default_params;
    fn = fieldnames(search_params);
    for i = [1:length(fn)],
        name = cell2mat(fn(i));
        params.(name) = search_params.(name);
    end
    if ~isnumeric(params.algorithm),
        params.algorithm = value2id(algos,params.algorithm);
    end
    if ~isnumeric(params.centers_init),
        params.centers_init = value2id(center_algos,params.centers_init);
    end
    if ~isnumeric(params.log_level),
        params.log_level = value2id(log_levels,params.log_level);
    end

    if (size(data,1)==1 && size(data,2)==1)
        % we already have an index
        [indices,dists] = nearest_neighbors('index_find_rn', data, testset, radius, params);
    else
        % create the index and search
        [indices,dists] = nearest_neighbors('find_rn', data, testset, radius, params);
    end
end
