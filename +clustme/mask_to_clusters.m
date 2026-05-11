function [starts, ends_] = mask_to_clusters(mask)
% mask_to_clusters - Convert a logical time-series mask into contiguous cluster boundaries
%
%   mask_to_clusters scans a logical array to identify independent runs of 
%   true values. It returns the exact start and end indices of each 
%   contiguous cluster.
%
% Syntax:
%   [starts, ends_] = clustme.mask_to_clusters(mask)
%
% Inputs:
%   mask   - [logical] A binary array over time (any shape). True (1) 
%            entries designate the presence of a candidate sample.
%
% Outputs:
%   starts - [K×1 double] 1-based start indices for the K detected clusters.
%   ends_  - [K×1 double] 1-based end indices for the K detected clusters.
%
% BEHAVIOUR
% ---------
% • The function internally flattens the input mask to a column vector.
% • A "cluster" is defined as any contiguous, unbroken run of true values.
% • If the mask is entirely false (no clusters), the function gracefully 
%   returns empty arrays to prevent downstream indexing errors.
%
% Dependencies:
%   - MATLAB R2019b or newer.
%   - Part of the ClustME Package (+clustme).
%
% Reference:
%   Yona, G., & Magill, P. J. Fast Cluster-Based Permutation Testing with 
%   Linear Mixed-Effects Models.
%
% Version: 1.0.0
% Last Modified: 29 April 2026
%
% Copyright (C) 2026 University of Oxford
% Author: Guy Yona
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.

    idx = find(mask(:));
    if isempty(idx)
        starts = [];
        ends_  = [];
        return;
    end

    isStart = [true;           diff(idx) > 1];
    isEnd   = [diff(idx) > 1;  true      ];

    starts = idx(isStart);
    ends_  = idx(isEnd);
end