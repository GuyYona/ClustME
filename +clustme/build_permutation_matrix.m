function permMatrix = build_permutation_matrix(nTrials, groupingIdx, maxPerms, method)
% build_permutation_matrix - Generate a pre-computed matrix of null randomisation instructions
%
%   build_permutation_matrix constructs the full set of randomisation
%   instructions (signs or indices) required for the resampling loop. 
%   Pre-computing the matrix provides a single reusable set of randomisation 
%   instructions for threshold estimation and max-cluster inference. 
%   Compact integer types reduce memory use, but very large B × N matrices 
%   can still be memory-limiting.
%
% Syntax:
%   permMatrix = clustme.build_permutation_matrix(nTrials, groupingIdx, maxPerms, method)
%
% Inputs:
%   nTrials     - [1×1 double] Total number of trials (N).
%   groupingIdx - [N×1 double] Vector of exchangeability block IDs (e.g., Subject IDs).
%                 Leave empty ([]) for unconstrained global row shuffling, corresponding to trial-level exchangeability
%   maxPerms    - [1×1 double] the requested upper bound of permutations (B). 
%                 If the exact non-identity randomisation space is smaller, exhaustive 
%                 enumeration is returned and the output may contain fewer rows.
%   method      - [char] Null generation scheme. (Default: 'signFlip')
%
% Outputs:
%   permMatrix  - [B×N int8] for 'signFlip'.
%                 [B×N uint32] for 'withinSubject'.
%                 [B×nUnits uint32] for 'groupLabel'.
%                 The pre-computed randomisation instructions.
%
% RANDOMISATION BEHAVIOUR (method)
% --------------------------------
% • 'signFlip' (Default): Returns a B × nTrials matrix of ±1 (int8).
%   Applies a Rademacher multiplier. If groupingIdx is provided, one 
%   Rademacher multiplier is generated per exchangeability block and expanded 
%   to all rows in that block. In ClustME.m, this same instruction matrix 
%   is used by the wild-bootstrap residual pathway after wildBootstrap is aliased to signFlip
% • 'withinSubject': Returns a B × nTrials matrix of trial indices 1..N (uint32).
%   Performs strict WITHIN-BLOCK shuffling of trials. Automatically falls back 
%   to Monte Carlo sampling if the design is unbalanced.
% • 'groupLabel': Returns a B × nUnits matrix of unit indices 1..nUnits (uint32).
%   Performs BETWEEN-BLOCK shuffling of unit identities. Valid only for balanced
%   between-subject designs under homoscedasticity.
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
% Last Modified: 7 May 2026
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

if nargin < 4, method = 'signFlip'; end

% 1. Determine unit structure
if isempty(groupingIdx)
    nUnits = nTrials;
else
    % Force contiguous indexing (1..N) to match ClustME storage
    [~, ~, groupingIdx] = unique(groupingIdx);
    nUnits = max(groupingIdx);
end

% Check whether unequal block sizes in a within-subject design require 
% Monte Carlo shuffling or exhaustive enumeration.
forceMonteCarlo = false;
if strcmp(method, 'withinSubject') && ~isempty(groupingIdx)
    uIds = unique(groupingIdx);
    firstSize = sum(groupingIdx == uIds(1));
    for k = 2:numel(uIds)
        if sum(groupingIdx == uIds(k)) ~= firstSize
            forceMonteCarlo = true;
            fprintf('NOTE: Unbalanced design detected. Using Monte Carlo shuffle instead of exhaustive enumeration.\n');
            break;
        end
    end
end

% 2. Check for exhaustive enumeration
if strcmp(method, 'signFlip') && nUnits > 0 && nUnits <= 20
    % Exhaustive sign-flip enumeration collapses global sign complements and 
    % excludes the identity, giving 2^(nUnits-1)-1 non-trivial sign patterns.
    maxNonTriv = 2^(nUnits - 1) - 1;
elseif strcmp(method, 'groupLabel') && nUnits > 0 && nUnits <= 8 && ~forceMonteCarlo
    maxNonTriv = factorial(nUnits);
elseif strcmp(method, 'withinSubject') && ~forceMonteCarlo
    if isempty(groupingIdx)
        % Global unconstrained shuffle: N!
        maxNonTriv = factorial(nTrials);
        if nTrials > 12; maxNonTriv = inf; end % Guard against Inf
    else
        % Blocked within-subject shuffle: (K!)^S
        trialsPerUnit = nTrials / nUnits;
        maxNonTriv = factorial(trialsPerUnit)^nUnits;
    end
else
    maxNonTriv = inf;
end

% 3. Generate Matrix
if maxNonTriv <= maxPerms
    % --- Case A: Exhaustive Enumeration (Exact) ---
    B = maxNonTriv;

    if strcmp(method, 'signFlip')
        % Standard Sign Flipping (2^(N-1)-1)
        permMatrix = zeros(B, nTrials, 'int8');
        allKs = 1:(2^(nUnits - 1) - 1);

        for j = 1:B
            k    = allKs(j);
            bits = bitget(k, 1:nUnits);
            w    = int8(bits(:)) * 2 - 1;

            if isempty(groupingIdx)
                permMatrix(j,:) = w.';
            else
                permMatrix(j,:) = w(groupingIdx).';
            end
        end

    elseif strcmp(method, 'groupLabel')
        % Exhaustive Between-Subject Permutation (N!)
        unitPerms = perms(1:nUnits); 
        
        % Exclude the identity permutation because the observed 
        % data are accounted for by the +1 finite-sample p-value correction.
        isIdentity = all(unitPerms == 1:nUnits, 2);
        unitPerms(isIdentity, :) = [];
        B = size(unitPerms, 1);
        
        % For between-subject, return the unit-level permutations directly.
        permMatrix = uint32(unitPerms);
        
    elseif strcmp(method, 'withinSubject')
        if isempty(groupingIdx)
            % Exhaustive Global Shuffling (N!)
            unitPerms = perms(1:nTrials);
            isIdentity = all(unitPerms == 1:nTrials, 2);
            unitPerms(isIdentity, :) = [];
            
            permMatrix = uint32(unitPerms);
        else
            % Exhaustive Within-Subject Permutation ((K!)^S)
            trialsPerUnit = nTrials / nUnits;
            basePerms = perms(1:trialsPerUnit); 
            nBase = size(basePerms, 1);
            
            % Generate Cartesian product of block permutations using ndgrid
            args = repmat({1:nBase}, 1, nUnits);
            grid = cell(1, nUnits);
            [grid{:}] = ndgrid(args{:});
            combIdx = reshape(cat(nUnits+1, grid{:}), [], nUnits);
            
            % Locate and exclude the identity combination across all blocks
            baseIdentityIdx = find(all(basePerms == 1:trialsPerUnit, 2));
            isIdentityComb  = all(combIdx == baseIdentityIdx, 2);
            combIdx(isIdentityComb, :) = [];
            
            B = size(combIdx, 1);
            permMatrix = zeros(B, nTrials, 'uint32');
            
            uIds = unique(groupingIdx);
            blocks = cell(nUnits, 1);
            for k = 1:nUnits
                blocks{k} = find(groupingIdx == uIds(k));
            end
            
            % Map the specific combinations back to the global trial row
            for j = 1:B
                idxRow = zeros(1, nTrials);
                for k = 1:nUnits
                    pVec = basePerms(combIdx(j, k), :);
                    destSlots = blocks{k};
                    idxRow(destSlots) = destSlots(pVec);
                end
                permMatrix(j,:) = idxRow;
            end
        end
    end

else
    % --- Case B: Random Sampling (Monte Carlo) ---
    B = maxPerms;

    if strcmp(method, 'signFlip')
        % --- Sign Flipping / Wild bootstrap ---
        permMatrix = zeros(B, nTrials, 'int8');
        for sh = 1:B
            if isempty(groupingIdx)
                w = randi([0,1], nTrials, 1) * 2 - 1;
                permMatrix(sh,:) = w.';
            else
                uSigns = randi([0,1], nUnits, 1) * 2 - 1;
                permMatrix(sh,:) = uSigns(groupingIdx).';
            end
        end

    elseif strcmp(method, 'withinSubject')
        % --- Within-Block Shuffling  ---
        % Shuffle trial indices independently within each grouping unit.
        % This preserves the block structure (Subject 1's data stays with Subject 1)
        % but randomises the condition labels via the residuals.

        permMatrix = zeros(B, nTrials, 'uint32');

        % Identify blocks once
        if isempty(groupingIdx)
            % Global shuffle
            for sh = 1:B
                permMatrix(sh,:) = randperm(nTrials);
            end
        else
            % Pre-calculate block indices for speed
            uIds = unique(groupingIdx);
            blocks = cell(numel(uIds), 1);
            for k = 1:numel(uIds)
                blocks{k} = find(groupingIdx == uIds(k));
            end

            for sh = 1:B
                pIndices = zeros(1, nTrials);
                for k = 1:numel(uIds)
                    idx = blocks{k};
                    % Shuffle indices strictly within this block
                    pIndices(idx) = idx(randperm(numel(idx)));
                end
                permMatrix(sh,:) = pIndices;
            end
        end
    elseif strcmp(method, 'groupLabel')
        % --- Group label shuffling for exchangeable between-subject units ---
        % Returns B x nUnits matrix of indices (1..nUnits).

        permMatrix = zeros(B, nUnits, 'uint32');
        for sh = 1:B
            permMatrix(sh,:) = randperm(nUnits);
        end
    else
        error('ClustME:UnknownMethod', 'Permutation method "%s" not recognized.', method);
    end
end

end