function results = select_best_lmm(responses, candidateFormulas, design, t, PreselectedCluster, varargin)
% select_best_lmm - Evaluate and rank candidate LME formulas using ClustME
%
%   select_best_lmm automatically evaluates a set of candidate Wilkinson 
%   formulas against a dataset. It executes the ClustME pipeline for each 
%   candidate and ranks the models based on the Akaike Information Criterion 
%   (AIC) of their underlying Static-V representations.
%
% Syntax:
%   results = clustme.select_best_lmm(responses, candidateFormulas, design, ...
%                                     t, PreselectedCluster, varargin)
%
% Inputs:
%   responses          - [N×T double] Continuous 2D data matrix (trials × time).
%   candidateFormulas  - [1×C cell] Cell array of Wilkinson strings to evaluate.
%   design             - [N×V table] Master design table.
%   t                  - [1×T double] Time vector corresponding to the data columns.
%   PreselectedCluster - [1×2 double] Predefined temporal region [tStart, tEnd] 
%                        used to anchor the Static-V evaluation.
%   varargin           - Name-Value pairs passed directly to the underlying 
%                        ClustME engine (e.g., 'numPerms', 'alphaValue').
%
% Outputs:
%   results - [1×C struct] Array of model evaluations sorted from best to worst 
%             (lowest Static AIC first). Contains fields:
%       .formula         [char] The evaluated Wilkinson string.
%       .StaticAIC       [double] AIC of the static LME (primary ranking metric).
%       .StaticVarRatios [double] Variance ratios of the static LME.
%       .clusters        [1×K struct] The standard ClustME clusters output.
%       .nClusters       [double] Total number of candidate clusters found.
%       .nSig            [double] Number of statistically significant clusters.
%       .pickedIdx       [double] Index of the most prominent cluster.
%       .pickedP         [double] Permutation p-value of the chosen cluster.
%       .promAvgAbs      [double] Average absolute grand-mean signal in the cluster.
%       .promPeakAbs     [double] Peak absolute grand-mean signal in the cluster.
%       .AIC             [1×K double] Post-hoc descriptive AICs for all clusters.
%       .AIC_selected    [double] Post-hoc descriptive AIC of the chosen cluster.
%       .varianceRatios  [1×K cell] Post-hoc descriptive variance ratios.
%
% BEHAVIOUR & ESTIMATION LOGIC
% ----------------------------
% 1) Time Anchoring: The function automatically extracts the midpoint of the 
%    provided `PreselectedCluster` and passes it to ClustME as the `TimeAnchor`.
% 2) ML Enforcement: To ensure that AIC comparisons across formulas with 
%    potentially varying fixed effects remain mathematically valid, this wrapper 
%    strictly forces `FitMethod = 'ML'` internally, overriding default REML.
% 3) Ranking: Models are sorted primarily by their `StaticAIC`. If ties occur, 
%    they are broken using the maximum signal prominence (`promAvgAbs`).
%
% Dependencies:
%   - MATLAB R2019b or newer.
%   - Statistics and Machine Learning Toolbox.
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


    % pull alphaValue if present, default 0.05
    alphaValue = 0.05;
    for a = 1:2:numel(varargin)
        if strcmpi(varargin{a}, 'alphaValue')
            alphaValue = varargin{a+1};
        end
    end

    % Grand-mean signal across all trials
    grandMean = mean(responses, 1, 'omitnan');     % 1 × T

    % --- Required inputs validation (t, PreselectedCluster) ---
    T = size(responses, 2);

    if ~isvector(t) || numel(t) ~= T
        error('select_best_lmm:BadTimeVector', ...
              'Length of t (%d) must match the time dimension T (%d).', numel(t), T);
    end
    if isempty(PreselectedCluster) || size(PreselectedCluster,2) ~= 2
        error('select_best_lmm:BadPreselectedCluster', ...
              'PreselectedCluster must be K×2 [tStart tEnd] in the same units as t.');
    end

    if ~istable(design)
        error('select_best_lmm:InvalidDesign', 'The "design" argument must be a table.');
    end

    % ------------------------------------------------------------
    % Define TimeAnchor from the middle of the pre-selected cluster
    % ------------------------------------------------------------
    timeAnchorMid = (PreselectedCluster(1,1) + PreselectedCluster(1,2)) / 2;

    % ------------------------------------------------------------
    nC = numel(candidateFormulas);
    emptyRes = struct('formula',[],'clusters',[],'pickedIdx',NaN, ...
                      'AIC',[], ...
                      'AIC_selected',NaN, ...
                      'varianceRatios',{{}}, ...
                      'StaticAIC',NaN, ...
                      'StaticVarRatios',[], ...
                      'pickedP',NaN,'promAvgAbs',NaN,'promPeakAbs',NaN, ...
                      'nClusters',0, 'nSig',0);
    results = repmat(emptyRes, 1, nC);

    for i = 1:nC
        fml = candidateFormulas{i};
        
        % Use ML estimation so AIC comparisons across different fixed-effect 
        % specifications are comparable.
        [clusters, mStats, ~] = ClustME(responses, design, fml, ...
            't', t, 'PreselectedCluster', PreselectedCluster, ...
            'TimeAnchor', timeAnchorMid, ...     
            'FitMethod', 'ML', ...
            varargin{:});

        % Store results
        results(i).formula         = fml;
        results(i).clusters        = clusters;
        results(i).StaticAIC       = mStats.AIC;
        results(i).StaticVarRatios = mStats.VarRatios;

        % Print AIC and variance ratios
        if isempty(mStats.VarRatios)
            vrStr = '[]';
        else
            vrStr = mat2str(mStats.VarRatios(:).', 3);
        end
        fprintf('Model %d: %s\n  Static AIC = %.3f\n  Static VarRatios = %s\n', ...
                i, fml, mStats.AIC, vrStr);

        % Select the most prominent significant cluster, falling back to the most prominent candidate cluster
        K = numel(clusters);
        results(i).nClusters = K;
        if K == 0, continue; end

        promAvgAbs  = nan(1,K);
        promPeakAbs = nan(1,K);
        for k = 1:K
            cols = clusters(k).start : clusters(k).end;
            promAvgAbs(k)  = mean(abs(grandMean(cols)), 'omitnan');
            promPeakAbs(k) = max( abs(grandMean(cols)) );
        end

        p   = [clusters.p_value];
        sig = p < alphaValue;
        cand = promAvgAbs;
        if any(sig), cand(~sig) = -inf; end
        
        [~, idx] = max(cand);
        if ~isfinite(cand(idx))
            [~, idx] = max(promAvgAbs);
        end

        results(i).pickedIdx   = idx;
        results(i).pickedP     = clusters(idx).p_value;
        results(i).promAvgAbs  = promAvgAbs(idx);
        results(i).promPeakAbs = promPeakAbs(idx);

        % Collect cluster-level AIC & VarRatios (from clusters struct)
        if isfield(clusters, 'AIC') && ~isempty([clusters.AIC])
            results(i).AIC          = [clusters.AIC];
            results(i).AIC_selected = clusters(idx).AIC;
        end
        if isfield(clusters, 'varianceRatios')
            results(i).varianceRatios = arrayfun(@(c) c.varianceRatios, clusters, 'UniformOutput', false);
        end
        results(i).nSig = sum(sig);
    end

    % Ranking: lower Static-V AIC is preferred; signal prominence is used only as a tie-breaker
    F = [results.StaticAIC].';  F(isnan(F)) = inf;
    P = [results.promAvgAbs].'; P(isnan(P)) = -inf;
    [~, ord] = sortrows([F, -P], [1 2]);
    results = results(ord);
end