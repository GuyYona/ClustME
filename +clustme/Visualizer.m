function h = Visualizer(vis_data, plotType, options)
% Visualizer - Unified plotting interface for ClustME permutation results
%
%   Visualizer provides quick diagnostic plots from the vis_data struct
%   returned by ClustME. It can display the observed GLS t-map, the
%   cluster-forming threshold, and the empirical max-cluster null
%   distribution. The function is intended for rapid inspection and
%   troubleshooting.
%
% Syntax:
%   h = clustme.Visualizer(vis_data)
%   h = clustme.Visualizer(vis_data, plotType)
%   h = clustme.Visualizer(vis_data, plotType, Name, Value)
%
% Inputs:
%   vis_data - [struct] The exact vis_data struct exported by ClustME.
%   plotType - [char] The type of plot to generate: 'all', 'tmap', or 
%              'hist'. (Default: 'all')
%
% Options (Name-Value arguments):
%   .targetAx          [matlab.graphics.axis.Axes] Target axes for the T-map. (Default: [])
%   .nullHistAx        [matlab.graphics.axis.Axes] Target axes for the Histogram. (Default: [])
%   .t                 [1×T double] Override the time vector from vis_data. (Default: [])
%   .showClusterLevel  [logical] Overlay cluster-level data means on a right y-axis. (Default: true)
%   .showDirectionText [logical] Show directionality text (e.g., A > B). (Default: true)
%   .shadowAlpha       [double] Alpha transparency for the permutation envelope. (Default: 0.4)
%   .shadeClusters     [logical] Shade the background of significant clusters. (Default: false)
%
% Outputs:
%   h - [struct] Graphics handles for the generated plots with fields:
%       .tmap [struct] Handles for the T-map axes, lines, and patches.
%       .hist [struct] Handles for the histogram axes, lines, and legend.
%
% Plot types:
%   'tmap' - Plots the observed GLS t-statistic over time and overlays the
%            cluster-forming threshold envelope.
%   'hist' - Plots the empirical max-cluster null distribution and overlays
%            the observed candidate-cluster statistics with their cluster-level
%            p-values.
%   'all'  - Generates both plots.
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

arguments
    vis_data (1,1) struct
    plotType (1,:) char {mustBeMember(plotType,{'all','tmap','hist'})} = 'all'
    options.targetAx = []       % For Tmap
    options.nullHistAx = []     % For Histogram
    options.t double = []       % Override time vector
    options.showClusterLevel (1,1) logical = true   
    options.showDirectionText (1,1) logical = true
    options.shadowAlpha (1,1) double = 0.4
    options.shadeClusters (1,1) logical = false
end

h = struct();

% 1. Plot T-Map
if ismember(plotType, {'all', 'tmap'})
    % Use time from vis_data if not overridden
    if isempty(options.t), t_use = vis_data.t; else, t_use = options.t; end

h.tmap = plot_tmap(vis_data, ...
        'targetAx', options.targetAx, ...
        't', t_use, ...
        'showClusterLevel', options.showClusterLevel, ...
        'showDirectionText', options.showDirectionText, ...
        'shadowAlpha', options.shadowAlpha, ...
        'shadeClusters', options.shadeClusters);
end

% 2. Plot Null Histogram
if ismember(plotType, {'all', 'hist'})
    h.hist = plot_null_hist(vis_data.nullStats, ...
        vis_data.obsClusterMass, ...
        vis_data.pVals, ...
        nullHistAx = options.nullHistAx, alphaValue = vis_data.alpha);
end
end

function h = plot_tmap(stats, options)
% plot_tmap - Plots the observed t-map with threshold envelopes and optional cluster shading
%
% Internal Context:
%   A local helper function executed by the Visualizer wrapper when plotType 
%   is 'all' or 'tmap'. It renders the continuous Generalised Least Squares (GLS) 
%   t-statistic trace across the evaluated epoch. It provides visual inference 
%   support by overlaying the permutation threshold envelope, optionally shading 
%   suprathreshold candidate clusters, and plotting descriptive cluster-level means 
%   on a decoupled axis.
%
% Inputs:
%   stats             - [struct] High-density visualisation arrays. Must contain:
%                       .Tmap, .Fs, .coefName, .alpha. 
%                       Optionally: .tcrit, .tcritApprox, .sigMask, .cStarts, .cEnds, .clusterLevel.
%
% Options (Name-Value arguments):
%   .t                 [1 × T double] Optional temporal vector override.
%   .targetAx          [matlab.graphics.axis.Axes] Target axes for the plot.
%   .showTcrit         [logical] Toggle to overlay the threshold envelope. (Default: true)
%   .shadeSig          [logical] True to shade the raw significance mask; false to shade final clusters. (Default: true)
%   .showClusterLevel  [logical] Overlay cluster-level data means on a right y-axis. (Default: true)
%   .showDirectionText [logical] Annotate the polarity of the contrast. (Default: true)
%   .shadowAlpha       [double] Alpha transparency for the permutation envelope. (Default: 0.4)
%   .shadeClusters     [logical] Master toggle to enable background shading. (Default: false)
%
% Outputs:
%   h                 - [struct] Graphics handles for the generated plot components:
%       .ax             [matlab.graphics.axis.Axes] Handle to the parent axes.
%       .tLine          [matlab.graphics.chart.primitive.Line] Handle for the GLS trace.
%       .shadePatches   [1 × P matlab.graphics.primitive.Patch] Handles for shaded background regions.
%       .dataLines      [1 × L matlab.graphics.chart.primitive.Line] Handles for mean cluster levels.
%
% Algorithmic & Exactness Notes:
%   * Time Vector Validation: Explicitly verifies that the length of a user-provided 
%     temporal vector strictly matches the length of the evaluated Tmap, throwing a 
%     fatal error to prevent dimension mismatch crashes during polygon rendering.
%   * Boundary Extraction Fallback: If explicit temporal boundaries (cStarts/cEnds) 
%     are missing from the struct but a significance mask (sigMask) is present, it 
%     mathematically derives the contiguous cluster bounds on the fly via an internal 
%     mask_to_clusters call.
%   * Envelope Rendering Logic: Dynamically constructs a symmetric polygon array 
%     to accurately render the continuous, time-varying empirical null envelope (tcrit). 
%     If an approximate static threshold is provided instead (tcritApprox), it falls 
%     back to rendering a flat rectangular bound.
%   * Shading Geometry: Utilises patch objects to construct translucent bounding 
%     boxes over significant epochs. It differentiates between shading the raw, 
%     unfiltered significance mask versus the final, duration-filtered clusters 
%     based on the shadeSig toggle.
%   * Dual-Axis Decoupling: Decouples the inferential Generalised Least 
%     Squares scores (left y-axis) from the descriptive post-hoc cluster means 
%     (right y-axis) using the native yyaxis system to prevent scale distortion.
%   * Z-Order Management: Enforces a strict z-ordering (via uistack) at the end 
%     of execution to guarantee that primary statistical lines are never visually 
%     obscured by background patch objects.
%   * Directionality Guard: Explicitly checks if the tested coefficient is the 
%     'Intercept' and suppresses polarity annotations (A > B) to prevent 
%     illogical directional claims on one-sample tests.

arguments
    stats struct
    options.t double = []
    options.targetAx = []
    options.showTcrit (1,1) logical = true    % overlay ±tcrit(t) from permutation cutoffs
    options.shadeSig  (1,1) logical = true    % True to shade the suprathreshold mask; false to shade candidate clusters.
    options.showClusterLevel (1,1) logical = true   
    options.showDirectionText (1,1) logical = true
    options.shadowAlpha (1,1) double = 0.4
    options.shadeClusters (1,1) logical = false
end

% ---- time base ----
T = numel(stats.Tmap);
if isempty(options.t)
    tvec = (0:T-1)/stats.Fs;
else
    tvec = options.t(:).';
    if numel(tvec) ~= T
        error('plot_tmap:BadTimeVector','Length of t (%d) must match Tmap (%d).', numel(tvec), T);
    end
end

% ---- cluster bounds (prefer final clusters; fallback to mask) ----
cStarts = []; cEnds = [];
if isfield(stats,'cStarts') && ~isempty(stats.cStarts), cStarts = stats.cStarts(:); end
if isfield(stats,'cEnds')   && ~isempty(stats.cEnds),   cEnds   = stats.cEnds(:);   end
if (isempty(cStarts) || isempty(cEnds)) && isfield(stats,'sigMask') && any(stats.sigMask)
    [cStarts, cEnds] = mask_to_clusters(stats.sigMask);
end
K = numel(cStarts);


% ---- axes ----
if ~isempty(options.targetAx) && isgraphics(options.targetAx,'axes')
    ax = options.targetAx; axes(ax); %#ok<LAXES>
    hold(ax,'on');
else
    figure('Name','ClustME — Tmap + Cluster Data'); ax = axes; hold(ax,'on');
end
h = struct('ax',ax,'tLine',[],'shadePatches',gobjects(0),'dataLines',gobjects(0));

% ---- Left y: Tmap ----
if options.showClusterLevel
    yyaxis(ax,'left');
end
h.tLine = plot(ax, tvec, stats.Tmap(:), 'LineWidth', 1.25);
leg = {'GLS score (chosen coef)'};

if options.showTcrit
    if isfield(stats,'tcrit') && ~isempty(stats.tcrit)
        % Create a shaded polygon for the time-varying null envelope
        X = [tvec, fliplr(tvec)];
        Y = [stats.tcrit(:)', fliplr(-stats.tcrit(:)')];
        fill(ax, X, Y, [0.4 0.4 0.4], 'EdgeColor', 'none', 'FaceAlpha', options.shadowAlpha, 'HitTest', 'off');
        leg{end+1} = '|t| < t_{crit}(t) envelope';
    elseif isfield(stats,'tcritApprox') && ~isempty(stats.tcritApprox)
        % Create a rectangular shaded polygon for a static threshold
        X = [tvec(1), tvec(end), tvec(end), tvec(1)];
        Y = [stats.tcritApprox, stats.tcritApprox, -stats.tcritApprox, -stats.tcritApprox];
        fill(ax, X, Y, [0.4 0.4 0.4], 'EdgeColor', 'none', 'FaceAlpha', options.shadowAlpha, 'HitTest', 'off');
        leg{end+1} = sprintf('|t| < %.3g envelope', stats.tcritApprox);
    end
end


% shade p<alpha mask OR final clusters
if options.shadeClusters    
    shadeColor = [0.9 0.95 1.0];
    if options.shadeSig && isfield(stats,'sigMask') && any(stats.sigMask)
        yl = ylim(ax);
        idx     = find(stats.sigMask(:).');
        isStart = [true,           diff(idx) > 1];
        isEnd   = [diff(idx) > 1,  true       ];
        starts  = idx(isStart);
        ends    = idx(isEnd);
        for kk = 1:numel(starts)
            xs = tvec(starts(kk)); xe = tvec(ends(kk));
            h.shadePatches(end+1) = patch('Parent',ax, ...
                'XData',[xs xe xe xs], 'YData',[yl(1) yl(1) yl(2) yl(2)], ...
                'FaceColor',shadeColor,'FaceAlpha',0.25,'EdgeColor','none'); %#ok<AGROW>
        end
    elseif ~options.shadeSig && K>0
        yl = ylim(ax);
        for kk = 1:K
            xs = tvec(cStarts(kk)); xe = tvec(cEnds(kk));
            h.shadePatches(end+1) = patch('Parent',ax, ...
                'XData',[xs xe xe xs], 'YData',[yl(1) yl(1) yl(2) yl(2)], ...
                'FaceColor',shadeColor,'FaceAlpha',0.25,'EdgeColor','none'); %#ok<AGROW>
        end
    end
end

uistack(findobj(ax,'Type','line'),'top');
ylabel(ax, 'GLS score (perm-calibrated)');
grid(ax,'on');

yline(ax, 0, 'k-', 'Alpha', 0.3); 

if options.showDirectionText && ~strcmpi(stats.coefName, 'Intercept')
    yl = ylim(ax);
    % Annotate the positive and negative directions relative to the reference level
    text(ax, tvec(1), yl(2), sprintf('  %s > Ref', stats.coefName), ...
        'VerticalAlignment','top', 'Color',[0 0.5 0], 'FontSize', 8, 'BackgroundColor','w');
    text(ax, tvec(1), yl(1), sprintf('  Ref > %s', stats.coefName), ...
        'VerticalAlignment','bottom', 'Color',[0.5 0 0], 'FontSize', 8, 'BackgroundColor','w');
end

% ---- Right y: use provided per-cluster levels (solid lines) ----
if options.showClusterLevel
    yyaxis(ax,'right');
    ylabel(ax, 'Mean data across cluster');

    if K>0 && isfield(stats,'clusterLevel') && numel(stats.clusterLevel)==K
        for kk = 1:K
            xs = tvec(cStarts(kk)); xe = tvec(cEnds(kk));
            lvl = stats.clusterLevel(kk);
            h.dataLines(end+1) = line(ax, [xs xe], [lvl lvl], ...
                'LineWidth', 2, 'LineStyle','-'); %#ok<AGROW>
        end
    end
end

title(ax, sprintf('GLS score map + cluster data — %s (\\alpha=%.3g)', stats.coefName, stats.alpha));
legend(ax, leg, 'Location','best');
end

function h = plot_null_hist(nullStats, obsClusterMass, pVals, options)
% plot_null_hist - Generates an empirical null distribution histogram with observed overlays
%
% Internal Context:
%   A local helper function executed by the Visualizer wrapper when plotType 
%   is 'all' or 'hist'. It renders the empirical max-statistic null distribution 
%   generated during the ClustME permutation loop. By overlaying the observed 
%   cluster masses and the critical alpha boundary, it provides a direct visual 
%   inspection of how observed candidate-cluster statistics compare with
%   the empirical max-cluster null distribution.
%
% Inputs:
%   nullStats      - [B × 1 double] Max-statistic empirical null distribution.
%   obsClusterMass - [1 × K double] Array of observed cluster-mass statistics.
%   pVals          - [1 × K double] Cluster-level p-values from the max-cluster procedure.
%
% Options (Name-Value arguments):
%   .nullHistBins  [double] Number of bins for the histogram. (Default: 50)
%   .nullHistAx    [matlab.graphics.axis.Axes] Target axes for the plot. (Default: [])
%   .alphaValue    [double] Family-wise alpha level. (Default: 0.05)
%
% Outputs:
%   h              - [struct] Graphics handles for the generated plot components:
%       .ax          [matlab.graphics.axis.Axes] Handle to the parent axes.
%       .hist        [matlab.graphics.chart.primitive.Histogram] Handle to the histogram.
%       .obsLines    [1 × K matlab.graphics.chart.primitive.ConstantLine] Handles for observed mass lines.
%       .critLine    [matlab.graphics.chart.primitive.ConstantLine] Handle for the critical threshold line.
%       .legend      [matlab.graphics.illustration.Legend] Handle to the figure legend.
%
% Algorithmic & Exactness Notes:
%   * Non-Finite & Empty State Guards: Actively filters out non-finite (NaN/Inf) 
%     values prior to evaluation. If the filtered array is empty (e.g., total 
%     upstream solver failure), it intercepts execution, annotates the axes as 
%     empty, and safely returns to prevent downstream rendering crashes.
%   * Dynamic Colour Mapping: When plotting the K observed clusters, it implements 
%     safe modulo arithmetic against the default lines() colormap. This mathematically 
%     guarantees that line colours will safely loop without crashing if the number 
%     of surviving clusters exceeds the default colormap length.
%   * Legend Encapsulation: Strictly forces the 'HandleVisibility' property to 'on' 
%     specifically for the critical threshold and observed mass lines. This prevents 
%     the dynamically assembled legend from accidentally capturing or omitting 
%     background graphic objects.
%   * Threshold Extraction: Mathematically derives the discrete critical boundary 
%     (q_{1-\alpha}) directly from the finite sample null distribution via exact 
%     quantile evaluation, ensuring the plotted line perfectly matches the inferential 
%     FWER logic.
arguments
    nullStats
    obsClusterMass
    pVals

    options.nullHistBins (1,1) double = 50;
    options.nullHistAx = []
    options.alphaValue (1,1) double = 0.05;
end

% Prepare axes
if ~isempty(options.nullHistAx) && isgraphics(options.nullHistAx,'axes')
    ax = options.nullHistAx; axes(ax); %#ok<LAXES>
    hold(ax,'on');
else
    figure('Name','ClustME — Null vs Observed Cluster Mass');
    ax = axes; hold(ax,'on');
end
h = struct('ax',ax,'hist',[],'obsLines',gobjects(0),'critLine',gobjects(0),'legend',[]);

ns = nullStats(:);
ns = ns(isfinite(ns));  % guard against NaN/Inf
if isempty(ns)
    title(ax,'Null distribution empty'); grid(ax,'on'); return;
end

% Histogram of the null (max-statistic) distribution
h.hist = histogram(ax, ns, options.nullHistBins, 'EdgeColor','none'); %#ok<CPROP>
xlabel(ax, 'Cluster mass statistic (max over clusters per shuffle)');
ylabel(ax, 'Count');
grid(ax,'on');

% Critical quantile for reference (1 - alpha)
qcrit = quantile(ns, 1 - options.alphaValue);
h.critLine = xline(ax, qcrit, '--', sprintf('q_{1-\\alpha}=%.3g', qcrit), ...
    'LabelOrientation','horizontal', 'HandleVisibility','on');

% Plot each observed cluster mass with its p-value
K = numel(obsClusterMass);
cmap = lines(max(K,1));
L = strings(1, K+1);
L(1) = "null (max)";
for k = 1:K
    xk = obsClusterMass(k);
    pv = pVals(k);
    h.obsLines(k) = xline(ax, xk, '-', sprintf('obs %d (p=%.3g)', k, pv), ...
        'Color', cmap(1+mod(k-1,size(cmap,1)),:), ...
        'LabelOrientation','horizontal', 'HandleVisibility','on', ...
        'LineWidth', 1.5);
    L(k+1) = sprintf('obs %d', k);
end

title(ax, sprintf('Permutation null (N=%d shuffles) | \\alpha=%.3g', numel(ns), options.alphaValue));

% Build legend with null + all obs lines
h.legend = legend(ax, 'Location', 'best');

end