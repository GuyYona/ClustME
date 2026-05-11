% run_sensitivity_validation - Sensitivity & Power Analysis Benchmark
%
%   This script serves as a scientific validation benchmark to 
%   characterize the signal recovery limits (Power vs. SNR) of the ClustME engine. 
%   This script generates empirical manuscript data to demonstrate the Differential 
%   Sensitivity of adaptiveLocal covariance modeling compared to Naive OLS in 
%   non-stationary noise.
%
% How to Run:
%   run_sensitivity_validation()
%
% Validation Targets:
%   1. Signal Recovery Limits: Calculate the Probability of Detection (PoD) 
%      across varying Signal-to-Noise Ratios (SNR: 0.6 to 1.6).
%   2. Algorithmic Advantage: Compare the statistical power of the GLS 
%      Adaptive approach against Naive OLS (V=I) and fixed-window Holm-Bonferroni baselines.
%
% Outputs:
%   - Terminal: Probability of Detection (PoD) metrics.
%   - Figures: Figure 2 (Panel F: Power Curves, Panel G: Signal Visualizer).
%   - Data: Exports empirical simulation arrays to 'Figure2-FG_data.mat'.
%
% Dependencies:
%   - MATLAB R2019b or newer.
%   - validation_settings.json (for data export and load flags).
%   - Part of the ClustME Validation Suite.
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

clear; clc; close all;

p = gcp('nocreate');
if isempty(p)
    parpool;  % or parpool('threads') / parpool('local', N)
end

%% 1. Configuration

% --- Load Export Settings ---
settingsFile = 'validation_settings.json';
if ~isfile(settingsFile)
    error('Settings file "%s" not found. Please create it in the script directory.', settingsFile);
end
settings = jsondecode(fileread(settingsFile));

loadData = settings.loadSavedData;
exportDir = settings.exportDir;

% --- EXECUTION CONTROL ---

switch settings.runMode
    case 'debug'
        nIterations = 10;  
        fprintf('Mode: DEBUG (Fast mechanical test)\n');
    case 'fast'
        nIterations = 200;  
        fprintf('Mode: FAST (partial run for sanity check)\n');
    case 'publication'
        nIterations = 1000;  
        fprintf('Mode: PUBLICATION (Recreating full manuscript run)\n');
    otherwise
        error('Unknown runMode "%s". Use "debug","fast", or "publication"', settings.runMode);
end

SNR_levels  = [0.6 0.8 1 1.2 1.4 1.6];

alphaLevel  = 0.05;

% Define Export Path
baseFileName = 'Figure2-FG';
dataFilePath = fullfile(exportDir, [baseFileName '_data.mat']);

% --- Simulation Parameters ---
config.nSubj       = 25;            % Sample size (One-Sample)
config.nTrials     = 40;            % Trials per subject
config.TrialCountCV        = 0.8;
config.MinTrialsPerSubject = 5;
config.SubjectVar  = 2.0;
config.SubjectNoiseCV = 0.0;
config.TrialNoiseDriftMeanStep = 0.00;
config.TrialNoiseDriftStepSD   = 0.00;
config.design      = 'one-sample';
config.noiseMode   = 'complex';     % 1/f Noise + Non-stationary variance
config.RampStrength= 2.0;           % Variance ramp (end/start ratio)
config.Fs          = 100;
config.signalWidth = 0.10;          % 100ms FWHM
config.minClusterSize = 0;
config.eventTime    = 0.0;  
config.AnchorOffset = 0.00;
config.jitterRange  = [-0.1 0.1];  % signal latency
config.TimeRange    = [-0.5 1.0];  

% --- Post-event regime: constant higher noise + sparse short bursts (trial-specific) ---
config.PostEventNoiseScale  = 1.0;   % SD multiplier for t >= eventTime
config.PostEventBurstRate   = 0.0;   % bursts/sec per trial (post-event only)
config.PostEventBurstWidth  = 0.07;  % seconds (FWHM)
config.PostEventBurstAmpSD  = 0.8;   % burst amplitude in SD units of post-event baseline

config.EventNoiseSubjectSD  = 2.0;
config.EventNoiseWidth      = 0.30;

if loadData
    load(dataFilePath);
    fprintf('\nLoaded simulation data from: %s\n', dataFilePath);
else

    %% Fixed-window naive baseline (10 non-overlapping windows + gaps; Holm-Bonferroni)
    nWin = 15;

    totalEdges = linspace(config.TimeRange(1), config.TimeRange(2), nWin + 1);
    winStarts  = totalEdges(1:end-1);
    winEnds    = totalEdges(2:end) -  1/config.Fs;

    % --- Storage ---
    results = struct('SNR', SNR_levels, ...
        'PoD_Adapt', zeros(size(SNR_levels)), ...
        'PoD_OLS',   zeros(size(SNR_levels)), ...
        'PoD_Naive', zeros(size(SNR_levels)));

    % We use a cell array to avoid indexing issues inside parfor
    rep_data_cell = cell(numel(SNR_levels), 1);
    rep_SNR = 1.0;

    % Container for representative trace (Panel G)
    rep_data = [];

    fprintf('========================================================\n');
    fprintf('Running Validation: Sensitivity Analysis (N=%d)\n', nIterations);
    fprintf('Comparison: GLS vs. OLS vs. Holm corrected window detection \n');
    fprintf('========================================================\n\n');

    %% 2. The Simulation Loop

    startTotal = tic;

    for i_snr = 1:numel(SNR_levels)
        curr_snr = SNR_levels(i_snr);

        fprintf('Running SNR = %.1f ... ', curr_snr);

        % Pre-allocate parallel arrays
        hits_adapt = false(nIterations, 1);
        hits_ols   = false(nIterations, 1);
        hits_naive = false(nIterations, 1);

        % Capture seed base for reproducibility
        base_seed = 100000 * round(1000 * curr_snr);
        
        rep_worker_storage = cell(nIterations, 1);

        tVec = config.TimeRange(1):(1/config.Fs):config.TimeRange(2);
        ClustME_Args = { 't', tVec, ...
                         'TimeAnchor', config.eventTime + config.AnchorOffset, ...
                         'parallel', false, ...
                         'verbose', false, ...
                         'minClusterSize', config.minClusterSize, ...
                         'alphaValue', alphaLevel};

        % Parallel Loop
        parfor i = 1:nIterations
            iter_seed = base_seed + i;

            % 1. Generate Data (Complex Noise + Signal)
            % Note: bench_generator handles the random seed
            genConfig = struct('designType', config.design, ...
                'targetSNR', curr_snr, ...
                'nTrials', config.nTrials);

            rng(iter_seed, 'twister');  % ensure jitter reproducible per iter
            sigTime = config.jitterRange(1) + (config.jitterRange(2)-config.jitterRange(1)) * rand(1,1);

            [resp, tbl, gtStruct] = clustme.bench_generator(config.nSubj, genConfig, ...
                'noiseMode', config.noiseMode, ...
                'SubjectVar', config.SubjectVar, ...
                'SubjectNoiseCV', config.SubjectNoiseCV, ...
                'TrialNoiseDriftMeanStep', config.TrialNoiseDriftMeanStep, ...
                'TrialNoiseDriftStepSD',   config.TrialNoiseDriftStepSD, ...
                'RampStrength', config.RampStrength, ...
                'signalWidth', config.signalWidth, ...
                'signalTime', sigTime, ...
                'eventTime',  config.eventTime, ...
                'TimeRange', config.TimeRange, ...
                'TrialCountCV', config.TrialCountCV, ...
                'MinTrialsPerSubject', config.MinTrialsPerSubject, ...
                'EventNoiseSubjectSD', config.EventNoiseSubjectSD, ...
                'EventNoiseWidth', config.EventNoiseWidth, ...       
                'PostEventNoiseScale', config.PostEventNoiseScale, ...
                'PostEventBurstRate',  config.PostEventBurstRate, ...
                'PostEventBurstWidth', config.PostEventBurstWidth, ...
                'PostEventBurstAmpSD', config.PostEventBurstAmpSD, ...
                'RandomSeed', iter_seed, ...
                'Fs', config.Fs);

            % 2. Run Adaptive-Local Mode 
            rng(iter_seed);
            [cl_adapt, ~, vd_adapt] = ClustME(resp, tbl, 'response~1+(1|Subject)', ...
                'whitening', false, ...
                'robustSE', 'none', ...
                'Vmode', 'adaptiveLocal', ClustME_Args{:});

            % 3. Run Naïve OLS (Standard Tool Baseline)
            % OLS-equivalent baseline using V=I
            rng(iter_seed);
            [cl_ols, ~, vd_ols] = ClustME(resp, tbl, 'response~1+(1|Subject)', ...
                'Vmode', 'identity', ClustME_Args{:});

            % 4. Score recovery using the predefined overlap and peak-proximity criteria
            hits_adapt(i) = check_recovery(cl_adapt, vd_adapt, gtStruct, alphaLevel);
            hits_ols(i)    = check_recovery(cl_ols, vd_ols, gtStruct, alphaLevel);
            hits_naive(i) = naive_fixed_window_holm(resp, tbl, tVec, gtStruct, alphaLevel, winStarts, winEnds);

            % 5. Capture data from the first worker only for the target SNR
            if (curr_snr == rep_SNR) && (i == 1)
                temp_rep = struct();
                temp_rep.gt  = gtStruct;
                temp_rep.vis = vd_adapt;

                [G_rep, ~] = findgroups(tbl.Subject);
                temp_rep.resp = splitapply(@(x) mean(x,1), resp, G_rep);

                rep_worker_storage{i} = temp_rep;
            end

        end

        % --- After the parfor finishes, move the data to the final container ---
        if (curr_snr == rep_SNR)
            rep_data_cell{i_snr} = rep_worker_storage{1};
        end

        % Store Results
        results.PoD_Adapt(i_snr)  = mean(hits_adapt);
        results.PoD_OLS(i_snr)    = mean(hits_ols);
        results.PoD_Naive(i_snr)  = mean(hits_naive);

        fprintf('Done. PoD: Adapt=%.2f | OLS=%.2f | NaiveWin=%.2f\n', ...
            results.PoD_Adapt(i_snr), results.PoD_OLS(i_snr), results.PoD_Naive(i_snr));
    end

    targetIdx = find(SNR_levels == rep_SNR, 1);
    if ~isempty(rep_data_cell{targetIdx})
        rep_data = rep_data_cell{targetIdx};
    else
        warning('Target SNR for Figure 2G was not found or captured.');
    end

    totalTime = toc(startTotal);
    fprintf('\nTotal Benchmark Time: %.1f minutes\n', totalTime/60);

    %% 3. Save Simulation Data
    if settings.exportData
        fprintf('\nSaving simulation data to: %s\n', dataFilePath);
        save(dataFilePath, 'results', 'rep_data', 'config', 'nIterations', 'alphaLevel', 'totalTime');
    else
        fprintf('\nData export disabled in settings. Skipping .mat save.\n');
    end
end

%% 4. Visualization (Figure 2 F-G)

% Figure Setup (Consistent with Fig 2 D-E)
figWidth  = 18;   
figHeight = 6;    
fontSize  = 9;    
fontName  = 'Arial';

% Colour definitions
col_Safe = [0 0.447 0.741];     % Blue (Adaptive/Safe)
col_GT   = [0 0 0];             % Black (Ground Truth)
col_Mean = [0.3 0.3 0.3];       % Dark Grey (Observed Mean)
col_SEM  = [0.7 0.7 0.7];
col_OLS   = [0.850 0.325 0.098]; % Red/Orange (Naïve OLS)

f = figure('Name', 'Fig 2 FG: Sensitivity', 'Color', 'w', 'Units', 'centimeters', 'Position', [5 5 figWidth figHeight]);

% Layout Logic
l_F = 0.08; 
w_panel = 0.38; 
gap_mid = 0.12; 
l_G = l_F + w_panel + gap_mid;
b_panel = 0.18;
h_panel = 0.70;

pos_F = [l_F, b_panel, w_panel, h_panel];
pos_G = [l_G, b_panel, w_panel, h_panel];


% --- Panel F: Power Curves ---
axF = axes('Position', pos_F); hold on;

hOLS  = plot(results.SNR, results.PoD_OLS,    '--x', 'Color', col_OLS,  'LineWidth', 1.5, 'MarkerSize', 6);
hAd   = plot(results.SNR, results.PoD_Adapt,  '-s',  'Color', col_Safe,  'LineWidth', 1.8, 'MarkerSize', 6, 'MarkerFaceColor', col_Safe);
hNW   = plot(results.SNR, results.PoD_Naive,  ':o',  'Color', [0 0 0],  'LineWidth', 1.2, 'MarkerSize', 5, 'MarkerFaceColor', 'w');

lgd = legend([hNW hOLS hAd], {'Fixed-window (Holm)', 'OLS (V=I)', 'GLS'}, ...
    'Location','northwest', 'Box','off', 'FontSize',8, 'FontName', fontName);
lgd.ItemTokenSize = [20, 18];

ylim([0 1]);
xlabel("SNR");
ylabel("Detection Rate");

% Calculate Total Gain (Best Adaptive vs Standard OLS)
auc_ols  = trapz(results.SNR, results.PoD_OLS);
auc_ad   = trapz(results.SNR, results.PoD_Adapt);

gain_GLS_vs_OLS = 100*(auc_ad - auc_ols)/max(eps, auc_ols);

text(0.50, 0.50, sprintf('%+0.0f%% AUC vs OLS', gain_GLS_vs_OLS), ...
    'Units','normalized', 'Color', col_Safe, 'FontWeight','bold', 'FontSize',8.5, ...
    'FontName', fontName, 'Interpreter', 'tex');

set(gca, 'FontSize', fontSize, 'FontName', fontName, 'Box', 'off', 'TickDir', 'out');
add_panel_label(f, l_F, 'F');


% --- Panel G: Signal Visualizer ---
axG = axes('Position', pos_G); hold on;

tVec = rep_data.gt.tVec;
% 1. Plot the observed grand-average response
obs_mean = mean(rep_data.resp, 1);
obs_sem  = std(rep_data.resp, 0, 1) / sqrt(size(rep_data.resp, 1));
upper = obs_mean + obs_sem;
lower = obs_mean - obs_sem;

fill([tVec(:)', fliplr(tVec(:)')], [upper(:)', fliplr(lower(:)')], col_SEM, 'FaceAlpha', 0.6, 'EdgeColor', 'none');

plot(tVec, obs_mean, 'Color', col_Mean, 'LineWidth', 1);

% 2. Plot Ground Truth Signal (Clean)
plot(tVec, rep_data.gt.signalVec, 'Color', col_GT, 'LineWidth', 1.5);

% 3. Highlight Detected Clusters (Adaptive)
sig_mask_ad = rep_data.vis.sigMask;
y_lims = ylim;
% Auto-scale Y if signal is small
if max(abs(obs_mean)) < 1
    ylim([-1 1] * max(abs(obs_mean))*1.5);
    y_lims = ylim;
end

% Shade the detected region (Adaptive = Blue Shade)
if any(sig_mask_ad)
    x_shade = tVec(sig_mask_ad);
    [starts, ends] = clustme.mask_to_clusters(sig_mask_ad);
    for k = 1:numel(starts)
        xs = tVec(starts(k):ends(k));
        patch([xs, fliplr(xs)], ...
              [repmat(y_lims(1),1,numel(xs)), repmat(y_lims(2),1,numel(xs))], ...
              col_Safe, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    end
end

xlabel('Time (s)', 'FontName', fontName, 'FontSize', fontSize);
ylabel('Amplitude (a.u.)', 'FontName', fontName, 'FontSize', fontSize);

% Custom Legend logic
h1 = plot(nan, nan, 'Color', col_Mean, 'LineWidth', 1);
h2 = plot(nan, nan, 'Color', col_GT, 'LineWidth', 1.5);
h3 = patch(nan, nan, col_Safe, 'FaceAlpha', 0.15, 'EdgeColor', 'none');

lgd_y_start = lgd.Position(2);

lgd = legend([h1 h2 h3], {'Observed Mean \pmSEM', 'Ground Truth', 'Detected Cluster'}, ...
    'Location', 'none', 'Box', 'off', 'FontSize', 8, 'FontName', fontName, 'Interpreter', 'tex');
lgd.ItemTokenSize = [16, 18];
lgd.Position(1:2) = [0.75, lgd_y_start];

text(lgd.Position(1) - lgd.Position(3) + 0.02, lgd.Position(2), ...
    sprintf('(GLS, $N = %d$)', config.nSubj), ...
    'Units','normalized', 'Color', 'k', 'FontWeight','bold', 'FontSize',8, ...
    'FontName', fontName, 'Interpreter', 'latex');

text(0.07, 0.91, 'SNR = 1', ...
    'Units','normalized', 'Color', 'k', 'FontWeight','bold', 'FontSize',9, ...
    'FontName', fontName, 'Interpreter', 'tex');

set(gca, 'FontSize', fontSize, 'FontName', fontName, 'Box', 'off', 'TickDir', 'out');
add_panel_label(f, l_G, 'G');


% --- Export ---
if settings.exportFigures
    pdfPath = fullfile(exportDir, [baseFileName '.pdf']);
    fprintf('\nExporting Fig 2 FG to %s...\n', pdfPath);
    exportgraphics(f, pdfPath, 'ContentType', 'vector', 'BackgroundColor', 'none');

    tiffPath = fullfile(exportDir, [baseFileName '.tif']);
    fprintf('Exporting Fig 2 FG TIFF to %s...\n', tiffPath);
    exportgraphics(f, tiffPath, 'Resolution', 300);
    
    fprintf('Done.\n');
else
    fprintf('\nFigure export disabled in settings. Skipping figure save.\n');
end


%% --- Helper Functions ---

function add_panel_label(fig, left_x, labelChar)
    annotation(fig, 'textbox', [left_x - 0.06, 0.94, 0.05, 0.05], ...
        'String', labelChar, 'EdgeColor', 'none', ...
        'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Arial');
end

function isHit = check_recovery(clusters, vis_data, gtStruct, alpha)
% check_recovery - Evaluates cluster detection against ground truth criteria
%
% Internal Context:
%   A helper function used to score signal recovery. It determines whether 
%   detected clusters constitute a "hit" based on their spatial and 
%   statistical alignment with the known injected signal.
%
% Inputs:
%   clusters - [struct] Detected clusters.
%   vis_data - [struct] Visualisation data containing the statistical T-map.
%   gtStruct - [struct] Ground truth metadata from the data generator.
%   alpha    - [double] Significance threshold.
%
% Outputs:
%   isHit    - [logical] True if a cluster meets all validation criteria.
%
% Algorithmic & Exactness Notes:
%   * Strict Hit Logic: A successful recovery requires p < alpha, a 
%     temporal overlap of >= 50% with the ground truth FWHM, and the peak 
%     cluster T-statistic to be within +/- 1 FWHM of the true signal peak.
%   * Peak Identification: Proximity is validated by finding the absolute 
%     maximum T-statistic specifically within the candidate cluster bounds 
%     rather than using a simple cluster centroid.


    isHit = false;
    
    % If no clusters at all
    if isempty(clusters) || isempty(clusters(1).p_value), return; end
    
    % Filter for significant clusters
    sigIdx = find([clusters.p_value] < alpha);
    
    if isempty(sigIdx), return; end
    
    % Ground Truth Bounds (Indices)
    if isempty(gtStruct.boundsFWHM), return; end % Should not happen if SNR > 0
    
    gt_start = gtStruct.boundsFWHM(1);
    gt_end   = gtStruct.boundsFWHM(2);
    gt_len   = gt_end - gt_start + 1;
    gt_peak_idx = gtStruct.peakIdx;
    gt_fwhm_samples = gtStruct.FWHM * gtStruct.Fs; 
    
    for k = sigIdx
        c_start = clusters(k).start;
        c_end   = clusters(k).end;
        
        % 1. Check Temporal Overlap
        i_start = max(c_start, gt_start);
        i_end   = min(c_end, gt_end);
        
        if i_end >= i_start
            overlap_len = i_end - i_start + 1;
            overlap_pct = overlap_len / gt_len;
            
            if overlap_pct >= 0.50
                % 2. Check Peak Proximity
                % Find peak T-stat index within this cluster
                cluster_t = abs(vis_data.Tmap(c_start:c_end));
                [~, loc_in_cluster] = max(cluster_t);
                peak_idx_abs = c_start + loc_in_cluster - 1;
                
                dist_samples = abs(peak_idx_abs - gt_peak_idx);
                
                % Rule: Within +/- 1 FWHM
                if dist_samples <= gt_fwhm_samples
                    isHit = true;
                    return; % Found a valid hit, break early
                end
            end
        end
    end
end

function hit = naive_fixed_window_holm(resp, tbl, tVec, gtStruct, alpha, winStarts, winEnds)
% naive_fixed_window_holm - Detects signals using fixed-window t-tests
%
% Internal Context:
%   A traditional detection baseline that applies one-sample t-tests to 
%   subject-level means within predefined temporal windows.
%
% Inputs:
%   resp      - [N × T double] Response matrix.
%   tbl       - [N × V table] Metadata for subject grouping.
%   tVec      - [1 × T double] Continuous time vector.
%   gtStruct  - [struct] Metadata detailing the ground truth signal.
%   alpha     - [double] Family-wise significance level.
%   winStarts - [1 × W double] Onset times for detection windows.
%   winEnds   - [1 × W double] Offset times for detection windows.
%
% Outputs:
%   hit       - [logical] True if the window containing the ground truth peak 
%               is significant after Holm-Bonferroni correction.
%
% Algorithmic & Exactness Notes:
%   * Targeted Hit Criteria: Success is strictly defined as the statistical 
%     rejection of the null hypothesis specifically for the window 
%     encompassing the ground truth peak time.
%   * Hierarchical Reduction: Collapses trial-level data into subject-level 
%     averages within each window before performing the group t-test.
%   * Row Alignment Guard: Synchronises the design table with the response 
%     matrix to prevent misaligned subject grouping during hierarchical 
%     mean calculation.

    % --- Normalise inputs (bench_generator sometimes returns cell wrappers) ---
    if iscell(tbl),  tbl  = tbl{1};  end

    % --- Enforce alignment between response rows and design table rows ---
    nRows = size(resp,1);
    if height(tbl) ~= nRows
        if height(tbl) > nRows
            % Keep the first nRows to match resp (conservative; avoids out-of-bounds)
            tbl = tbl(1:nRows, :);
        else
            error('naive_fixed_window_holm: tbl has fewer rows (%d) than resp (%d).', height(tbl), nRows);
        end
    end

    % --- Determine which window contains the GT peak time ---
    [~, idxPeak] = max(abs(gtStruct.signalVec(:)));
    tPeak = tVec(idxPeak);

    wTarget = find(tPeak >= winStarts & tPeak < winEnds, 1, 'first');
    if isempty(wTarget)
        hit = false; % peak landed in a gap => baseline cannot detect by design
        return;
    end

    % --- Subject grouping ---
    if ismember('Subject', tbl.Properties.VariableNames)
        subj = tbl.Subject;
    else
        error('naive_fixed_window_holm: tbl has no Subject column.');
    end

    % Robust grouping (categorical / numeric / string)
    [G, subjLevels] = findgroups(subj);
    nSubj = numel(subjLevels);

    pvals = nan(numel(winStarts),1);

    for w = 1:numel(winStarts)
        idx = (tVec >= winStarts(w)) & (tVec < winEnds(w));
        if ~any(idx)
            pvals(w) = 1; %#ok<*AGROW>
            continue;
        end

        subjMeans = nan(nSubj,1);
        for s = 1:nSubj
            rowIdx = find(G == s);
            subjMeans(s) = mean(resp(rowIdx, idx), 'all');

        end

        % one-sample t-test across subjects
        [~, p] = ttest(subjMeans, 0, 'Alpha', alpha);
        if isnan(p), p = 1; end
        pvals(w) = p;
    end

    reject = holm_bonferroni(pvals, alpha);
    hit = reject(wTarget);
end


function reject = holm_bonferroni(pvals, alpha)
%holm_bonferroni - Performs Holm-Bonferroni step-down correction
%
% Internal Context:
%   A statistical helper used to control the Family-Wise Error Rate (FWER) 
%   for the fixed-window baseline in sensitivity validation trials.
%
% Inputs:
%   pvals  - [m × 1 double] Vector of uncorrected p-values.
%   alpha  - [double] Target family-wise significance level.
%
% Outputs:
%   reject - [m × 1 logical] Boolean mask of rejected null hypotheses.
%
% Algorithmic & Exactness Notes:
%   * Step-Down Procedure: Evaluates p-values in ascending order against 
%     the adaptive threshold alpha/(m-k+1). Rejection strictly halts at 
%     the first non-significant result.
%   * Order Restoration: Maps rejections back to the original input 
%     indices to ensure significance remains correctly aligned with 
%     the temporal windows from the calling function.
    m = numel(pvals);
    [ps, ord] = sort(pvals(:), 'ascend');

    rej_sorted = false(m,1);
    for k = 1:m
        thresh = alpha / (m - k + 1);
        if ps(k) <= thresh
            rej_sorted(k) = true;
        else
            break; % step-down stops at first non-rejection
        end
    end

    reject = false(m,1);
    reject(ord) = rej_sorted;
end
