% run_fwer_validation - Factorial FWER Benchmark & Whitening Necessity
%
%   This script serves as a rigorous scientific validation benchmark to prove 
%   Type I Error (FWER) control across standard and extreme data configurations. 
%   Distinct from routine software integrity tests, this script generates the 
%   empirical simulation data used to construct the manuscript's core arguments 
%   regarding the necessity of Generalized Least Squares (GLS) and Wild Bootstrapping.
%
% How to Run:
%   run_fwer_validation()
%
% Validation Targets:
%   1. Baseline Safety: One-Sample (Global V).
%   2. Baseline Safety: Within-Subject (Shuffle).
%   3. Advanced Safety: One-Sample (Complex Noise, Adaptive V).
%   4. The Trap: OLS + Permutation in Unbalanced Heteroscedastic designs.
%   5. The Partial Fix: GLS + Permutation.
%   6. The Solution: GLS + Wild Bootstrap.
%   7. Baseline Safety: Balanced Between-Subject (groupLabel).
%   8. Mechanism Check: GLS + Wild Bootstrap (Leverage Correction OFF).
%
% Outputs:
%   - Terminal: Empirical FWER percentages for all configurations.
%   - Figures: Figure 2 (Panels D-E) and Supplementary Figure 1 (Panels A-B).
%   - Data: Exports empirical simulation arrays to 'Figure2-DEF_data.mat'.
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

%% 1. Configuration

settingsFile = 'validation_settings.json';
if ~isfile(settingsFile)
    error('Settings file "%s" not found. Please create it in the script directory.', settingsFile);
end
settings = jsondecode(fileread(settingsFile));

alphaLevel  = 0.05;

% --- EXECUTION CONTROL ---
% Define which configs to calculate fresh. Others will be loaded.
N_CONF = 8;
configsToRun = 3;% 1:N_CONF;  % run all configurations. Use 1:6 for paper reproduction

switch settings.runMode
    case 'debug'
        nIterations = 10;  
        numPerms   = 200; 
        fprintf('Mode: DEBUG (Fast mechanical test)\n');
        
    case 'fast'
        nIterations = 200;  
        numPerms   = 1000; 
        fprintf('Mode: FAST (partial run for sanity check)\n');
        
    case 'publication'
        nIterations = 1000;  
        numPerms   = 2000; 
        fprintf('Mode: PUBLICATION (Recreating full manuscript run)\n');
        
    otherwise
        error('Unknown runMode "%s". Use "debug","fast", or "publication"', settings.runMode);
end

% --- Export Settings ---

exportDir = settings.exportDir;
baseFileName = 'Figure2-DEF';
dataFilePath = fullfile(exportDir, [baseFileName '_data.mat']);

% --- Load Previous Data (if needed) ---
simData = [];

if settings.loadSavedData || length(configsToRun)<N_CONF
    if isfile(dataFilePath)
        fprintf('Loading existing data from: %s\n', dataFilePath);
        loaded = load(dataFilePath);

        if isfield(loaded, 'simData')
            simData = loaded.simData;
        else
            warning('Old data format detected in .mat file. Running all configurations to generate new struct.');
            configsToRun = 1:N_CONF;
        end
    else
        warning('Data file not found. Running all configurations.');
        configsToRun = 1:N_CONF;
    end
end

% --- Test Configurations ---
% Define a baseline configuration with default values
baseConfig = struct(...
    'name', 'Baseline', ...
    'design', 'one-sample', ...
    'formula', 'response ~ 1 + (1|Subject)', ... 
    'coef', '', ...
    'noise', 'gaussian', ...
    'Vmode', 'global', ...
    'whitening', true, ...
    'nSubj', 20, ...
    'method', 'signFlip', ...
    'tcritMode', 'permutation', ...
    'SubjectVar', 2.0, ...
    'nTrials', 30, ... 
    'NoiseAlpha', 1.0, ...
    'RampStrength', 3.0, ...
    'GroupNoiseScale', [1, 1], ...
    'trialDrop', struct('enable', false, 'keepProb', [1, 1], 'minKeep', 0), ...
    'wbLeverage', true, ...
    'seedOffset', 0 ...
);

% Preallocate configurations by inheriting the baseline
configs = repmat(baseConfig, 1, N_CONF);

% 1. Baseline Safety: One-Sample, Gaussian Noise, Global V (Standard)
configs(1).name      = 'OneSample (Global)';
configs(1).seedOffset = 10000;

% 2. Baseline Safety: Within-Subject, Gaussian Noise (Standard)
configs(2).name      = 'Within (Shuffle)';
configs(2).design    = 'within';
configs(2).method    = 'withinSubject';
configs(2).formula   = 'response ~ 1 + Condition + (1|Subject)'; 
configs(2).coef      = 'Condition_B';   
configs(2).seedOffset = 20000;

% 3. Advanced Safety: One-Sample, Complex Noise, Adaptive V (Stress Test)
configs(3).name      = 'Adaptive (Complex)';
configs(3).noise     = 'complex';
configs(3).Vmode     = 'adaptiveLocal'; 
configs(3).seedOffset = 31000;

% 4. Unbalanced Heteroscedastic Benchmark: OLS (V=I) vs GLS (V=Global)
configs(4).name      = 'Behrens-Fisher';
configs(4).design    = 'between';
configs(4).formula   = 'response ~ 1 + Group + (1|Subject)';    
configs(4).coef      = 'Group_Patient';                          
configs(4).noise     = 'complex'; 
configs(4).Vmode     = 'adaptiveLocal';
configs(4).nSubj     = [20 10];     
configs(4).method    = 'groupLabel';
configs(4).SubjectVar = 0.20;   
configs(4).nTrials   = 10;      
configs(4).NoiseAlpha   = 1.0;
configs(4).RampStrength = 1.0;  
configs(4).GroupNoiseScale = [1, 3];
configs(4).whitening  = false;     
configs(4).Vmode      = 'identity'; 
configs(4).seedOffset = 40000;

% 5. GLS + label permutation under heteroscedastic imbalance
configs(5) = configs(4);
configs(5).name       = 'GLS + Permutation';
configs(5).whitening  = true;
configs(5).Vmode      = 'adaptiveLocal'; 

% 6. GLS + wild bootstrap under heteroscedastic imbalance
configs(6) = configs(5);
configs(6).name       = 'GLS + Wild Bootstrap';
configs(6).method     = 'wildBootstrap';

% 7. Balanced homoscedastic between-subject label-permutation benchmark
configs(7) = baseConfig;
configs(7).name       = 'Balanced Between-Subject';
configs(7).design     = 'between';
configs(7).formula    = 'response ~ 1 + Group + (1|Subject)';    
configs(7).coef       = 'Group_Patient';                          
configs(7).noise      = 'gaussian'; 
configs(7).nSubj      = [20 20];     
configs(7).method     = 'groupLabel';
configs(7).GroupNoiseScale = [1 1];
configs(7).seedOffset = 50000;

% 8. GLS + wild bootstrap without leverage correction
configs(8) = configs(6);
configs(8).name       = 'GLS + WB (No Leverage)';
configs(8).wbLeverage = false;
% Use the same seed family as configurations 4-6 for matched comparison.

% Storage
nConfigs = numel(configs);
if isempty(simData) 
    % Initialize a clean 1xN struct array if no data was loaded
    simData = struct('isSig', cell(1, nConfigs), ...
                     'maxStats', cell(1, nConfigs), ...
                     'crit', cell(1, nConfigs));
end

fprintf('========================================================\n');
fprintf('          Factorial FWER Benchmark (N=%d)\n', nIterations);
fprintf('========================================================\n\n');

%% 2. The Simulation Loop

startTotal = tic;

for c = 1:nConfigs
    if ~ismember(c, configsToRun)
        fprintf('Config %d: %s... [LOADED/SKIPPED]\n', c, configs(c).name);
        continue;
    end

    fprintf('Config %d: %s... ', c, configs(c).name);
    cfg = configs(c);
    
    sigFlags = nan(nIterations, 1);
    maxStats = nan(nIterations, 1);
    critVals = nan(nIterations, 1);

    for i = 1:nIterations
        % Unique seed per iteration, but shared across configs 4,5,6
        currentSeed = i + cfg.seedOffset;
        
        % A. Generate Null Data
        genConfig = struct('designType', cfg.design, 'nTrials', cfg.nTrials, 'effectSize', 0);
        [resp, designTbl, ~] = clustme.bench_generator(cfg.nSubj, genConfig, ...
            'noiseMode', cfg.noise, ...
            'NoiseAlpha', cfg.NoiseAlpha, ...
            'GroupNoiseScale', cfg.GroupNoiseScale, ... 
            'SubjectVar', cfg.SubjectVar, ...
            'RampStrength', cfg.RampStrength, ...
            'RandomSeed', currentSeed + 30); 

        % B. Trial attrition / QC rejection
        if isfield(cfg, 'trialDrop') && cfg.trialDrop.enable
            rng(currentSeed + 7777); % Ensure dropout is also perfectly paired

            keepMask = true(height(designTbl), 1);
            subjVals = unique(designTbl.Subject);

            for s = 1:numel(subjVals)
                idxS = (designTbl.Subject == subjVals(s));

                % Determine keep probability by group (between-subject designs)
                pKeep = cfg.trialDrop.keepProb(1); % default (Control)
                if ismember('Group', designTbl.Properties.VariableNames)
                    grp = designTbl.Group(find(idxS, 1, 'first'));
                    if grp == categorical("Patient")
                        pKeep = cfg.trialDrop.keepProb(2);
                    end
                end

                idxRows = find(idxS);
                k = (rand(numel(idxRows), 1) < pKeep);

                % Enforce minimum kept trials per subject
                if sum(k) < cfg.trialDrop.minKeep
                    k(:) = false;
                    nForce = min(cfg.trialDrop.minKeep, numel(idxRows));
                    k(randperm(numel(idxRows), nForce)) = true;
                end

                keepMask(idxRows) = k;
            end

            % Apply to design + all response matrices
            designTbl = designTbl(keepMask, :);
            resp = resp(keepMask, :);
        end

        % C. Run ClustME
        try
            [sigFlags(i), maxStats(i), critVals(i)] = run_method_variant(resp, designTbl, cfg, numPerms, alphaLevel);
        catch ME
            fprintf('\nError iter %d: %s\n', i, ME.message);
        end

        if mod(i, 100) == 0
            fprintf('   > Iter %d/%d | Current FWER: %.1f%%\n', i, nIterations, mean(sigFlags(1:i), 'omitnan') * 100);
        end
    end
    
    % Store directly into the unified struct
    simData(c).isSig = sigFlags;
    simData(c).maxStats = maxStats;
    simData(c).crit = critVals;

    % Report 
    empFWER = mean(sigFlags, 'omitnan');
    fprintf('FWER = %.1f%%\n', empFWER * 100);
end

totalTime = toc(startTotal);
fprintf('\nTotal Benchmark Time: %.1f minutes\n', totalTime/60);

%% 2.5 Save Simulation Data

if settings.exportData
    dataFilePath = fullfile(exportDir, [baseFileName '_data.mat']);
    
    fprintf('\nSaving simulation data to: %s\n', dataFilePath);
    save(dataFilePath, 'simData', 'configs', 'nIterations', 'alphaLevel', 'totalTime');
    fprintf('Data saved successfully.\n');
else
    fprintf('\nData export disabled in settings. Skipping .mat save.\n');
end

%% 2.6 Print Final FWER Summary
fprintf('\n========================================================\n');
fprintf('FINAL EMPIRICAL FWER SUMMARY (N=%d)\n', nIterations);
fprintf('========================================================\n');
fprintf('1. 1-Sample (Global) baseline     : %0.1f%%\n', mean(simData(1).isSig, 'omitnan') * 100);
fprintf('2. Within-Subject baseline        : %0.1f%%\n', mean(simData(2).isSig, 'omitnan') * 100);
fprintf('3. 1-Sample (Adaptive) baseline   : %0.1f%%\n', mean(simData(3).isSig, 'omitnan') * 100);
fprintf('4. OLS + Perm (Between-Subject)   : %0.1f%%\n', mean(simData(4).isSig, 'omitnan') * 100);
fprintf('5. GLS + Perm (Between-Subject)   : %0.1f%%\n', mean(simData(5).isSig, 'omitnan') * 100);
fprintf('6. GLS + WB (Between-Subject)     : %0.1f%%\n', mean(simData(6).isSig, 'omitnan') * 100);
fprintf('7. Balanced Between-Subject (groupLabel): %0.1f%%\n', mean(simData(7).isSig, 'omitnan') * 100);
fprintf('8. GLS + WB (No Leverage)         : %0.1f%%\n', mean(simData(8).isSig, 'omitnan') * 100);
fprintf('========================================================\n\n');

%% 3. Visualization (Figure 2 D-E)

% Figure Setup
figWidth  = 18;   
figHeight = 6;    
fontSize  = 9;    
fontName  = 'Arial';

f = figure('Name', 'Fig 2 DE: FWER & Necessity', 'Color', 'w', 'Units', 'centimeters', 'Position', [5 5 figWidth figHeight]);

% Layout Logic: Physical vertical split for Panel D
b_panel = 0.18; 
h_total = 0.70; 
w_panel = 0.38; 
gap_mid = 0.14; % Horizontal gap between D and E

l_D = 0.08;
l_E = l_D + w_panel + gap_mid;

% Define Two Separate Viewports for Panel D (Bottom 70%, Top 25%)
y_split_gap = 0.015; 
h_bot = h_total * 0.70;
h_top = h_total * 0.25;
b_top = b_panel + h_bot + y_split_gap;

pos_D_bot = [l_D, b_panel, w_panel, h_bot];
pos_D_top = [l_D, b_top,   w_panel, h_top];
pos_E     = [l_E, b_panel, w_panel, h_total];

% --- Panel D: FWER Bar Chart (Split Axis) ---
% Pre-calculate stats directly from simData
fwerMeans = zeros(1, N_CONF);
for k = 1:N_CONF
    fwerMeans(k) = mean(simData(k).isSig, 'omitnan') * 100;
end
fwerCI = 1.96 * sqrt((fwerMeans/100) .* (1 - fwerMeans/100) / nIterations) * 100;

% Define positions: Left (OLS), Center (GLS Perm), Right (GLS WB)
x_base = 4 - 0.24;  % Left: Naive (Grey)
x_trap = 4;         % Center: Trap (Orange)
x_safe = 4 + 0.24;  % Right: Safe (Blue)
w_sub  = 0.22;

% Extract specific values for Configs 4, 5, 6
olsMean  = fwerMeans(4); olsCI  = fwerCI(4);
trapMean = fwerMeans(5); trapCI = fwerCI(5);
ctrlMean = fwerMeans(6); ctrlCI = fwerCI(6);

% Colors
col_Safe = [0 0.447 0.741];    
col_Trap = [0.85 0.325 0.098]; 
col_Base = [0.6 0.6 0.6];      

% Create Separate Axes
axD_bot = axes('Position', pos_D_bot); hold on;
axD_top = axes('Position', pos_D_top); hold on;

for ax = [axD_bot, axD_top]
    axes(ax); hold on;
    % Standard Configs
    plotOrder = [1, 3, 2]; % Reorders the first three bars to match the manual labels
    for x = 1:3
        cfgIdx = plotOrder(x);
        bar(x, fwerMeans(cfgIdx), 0.6, 'FaceColor', col_Safe, 'EdgeColor', 'none');
        errorbar(x, fwerMeans(cfgIdx), fwerCI(cfgIdx), 'k', 'CapSize', 3, 'LineStyle', 'none');
    end

    % Configs 4-6 Split
    bar(x_base, olsMean, w_sub, 'FaceColor', col_Base, 'EdgeColor', 'none');
    errorbar(x_base, olsMean, olsCI, 'k', 'CapSize', 0, 'LineStyle', 'none');

    bar(x_trap, trapMean, w_sub, 'FaceColor', col_Trap, 'EdgeColor', 'none');
    errorbar(x_trap, trapMean, trapCI, 'k', 'CapSize', 0, 'LineStyle', 'none');

    bar(x_safe, ctrlMean, w_sub, 'FaceColor', col_Safe, 'EdgeColor', 'none');
    errorbar(x_safe, ctrlMean, ctrlCI, 'k', 'CapSize', 0, 'LineStyle', 'none');
end

% Format Bottom Axis (0-12%)
axes(axD_bot);
ylim([0 12]); yticks([0 5 10]);
ylabel('FWER (%)', 'FontSize', fontSize, 'FontName', fontName);
set(gca, 'Box', 'off', 'TickDir', 'out', 'FontSize', fontSize, 'FontName', fontName);
% 1. Clear default labels so they don't conflict
set(axD_bot, 'XTick', [1 2 3 4], 'XTickLabel', [], 'XLim', [0.5 4.8]);

% 2. Define Multiline Labels manually
manualLabels = {
    {'1-Sample'; '(Global)'}, ...
    {'1-Sample'; '(Adaptive)'}, ...   
    {'Within-'; 'Subject'}, ...
    {'Between-'; 'Subject'}
};

% 3. Place text objects explicitly (using 'Clipping','off' to show below axis)
for k = 1:4
    text(axD_bot, k, -0.2, manualLabels{k}, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', ...
        'FontSize', fontSize, ...
        'FontName', fontName, ...
        'Clipping', 'off'); % Ensure it renders outside the Y-limits
end

yline(5, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5); % Target Line
text(0.5, 5.8, 'Target 5%', 'FontSize', 8, 'Color', [0.4 0.4 0.4]);

% Format Top Axis (25-40%)
axes(axD_top);
ylim([25 40]); yticks([30 40]);
set(gca, 'Box', 'off', 'TickDir', 'out', 'FontSize', fontSize, 'FontName', fontName);
set(gca, 'XColor', 'none', 'YColor', 'k'); % Hide X axis
set(gca, 'XLim', [0.5 4.8]);

axD_top.XAxis.Visible = 'off';
axD_top.XBaseline.Visible = 'off';
line(xlim(axD_top), [25 25], 'Color', 'w', 'LineWidth', 1);

% Draw Visual Break (Diagonal Lines)
annotation('line', [l_D-0.01 l_D+0.01], [b_top-0.005 b_top+0.005], 'LineWidth', 1, 'Color', 'k'); 
annotation('line', [l_D-0.01 l_D+0.01], [b_top+0.005 b_top+0.015], 'LineWidth', 1, 'Color', 'k'); 

% Create Legend
hold on;
% Create invisible dummy handles in the correct order
hL_Base = bar(nan, nan, 'FaceColor', col_Base, 'EdgeColor', 'none');
hL_Trap = bar(nan, nan, 'FaceColor', col_Trap, 'EdgeColor', 'none');
hL_Safe = bar(nan, nan, 'FaceColor', col_Safe, 'EdgeColor', 'none');

% Unified Legend: Naïve -> Trap -> Safe
legend([hL_Base hL_Trap hL_Safe], ...
    {'OLS + Permutation', 'GLS + Permutation', 'GLS + Wild Bootstrap'}, ...
    'Location', 'northwest', ... 
    'Box', 'off', ...
    'FontSize', 8, ...
    'FontName', fontName);

add_panel_label(f, l_D, 'D');

% --- Panel E: The Mechanism (Stacked Subplots) ---
% Layout math for 3 stacked subplots
gap_h = 0.04; 
h_sub = (h_total - 2*gap_h) / 3;

pos_E1 = [l_E, b_panel + 2*(h_sub + gap_h), w_panel, h_sub]; % Top: OLS
pos_E2 = [l_E, b_panel + 1*(h_sub + gap_h), w_panel, h_sub]; % Mid: GLS Perm
pos_E3 = [l_E, b_panel,                     w_panel, h_sub]; % Bot: GLS WB

% 1. Calculate Empirical (True) 95th Percentiles
emp_Base = prctile(simData(4).maxStats, 95);
emp_Trap = prctile(simData(5).maxStats, 95);
emp_Safe = prctile(simData(6).maxStats, 95);

% 2. Calculate Internal (Assumed) Thresholds
int_Base = mean(simData(4).crit, 'omitnan'); 
int_Trap = mean(simData(5).crit, 'omitnan'); 
int_Safe = mean(simData(6).crit, 'omitnan'); 

edges = linspace(0, 40, 50); 
ymax = 135;

% --- Subplot E1: OLS + Permutation (Base) ---
axE1 = axes('Position', pos_E1); hold on;
histogram(simData(4).maxStats, edges, 'FaceColor', col_Base, 'FaceAlpha', 0.6, 'EdgeColor', 'none');
ylim([0 ymax]); xlim([0 max(edges)]);
line([emp_Base emp_Base], [0 60], 'Color', 'k', 'LineWidth', 1.5);
line([int_Base int_Base], [0 60], 'Color', 'r', 'LineWidth', 1.5, 'LineStyle', '-');
text(0.5, 1, 'OLS + Permutation (Fail)', 'Units', 'normalized', ...
    'FontSize', fontSize-1, 'FontName', fontName, 'Color', col_Base, 'HorizontalAlignment', 'center');
set(axE1, 'Box', 'off', 'TickDir', 'out', 'XTickLabel', [], 'FontSize', fontSize, 'FontName', fontName);
%ylabel('Count', 'FontSize', fontSize, 'FontName', fontName);

% --- Subplot E2: GLS + Permutation (Trap) ---
axE2 = axes('Position', pos_E2); hold on;
histogram(simData(5).maxStats, edges, 'FaceColor', col_Trap, 'FaceAlpha', 0.6, 'EdgeColor', 'none');
ylim([0 ymax]); xlim([0 max(edges)]);
line([emp_Trap emp_Trap], [0 60], 'Color', 'k', 'LineWidth', 1.5);
line([int_Trap int_Trap], [0 60], 'Color', 'r', 'LineWidth', 1.5, 'LineStyle', '-');
text(0.5, 0.92, 'GLS + Permutation (Fail)', 'Units', 'normalized', ...
    'FontSize', fontSize-1, 'FontName', fontName, 'Color', col_Trap, 'HorizontalAlignment', 'center');
set(axE2, 'Box', 'off', 'TickDir', 'out', 'XTickLabel', [], 'FontSize', fontSize, 'FontName', fontName);
ylabel('Count', 'FontSize', fontSize, 'FontName', fontName);

% --- Subplot E3: GLS + Wild Bootstrap (Safe) ---
axE3 = axes('Position', pos_E3); hold on;
histogram(simData(6).maxStats, edges, 'FaceColor', col_Safe, 'FaceAlpha', 0.6, 'EdgeColor', 'none');
ylim([0 ymax]); xlim([0 max(edges)]);
line([emp_Safe emp_Safe], [0 60], 'Color', 'k', 'LineWidth', 1.5);
line([int_Safe int_Safe], [0 60], 'Color', 'r', 'LineWidth', 1.5, 'LineStyle', '-');
text(0.5, 0.85, 'GLS + Wild Bootstrap', 'Units', 'normalized', ...
    'FontSize', fontSize-1, 'FontName', fontName, 'Color', col_Safe, 'HorizontalAlignment', 'center');
set(axE3, 'Box', 'off', 'TickDir', 'out', 'FontSize', fontSize, 'FontName', fontName);
xlabel('Max Cluster Statistic', 'FontName', fontName, 'FontSize', fontSize);
%ylabel('Count', 'FontSize', fontSize, 'FontName', fontName);

% Add legend to the top subplot to explain the lines
lgd = legend(axE1, {'Null Dist.', 'Empirical 95%', 'Internal 95%'}, ...
    'Location', 'northeast', 'Box', 'off', 'FontSize', 8);
lgd.ItemTokenSize(1) = 11;
lgd.Position(1) = lgd.Position(1) + 0.025;
add_panel_label(f, l_E, 'E');

% Export Figure 1
if settings.exportFigures
    pdfPath = fullfile(exportDir, [baseFileName '.pdf']);
    fprintf('\nExporting Fig 1 to %s...\n', exportDir);
    exportgraphics(f, pdfPath, 'ContentType', 'vector', 'BackgroundColor', 'none');
    
    % TIFF Export at 300 DPI
    tiffPath = fullfile(exportDir, [baseFileName '.tif']);
    fprintf('Exporting Fig 1 TIFF to %s...\n', tiffPath);
    exportgraphics(f, tiffPath, 'Resolution', 300);
else
    fprintf('\nFigure export disabled in settings. Skipping Fig 1 save.\n');
end
%% 4. Supplementary Figure S1 (Convergence)

fS = figure('Name', 'Fig S1A', 'Color', 'w', 'Units', 'centimeters', 'Position', [10 5 6 6]);

% Define shared plotting settings struct
plotSettings = struct();
plotSettings.fontName = fontName;
plotSettings.fontSize = fontSize;

plotSettings.color = [0 0.447 0.741]; % Blue
plot_convergence(fS, simData(1).isSig, 'Global V', plotSettings);

plotSettings.color = [0.466 0.674 0.188]; % Green
plot_convergence(fS, simData(3).isSig, 'Adaptive V', plotSettings);

add_panel_label(fS, 0.06, 'A');


% --- Panel B: Balanced Between-Subject Convergence ---
fS_B = figure('Name', 'Fig S1B', 'Color', 'w', 'Units', 'centimeters', 'Position', [17 5 6 6]);

plotSettings.color = [0.4940 0.1840 0.5560]; 
plot_convergence(fS_B, simData(7).isSig, 'Balanced Between-Subject', plotSettings);

add_panel_label(fS_B, 0.06, 'B');


% Export Figure S1A+B
if settings.exportFigures
    supPath = fullfile(exportDir, 'FigureS1A_Convergence.pdf');
    fprintf('Exporting Fig S1A to %s...\n', supPath);
    annotation(fS, 'rectangle', [0 0 1 1], 'Color', 'none', 'EdgeColor', 'none'); % prevent exportgraphics auto-crop
    exportgraphics(fS, supPath, 'ContentType', 'vector', 'BackgroundColor', 'none');
    
    % TIFF Export
    tiffPath = fullfile(exportDir, 'FigureS1A_Convergence.tif');
    fprintf('Exporting Fig S1A TIFF to %s...\n', tiffPath);
    exportgraphics(fS, tiffPath, 'Resolution', 300);

    supPath = fullfile(exportDir, 'FigureS1B_Convergence.pdf');
    fprintf('Exporting Fig S1B to %s...\n', supPath);
    annotation(fS_B, 'rectangle', [0 0 1 1], 'Color', 'none', 'EdgeColor', 'none'); % prevent exportgraphics auto-crop
    exportgraphics(fS_B, supPath, 'ContentType', 'vector', 'BackgroundColor', 'none');
    
    % TIFF Export 
    tiffPath = fullfile(exportDir, 'FigureS1B_Convergence.tif');
    fprintf('Exporting Fig S1B TIFF to %s...\n', tiffPath);
    exportgraphics(fS_B, tiffPath, 'Resolution', 300);
else
    fprintf('Figure export disabled in settings. Skipping Fig S1A+B save.\n');
end
fprintf('Done.\n');

function ax = plot_convergence(fig, isSigArray, plotName, plotSettings)
% plot_convergence - Renders cumulative FWER traces for validation trials
%
% Internal Context:
%   A plotting function that overlays multiple cumulative error rate traces 
%   onto a single set of axes to visually compare method performance.
%
% Inputs:
%   fig          - [matlab.ui.Figure] Target figure handle.
%   isSigArray   - [N × 1 logical] Significance outcomes per iteration.
%   plotName     - [char] Legend label for the trace.
%   plotSettings - [struct] Visual parameters (.color, .fontName, .fontSize).
%
% Outputs:
%   ax           - [matlab.graphics.axis.Axes] Handle to the populated axes.
%
% Algorithmic & Exactness Notes:
%   * Cumulative Mean: Computes the rolling error rate via a cumulative sum 
%     divided by the iteration index.
%   * Static Reference Band: Draws a shaded 95% Confidence Interval polygon 
%     representing the expected binomial variance at 1,000 iterations.
%   * Dynamic Overlay: Retrieves existing axes to append subsequent traces, 
%     dynamically expanding the X-axis limits to fit the longest data vector.

    % Calculate local iterations for the current array
    nIters = length(isSigArray);
    localMax = min(max(nIters, 1), 1000); % Ensure at least 1, caps at 1000
    
    ax = findobj(fig, 'Type', 'axes');
    
    if isempty(ax)
        % 1. Create axes and base formatting
        ax = axes('Parent', fig, 'Position', [0.22 0.18 0.72 0.70]);
        hold(ax, 'on');
        
        % 2. Draw the nominal 95% binomial acceptance band used for the 
        %    manuscript-scale 1,000-iteration benchmark.
        fill(ax, [1:1000, 1000:-1:1], ...
             [repmat(0.0365, 1, 1000), repmat(0.0635, 1, 1000)], ...
             [0.9 0.9 0.9], 'EdgeColor', 'none', 'DisplayName', '95% CI');
             
        % 3. Plot 5% target line (hidden from legend)
        yline(ax, 0.05, '--', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
        
        % 4. Format Axes using the local array's length for initial bounds
        xlim(ax, [1 localMax]); ylim(ax, [0 0.15]);
        xlabel(ax, 'Iterations', 'FontName', plotSettings.fontName, 'FontSize', plotSettings.fontSize);
        ylabel(ax, 'Cumulative FWER', 'FontName', plotSettings.fontName, 'FontSize', plotSettings.fontSize);
        set(ax, 'FontSize', plotSettings.fontSize, 'FontName', plotSettings.fontName, 'Box', 'off');
    else
        ax = ax(1); % Use existing axes
        hold(ax, 'on');
        
        % Expand x-axis dynamically if subsequent lines are longer
        currentXlim = xlim(ax);
        if localMax > currentXlim(2)
            xlim(ax, [1 localMax]);
        end
    end
    
    % 5. Calculate and add the convergence line
    if nIters > 0
        runFWER = cumsum(isSigArray) ./ (1:nIters)';
        plot(ax, 1:nIters, runFWER, 'LineWidth', 1.5, 'Color', plotSettings.color, 'DisplayName', plotName);
    end
    
    % 6. Adapt legend automatically
    lgd = legend(ax, 'show');
    lgd.Box = 'off';
    lgd.Location = 'northeast';
    lgd.FontSize = 8;
    lgd.ItemTokenSize(1) = 12;
end

function add_panel_label(fig, left_x, labelChar)
    % left_x: The horizontal start of the panel
    annotation(fig, 'textbox', [left_x - 0.06, 0.94, 0.05, 0.05], ...
        'String', labelChar, 'EdgeColor', 'none', ...
        'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Arial');
end

function [isSig, maxStat, internalCrit, vis_data] = run_method_variant(resp, design, cfg, numPerms, alpha, fullLMEFlag)
% run_method_variant - Executes a single permutation trial for FWER validation
%
% Internal Context:
%   A helper function that runs a configured statistical trial. It bridges 
%   the synthetic data generation with the main clustering algorithm to 
%   derive empirical significance rates.
%
% Inputs:
%   resp        - [N × T double] Generated response matrix.
%   design      - [N × V table] Trial metadata table.
%   cfg         - [struct] Scenario configuration (Vmode, whitening, leverage).
%   numPerms    - [double] Target permutation count.
%   alpha       - [double] Significance level.
%   fullLMEFlag - [logical] Toggle for continuous exact refitting.
%
% Outputs:
%   isSig        - [logical] True if any cluster strictly satisfies p < alpha.
%   maxStat      - [double] Maximum observed cluster mass.
%   internalCrit - [double] Empirical critical threshold derived from the null.
%   vis_data     - [struct] Full visualisation arrays from the algorithm.
%
% Algorithmic & Exactness Notes:
%   * Structural override: allows intentionally invalid comparator branches used to 
%     demonstrate exchangeability failure.
%   * Empty Cluster Guard: Defaults the extracted maximum mass to 0 if the 
%     trace fails to cross the cluster-forming threshold, preventing indexing errors.
%   * Threshold extraction: estimates the empirical 1-alpha percentile from the 
%     permutation null distribution for comparison with maxStat.

    if nargin < 6 || isempty(fullLMEFlag)
        fullLMEFlag = false;
    end

    [clusters, ~, vis_data] = ClustME(resp, design, cfg.formula, ...
        'permutationMethod', cfg.method, ...
        'whitening', cfg.whitening, ...
        'overrideWhiteningCheck', true, ...
        'parallel', true, ...
        'robustSE', 'none', ...
        'Vmode', cfg.Vmode, ...
        'fullLME', fullLMEFlag, ...
        'tcritMode', cfg.tcritMode, ...
        'clusterMassMethod', 'mean', ...
        'numPerms', numPerms, ...
        'testCoefficient', cfg.coef, ...
        'alphaValue', alpha, ...
        'wbLeverage', cfg.wbLeverage, ...
        'verbose', false);

    isSig = any([clusters.p_value] < alpha);

    if isempty(clusters)
        maxStat = 0;
    else
        maxStat = max([clusters.mass]);
    end

    internalCrit = prctile(vis_data.nullStats, 100 * (1 - alpha));
end