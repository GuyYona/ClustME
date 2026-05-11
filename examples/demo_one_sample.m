%% demo_one_sample
% Example of a simple one-sample ClustME workflow
%
% This demo assumes that ClustME.m and the +clustme
% package folder are already on the MATLAB path.
%
% The example generates a synthetic one-sample hierarchical time series and
% tests whether the population-level response differs from zero. It uses
% subject-level sign flipping for the one-sample/intercept null and shows
% the result both on the original response trace and on the ClustME t-map.

%% 1. Generate a synthetic one-sample dataset
% Rows are trial-level observations nested within subjects. A positive
% event-related transient is added after time zero.

clear; clc; close all;
rng(42, 'twister');

Fs = 100;
timeRange = [-0.2 0.8];
nSubjects = 18;

config = struct();
config.designType = 'one-sample';
config.nTrials = 25;
config.targetSNR = 1.25;

[responses, design, groundTruth] = clustme.bench_generator(nSubjects, config, ...
    'Fs', Fs, ...
    'TimeRange', timeRange, ...
    'RandomSeed', 42, ...
    'noiseMode', 'complex', ...
    'SubjectVar', 2.0, ...
    'RampStrength', 2.0, ...
    'signalTime', 0.35, ...
    'signalWidth', 0.08);

design.Subject = categorical(design.Subject);
t = linspace(timeRange(1), timeRange(2), size(responses, 2));

fprintf('Generated %d observations x %d time samples from %d subjects.\n\n', ...
    size(responses, 1), size(responses, 2), nSubjects);

%% 2. Run ClustME
% The model tests the intercept while accounting for subject-level random
% intercepts. Global Static-V pooling is used here as a simple introductory
% choice, so the covariance estimate is not tied to a user-selected event
% latency.

lmeFormula = 'response ~ 1 + (1|Subject)';

rng(1042, 'twister');  % Reproducible permutation draws.

[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula, ...
    'permutationMethod', 'signFlip', ...
    'permuteUnit', 'Subject', ...
    'testCoefficient', '', ...         % an empty test coefficient tests for the intercept
    'numPerms', 1000, ...
    'BqTarget', 500, ...
    'Vmode', 'global', ...    
    'clusterMassMethod', 'mean', ...
    'clusterSummaryMetric', 'signedPeak', ...
    'Fs', Fs, ...
    't', t, ...
    'parallel', false);

%% 3. Summarise detected clusters
% Cluster p-values are evaluated against the max-cluster null, providing
% cluster-level FWER control over the analysed time series. ClusterMass is
% the inferential statistic; SummaryMetric is the descriptive response
% summary requested through clusterSummaryMetric.

if isempty(clusters)
    fprintf('No clusters survived cluster-level FWER control at alpha = 0.05.\n');
else
    nClusters = numel(clusters);

    clusterTable = table((1:nClusters)', ...
        arrayfun(@(c) t(c.start), clusters(:)), ...
        arrayfun(@(c) t(c.end), clusters(:)), ...
        [clusters(:).mass]', ...
        [clusters(:).p_value]', ...
        [clusters(:).measure]', ...
        'VariableNames', {'Cluster', 'Start_s', 'End_s', ...
                          'ClusterMass', 'PValue', 'SummaryMetric'});
    disp(clusterTable);
    fprintf(['ClusterMass is the inferential cluster statistic used for permutation testing.\n', ...
         'SummaryMetric is the descriptive signedPeak response summary within the cluster.\n']);

end

%% 4. Plot the response and mark significant clusters
% This is a standard data-space plot. clustme.plotClusterLines overlays the
% FWER-controlled cluster marker without replacing the user's own figure.

figure('Name', 'Synthetic response with FWER-controlled cluster marker');
axResponse = axes;
plot(axResponse, t, mean(responses, 1), 'LineWidth', 1.5);
hold(axResponse, 'on');
ylim(axResponse, [0 2]);  % Fixed for this synthetic example.

if ~isempty(clusters)
    clustme.plotClusterLines(axResponse, clusters, 'pvals', 'stars');
end

xline(axResponse, 0, ':');
xline(axResponse, groundTruth.peakTime, '--');

% The ground-truth marker is available only because this is a simulation; it
% is shown for orientation and is not used by ClustME.

labelY = 0.03;
text(axResponse, 0, labelY, 'Event', ...
    'Rotation', 90, ...
    'FontSize', 12, 'FontWeight', 'bold', ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
text(axResponse, groundTruth.peakTime, labelY, 'Ground truth', ...
    'Rotation', 90, ...
    'FontSize', 12, 'FontWeight', 'bold', ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');

ylabel(axResponse, 'Response amplitude');
xlabel(axResponse, 'Time (s)');
title(axResponse, 'Synthetic response with ClustME cluster marker');
grid(axResponse, 'on');

%% 5. Inspect the statistical t-map
% The t-map view shows the observed statistic and the empirical
% cluster-forming threshold. Candidate clusters are formed where the
% observed statistic exceeds this threshold.

clustme.Visualizer(vis_data, 'tmap', 't', t, ...
    'shadeClusters', false, ...
    'showClusterLevel', false, ...
    'showDirectionText', false);

%% 6. Record minimal provenance
% This short checklist records the main analysis choices used above. The
% toolbox and MATLAB versions are taken from mstats.Provenance.

fprintf('\nRun provenance:\n');
fprintf('  Formula:              %s\n', lmeFormula);
fprintf('  Tested coefficient:   Intercept\n');
fprintf('  Randomisation:        signFlip, permuteUnit = Subject\n');
fprintf('  numPerms / BqTarget:  1000 / 500\n');
fprintf('  Vmode:                global\n');
fprintf('  clusterMassMethod:    mean\n');
fprintf('  clusterSummaryMetric: signedPeak\n');
fprintf('  RNG seed before run:  1042\n');
fprintf('  ClustME version:      %s\n', string(mstats.Provenance.ClustMEVersion));
fprintf('  MATLAB version:       %s\n', string(mstats.Provenance.MATLABVersion));
fprintf('  Timestamp:            %s\n', string(mstats.Provenance.Timestamp));
