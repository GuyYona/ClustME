function [responses, designTable, groundTruth] = bench_generator(nSubjects, config, options)
% bench_generator - Synthetic dataset generator for ClustME validation
%
%   bench_generator creates hierarchical datasets with controlled 
%   signal-to-noise ratios (SNR), complex noise textures (1/f), and 
%   specific experimental designs. It is designed to rigorously validate 
%   cluster-based permutation algorithms against known ground truths.
%
% Syntax:
%   [responses, designTable, groundTruth] = clustme.bench_generator(nSubjects, config)
%   [responses, designTable, groundTruth] = clustme.bench_generator(nSubjects, config, options)
%
% Inputs:
%   nSubjects - [double] Number of subjects. Scalar for 'one-sample' and 
%               'within' designs. A 1×2 vector [N_GrpA, N_GrpB] for 'between'.
%   config    - [struct] Core experiment definition with fields:
%       .designType       [char] 'one-sample', 'within', or 'between'.
%       .nTrials          [double] Base trials per condition/subject. (Default: 30)
%       .targetSNR        [double] Target local SNR. Overrides effectSize. (Default: NaN)
%       .effectSize       [double] Peak amplitude of signal. (Default: 0)
%
% Options (Name-Value arguments):
%   .Fs                      [double] Sampling rate in Hz. (Default: 100)
%   .TimeRange               [1×2 double] [min max] epoch in seconds. (Default: [-0.2 0.8])
%   .RandomSeed              [double] Random seed for reproducibility. (Default: [])
%   .noiseMode               [char] 'gaussian' or 'complex' (1/f). (Default: 'gaussian')
%   .NoiseAlpha              [double] 1/f^alpha scaling (1.0 = Pink). (Default: 1.0)
%   .RampStrength            [double] End-of-epoch variance multiplier. (Default: 3.0)
%   .GroupNoiseScale         [1×2 double] Multipliers for residual noise. (Default: [1 1])
%   .signalWidth             [double] FWHM of Gaussian signal in sec. (Default: 0.05)
%   .signalTime              [double] Peak time of injection in sec. (Default: 0.4)
%   .eventTime               [double] Time of experimental event in sec. (Default: 0.0)
%   .SubjectVar              [double] Variance of random intercepts. (Default: 2.0)
%   .SubjectNoiseCV          [double] Coeff of variation for subject noise. (Default: 0.0)
%   .TrialCountCV            [double] CV for unbalanced trial counts. (Default: 0.0)
%   .EventNoiseSubjectSD     [double] SD of shared subject noise at event. (Default: 0.0)
%   .PostEventNoiseScale     [double] Multiplier for SD after eventTime. (Default: 1.0)
%   .PostEventBurstRate      [double] Bursts/sec post-event. (Default: 0.0)
%   .TrialNoiseDriftMeanStep [double] Mean fractional noise step per trial. (Default: 0.0)
%
% Outputs:
%   responses   - [N×T double] Generated data matrix (trials × time).
%   designTable - [N×3 table] Table with variables 'Subject', 'Trial', and 
%                 either 'Condition' or 'Group'.
%   groundTruth - [struct] Metadata detailing the exact injected parameters:
%       .signalVec       [1×T double] The noiseless signal vector.
%       .maskFWHM        [1×T logical] Mask defined by FWHM (>= 50% peak).
%       .maskExtent      [1×T logical] Mask for visual boundaries (>= 1% peak).
%       .peakTime        [double] Exact time of peak (seconds).
%       .FWHM            [double] Full Width at Half Maximum (seconds).
%       .amplitude       [double] Injected peak value (derived from SNR).
%       .sigmaGlobal     [double] RMS of noise across the entire epoch.
%       .sigmaLocal      [double] Noise SD at the time of injection.
%       .SNR             [double] Local SNR (EffectSize / sigmaLocal).
%       .peakIdx         [double] Sample index of the peak.
%       .boundsFWHM      [1×2 double] [Start End] indices of the FWHM window.
%       .boundsExtent    [1×2 double] [Start End] indices of the 1% visual window.
%       .signalCondition [char] Condition containing the signal ('B', 'Patient', 'all').
%
% BEHAVIOUR & INJECTION LOGIC
% ---------------------------
% • Spectral Synthesis: If noiseMode='complex', the function uses a custom 
%   Fast Fourier Transform (FFT) approach to generate true 1/f^alpha noise, 
%   followed by an optional non-stationary variance ramp.
% • Adaptive Targeting: If config.targetSNR is provided, the function dynamically 
%   calculates the local noise standard deviation at options.signalTime and 
%   forces the injected effect size to perfectly match the requested SNR.
% • Unbalanced Designs: Using TrialCountCV > 0 introduces realistic dropout 
%   rates, forcing ClustME to handle varying numbers of trials per subject.
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
        nSubjects          double {mustBePositive, mustBeInteger}
        config             struct

        options.noiseMode       (1,:) char {mustBeMember(options.noiseMode, {'gaussian','complex'})} = 'gaussian'
        options.Fs              (1,1) double {mustBePositive} = 100
        options.TimeRange       (1,2) double = [-0.2 0.8]
        options.RandomSeed      double = []

        % --- Unbalanced trial counts (dropout / QC rejection) ---
        options.TrialCountCV          (1,1) double {mustBeNonnegative} = 0.0
        options.MinTrialsPerSubject   (1,1) double {mustBePositive, mustBeInteger} = 10


        % Signal Morphometry
        options.signalWidth     (1,1) double = 0.05  % FWHM in seconds
        options.signalTime      (1,1) double = 0.4   % Peak time in seconds
        options.eventTime       (1,1) double = 0.0
        
        % Noise/Variance Parameters
        options.SubjectVar      (1,1) double = 2.0   % Variance of random intercepts
        options.SubjectNoiseCV  (1,1) double {mustBeNonnegative} = 0.0
        options.RampStrength    (1,1) double = 3.0   % End-of-epoch variance multiplier (if complex)
        options.NoiseAlpha      (1,1) double = 1.0   % 1/f^alpha (1.0 = Pink noise)
        % Event-coupled SUBJECT-shared noise (induces local trial covariance near event)
        options.EventNoiseSubjectSD   (1,1) double {mustBeNonnegative} = 0.0  % 0 = off
        options.EventNoiseWidth       (1,1) double {mustBePositive}    = 0.15 % FWHM (s)
        options.EventNoiseMode        (1,:) char {mustBeMember(options.EventNoiseMode, {'gaussian','postEventRise'})} = 'gaussian'
        options.EventNoiseTau         (1,1) double {mustBePositive} = 0.40   % seconds (used for postEventRise)
        

        % --- Optional post-event noise and burst parameters; inactive unless scale/rate/amplitude are increased ---
        options.PostEventNoiseScale   (1,1) double {mustBePositive} = 1.0      % PostEventNoiseScale multiplies the SD for t >= eventTime
        options.PostEventBurstRate    (1,1) double {mustBeNonnegative} = 0.0   % Burst process is active only for t >= eventTime, bursts/second per trial
        options.PostEventBurstWidth   (1,1) double {mustBePositive} = 0.05     % Burst shape: Gaussian with FWHM in seconds
        options.PostEventBurstAmpSD   (1,1) double {mustBeNonnegative} = 0.0   % Burst amplitude in units of the post-event SD 


        options.TrialNoiseDriftMeanStep (1,1) double {mustBeNonnegative} = 0.0  % mean fractional step per trial (0 = off)
        options.TrialNoiseDriftStepSD   (1,1) double {mustBeNonnegative} = 0.0  % SD of fractional step per trial
        options.TrialNoiseDriftNormaliseMean (1,1) logical = true               % keep mean multiplier ~= 1 per subject

        % Multipliers for residual noise in [GroupA, GroupB]. 
        % Use [1, 5] to simulate high noise in the second group.
        options.GroupNoiseScale (1,2) double {mustBePositive} = [1 1]
    end

    %% 1. Setup & Validation
 
    % Internal field extraction for config
    % Set defaults for config fields if missing
    if ~isfield(config, 'designType'), error('config.designType is required'); end
    if ~isfield(config, 'nTrials'),    config.nTrials = 30; end
    if ~isfield(config, 'effectSize'), config.effectSize = 0; end
    if ~isfield(config, 'targetSNR'),  config.targetSNR = NaN; end
    
    if ~isempty(options.RandomSeed)
        rng(options.RandomSeed);
    end

    % Validate nSubjects against Design Type
    if strcmp(config.designType, 'between')
        if numel(nSubjects) ~= 2
            error('bench_generator:InvalidSubjectCount', ...
                'For "between" design, nSubjects must be [N_GroupA, N_GroupB].');
        end
        nSubjA = nSubjects(1);
        nSubjB = nSubjects(2);
        totalSubjects = sum(nSubjects);
    else
        if numel(nSubjects) ~= 1
            error('bench_generator:InvalidSubjectCount', ...
                'For "%s" design, nSubjects must be a scalar integer.', config.designType);
        end
        nSubjA = nSubjects; % Treat as total N
        totalSubjects = nSubjects;
    end

    subjNoiseScale = ones(totalSubjects,1);

    if options.SubjectNoiseCV > 0
        % Convert desired linear CV to a lognormal parameter 
        % This gives positive multipliers with approximately the requested CV.
        sigma = sqrt(log(1 + options.SubjectNoiseCV^2));
        mu    = -0.5 * sigma^2;  % ensures E[exp(mu + sigma Z)] = 1
        subjNoiseScale = exp(mu + sigma * randn(totalSubjects,1));
    end

    % Trial counts per subject (supports unbalanced designs)
    mu = config.nTrials;
    if options.TrialCountCV > 0
        nTrialsBySubject = round(mu + options.TrialCountCV * mu * randn(totalSubjects,1));
        nTrialsBySubject = max(options.MinTrialsPerSubject, nTrialsBySubject);
    else
        nTrialsBySubject = repmat(mu, totalSubjects, 1);
    end

    % Total rows
    switch config.designType
        case 'one-sample'
            nTotalRows = sum(nTrialsBySubject);
        case 'within'
            nTotalRows = 2 * sum(nTrialsBySubject);
        case 'between'
            nTotalRows = sum(nTrialsBySubject);
    end

    % Optional monotonic across-trial residual-noise drift, 
    % generated from an isolated RNG so enabling drift 
    % does not change the noise draws.
    trialNoiseScale = cell(totalSubjects,1);

    rngState_drift = rng;
    if ~isempty(options.RandomSeed)
        rng(options.RandomSeed + 7919);  % deterministic, independent sub-seed
    end

    if (options.TrialNoiseDriftMeanStep > 0) || (options.TrialNoiseDriftStepSD > 0)
        for s = 1:totalSubjects
            nT = nTrialsBySubject(s);

            % Non-negative steps => monotone non-decreasing drift (skewed to increasing noise)
            steps = options.TrialNoiseDriftMeanStep + options.TrialNoiseDriftStepSD * randn(nT,1);
            steps = max(0, steps);

            drift = 1 + cumsum(steps);

            % Hard-coded safety clamps (avoid degeneracy)
            drift = max(drift, 0.05);
            drift = min(drift, 1e6);

            % Normalise mean so overall noise level is not inflated
            if options.TrialNoiseDriftNormaliseMean
                drift = drift / mean(drift);
            end

            trialNoiseScale{s} = drift;
        end
    else
        for s = 1:totalSubjects
            trialNoiseScale{s} = ones(nTrialsBySubject(s),1);
        end
    end

    rng(rngState_drift);

    % Time vector
    dt = 1 / options.Fs;
    
    % Note: The end time might be slightly adjusted to fit the integer sample count
    tVec = options.TimeRange(1) : dt : options.TimeRange(2); 
    T_samples = numel(tVec);
    [~, tIdx] = min(abs(tVec - options.signalTime)); 

    %% 2. Generate Noise Texture (Batch Mode)
    % We generate all noise at once for efficiency, then reshape/inject.
    
    if strcmp(options.noiseMode, 'gaussian')
        % Standard White Noise
        noiseBlock = randn(nTotalRows, T_samples);
    else
        % Complex: 1/f Spectral Synthesis + Non-Stationary Ramp
        noiseBlock = generate_pink_noise(nTotalRows, T_samples, options.NoiseAlpha);
        
        % Apply Variance Ramp (Non-Stationarity)
        % Linearly scale SD from 1.0 to sqrt(options.RampStrength)
        rampProfile = linspace(1, sqrt(options.RampStrength), T_samples);
        noiseBlock  = noiseBlock .* rampProfile;
    end

    % --- Post-event higher noise regime (constant SD step at eventTime) ---
    if options.PostEventNoiseScale ~= 1.0
        postMask = (tVec >= options.eventTime);
        if any(postMask)
            noiseBlock(:, postMask) = noiseBlock(:, postMask) * options.PostEventNoiseScale;
        end
    end


    %% 3. Generate Base Signal (Ground Truth, Adaptive amplitude)

    % Determine Local Noise Level at Injection Time
    sigmaLocal = std(noiseBlock(:, tIdx));

    % Determine Signal Amplitude
    if ~isnan(config.targetSNR)
        % Priority: Target SNR -> Calculate required amplitude
        amplitude = config.targetSNR * sigmaLocal;
    else
        % Fallback: Fixed Effect Size
        amplitude = config.effectSize;
    end

    % Gaussian bump: A * exp( - (t - t0)^2 / (2*sigma^2) )
    % Convert FWHM to sigma: sigma = FWHM / (2 * sqrt(2 * log(2)))

    if amplitude == 0
        rawSignal = zeros(1, T_samples);
    else
        sigma_s = options.signalWidth / (2 * sqrt(2 * log(2)));
        rawSignal = amplitude * exp(-((tVec - options.signalTime).^2) / (2 * sigma_s^2));
    end

    % Event-coupled subject noise (shared across trials of same subject)
    % Gaussian envelope centred at options.eventTime; 
    sigma_ev = options.EventNoiseWidth / (2 * sqrt(2 * log(2)));
    eventEnv = exp(-((tVec - options.eventTime).^2) / (2 * sigma_ev^2));   % 1 x T (event-centred)


    % Per-subject amplitudes for the shared event noise component
    subjEventAmp = randn(totalSubjects, 1) * options.EventNoiseSubjectSD;  % Nsubj x 1


    %% 4. Prepare Data Containers
    
    dataMatrix = zeros(nTotalRows, T_samples);
    subjList   = cell(nTotalRows, 1);
    condList   = cell(nTotalRows, 1);
    trialList  = zeros(nTotalRows, 1);
    
  
    %% 5. Assemble Data (Hierarchical Injection)
    
    row = 0;
    
    % --- Random Intercept Generation ---
    % Generate subject-level offsets (Gaussian, mean 0, var=SubjectVar)
    subjOffsets = randn(totalSubjects, 1) * sqrt(options.SubjectVar);

    % --- Design Loop ---
    switch config.designType
        
        case 'one-sample'
            % All trials get Signal (if EffectSize > 0)
            for s = 1:totalSubjects
                sID = sprintf('S%03d', s);
                offset = subjOffsets(s);
                
                for t = 1:nTrialsBySubject(s)
                    row = row + 1;
                    % Add Signal + Noise + Random Intercept
                    baseTrace = noiseBlock(row, :) * subjNoiseScale(s) * trialNoiseScale{s}(t);
                    burstVec  = generate_post_event_bursts(tVec, options.eventTime, options.PostEventBurstRate, options.PostEventBurstWidth, options.PostEventBurstAmpSD, baseTrace);
                    dataMatrix(row, :) = baseTrace + rawSignal + offset + subjEventAmp(s) * eventEnv + burstVec;

                    subjList{row}  = sID;
                    condList{row}  = 'OneSample';
                    trialList(row) = t;
                end
            end
            
        case 'within'
            % Condition A = Noise, Condition B = Signal
            % Correlated: Same subject offset for both conditions
            for s = 1:totalSubjects
                sID = sprintf('S%03d', s);
                offset = subjOffsets(s);
                
                for t = 1:nTrialsBySubject(s)
                    % Cond A (Noise)
                    row = row + 1;
                    baseTrace = noiseBlock(row, :) * subjNoiseScale(s) * trialNoiseScale{s}(t);
                    burstVec  = generate_post_event_bursts(tVec, options.eventTime, options.PostEventBurstRate, options.PostEventBurstWidth, options.PostEventBurstAmpSD, baseTrace);
                    dataMatrix(row, :) = baseTrace + offset + subjEventAmp(s) * eventEnv + burstVec; % No Signal

                    subjList{row} = sID; condList{row} = 'A'; trialList(row) = t;
                    
                    % Cond B (Signal)
                    row = row + 1;
                    dataMatrix(row, :) = noiseBlock(row, :) * subjNoiseScale(s) * trialNoiseScale{s}(t) + rawSignal + offset + subjEventAmp(s) * eventEnv;

                    subjList{row} = sID; condList{row} = 'B'; trialList(row) = t;
                end
            end
            
 case 'between'
            % Group A (Control/Noise) vs Group B (Patient/Signal)
            
            % Process Group A (Control)
            scaleA = options.GroupNoiseScale(1); % Extract scale
            
            for s = 1:nSubjA
                sID = sprintf('C%03d', s);
                offset = subjOffsets(s); 
                
                for t = 1:nTrialsBySubject(s)
                    row = row + 1;
                    % Apply ScaleA to the noise block
                    scaledNoise = noiseBlock(row, :) * scaleA * subjNoiseScale(s) * trialNoiseScale{s}(t);
                    burstVec = generate_post_event_bursts(tVec, options.eventTime, options.PostEventBurstRate, options.PostEventBurstWidth, options.PostEventBurstAmpSD, scaledNoise);
                    dataMatrix(row, :) = scaledNoise + offset + subjEventAmp(s) * eventEnv + burstVec; % No Signal
                    subjList{row} = sID; condList{row} = 'Control'; trialList(row) = t;
                end
            end
            
            % Process Group B (Patient)
            scaleB = options.GroupNoiseScale(2); % Extract scale
            
            for s = 1:nSubjB
                sID = sprintf('P%03d', s);
                subjIdx = nSubjA + s;
                offset = subjOffsets(subjIdx);
                
                for t = 1:nTrialsBySubject(subjIdx)
                    row = row + 1;
                    % Apply ScaleB to the noise block
                    scaledNoise = noiseBlock(row, :) * scaleB * subjNoiseScale(subjIdx) * trialNoiseScale{subjIdx}(t);
                    burstVec = generate_post_event_bursts(tVec, options.eventTime, options.PostEventBurstRate, options.PostEventBurstWidth, options.PostEventBurstAmpSD, scaledNoise);
                    dataMatrix(row, :) = scaledNoise + rawSignal + offset + subjEventAmp(subjIdx) * eventEnv + burstVec;
                    subjList{row} = sID; condList{row} = 'Patient'; trialList(row) = t;
                end
            end
    end

    %% 6. Output Packaging
    
    responses = dataMatrix;
    
    % Determine variable name for Condition column based on design
    if strcmp(config.designType, 'between')
        varName = 'Group';
    else
        varName = 'Condition';
    end
    
    designTable = table(categorical(condList), categorical(subjList), trialList, ...
        'VariableNames', {varName, 'Subject', 'Trial'});
    
    % Ground Truth Metadata 
    groundTruth = struct();

    % Metadata: Which condition contains the signal?
    switch config.designType
        case 'within',      groundTruth.signalCondition = 'B';
        case 'between',     groundTruth.signalCondition = 'Patient';
        case 'one-sample',  groundTruth.signalCondition = 'all';
    end

    groundTruth.signalVec = rawSignal;

    % Helper to extract bounds [Start, End] from a logical mask
    getBounds = @(m) [find(m,1,'first'), find(m,1,'last')];

    % Define Mask Logic to be Sign-Agnostic (Magnitude-based)
    thresholdFWHM   = 0.50 * abs(amplitude);
    thresholdExtent = 0.01 * abs(amplitude);
    if abs(amplitude) > 0
        groundTruth.maskFWHM     = abs(rawSignal) >= thresholdFWHM;
        groundTruth.maskExtent   = abs(rawSignal) >= thresholdExtent;
        groundTruth.boundsFWHM   = getBounds(groundTruth.maskFWHM);
        groundTruth.boundsExtent = getBounds(groundTruth.maskExtent);
    else
        groundTruth.maskFWHM     = false(size(rawSignal));
        groundTruth.maskExtent   = false(size(rawSignal));
        groundTruth.boundsFWHM   = [];
        groundTruth.boundsExtent = [];
    end

    groundTruth.peakTime  = options.signalTime;
    groundTruth.FWHM      = options.signalWidth;
    groundTruth.Fs        = options.Fs;
    groundTruth.tVec      = tVec;
    groundTruth.peakIdx   = tIdx;
    groundTruth.nTrialsBySubject = nTrialsBySubject;
    groundTruth.subjNoiseScale = subjNoiseScale;
    groundTruth.SubjectNoiseCV = options.SubjectNoiseCV;
    groundTruth.eventTime = options.eventTime;
    groundTruth.trialNoiseScale = trialNoiseScale;
    groundTruth.TrialNoiseDriftMeanStep = options.TrialNoiseDriftMeanStep;
    groundTruth.TrialNoiseDriftStepSD   = options.TrialNoiseDriftStepSD;

    % --- SNR Calculation (Local vs Global Difficulty) ---

    % Global noise level across the entire epoch for reference
    groundTruth.sigmaGlobal = std(noiseBlock(:));
    
    % Report the noise standard deviation at the exact time of the injection site
    groundTruth.amplitude  = amplitude;   % The actual injected peak value
    groundTruth.sigmaLocal = sigmaLocal;
    
    % Effective SNR reflects the true local task difficulty
    if sigmaLocal > 0
        groundTruth.SNR = amplitude / sigmaLocal;
    else
        groundTruth.SNR = Inf;
    end

end

%% HELPER FUNCTIONS %%
 
function noise = generate_pink_noise(nRows, nSamples, alpha)
% generate_pink_noise - Generates 1/f^alpha noise using spectral synthesis
%
% Internal Context:
%   A helper function for bench_generator.m to create autocorrelated noise 
%   traces (e.g., pink noise) when the 'complex' noise mode is selected.
%
% Inputs:
%   nRows    - [double] Number of independent trial traces to generate.
%   nSamples - [double] Number of time samples per trace.
%   alpha    - [double] Spectral power-law exponent (e.g., 1.0 for pink noise, 
%              0.0 for white noise).
%
% Outputs:
%   noise    - [nRows × nSamples double] Matrix of generated noise traces.
%
% Algorithmic & Exactness Notes:
%   * DC Singularity Guard: The 0 Hz (DC) component scaling is hardcoded to 0 
%     to prevent a 1/f division-by-zero singularity and enforce a zero-mean baseline.
%   * Hermitian Symmetry: Dynamically constructs a conjugate-symmetric frequency 
%     spectrum for both even and odd sample lengths, and applies real(ifft(...)) 
%     to guarantee strictly real-domain outputs.
%   * Row-Wise Standardisation: Normalises the empirical standard deviation of 
%     each individual trace (row) to exactly 1.0. This is structurally required 
%     so downstream Signal-to-Noise Ratio (SNR) injections scale predictably.
    
    % 1. Define Frequency Grid
    % Only need positive frequencies for real signal synthesis
    if mod(nSamples, 2) == 0
        numFreqs = nSamples/2 + 1;
    else
        numFreqs = (nSamples+1)/2;
    end
    
    % 2. Generate Random Phase (White Noise in Freq Domain)
    % Random complex numbers: exp(i * phase)
    phases = rand(nRows, numFreqs) * 2 * pi;
    spect  = complex(cos(phases), sin(phases));
    
    % 3. Scale Amplitudes by 1/f^alpha
    % Frequencies: 0, 1, ..., Nyquist
    freqs = (0:numFreqs-1); 
    
    % Avoid singularity at DC (f=0): set scaling to 0 or 1.
    % We typically set DC=0 to center the noise, then add intercepts later.
    scaling = ones(1, numFreqs);
    scaling(2:end) = (1 ./ freqs(2:end)) .^ (alpha/2); % Power spectrum is 1/f^a, so amp is 1/f^(a/2)
    scaling(1) = 0; % Zero mean
    
    % Apply scaling
    spect = spect .* scaling;
    
    % 4. Construct Full Spectrum (Hermitian Symmetry for Real IFFT)
    if mod(nSamples, 2) == 0
        % Even length: DC, pos, Nyquist, neg
        % spect has: [DC, 1...N/2-1, Nyquist]
        % reconstruct: [DC, pos, Nyquist, conj(flip(pos))]
        specFull = [spect, conj(spect(:, end-1:-1:2))];
    else
        % Odd length: DC, pos, neg
        specFull = [spect, conj(spect(:, end:-1:2))];
    end
    
    % 5. IFFT to Time Domain
    noise = real(ifft(specFull, [], 2));
    
    % 6. Normalize to Unit Variance
    % (Standardize so effect size math is consistent)
    targetSTD = 1.0;
    currentSTD = std(noise, 0, 2);
    noise = noise ./ currentSTD * targetSTD;

end

function burstVec = generate_post_event_bursts(tVec, eventTime, rateHz, fwhm, ampSD, baseTrace)
% generate_post_event_bursts - Synthesises trial-specific sparse Gaussian bursts in the post-event epoch
%
% Internal Context:
%   A local helper function executed inside the hierarchical generation loops 
%   of bench_generator.m. It injects high-frequency, transient artefacts 
%   (bursts) that occur exclusively after the experimental event. This simulates 
%   realistic trial-to-trial heterogeneity and non-stationary noise topologies.
%
% Inputs:
%   tVec      - [1 × T double] Continuous time vector for the epoch.
%   eventTime - [double] Temporal onset of the experimental event (in seconds).
%   rateHz    - [double] Mean occurrence rate of bursts per second.
%   fwhm      - [double] Full Width at Half Maximum (FWHM) of each burst (in seconds).
%   ampSD     - [double] Standard deviation multiplier for burst amplitudes.
%   baseTrace - [1 × T double] The generated baseline noise trace for the current trial.
%
% Outputs:
%   burstVec  - [1 × T double] An isolated vector containing only the synthesised bursts.
%
% Algorithmic & Exactness Notes:
%   * Localised Variance Tethering: Burst amplitudes are strictly scaled against 
%     the standard deviation of the current trial's specific post-event baseline, 
%     ensuring artefacts scale proportionally to the local noise regime rather 
%     than the global variance.
%   * Temporal & Stochastic Bounding: Artefact injection is strictly confined 
%     to the post-event window. The total burst count is drawn from a Poisson 
%     distribution proportional to the available post-event duration.

    T = numel(tVec);
    burstVec = zeros(1, T);

    if rateHz <= 0 || ampSD <= 0
        return;
    end

    postMask = (tVec >= eventTime);
    if ~any(postMask)
        return;
    end

    % Duration of post-event region in seconds
    postDur = tVec(find(postMask, 1, 'last')) - eventTime;
    if postDur <= 0
        return;
    end

    % Reference SD from the post-event portion of this trial's baseline trace
    sigmaRef = std(baseTrace(postMask));
    if ~isfinite(sigmaRef) || sigmaRef <= 0
        sigmaRef = 1.0;
    end

    % Number of bursts: Poisson(rateHz * postDur)
    nBursts = poissrnd(rateHz * postDur);
    if nBursts <= 0
        return;
    end

    sigma_b = fwhm / (2 * sqrt(2 * log(2)));

    for k = 1:nBursts
        t0  = eventTime + rand() * postDur;           % uniform in post-event region
        amp = randn() * ampSD * sigmaRef;             % amplitude in SD units
        burstVec = burstVec + amp * exp(-((tVec - t0).^2) / (2 * sigma_b^2));
    end
end
