classdef TestBenchGenerator < matlab.unittest.TestCase
    % TestBenchGenerator - Software integrity and unit test suite for bench_generator
    %
    %   TestBenchGenerator verifies the software architecture and synthesis
    %   mechanics of the ClustME synthetic dataset generator. This suite ensures the 
    %   generator correctly parses configurations, handles edge cases (e.g., negative signals),
    %   enforces group boundaries, and accurately scales spectral noise profiles.
    %
    % How to Run:
    %   results = runtests('validation/TestBenchGenerator.m');
    %
    % Test Inventory:
    %   1. testDefaults_AreApplied       - Verifies fallback to default configurations.
    %   2. testOneSample_TableSemantics  - Checks trial and subject column generation.
    %   3. testTargetSNR_IsConsistent    - Confirms dynamic amplitude scaling via local noise.
    %   4. testWithin_ConditionsAndSignalInB - Verifies signal localization in 'B' trials.
    %   5. testBetween_GroupColumnAndSignalInPatient - Verifies signal in 'Patient' trials.
    %   6. testBetween_GroupNoiseScale_ScalesWithinRowStd - Tests heteroscedastic noise.
    %   7. testComplexNoise_RampStrength_IncreasesLateAcrossRowSD - Verifies variance ramps.
    %   8. testComplexNoise_NoiseAlpha_IncreasesLowFreqPower - Confirms 1/f spectral scaling.
    %   9. testFWHMBounds_ApproxMatchesSignalWidth - Checks time-domain bounding box math.
    %  10. testInvalidSubjectCount_Throws - Confirms input guards for group sizes.
    %  11. testNegativeEffectSize_GeneratesValidMasks - Verifies mask logic for negative peaks.
    %  12. testSignalAbsentInNoiseCondition - Ensures control groups have zero injected signal.
    %
    % Dependencies:
    %   - MATLAB R2019b or newer.
    %   - Part of the ClustME Validation Suite.
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

    methods (Test)

        function testDefaults_AreApplied(tc)
            % Only provide required config field (designType).
            cfg = struct('designType', 'one-sample');

            [responses, designTable, gt] = clustme.bench_generator(3, cfg);

            data = tc.getDataMatrix(responses);

            % Defaults from arguments block
            tc.verifyEqual(gt.Fs, 100);
            tc.verifyEqual(gt.peakTime, 0.4);
            tc.verifyEqual(gt.FWHM, 0.05);
            tc.verifyEqual(gt.tVec(1), -0.2, "AbsTol", 1e-10);
            tc.verifyEqual(gt.tVec(end), 0.8, "AbsTol", 1e-10);
            tc.verifyEqual(mean(diff(gt.tVec)), 1/gt.Fs, "AbsTol", 1e-10);

            % Default config: nTrials=30, effectSize=0, targetSNR=NaN -> null signal
            tc.verifySize(data, [3*30, numel(gt.tVec)]);
            tc.verifyEqual(height(designTable), 3*30);

            tc.verifyEqual(designTable.Properties.VariableNames, {'Condition','Subject','Trial'});
            tc.verifyEqual(categories(designTable.Condition), {'OneSample'});
            tc.verifyEqual(gt.signalCondition, 'all');

            tc.verifyEqual(gt.amplitude, 0);
            tc.verifyFalse(any(gt.maskFWHM));
            tc.verifyFalse(any(gt.maskExtent));
            tc.verifyEmpty(gt.boundsFWHM);
            tc.verifyEmpty(gt.boundsExtent);
            tc.verifyEqual(gt.SNR, 0, "AbsTol", 1e-12);
        end

        function testOneSample_TableSemantics(tc)
            cfg = struct('designType','one-sample', 'nTrials', 5);

            [responses, designTable, gt] = clustme.bench_generator(4, cfg, RandomSeed=1);

            data = tc.getDataMatrix(responses);
            tc.verifySize(data, [4*5, numel(gt.tVec)]);

            tc.verifyEqual(designTable.Properties.VariableNames, {'Condition','Subject','Trial'});
            tc.verifyEqual(sort(categories(designTable.Condition)), {'OneSample'}');

            % Trial column must be numeric and run 1..nTrials
            tc.verifyClass(designTable.Trial, 'double');
            tc.verifyEqual(unique(designTable.Trial).', 1:cfg.nTrials);

            % Each subject appears exactly nTrials rows
            subjCounts = countcats(designTable.Subject);
            tc.verifyEqual(subjCounts(:), repmat(cfg.nTrials, numel(subjCounts), 1));
        end

        function testTargetSNR_IsConsistent(tc)
            cfg = struct('designType','one-sample', 'nTrials', 10, 'targetSNR', 2.0);

            [responses, ~, gt] = clustme.bench_generator(20, cfg, RandomSeed=42);
            %#ok<NASGU>
            tc.getDataMatrix(responses);

            tc.verifyGreaterThan(gt.sigmaLocal, 0);
            tc.verifyEqual(gt.SNR, cfg.targetSNR, "AbsTol", 1e-10);
            tc.verifyEqual(gt.amplitude, cfg.targetSNR * gt.sigmaLocal, "AbsTol", 1e-10);

            tc.verifyEqual(gt.peakIdx, find(abs(gt.tVec - gt.peakTime) == min(abs(gt.tVec - gt.peakTime)), 1));
            tc.verifyTrue(gt.maskFWHM(gt.peakIdx));
            tc.verifyTrue(gt.maskExtent(gt.peakIdx));
            tc.verifyEqual(numel(gt.boundsFWHM), 2);
            tc.verifyEqual(numel(gt.boundsExtent), 2);
        end

        function testWithin_ConditionsAndSignalInB(tc)
            cfg = struct('designType','within', 'nTrials', 8, 'effectSize', 1.0);

            % Use seed for determinism; no need to override any default options otherwise
            [responses, designTable, gt] = clustme.bench_generator(30, cfg, RandomSeed=7);

            data = tc.getDataMatrix(responses);
            tc.verifySize(data, [30*cfg.nTrials*2, numel(gt.tVec)]);

            tc.verifyEqual(designTable.Properties.VariableNames, {'Condition','Subject','Trial'});
            tc.verifyEqual(sort(categories(designTable.Condition)), {'A';'B'});

            tc.verifyEqual(gt.signalCondition, 'B');

            % Strong check at the peak: mean(B) > mean(A)
            idxPeak = gt.peakIdx;
            rowsA = designTable.Condition == 'A';
            rowsB = designTable.Condition == 'B';
            tc.verifyGreaterThan(mean(data(rowsB, idxPeak)) - mean(data(rowsA, idxPeak)), 0.25*gt.amplitude);

            % Per subject: exactly nTrials of A and nTrials of B
            subs = categories(designTable.Subject);
            for i = 1:numel(subs)
                sMask = designTable.Subject == subs{i};
                tc.verifyEqual(sum(sMask & rowsA), cfg.nTrials);
                tc.verifyEqual(sum(sMask & rowsB), cfg.nTrials);
            end
        end

        function testBetween_GroupColumnAndSignalInPatient(tc)
            cfg = struct('designType','between', 'nTrials', 6, 'effectSize', 1.0);

            % Set SubjectVar=0 to avoid random intercept imbalances between groups
            [responses, designTable, gt] = clustme.bench_generator([25 20], cfg, SubjectVar=0, RandomSeed=11);

            data = tc.getDataMatrix(responses);
            tc.verifySize(data, [(25+20)*cfg.nTrials, numel(gt.tVec)]);

            tc.verifyEqual(designTable.Properties.VariableNames, {'Group','Subject','Trial'});
            tc.verifyEqual(sort(categories(designTable.Group)), {'Control';'Patient'});

            tc.verifyEqual(gt.signalCondition, 'Patient');

            idxPeak = gt.peakIdx;
            rowsC = designTable.Group == 'Control';
            rowsP = designTable.Group == 'Patient';
            tc.verifyGreaterThan(mean(data(rowsP, idxPeak)) - mean(data(rowsC, idxPeak)), 0.25*gt.amplitude);
        end

        function testBetween_GroupNoiseScale_ScalesWithinRowStd(tc)
            cfg = struct('designType','between', 'nTrials', 2, 'effectSize', 0);

            [responses, designTable, gt] = clustme.bench_generator([50 50], cfg, GroupNoiseScale=[1 5], RandomSeed=123);

            data = tc.getDataMatrix(responses);

            % within-row std across time (offsets don't matter; they're constant over time)
            rowStd = std(data, 0, 2);
            stdC = median(rowStd(designTable.Group == 'Control'));
            stdP = median(rowStd(designTable.Group == 'Patient'));

            tc.verifyGreaterThan(stdP/stdC, 4.0);
            tc.verifyLessThan(stdP/stdC, 6.5);

            tc.verifyEqual(gt.amplitude, 0);
            tc.verifyFalse(any(gt.maskFWHM));
        end

        function testComplexNoise_RampStrength_IncreasesLateAcrossRowSD(tc)
            cfg = struct('designType','one-sample', 'nTrials', 10, 'effectSize', 0);

            % Must set SubjectVar=0 here; otherwise random intercepts dominate across-row SD.
            [responses, ~, ~] = clustme.bench_generator(60, cfg, ...
                noiseMode='complex', RampStrength=9.0, NoiseAlpha=1.0, SubjectVar=0, RandomSeed=77);

            data = tc.getDataMatrix(responses);

            earlySD = std(data(:, 1));
            lateSD  = std(data(:, end));

            % Expect ~sqrt(9)=3x; allow slack for finite sample fluctuations
            tc.verifyGreaterThan(lateSD/earlySD, 2.0);
            tc.verifyLessThan(lateSD/earlySD, 4.5);
        end

        function testComplexNoise_NoiseAlpha_IncreasesLowFreqPower(tc)
            cfg = struct('designType','one-sample', 'nTrials', 5, 'effectSize', 0);

            [r0, ~, ~] = clustme.bench_generator(40, cfg, ...
                noiseMode='complex', RampStrength=1.0, NoiseAlpha=0.0, SubjectVar=0, RandomSeed=202);

            [r2, ~, ~] = clustme.bench_generator(40, cfg, ...
                noiseMode='complex', RampStrength=1.0, NoiseAlpha=2.0, SubjectVar=0, RandomSeed=202);

            x0 = tc.getDataMatrix(r0);
            x2 = tc.getDataMatrix(r2);

            ratio0 = tc.lowHighBandPowerRatio(x0);
            ratio2 = tc.lowHighBandPowerRatio(x2);

            tc.verifyGreaterThan(ratio2, ratio0 * 1.2);
        end

        function testFWHMBounds_ApproxMatchesSignalWidth(tc)
            cfg = struct('designType','one-sample', 'nTrials', 3, 'effectSize', 1.0);

            % Use higher Fs for better discretisation (this test targets Fs/signal width behaviour)
            [~, ~, gt] = clustme.bench_generator(30, cfg, Fs=200, RandomSeed=9);

            tc.verifyEqual(gt.Fs, 200);
            dt = 1/gt.Fs;

            b = gt.boundsFWHM;
            tc.verifyEqual(numel(b), 2);

            widthSec = (b(2) - b(1)) * dt;
            tc.verifyEqual(widthSec, gt.FWHM, "AbsTol", 2*dt);
            tc.verifyTrue(gt.maskFWHM(gt.peakIdx));
        end

        function testInvalidSubjectCount_Throws(tc)
            cfgBetween = struct('designType','between');
            tc.verifyError(@() clustme.bench_generator(5, cfgBetween), ...
                'bench_generator:InvalidSubjectCount');

            cfgWithin = struct('designType','within');
            tc.verifyError(@() clustme.bench_generator([3 3], cfgWithin), ...
                'bench_generator:InvalidSubjectCount');
        end

        function testNegativeEffectSize_GeneratesValidMasks(tc)
            % Critical for detecting "dips"
            cfg = struct('designType','one-sample', 'nTrials', 5, 'effectSize', -2.0);

            [responses, ~, gt] = clustme.bench_generator(10, cfg, RandomSeed=101);

            % 1. Verify negative peak amplitude
            % Peak should be approximately -2.0
            tc.verifyEqual(min(gt.signalVec), -2.0, "AbsTol", 1e-5);

            % 2. Verify zero-valued baseline
            tc.verifyEqual(max(gt.signalVec), 0, "AbsTol", 1e-12);

            % 3. Verify FWHM mask generation
            tc.verifyTrue(any(gt.maskFWHM), 'Masks must be generated for negative signals');
            tc.verifyTrue(gt.maskFWHM(gt.peakIdx), 'Peak index must be inside the mask');

            % 4. Verify non-empty FWHM bounds
            tc.verifyNotEmpty(gt.boundsFWHM);
        end

        function testSignalAbsentInNoiseCondition(tc)
            % Verify that 'Condition A' (Within) or 'Control' (Between) is truly empty of signal
            cfg = struct('designType','within', 'nTrials', 20, 'effectSize', 5.0); % Huge signal
            
            [responses, designTable, gt] = clustme.bench_generator(10, cfg, RandomSeed=102);
            data = tc.getDataMatrix(responses);
            
            rowsA = designTable.Condition == 'A';
            dataA = data(rowsA, :);
            
            % Check correlation with signal vector in noise trials
            % Should be near zero (random), definitely not 1.0
            sigCorr = corr(mean(dataA)', gt.signalVec');
            
            tc.verifyLessThan(abs(sigCorr), 0.3, 'Noise condition has high correlation with signal!');
        end
    end

    methods (Static, Access = private)

        function data = getDataMatrix(responses)
            if ~isnumeric(responses)
                error('Expected responses to be a numeric data matrix.');
            end
            data = responses;
        end

        function ratio = lowHighBandPowerRatio(x)
            % Robust low/high power ratio via FFT.
            % x: (nRows x T)
            if size(x,1) > 20
                x = x(1:20, :);
            end
            x = x - mean(x, 2); % remove DC

            T = size(x,2);
            X = fft(x, [], 2);
            P = abs(X).^2;

            kMax = floor(T/2);
            P1 = P(:, 2:kMax); % exclude DC

            nBins = size(P1,2);
            nBand = max(5, floor(0.10 * nBins));

            Plow  = mean(P1(:, 1:nBand), 2);
            Phigh = mean(P1(:, end-nBand+1:end), 2);

            ratio = mean(Plow ./ Phigh);
        end
    end
end
