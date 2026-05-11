classdef TestClustME < matlab.unittest.TestCase
    % TestClustME - Software integrity and unit test suite for the ClustME engine
    %
    %   TestClustME verifies the software architecture and functional mechanics
    %   of the ClustME toolbox. This suite ensures the
    %   code executes correctly, generates exact mathematical sets, survives
    %   edge cases (NaNs, N=1), and catches illegal configurations.
    %
    % How to Run:
    %   results = runtests('validation/TestClustME.m')
    %
    % Test Inventory:
    %   1. testNull_Intercept_SignFlip       - Executes 1-sample null using SignFlip.
    %   2. testNull_Condition_WithinSubject  - Executes 2-sample null using withinSubject.
    %   3. testSensitivity_Condition_Within  - Checks temporal detection of a known signal.
    %   4. testSensitivity_Intercept_SignFlip- Checks detection on subset data.
    %   5. testOutputArchitecture            - Verifies output struct fields decoupling.
    %   6. testBetweenSubject_GroupLabel     - Tests unbalanced between-subject designs.
    %   7. test_ExactPermutations_N6         - Verifies exhaustive math (N=6 -> 31 or 719).
    %   8. test_ConfigurationGuards          - Tests hard-fails (e.g., missing whitening).
    %   9. testIntegrity_MissingData         - Ensures survival against sparse NaNs.
    %  10. testIntegrity_SingleSubject       - Verifies error on N=1 limits.
    %  11. testROIMode_Parametric        - Verifies fixed-window extraction and parametric thresholding.
    %
    % Dependencies:
    %   - MATLAB R2019b or newer.
    %   - Statistics and Machine Learning Toolbox.
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
    
    properties
        % Common simulation parameters
        Fs = 100
        TVec
        LMEFormula = 'response ~ 1 + Condition + (1|Subject)'
        
        % Data containers
        ResponsesNull
        ResponsesSignal
        DesignTable     

        % --- Between-Subject Data Containers ---
        ResponsesBetween
        DesignBetween
        FormulaBetween = 'response ~ 1 + Group + (1|Subject)'
        
        % Simulation config
        nSubjects = 20
        nTrialsPerCond = 30
    end
    
    methods (TestClassSetup)
      function generateSimulationData(testCase)
            % Generate shared synthetic datasets for the test class using bench_generator.
            
            % 1. Config for Within-Subject Data
            configWithin.designType = 'within';
            configWithin.nTrials    = testCase.nTrialsPerCond;
            
            % A) Generate Null Data (Effect = 0)
            configWithin.effectSize = 0; 
            [respNull, dtNull, gtNull] = clustme.bench_generator(testCase.nSubjects, configWithin, ...
                'Fs', testCase.Fs, 'noiseMode', 'complex', 'RampStrength', 3.0, 'RandomSeed', 42);
            
            testCase.ResponsesNull = respNull;
            testCase.DesignTable   = dtNull;
            testCase.TVec          = gtNull.tVec; % Sync time vector
            
            % B) Generate Signal Data (Effect = 2.0)
            % Note: We reuse the design structure since N/Trials are identical
            configWithin.effectSize = 2.0;
            [respSig, ~, ~] = clustme.bench_generator(testCase.nSubjects, configWithin, ...
                'Fs', testCase.Fs, 'noiseMode', 'complex', 'RampStrength', 3.0, ...
                'signalTime', 0.4, 'signalWidth', 0.05, 'RandomSeed', 42);
            
            testCase.ResponsesSignal = respSig;
            
            % 2. Config for Between-Subject Data (Unbalanced 12 vs 8)
            configBetween.designType = 'between';
            configBetween.nTrials    = 20;
            configBetween.effectSize = 3.0;
            
            [respBetw, dtBetw, ~] = clustme.bench_generator([12 8], configBetween, ...
                'Fs', testCase.Fs, 'noiseMode', 'gaussian', ...
                'signalTime', 0.2, 'RandomSeed', 99);
                
            testCase.ResponsesBetween = respBetw;
            testCase.DesignBetween    = dtBetw;
        end
    end
    
    methods (Test)

        function testNull_Intercept_SignFlip(testCase)
            % [TEST A] Null Check: One-Sample Intercept using Sign-Flip
            fprintf('\n--- Test: Null (Intercept / SignFlip) ---\n');

            formIntercept = 'response ~ 1 + (1|Subject)';

            [clusters, ~, ~] = ClustME(...
                testCase.ResponsesNull, testCase.DesignTable, formIntercept, ...
                'Fs', testCase.Fs, 't', testCase.TVec, ...
                'permuteUnit', 'Subject', ...
                'permutationMethod', 'signFlip', ...    % Explicit SignFlip
                'numPerms', 100, ...
                'whitening', true, ...
                'testCoefficient', '', ...              % Target Intercept
                'minClusterSize', 50);

            % Assertion: No significant clusters in noise
            if ~isempty(clusters)
                pVals = [clusters.p_value];
                testCase.verifyTrue(all(pVals >= 0.05), ...
                    sprintf('False positive Intercept cluster (p=%.4f)', min(pVals)));
            end
        end

        function testNull_Condition_WithinSubject(testCase)
            % [TEST A2] Null Check: Within-Subject Condition using withinSubject
            fprintf('\n--- Test: Null (Condition / withinSubject) ---\n');

            [clusters, ~, ~] = ClustME(...
                testCase.ResponsesNull, testCase.DesignTable, testCase.LMEFormula, ...
                'Fs', testCase.Fs, 't', testCase.TVec, ...
                'permuteUnit', 'Subject', ...
                'permutationMethod', 'withinSubject', ...     % Explicit withinSubject
                'numPerms', 100, ...
                'whitening', true, ...
                'testCoefficient', 'Condition_B', ...   % Target Condition
                'minClusterSize', 50);

            % Assertion: No significant clusters in noise
            if ~isempty(clusters)
                pVals = [clusters.p_value];
                testCase.verifyTrue(all(pVals >= 0.05), ...
                    sprintf('False positive Condition cluster (p=%.4f)', min(pVals)));
            end
        end

        function testSensitivity_Condition_WithinSubject(testCase)
            % [TEST B2] Sensitivity: Two-Sample Condition using withinSubject
            fprintf('\n--- Test: Sensitivity (Condition / withinSubject) ---\n');

            [clusters, ~, ~] = ClustME(...
                testCase.ResponsesSignal, testCase.DesignTable, testCase.LMEFormula, ...
                'Fs', testCase.Fs, 't', testCase.TVec, ...
                'permuteUnit', 'Subject', ...
                'permutationMethod', 'withinSubject', ...     % Explicit withinSubject
                'numPerms', 100, ...
                'whitening', true, ...
                'testCoefficient', 'Condition_B', ...
                'minClusterSize', 50);

            % 1. Verify significance
            sigClusters = clusters([clusters.p_value] < 0.05);
            testCase.verifyNotEmpty(sigClusters, 'Failed to detect Condition signal.');

            % 2. Verify temporal accuracy (Signal at 0.4s)
            foundTarget = false;
            for k = 1:numel(sigClusters)
                tCent = mean(testCase.TVec(sigClusters(k).start : sigClusters(k).end));
                if abs(tCent - 0.4) < 0.1, foundTarget = true; end
            end
            testCase.verifyTrue(foundTarget, 'Signal detected, but timing is wrong (expected ~0.4s).');
        end

        function testSensitivity_Intercept_SignFlip(testCase)
            % [TEST B] Sensitivity: One-Sample Intercept on Cond B Subset
            fprintf('\n--- Test: Sensitivity (Intercept / SignFlip / Subset) ---\n');

            % 1. Subset Data (Extract only Condition B)
            rowsB    = testCase.DesignTable.Condition == 'B';
            designB  = testCase.DesignTable(rowsB, :);

            respB = testCase.ResponsesSignal(rowsB, :);

            formIntercept = 'response ~ 1 + (1|Subject)';

            [clusters, ~, ~] = ClustME(...
                respB, designB, formIntercept, ...
                'Fs', testCase.Fs, 't', testCase.TVec, ...
                'permuteUnit', 'Subject', ...
                'permutationMethod', 'signFlip', ... % Explicit SignFlip
                'numPerms', 100, ...
                'whitening', true, ...
                'testCoefficient', '', ...          % Target Intercept
                'minClusterSize', 50);

            % Verify detection
            sigClusters = clusters([clusters.p_value] < 0.05);
            testCase.verifyNotEmpty(sigClusters, 'Failed to detect Intercept signal in subset.');
        end

        function testOutputArchitecture(testCase)
            % Verify that the function returns the correct decoupled structs.

            fprintf('\n--- Running Test: Output Structure and Field Integrity ---\n');
            [clusters, mstats, vis_data] = ClustME(...
                testCase.ResponsesNull, testCase.DesignTable, testCase.LMEFormula, ...
                'permuteUnit', 'Subject', ...
                'Fs', testCase.Fs, ...
                't', testCase.TVec, ...
                'numPerms', 2);

            testCase.verifyTrue(isfield(mstats, 'AIC'), 'mstats missing AIC');
            testCase.verifyTrue(isfield(vis_data, 'nullStats'), 'vis_data missing nullStats');
            testCase.verifyTrue(isfield(clusters, 'p_value'), 'clusters missing p_value');
        end

        function testBetweenSubject_GroupLabel(testCase)
            % [TEST C] Between-Subject Unbalanced (GroupLabel)
            fprintf('\n--- Test: Between-Subject (GroupLabel / Unbalanced) ---\n');

            [clusters, ~, ~] = ClustME(...
                testCase.ResponsesBetween, testCase.DesignBetween, testCase.FormulaBetween, ...
                'Fs', testCase.Fs, 't', testCase.TVec, ...
                'permuteUnit', 'Subject', ...
                'permutationMethod', 'groupLabel', ...
                'whitening', true, ...
                'numPerms', 100, ...
                'minClusterSize', 50, ...
                'testCoefficient', 'Group_Patient'); % Ensure coefficient is explicit

            testCase.verifyNotEmpty(clusters, 'Failed to return clusters.');
            sigClusts = clusters([clusters.p_value] < 0.05);
            testCase.verifyNotEmpty(sigClusts, 'Failed to detect Group signal.');

            % Check timing around the injected group signal at 0.2 s
            tMean = mean(testCase.TVec(sigClusts(1).start : sigClusts(1).end));
            testCase.verifyTrue(abs(tMean - 0.2) < 0.1, ...
                sprintf('Cluster timing wrong. Expected ~0.2s, got %.2fs', tMean));
        end

        function test_ExactPermutations_SmallN(testCase)
            % Verify Exact Test Logic (Small N)
            % Checks that the toolbox correctly identifies exhaustive permutation sets
            % when N is small, overriding the requested numPerms.
            
            fprintf('\n--- Test: Exact Permutations (Small N) ---\n');

            % --- Sub-Test 1: Sign-Flip Exactness (2^(6-1) - 1 = 31) ---
            % 1. Generate N=6 Data
            configN6.designType = 'within';
            configN6.nTrials    = 10; 
            configN6.effectSize = 2.0; 
            
            [respN6, dtN6, gtN6] = clustme.bench_generator(6, configN6, ...
                'Fs', testCase.Fs, 'noiseMode', 'gaussian', 'RandomSeed', 123);
            
            fprintf('   Verifying Sign-Flip (Expect 31 null perms)...\n');
            [~, ~, vis_SF] = ClustME(...
                respN6, dtN6, 'response ~ 1 + (1|Subject)', ...
                'Fs', testCase.Fs, 't', gtN6.tVec, ...
                'permuteUnit', 'Subject', ...
                'permutationMethod', 'signFlip', ...
                'numPerms', 1000, ...      
                'whitening', true, ...
                'minClusterSize', 1, ...
                'verbose', false);
            
            nPermsSF = numel(vis_SF.nullStats);
            testCase.verifyEqual(nPermsSF, 31, ...
                'SignFlip should return 2^(N-1) - 1 (31) permutations.');
            
            % --- Sub-Test 2: withinSubject Exactness ((K!)^S - 1) ---
            % We build a custom block design: 4 subjects (S=4), 3 trials per subject (K=3).
            % Exhaustive Cartesian combinations = (3!)^4 = 6^4 = 1296. Minus identity = 1295.
            fprintf('   Verifying withinSubject Cartesian Exhaustive (Expect 1295 null perms)...\n');
            
            S = categorical(repelem((1:4)', 3));
            C = categorical(repmat((1:3)', 4, 1));
            dtBlock = table(S, C, 'VariableNames', {'Subject', 'Condition'});
            respBlock = randn(12, 10); % 12 total trials, 10 arbitrary timepoints
            tBlock = (1:10) / testCase.Fs;
            
            runShuffle = @() ClustME(...
                respBlock, dtBlock, 'response ~ 1 + Condition + (1|Subject)', ...
                'Fs', testCase.Fs, 't', tBlock, ...
                'permuteUnit', 'Subject', ...
                'permutationMethod', 'withinSubject', ...
                'testCoefficient', 'Condition_2', ...
                'numPerms', 2000, ...      
                'whitening', true, ...
                'minClusterSize', 1, ...
                'verbose', false);

            [clust_Shuf1, ~, vis_Shuf1] = runShuffle();
            
            nPermsShuf = numel(vis_Shuf1.nullStats);
            testCase.verifyEqual(nPermsShuf, 1295, ...
                'withinSubject should return (K!)^S - 1 (1295) permutations.');

            % --- Sub-Test 3: Stability Check (Determinism) ---
            fprintf('   Verifying Stability (3 Repeats)...\n');
            
            [clust_Shuf2, ~] = runShuffle();
            [clust_Shuf3, ~] = runShuffle();
            
            if ~isempty(clust_Shuf1)
                 p1 = [clust_Shuf1.p_value];
                 p2 = [clust_Shuf2.p_value];
                 p3 = [clust_Shuf3.p_value];
                 
                 testCase.verifyEqual(p1, p2, 'Run 1 and Run 2 p-values mismatch.');
                 testCase.verifyEqual(p1, p3, 'Run 1 and Run 3 p-values mismatch.');
            end
            
            fprintf('   [PASS] Exact Test Logic verified.\n');
        end

        function test_ConfigurationGuards(testCase)
            % Verify Configuration Guards
            % Ensures the toolbox errors gracefully when users request invalid
            % or mathematically unsafe configurations for groupLabel permutation.
            
            fprintf('\n--- Test: Configuration Guards (groupLabel) ---\n');

            % 1. Guard A: Whitening Required for groupLabel
            fprintf('   Verifying Guard: Whitening=False (Expect Error)...\n');
            
            try
                ClustME(...
                    testCase.ResponsesBetween, testCase.DesignBetween, testCase.FormulaBetween, ...
                    'Fs', testCase.Fs, 't', testCase.TVec, ...
                    'permuteUnit', 'Subject', ...
                    'permutationMethod', 'groupLabel', ...
                    'testCoefficient', 'Group_Patient', ...
                    'whitening', false, ...     % Invalid for groupLabel; should trigger guard
                    'numPerms', 10);
                
                testCase.verifyFail('Failed to error on groupLabel with whitening=false.');
            catch ME
                testCase.verifyEqual(ME.identifier, 'ClustME:UnsupportedConfiguration', ...
                    'Incorrect error ID for whitening guard.');
                fprintf('   [PASS] Caught expected error: %s\n', ME.message);
            end

            % 2. Guard B: Varying Regressor within Unit
            fprintf('   Verifying Guard: Varying Regressor (Expect Error)...\n');
            
            % Create a broken design table
            badDesign = testCase.DesignBetween;
            
            % Use exact subject ID 'C001' as per generator standards
            rows = find(badDesign.Subject == 'C001');
            
            % Force mixing groups: First half Control, second half Patient
            half = floor(numel(rows)/2);
            
            % Direct assignment (assumes categorical/cellstr compatibility from generator)
            badDesign.Group(rows(1:half))     = 'Control';
            badDesign.Group(rows(half+1:end)) = 'Patient';
            
            try
                ClustME(...
                    testCase.ResponsesBetween, badDesign, testCase.FormulaBetween, ...
                    'Fs', testCase.Fs, 't', testCase.TVec, ...
                    'permuteUnit', 'Subject', ...
                    'permutationMethod', 'groupLabel', ...
                    'testCoefficient', 'Group_Patient', ...
                    'whitening', true, ...
                    'numPerms', 10);
                
                testCase.verifyFail('Failed to error on varying regressor within unit.');
            catch ME
                testCase.verifyEqual(ME.identifier, 'ClustME:VaryingRegressor', ...
                    'Incorrect error ID for varying regressor guard.');
                fprintf('   [PASS] Caught expected error: %s\n', ME.message);
            end
        end

        function testIntegrity_MissingData(testCase)
            % [TEST E] Integrity: Handling of NaN (Missing Data)
            % Goal: Ensure code explicitly errors when non-finite data (NaNs) are present.
            fprintf('\n--- Test: Integrity (Sparse NaNs) ---\n');

            % 1. Create data with a hole
            respWithNaN = testCase.ResponsesSignal;
            
            % Drop samples 30:40 for the first trial (guarantees NaNs are injected)
            respWithNaN(1, 30:40) = NaN;

            % 2. Run ClustME and expect a hard fail
            fprintf('   Verifying Guard: Non-Finite Data (Expect Error)...\n');
            
            runClustME = @() ClustME(...
                respWithNaN, testCase.DesignTable, testCase.LMEFormula, ...
                'Fs', testCase.Fs, 't', testCase.TVec, ...
                'permuteUnit', 'Subject', ...
                'permutationMethod', 'withinSubject', ...
                'numPerms', 10, ...
                'whitening', true, ...
                'testCoefficient', 'Condition_B', ...
                'minClusterSize', 20);

            % Assert that it throws the exact missing data error
            testCase.verifyError(runClustME, 'ClustME:NonFiniteDataDetected', ...
                'Failed to error on non-finite data (NaNs) in the response matrix.');
                
            fprintf('   [PASS] Caught expected error: ClustME:NonFiniteDataDetected\n');
        end

        function testIntegrity_SingleSubject(testCase)
            % [TEST F] Integrity: Single Subject Edge Case
            % Goal: Ensure code fails predictably for N=1 case 
            fprintf('\n--- Test: Integrity (Single Subject) ---\n');

            rows = testCase.DesignTable.Subject == 'S001';

            % Guard: Ensure we actually found the subject
            if nnz(rows) == 0
                error('Test setup failed: Subject S01 not found in Design Table.');
            end

            designS1 = testCase.DesignTable(rows, :);
            dataS1   = testCase.ResponsesSignal(rows, :);

            try
                ClustME(dataS1, designS1, testCase.LMEFormula, ...
                    'Fs', testCase.Fs, 't', testCase.TVec, ...
                    'permuteUnit', 'Subject', ...
                    'numPerms', 10);

                testCase.verifyFail('Code should have errored for N=1 subject but did not.');

            catch ME
                % We expect this to fail because N=1 subject implies 0 degrees of freedom
                % for Subject-level permutations or random effects.
                % The success condition is that it fails *predictably*.

                fprintf('Received Expected Error for N=1: %s\n', ME.message);

                % Verify it didn't crash on the design table check again
                testCase.verifyFalse(strcmp(ME.identifier, 'ClustME:DesignRequired'), ...
                    'Should not trigger DesignRequired error if design table is valid.');
            end
        end

        function testROIMode_Parametric(testCase)
            % [TEST G] ROI Mode + Parametric Threshold
            % Verifies the engine can calculate stats for a fixed temporal window
            % using parametric t-critical values instead of max-statistic permutations.
            fprintf('\n--- Test: ROI Mode (Parametric Threshold) ---\n');

            [clusters, ~, ~] = ClustME(...
                testCase.ResponsesSignal, testCase.DesignTable, testCase.LMEFormula, ...
                'Fs', testCase.Fs, 't', testCase.TVec, ...
                'PreselectedCluster', [0.3 0.5], ...   % Target ROI
                'tcritMode', 'parametric', ...         % Force Parametric
                'numPerms', 50, ...                    
                'verbose', false);

            % 1. Verify a cluster was actually returned
            testCase.verifyNotEmpty(clusters, 'ROI Mode returned empty clusters.');

            % 2. Verify the returned cluster bounds align with the requested ROI
            tStart = testCase.TVec(clusters(1).start);
            tEnd   = testCase.TVec(clusters(1).end);

            % Allow a tiny floating point tolerance based on Fs
            tol = 1.5 * (1 / testCase.Fs);
            testCase.verifyTrue(abs(tStart - 0.3) < tol, sprintf('ROI start mismatch: Got %.3f, Expected 0.3', tStart));
            testCase.verifyTrue(abs(tEnd - 0.5) < tol, sprintf('ROI end mismatch: Got %.3f, Expected 0.5', tEnd));
        end
    end
end