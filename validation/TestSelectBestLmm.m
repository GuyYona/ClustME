classdef TestSelectBestLmm < matlab.unittest.TestCase
    % TestSelectBestLmm - Software integrity and unit test suite for select_best_lmm
    %
    %   TestSelectBestLmm verifies the execution and output integrity of the 
    %   ClustME model selection utility using minimal synthetic dummy data.
    %
    % How to Run:
    %   results = runtests('validation/TestSelectBestLmm.m');
    %
    % Test Inventory:
    %   1. testExecutionAndOutput - Runs the selection utility and verifies 
    %      the output struct contains the required AIC and cluster metrics.
    %
    % Dependencies:
    %   - MATLAB R2019b or newer.
    %   - Statistics and Machine Learning Toolbox.
    %   - Part of the ClustME Validation Suite.
    %
    % Reference:
    %   Yona, G., & Magill, P. J. Fast Cluster-Based Permutation Testing with
    %   Linear Mixed-Effects Models.
    %
    % Version: 1.0.0
    % Last Modified: 30 April 2026
    %
    % Copyright (C) 2026 University of Oxford
    % Author: Guy Yona
    %
    % This program is free software: you can redistribute it and/or modify
    % it under the terms of the GNU General Public License as published by
    % the Free Software Foundation, either version 3 of the License, or
    % (at your option) any later version.

    methods (Test)

        function testExecutionAndOutput(tc)
            % 1. Setup minimal dummy data (Identical to script)
            nTrials = 20;
            nTime   = 50;
            t       = linspace(0, 1, nTime);

            % Create Design Table
            design = table();
            design.Trial = (1:nTrials)';
            design.Subject = categorical(repmat((1:5)', 4, 1)); % 5 subjects, 4 trials each

            % Create Response Data 
            responses = randn(nTrials, nTime);

            % Define Inputs
            formulas = {'response ~ 1 + (1|Subject)', 'response ~ 1'};
            roi      = [0.4 0.6]; % Preselected cluster window

            fprintf('Running select_best_lmm test...\n');

            % 2. Call the function
            results = clustme.select_best_lmm(responses, formulas, design, t, roi, ...
                                      'verbose', false, 'numPerms', 10); 

            % 3. Check output using Unit Testing assertions
            tc.verifyNotEmpty(results, 'Results are empty.');
            tc.verifyTrue(isfield(results, 'formula'), 'Missing formula field.');
            
            tc.verifyTrue(isfield(results, 'StaticAIC'), 'Missing StaticAIC field.');
            tc.verifyTrue(isfield(results, 'nClusters'), 'Missing nClusters field.');
            
            % Print original output summary table
            fprintf('Success! Output summary:\n');
            disp(table({results.formula}', [results.StaticAIC]', [results.nClusters]', ...
                 'VariableNames', {'Formula', 'StaticAIC', 'NumClusters'}));
        end
    end
end