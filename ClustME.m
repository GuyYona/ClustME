function [clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula, options)

% ClustME - Cluster-based permutation testing with linear mixed-effects models
%
%   ClustME implements a cluster-based permutation test for hierarchical 
%   time-series data. The test statistic at each time sample is a Generalized 
%   Least Squares (GLS) contrast derived from a linear mixed-effects (LME) model. 
%
%   By estimating the marginal covariance matrix (Static-V) at a reference 
%   timepoint and reusing it across permutations, the engine avoids exhaustive 
%   refitting inside the resampling loop. This provides massive computational 
%   speed gains while maintaining asymptotic permutation validity under 
%   exchangeability assumptions.
%
% Syntax:
%   clusters = ClustME(responses, design, lmeFormula)
%   [clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula, options)
%
% Inputs:
%   responses  - [N×T double] Continuous 2D data matrix (trials × time samples).
%   design     - [N×V table] Trial-level metadata containing predictors and grouping variables.
%   lmeFormula - [char] Wilkinson string for fitlme (e.g., 'response ~ 1 + Group + (1|Subject)').
%
% Options (Name-Value arguments):
%   .permutationMethod      [char]    Null generation method: 'signFlip', 'withinSubject', 
%                                     'groupLabel', or 'wildBootstrap'. (Default: 'signFlip')
%   .fullLME                [logical] Bypass Static-V for continuous exhaustive refits. (Default: false)
%   .minClusterSize         [double]  Minimum cluster length in ms. (Default: 0)
%   .numPerms               [double]  Target permutations for the null distribution. (Default: 5000)
%   .BqTarget               [double]  Target permutations for empirical thresholding. (Default: 2000)
%   .alphaValue             [double]  Family-wise alpha level. (Default: 0.05)
%   .parallel               [logical] Enable parallel pool execution. (Default: false)
%   .clusterMassMethod      [char]    Statistic defining cluster mass: 'sum', 'mean'. (Default: 'mean')
%   .testCoefficient        [char]    Specific fixed effect to test. (Default: '' i.e., Intercept)
%   .Fs                     [double]  Sampling frequency in Hz. (Default: 100)
%   .clusterSummaryMetric   [char]    Metric for descriptive cluster extraction: 'mean', 'sum', 
%                                     'median', 'signedPeak'. (Default: 'signedPeak')
%   .permuteUnit            [char]    Exchangeability unit. (Default: 'auto')
%   .whitening              [logical] Perform V^{-1/2} whitening. (Default: true)
%   .PreselectedCluster     [double]  K×2 [tStart tEnd] array of predefined ROIs. (Default: [])
%   .t                      [double]  1×T optional time vector. (Default: [])
%   .TimeAnchor             [double]  Scalar time for static V construction. (Default: [])
%   .Vmode                  [char]    Variance pooling strategy: 'adaptiveLocal', 'local', 
%                                     'global', 'identity'. (Default: 'adaptiveLocal')
%   .FitMethod              [char]    LME fitting method: 'REML', 'ML'. (Default: 'REML')
%   .tcritMode              [char]    Cluster-forming threshold logic: 'permutation', 
%                                     'parametric'. (Default: 'permutation')
%   .wbLeverage             [logical] Toggle reduced-model CR2 adjustment. (Default: true)
%   .verbose                [logical] Print internal diagnostics. (Default: false)
%
% Outputs:
%   clusters - [1×K struct] Array detailing each surviving cluster with fields:
%       .type           ['positive' | 'negative'] Direction of the effect.
%       .start, .end    [double] Sample indices of the cluster boundaries.
%       .mass           [double] Cluster-mass statistic (primary inference metric).
%       .measure        [double] Descriptive metric of the cluster data across trials.
%       .p_value        [double] FWER-corrected permutation p-value.
%       .lmeTStat       [double] Descriptive t-statistic from the post-hoc LME.
%       .lmePValue      [double] Descriptive p-value from the post-hoc LME.
%       .covVars        [double] Random-effect variances from the post-hoc LME.
%       .resVar         [double] Residual variance from the post-hoc LME.
%       .varianceRatios [double] Variance ratios from the post-hoc LME.
%       .AIC            [double] AIC of the post-hoc LME.
%
%   mstats   - [struct] Model statistics from the Static-V anchor fit:
%       .AIC            [double] AIC of the static covariance model.
%       .VarRatios      [double] Variance ratios of the static covariance model.
%       .Model          [object] The full fitlme model object.
%       .Provenance     [struct] contains vesion and timestamp metadata.
%
%   vis_data - [struct] High-density arrays for ClustME visualisations:
%       .Tmap           [1×T double] Observed GLS t-statistics.
%       .Fs             [double] Sampling frequency.
%       .coefName       [char] Name of the tested coefficient.
%       .alpha          [double] Family-wise alpha level.
%       .tcrit          [1×T double] Exact time-varying thresholds.
%       .sigMask        [1×T logical] Significance mask.
%       .cStarts        [1×K double] Cluster start indices.
%       .cEnds          [1×K double] Cluster end indices.
%       .clusterLevel   [1×K double] Cluster averages across trials for plotting.
%       .nullStats      [B×1 double] Max-statistic empirical null distribution.
%       .obsClusterMass [1×K double] Observed cluster masses.
%       .pVals          [1×K double] p-values per cluster.
%       .alphaValue     [double] Family-wise alpha level.
%       .t              [double] Time vector.
%       .clusterMassMethod   [char] Statistic used for cluster mass.
%       .vstat          [struct] Variance pooling diagnostics.
%       .idxAnchor      [double] Index of the time anchor.
%
% PIPELINE AND RATIONALE
% ----------------------
% 1) Clean inputs: Validates data consistency, formula compatibility, and strictly
%    checks for non-finite data (NaN/Inf) to prevent matrix solver crashes.
% 2) Long-format design: Resolves exchangeability units and grouping indices
%    (e.g., subjects, trials) required to constrain the null generation.
% 3) Per-time residual scales: Fits independent LMEs at every timepoint to extract
%    both the residual variance (SigmaVec, used solely to identify the baseline noise 
%    anchor) and the true time-varying standard error of the contrast (SeVec, used 
%    for downstream studentization). Both are stabilized using a running median.
% 4) Static Covariance (Static-V): Uses SigmaVec to automatically identify a 
%    representative TimeAnchor (if not user-specified). Constructs the marginal 
%    covariance matrix (V) at this anchor. Variance components are extracted and 
%    stabilized according to the specified Vmode pooling strategy.
% 5) Observed GLS T-Map Construction: Computes the GLS normal equations and
%    derives the observed t-statistic trace using the static V matrix. This includes 
%    optional Cholesky whitening and exact true-SE studentization using the relative 
%    standard errors (RelScale = SeVec / se_j).
% 6) Permutation Primitives (Freedman-Lane): Fits the reduced nuisance model
%    under the Static-V geometry to isolate baseline predictions (yHat0) from the
%    reduced-model residuals (e0). Applies CR2-style leverage adjustments if active.
% 6b) Generate Permutation Matrix: Pre-computes the full set of randomisation
%    instructions based on the selected exchangeability constraints.
% 7) Cluster-forming threshold: Evaluates the pointwise threshold (tcrit) via a
%    static parametric distribution or an empirical permutation envelope.
% 8) Cluster detection: Isolates contiguous temporal samples exceeding the threshold
%    and applies the minimum duration filter (minClusterSize).
% 9) Observed Cluster Statistics: Calculates the cluster mass (sum or mean of t^2)
%    for each observed candidate cluster across its temporal extent.
% 10) Null distribution: Executes the full resampling loop to build the max-statistic
%     empirical null distribution. Uses either fast matrix projections or exact
%     LME refits (Path A / Path B).
% 11) P-values: Evaluates the observed cluster statistics against the finite-sample
%     empirical null distribution to yield FWER-controlled p-values.
% 12) Post-hoc Cluster Characterization: Refits descriptive LMEs to the aggregated
%     data (e.g., peak, mean) within each cluster's boundaries for effect summaries.
% 13) Output Packaging: Assembles the validated arrays and metadata into the
%     clusters, mstats, and vis_data structs.
%
% ASSUMPTIONS & NOTES
% -------------------
% • Asymptotic Validity: Because the marginal covariance matrix (V) and the 
%   residual scales are estimated, the test is asymptotically valid under the 
%   chosen exchangeability assumptions, rather than strictly exact in finite samples.
% • Nuisance Preservation: The Static-V approximation avoids re-estimating 
%   nuisance parameters in each shuffle. The same linear map is applied to both 
%   observed and permuted data, maintaining asymptotic validity.
% • Exact Studentization: Per-time studentization is applied as a relative scale 
%   (SigmaVec/barSigma) in both whitened and non-whitened modes. Whitening 
%   changes the GLS projector but does not disable studentization.
% • Thresholding vs. Inference: The cluster-forming threshold establishes 
%   candidate boundaries but does not control error rates. FWER control is 
%   achieved exclusively by the max-statistic permutation procedure.
% • Post-Hoc Summaries: Reported cluster-level metrics (AIC, lmeTStat, covVars) 
%   from the cluster-averaged LME are strictly descriptive and are explicitly 
%   decoupled from the permutation test inference.
% • Small-N Thresholding: A small number of exchangeability units (N <= 10) can 
%   make empirical permutation thresholds unstable. The function will issue a 
%   diagnostic warning suggesting options.tcritMode = 'parametric'.
% • Exchangeability & Behrens-Fisher: Standard label permutation ('groupLabel') 
%   assumes homoscedasticity. Unbalanced between-subject designs with unequal 
%   variances violate this assumption. In such cases, 'wildBootstrap' 
%   (subject-level Rademacher sign-flipping) is mandated to preserve 
%   group-specific variance structures.
% • Unsigned Biphasic Clustering: Clusters are formed using a two-sided 
%   magnitude criterion (|t| >= tcrit) and evaluated using a non-negative mass 
%   (t^2). Rapidly biphasic responses crossing zero are mathematically bound 
%   as a single continuous temporal event.
% • Static Regressors Limitation: To leverage the Static-V acceleration, all 
%   fixed-effect regressors must be static, scalar trial-level values. Continuous 
%   time-varying covariates (e.g., sample-by-sample velocity) are not supported.
%
% Dependencies:
%   - MATLAB R2019b or newer. (Validated on R2025b)
%   - Statistics and Machine Learning Toolbox
%   - Parallel Computing Toolbox (Optional)
%   - ClustME Package (+clustme)
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

    arguments
        responses                      double
        design                         table
        lmeFormula                     (1,:) char

        options.permutationMethod      (1,:) char {mustBeMember(options.permutationMethod,{'signFlip','withinSubject','groupLabel','wildBootstrap'})} = 'signFlip'
        options.fullLME                (1,1) logical = false   % when TRUE, bypasses the fast matrix algebra and fits LMEs to every timepoint

        options.minClusterSize         (1,1) double {mustBeNonnegative}               = 0
        options.numPerms               (1,1) double {mustBeInteger,mustBeNonnegative} = 5000
        options.BqTarget               (1,1) double {mustBeInteger,mustBeNonnegative} = 2000
        options.alphaValue             (1,1) double {mustBePositive,mustBeLessThanOrEqual(options.alphaValue,1)} = 0.05
        options.parallel               (1,1) logical = false
        options.clusterMassMethod           (1,:) char {mustBeMember(options.clusterMassMethod,{ 'sum','mean' })}                         = 'mean'
        options.testCoefficient        (1,:) char = ''   % '' => Intercept; otherwise exact name in m.CoefficientNames
        options.Fs                     (1,1) double {mustBePositive} = 100
        options.clusterSummaryMetric         (1,:) char {mustBeMember(options.clusterSummaryMetric,{'mean','sum','median','signedPeak'})}   = 'signedPeak'
        options.permuteUnit            (1,:) char = 'auto';
        options.whitening              (1,1) logical = true    % performs V^{-1/2} whitening
        options.PreselectedCluster     double       = []       % K×2 [tStart tEnd] in same units as options.t
        options.t                      (1,:) double = []       % optional time vector (seconds or custom)
        options.TimeAnchor             double       = []       % optional scalar time (same units as options.t)
        options.Vmode                  (1,:) char {mustBeMember(options.Vmode,{'adaptiveLocal','local','global','identity'})}   = 'adaptiveLocal'
        options.FitMethod              (1,:) char {mustBeMember(options.FitMethod,{'REML','ML'})}                               = 'REML'
        options.tcritMode              (1,:) char {mustBeMember(options.tcritMode,{'permutation','parametric'})}                = 'permutation'
        options.robustSE               (1,:) char {mustBeMember(options.robustSE,{'none'})}                               = 'none'     
                 % option 'CR2' to be added in a future release after validation

        options.wbLeverage             (1,1) logical = true    % Toggle for reduced-model CR2 adjustment
        options.verbose                (1,1) logical = false   % print internal diagnostics (V stability, etc.)
        options.overrideWhiteningCheck (1,1) logical = false   % Validation use only: hidden flag to bypass whitening error
    end

    tic;

    mstats.AIC       = [];
    mstats.VarRatios = [];
    mstats.Model     = [];
    mstats.Provenance.ClustMEVersion = clustme.version();
    mstats.Provenance.MATLABVersion  = version;
    mstats.Provenance.Timestamp      = datetime("now");
    

    minClusterSamples = max(1, floor(options.minClusterSize/1e3 * options.Fs));
    hasPreselected = ~isempty(options.PreselectedCluster);

    if options.parallel
        try
            if isempty(gcp('nocreate'))
                 parpool('local');     
            end
        catch ME
            warning('ClustME:ParallelFailed', ...
                'Parallel requested but pool could not be started (%s). Running serial.', ME.message);
            options.parallel = false;
        end
    end

    %% 1) Clean inputs -----------------------------------------------------------

    % 1. Map Responses (N_total x T)

    allResponses = responses;
    allResponses_preWhite = allResponses;
    dur          = size(allResponses, 2);
    nTrials      = size(allResponses, 1);
   
    % --- Pre-flight Check: Missing or Infinite Data (NaNs/Infs) ---
    if any(~isfinite(allResponses), 'all')
        badMap = ~isfinite(allResponses);
        nTrialsWithBad = sum(any(badMap, 2));
        
        % Identify first problematic timepoint for context
        [~, c_idx] = find(badMap, 1);
        t_bad = c_idx; 
        if ~isempty(options.t), t_bad = options.t(c_idx); end
        
        error('ClustME:NonFiniteDataDetected', ...
            ['Non-finite data (NaN or Inf) detected in %d trials.\n' ...
             '      First detection at sample/time: %.2f\n' ...
             '      CRITICAL: The matrix solver requires strictly finite continuous data.\n' ...
             '      You must explicitly excise or impute these missing/infinite values \n' ...
             '      upstream before passing the response matrix to ClustME.'], ...
             nTrialsWithBad, t_bad);
    end

    % 2. Build the Master Design Table (tblTemplate)
    if height(design) ~= nTrials
        error('ClustME:DesignMismatch', ...
            'Height of design table (%d) must match total number of trials in responses (%d).', ...
            height(design), nTrials);
    end
    
    tblTemplate = design;
    
    % Ensure 'Trial' column exists (or add it)
    if ~ismember('Trial', tblTemplate.Properties.VariableNames)
        tblTemplate.Trial = (1:nTrials)';
    end

    % 3. Add Placeholder Response Column
    %    (This is overwritten during fitting, but needed for formula validation)
    tblTemplate.response = zeros(nTrials, 1);

    % 4. Validate Formula against Table
    allVars = setdiff(unique(regexp(lmeFormula, '[A-Za-z0-9_]+', 'match')), ...
                      {'response','1','x','log','exp','sqrt','abs'});
    
    missingVars = setdiff(allVars, tblTemplate.Properties.VariableNames);
    if ~isempty(missingVars)
        error('ClustME:MissingVariable', ...
            'Variable(s) "%s" required by formula not found in design table.', ...
            strjoin(missingVars, ', '));
    end

    % --- Diagnostic Warning: numeric fixed effect with few unique values ---
    testVar = options.testCoefficient;
    if ~isempty(testVar) && ismember(testVar, tblTemplate.Properties.VariableNames)
        vData = tblTemplate.(testVar);
        if isnumeric(vData) && all(mod(vData(~isnan(vData)), 1) == 0) && numel(unique(vData)) <= 10
            warning('ClustME:NumericCategorical', ...
                ['Fixed effect "%s" is numeric but has few unique values. It will be treated as a \n' ...
                 '      continuous linear slope. Cast to categorical if independent means are intended.'], testVar);
        end
    end

    % --- Method/Coefficient Compatibility Checks ---

    % 1. Warn if signFlip is used for non-Intercept tests

    if strcmp(options.permutationMethod, 'signFlip') && ~isempty(options.testCoefficient) && ~strcmpi(options.testCoefficient, '(Intercept)')
        warning('ClustME:MethodMismatch', ...
            ['"signFlip" selected for coefficient "%s". ', ...
            'Sign-flipping assumes a zero-symmetric null distribution, which is often violated in group comparisons.', ...
            'Consider "wildBootstrap" (robust) or "groupLabel" (balanced designs) instead.'], ...
            options.testCoefficient);
    end

    % Route wildBootstrap through the sign-flip residual engine 
    % while preserving the requested method for branch-specific logic.
    requestedPermutationMethod = options.permutationMethod;

    % Alias 'wildBootstrap' to 'signFlip' for the engine
    if strcmp(options.permutationMethod, 'wildBootstrap')
        options.permutationMethod = 'signFlip';
    end

    % 2. Warn if withinSubject/groupLabel is used for Intercept-only tests
    if ismember(options.permutationMethod, {'withinSubject', 'groupLabel'}) && isempty(options.testCoefficient)
        warning('ClustME:MethodMismatch', ...
            ['"%s" method selected without a coefficient (defaulting to Intercept). ', ...
            'Label-swapping tests between-group exchangeability. ', ...
            'For one-sample evaluations, "signFlip" is recommended.'], ...
            options.permutationMethod);
    end

    if strcmp(options.robustSE, 'CR2') && ~options.whitening
        error('ClustME:CR2RequiresWhitening', ...
            'options.robustSE="CR2" is supported only when options.whitening=true.');
    end

    if strcmp(options.robustSE, 'CR2') && options.fullLME
        error('ClustME:CR2UnsupportedFullLME', ...
            'options.robustSE="CR2" is currently supported only with options.fullLME=false.');
    end

    %% 2) Long‑format design -----------------------------------------------------
      
    % ---------------------------------------------------------------------
    % Resolve permutation unit & Grouping Indices
    % ---------------------------------------------------------------------
    % Resolve exchangeability units and validate the requested permutation unit against the design table.
    [groupingIdx, nUnits, permuteUnitName] = clustme.get_permut_unit(lmeFormula, tblTemplate, options.permuteUnit);

    % For trial-level permutation, use trial rows as exchangeability units 
    % when no grouping index is returned.
    if nUnits == 0, nUnits = nTrials; end

    if nUnits == 1 && ~strcmp(permuteUnitName, 'trial')
       error('ClustME:SingleExchangeabilityUnit', ...
            ['Only 1 exchangeability unit detected for "%s".\n' ...
            '      Permutation testing requires multiple exchangeable units to form a null distribution.\n' ...
            '      1. Check that the design table column "%s" contains multiple unique identifiers\n' ...
            '      2. For running a single-subject analysis, set options.permuteUnit = ''trial'''], ...
            permuteUnitName, permuteUnitName);
    end

    % --- Small-N exchangeability note when using permutation-based thresholds ---
    if nUnits > 0 && nUnits <= 10 && ~strcmp(options.tcritMode,'parametric')
       fprintf(['ClustME NOTE: Detected %d exchangeability units (%s).\n' ...
                '              Permutation-based thresholds may exhibit instability at this sample size.\n' ...
                '              Consider setting options.tcritMode="parametric" to use a parametric threshold for cluster formation, or review the generated threshold using clustme.Visualizer.\n'], ...
                nUnits, permuteUnitName);
    end

    % CR2 is active only on the whitened static-V path
    useCR2 = strcmp(options.robustSE, 'CR2') && ...
             options.whitening && ...
             ~strcmp(options.Vmode, 'identity') && ...
             ~options.fullLME;

    % Use the same independent units resolved by get_permut_unit()
    if useCR2
        if isempty(groupingIdx)
            % permuteUnit = 'trial'  -> singleton clusters
            cr2ClusterIdx = (1:nTrials).';
        else
            cr2ClusterIdx = groupingIdx(:);
        end
    else
        cr2ClusterIdx = [];
    end

    % Wild-bootstrap leverage correction is applied on the whitened path only.
    % Use the same independent units resolved by get_permut_unit().
    useWBLeverage = options.wbLeverage && ...
                    strcmp(requestedPermutationMethod, 'wildBootstrap') && ...
                    options.whitening && ...
                    ~options.fullLME;

    if useWBLeverage
        if isempty(groupingIdx)
            wbClusterIdx = (1:nTrials).';
        else
            wbClusterIdx = groupingIdx(:);
        end
    else
        wbClusterIdx = [];
    end

    %% 3) Per-time residual scales and coefficient selection

    % Fit the first timepoint to initialise coefficient selection and per-time scale extraction.

    mdl0  = fitlme(setfield(tblTemplate,'response',allResponses(:,1)),lmeFormula, 'FitMethod', options.FitMethod); %#ok<SFLD>

    % Decide which fixed-effect coefficient to test (default: Intercept)
    coefNames = mdl0.CoefficientNames;
    if ~isempty(options.testCoefficient)
        idxCoef = find(strcmp(coefNames, options.testCoefficient), 1);
        if isempty(idxCoef)
            error('ClustME:BadEffectName', 'No fixed effect named "%s". Available: %s', ...
                  options.testCoefficient, strjoin(coefNames, ', '));
        end
    else
        idxCoef = 1;  % Intercept
    end

    SigmaVec = nan(dur,1);        % per-time residual SD (sqrt MSE)
    SeVec    = nan(dur,1);        % per-time standard error of the coefficient
    SigmaVec(1) = sqrt(mdl0.MSE);
    SeVec(1)    = mdl0.Coefficients.SE(idxCoef);

    if options.parallel
        constTbl = parallel.pool.Constant(tblTemplate);
        parfor tp = 2:dur
            tbl = constTbl.Value;
            tbl.response = allResponses(:,tp);

            mse_tp = NaN;
            se_tp  = NaN;
            try
                m_tp   = fitlme(tbl, lmeFormula, 'FitMethod', options.FitMethod);
                mse_tp = m_tp.MSE;
                se_tp  = m_tp.Coefficients.SE(idxCoef);
            catch ME
                error('ClustME:LMEFitFailed', ...
                    'Per-time LME fit failed at timepoint %d (%s).', tp, ME.message);
            end

            if isnan(mse_tp)
                error('ClustME:LMEFitFailedNaN', ...
                    'Per-time LME fit at timepoint %d returned NaN MSE.', tp);
            end

            SigmaVec(tp) = sqrt(mse_tp);
            SeVec(tp)    = se_tp;
        end
    else
        for tp = 2:dur
            tblTemplate.response = allResponses(:,tp);

            mse_tp = NaN;
            se_tp  = NaN;
            try
                m_tp   = fitlme(tblTemplate, lmeFormula, 'FitMethod', options.FitMethod);
                mse_tp = m_tp.MSE;
                se_tp  = m_tp.Coefficients.SE(idxCoef);
            catch ME
                error('ClustME:LMEFitFailed', ...
                    'Per-time LME fit failed at timepoint %d (%s).', tp, ME.message);
            end

            if isnan(mse_tp)
                error('ClustME:LMEFitFailedNaN', ...
                    'Per-time LME fit at timepoint %d returned NaN MSE.', tp);
            end

            SigmaVec(tp) = sqrt(mse_tp);
            SeVec(tp)    = se_tp;
        end
    end


    % ---- stabilisation of SigmaVec -----------------------------------
    % Mild running-median smoothing over time and clipping of tiny sigmas.
    % This reduces the impact of noisy per-time MSE estimates without
    % fundamentally changing the scale pattern.

    if dur >= 5
        SigmaVec = movmedian(SigmaVec, 5, 'omitnan', 'Endpoints', 'shrink');
        SeVec    = movmedian(SeVec, 5, 'omitnan', 'Endpoints', 'shrink');
    end

    % Clip very small sigmas to a fraction of the median to avoid huge t-values
    medSigma = median(SigmaVec(SigmaVec > 0 & isfinite(SigmaVec)));
    if isfinite(medSigma) && medSigma > 0
        SigmaVec = max(SigmaVec, 0.05 * medSigma);   % lower bound = 5% of median
    end

    medSe = median(SeVec(SeVec > 0 & isfinite(SeVec)));
    if isfinite(medSe) && medSe > 0
        SeVec = max(SeVec, 0.05 * medSe);
    end


    %% 4) Static-V construction and diagnostics

    [V, mdlStatic, idxAnchor, vstat] = clustme.build_static_V(tblTemplate, allResponses, lmeFormula, SigmaVec, ...
                Vmode=options.Vmode, t=options.t, TimeAnchor=options.TimeAnchor, ...
                FitMethod=options.FitMethod, verbose=options.verbose);

    % Preserve outputs from the static model (AIC, VarRatios, and full object)
    mstats.AIC = mdlStatic.ModelCriterion.AIC;
    mstats.Model = mdlStatic;

    cpStatic = covarianceParameters(mdlStatic);
    if isempty(cpStatic)
        mstats.VarRatios = [];
    else
        varsStatic    = cellfun(@(C) diag(C), cpStatic, 'UniformOutput', false);
        covVarsStatic = vertcat(varsStatic{:});
        resVarStatic  = mdlStatic.MSE;
        mstats.VarRatios     = covVarsStatic ./ (covVarsStatic + resVarStatic);
    end

    %% 5) Observed GLS T-Map Construction
    % Fit the full model design (X), apply whitening (if requested), and compute
    % the Generalized Least Squares t-statistic for the observed data.

    [X] = designMatrix(mdlStatic,'Fixed');

    %  optional whiten (compute L from V, then transform rows)
    if options.whitening
        Vs = (V + V') * 0.5;
        try
            L = chol(Vs, 'lower');   % Vs = L * L'
        catch
            epsV = 1e-8 * max(1, median(full(diag(Vs))));
            L = chol(Vs + epsV * speye(size(Vs)), 'lower');
        end
        % row-transform (whiten)
        allResponses = L \ allResponses;   % n×T -> whitened
        X = L \ X;                         % n×p -> whitened design
        V = speye(size(Vs));               % downstream code expects V (now identity)
    end

    [solveV, ~] = make_V_solver(V); 

    % Precompute mapping a' * y -> t_j, and time-point threshold vector
    % Solve M1 = V^{-1} X using sparse backslash
    M1        = solveV(X);                         % n × p

    XtVinvX = X' * M1;                             % p × p

    % Guard: rank deficiency / ill-conditioning in GLS normal equations
    rc = rcond(full(XtVinvX));
    if ~isfinite(rc) || rc < 1e-12
        error('ClustME:IllConditionedXtVinvX', ...
            'Ill-conditioned XtVinvX (rcond=%g). Fixed-effect design may be rank-deficient/collinear.', rc);
    end

    e_j     = zeros(size(XtVinvX,1),1);
    e_j(idxCoef) = 1;
    Scol    = XtVinvX \ e_j;                       % (XtVinvX)^{-1} e_j  via solve
    se_j    = sqrt(Scol(idxCoef));
    w       = M1 * Scol;                           % n × 1
    w_row   = (w' / se_j);                         % 1 × n

    % Calculate the relative scale used for time-varying studentisation
    if strcmp(options.Vmode, 'identity')
        % --- OLS-equivalent identity-V scaling ---
        beta_ols  = X \ allResponses;
        res_ols   = allResponses - (X * beta_ols);
        df_ols    = size(X, 1) - rank(full(X));
        Sigma_ols = sqrt(sum(res_ols.^2, 1) / df_ols).';

        medSigOls = median(Sigma_ols(Sigma_ols > 0 & isfinite(Sigma_ols)));
        if isfinite(medSigOls) && medSigOls > 0
            Sigma_ols = max(Sigma_ols, 0.05 * medSigOls);
        end
        RelScale = Sigma_ols;

    elseif useCR2  % This option is not yet validated
        SeVec_CR2 = clustme.CR2.compute_SE(allResponses, X, XtVinvX, idxCoef, cr2ClusterIdx);

        medSeCR2 = median(SeVec_CR2(SeVec_CR2 > 0 & isfinite(SeVec_CR2)));
        if isfinite(medSeCR2) && medSeCR2 > 0
            SeVec_CR2 = max(SeVec_CR2, 0.05 * medSeCR2);
        end

        RelScale = SeVec_CR2 / se_j;

    else
        % Standard LME-derived relative scaling
        RelScale = SeVec / se_j;
    end

    if options.fullLME
        % === FULL LME MODE ===
        fprintf('ClustME: Full LME refit mode active (Observed Map). This will be slow.\n');
        
        tGLS_obs = zeros(1, dur);
        fit_failures = false(1, dur);
        
        % Use RAW data (fitlme handles variance internally).
        % (allResponses was potentially whitened above, so use the backup copy)
        data_for_fit = allResponses_preWhite; 

        if options.parallel
            % constTbl is available from Section 3.
            parfor t = 1:dur
                row_tbl = constTbl.Value;
                row_tbl.response = data_for_fit(:, t);

                try
                    m_full = fitlme(row_tbl, lmeFormula, 'FitMethod', options.FitMethod);
                    tGLS_obs(t) = m_full.Coefficients.tStat(idxCoef);
                catch
                    tGLS_obs(t) = 0;
                    fit_failures(t) = true;
                end
            end
        else
            warning('ClustME:FullLMESerial', ...
                'Executing Full LME mode without parallel processing. This will be extremely slow.');
            % Serial execution path
            row_tbl = tblTemplate;
            for t = 1:dur
                row_tbl.response = data_for_fit(:, t);

                try
                    m_full = fitlme(row_tbl, lmeFormula, 'FitMethod', options.FitMethod);
                    tGLS_obs(t) = m_full.Coefficients.tStat(idxCoef);
                catch
                    tGLS_obs(t) = 0;
                    fit_failures(t) = true;
                end
            end
        end

        if any(fit_failures)
             warning('ClustME:ConvergenceFailure', ...
                     'Full LME fit failed to converge at %d timepoints (T-statistics defaulted to 0).', sum(fit_failures));
        end

    else
        % === STANDARD FAST MODE (Static-V) ===    
        % Static-V GLS t for the chosen coefficient across all timepoints (observed)
        tGLS_obs = (w_row * allResponses) ./ RelScale.';   % 1×T 
    end
   

    %% 6) Permutation primitives (Freedman–Lane setup)
    % Construct the reduced model (X0) excluding the effect of interest.
    % Compute residuals (e0) and projected fits (yHat0) once using Static-V.
    %  For intercept-only designs this collapses to
    %  ŷ0 = 0 and e0 = y, i.e. standard one-sample sign-flip.

    p        = size(XtVinvX,1);
    allIdx   = 1:p;
    nuIdx    = allIdx(allIdx ~= idxCoef);  % nuisance fixed effects
    p0       = numel(nuIdx);

    if p0 == 0
        % Intercept-only (or single fixed effect) case: reduced model has no
        % fixed effects => ŷ0 = 0, e0 = y.
        yHat0_all = zeros(size(allResponses), 'like', allResponses);  % n×T
        e0_all    = allResponses;                                     % n×T
    else
        % GLS reduced-model fit with static V:
        %   β0(t)  = (X0' V^{-1} X0)^{-1} X0' V^{-1} y(t)
        %   ŷ0(t)  = X0 β0(t)
        %   e0(t)  = y(t) - ŷ0(t)
        X0       = X(:, nuIdx);               % n × p0
        M0       = solveV(X0);                % n × p0   (V^{-1} X0)

        XtVinvX0 = X0' * M0;                  % p0 × p0

        % Guard: rank deficiency / ill-conditioning in reduced-model normal equations
        rc0 = rcond(full(XtVinvX0));
        if ~isfinite(rc0) || rc0 < 1e-12
            error('ClustME:IllConditionedXtVinvX0', ...
                'Ill-conditioned XtVinvX0 (rcond=%g). Reduced-model design may be rank-deficient/collinear.', rc0);
        end

        % Solve V^{-1} Y once (static V) for all time points
        VinvY      = solveV(allResponses);    % n × T
        XtVinvY0   = X0' * VinvY;             % p0 × T
        beta0_all  = XtVinvX0 \ XtVinvY0;     % p0 × T

        yHat0_all  = X0 * beta0_all;          % n × T
        e0_all     = allResponses - yHat0_all;

        % Wild bootstrap: undo reduced-model residual shrinkage at the cluster level
        if useWBLeverage
            e0_all = clustme.CR2.adjust_residuals(e0_all, X0, XtVinvX0, wbClusterIdx);
        end
    end

    % Precompute projection of reduced-model fits through the GLS contrast.
    % This is the part of the statistic that does NOT change across shuffles.
    t0_base = w_row * yHat0_all;            % 1 × T projected reduced fit

    % --- Pre-computation for Group label X-Permutation ---
    if strcmp(options.permutationMethod, 'groupLabel')
        if isempty(options.testCoefficient)
            error('ClustME:GroupLabelRequiresCoef', 'Must specify options.testCoefficient for groupLabel permutation.');
        end
        if ~options.whitening && ~options.overrideWhiteningCheck
            error('ClustME:UnsupportedConfiguration', ...
                'The "groupLabel" method requires whitening=true for unbalanced designs.');
        end

        uIds = unique(groupingIdx);
        nUnits = numel(uIds);
        unitRows  = cell(nUnits,1);
        unitVals  = zeros(nUnits,1); % Tested-regressor value for each unit
        unitBasis = cell(nUnits,1);  % Whitened basis shape for a unit-level constant regressor
        
        % 1. Get RAW Regressor (to determine 0 vs 1 labels)
        X_raw_full = designMatrix(mdlStatic, 'Fixed'); 
        X_raw_col  = full(X_raw_full(:, idxCoef));
        
        % 2. Compute Whitened Unit-level Basis Vector
        % Applying L (from Section 5) to a vector of ones gives the shape
        % that a unit-level constant regressor takes after whitening.
        if exist('L', 'var')
            basis_vec = L \ ones(nTrials, 1);
        else
            basis_vec = ones(nTrials, 1);
        end

        for k = 1:nUnits
            rows = (groupingIdx == uIds(k));
            unitRows{k} = find(rows);

            % Validate using RAW values
            if range(X_raw_col(rows)) > 1e-9
                error('ClustME:VaryingRegressor', 'Regressor varies within unit. Cannot permute labels.');
            end
            
            % Store Raw Label (Magnitude)
            unitVals(k) = X_raw_col(find(rows,1));
            
            % Store Whitened Basis (Shape)
            % This is L_sub^-1 * 1
            unitBasis{k} = basis_vec(rows);
        end
    else
        % Placeholders
        unitRows = []; unitVals = []; unitBasis = []; nUnits = 0;
    end

    %% 6b) Generate Permutation Matrix (Built once, used for both T-crit and Null)

    permMatrix = clustme.build_permutation_matrix(nTrials, groupingIdx, options.numPerms, options.permutationMethod);
    numPermsTotal = size(permMatrix, 1);

    %% 7) Cluster-forming threshold and supra-threshold mask
    % We always compute tcritVec for plotting/reference, even in ROI mode
    % (PreselectedCluster). The sigMask used for threshold-based cluster
    % detection is only applied when PreselectedCluster is empty.

    switch options.tcritMode
        case 'permutation'
            % --- per-time empirical cutoff from the absolute permuted statistics ---
            
            % Reuse the master permutation matrix (take the first N rows)
            BqActual = min(options.BqTarget, numPermsTotal);
            permMatrix_q = permMatrix(1:BqActual, :);
            
            TPerm = zeros(BqActual, dur, 'like', allResponses);

            for b = 1:BqActual
                pRow = double(permMatrix_q(b, :)).'; 
                TPerm(b,:) = abs(compute_t_map(pRow, options.permutationMethod, e0_all, t0_base, w_row, RelScale, X, idxCoef, unitRows, unitVals, unitBasis));
            end
            
            tcritVec = quantile(TPerm, 1 - options.alphaValue, 1);  % 1×T

        case 'parametric'
            % --- parametric t-threshold based on #exchangeability units ---
            % nUnits was resolved in Section 2 via get_permut_unit
            
            % df ≈ nUnits-1; guard against degenerate cases
            df = max(nUnits - 1, 1);
            tcritScalar = tinv(1 - options.alphaValue/2, df);
            tcritVec    = repmat(abs(tcritScalar), 1, dur);
    end

    % Build sigMask only when we actually use threshold-based cluster detection.
    % In PreselectedCluster (ROI) mode, clusters come only from the preselected
    % windows; sigMask is left all-false so shading is optional and separate.
    if isempty(options.PreselectedCluster)
        sigMask = abs(tGLS_obs(:)) >= tcritVec(:);
    else
        sigMask = false(dur,1);
    end

    %% 8) Cluster detection ------------------------------------------------------
    if ~isempty(options.PreselectedCluster)
        % Use pre-identified time windows (no thresholding; no minClusterSize filter)
        if isempty(options.t)
            error('ClustME:PreselectedCluster','options.PreselectedCluster requires options.t (time vector).');
        end
        if size(options.PreselectedCluster,2) ~= 2
            error('ClustME:PreselectedClusterShape','options.PreselectedCluster must be K×2 [tStart tEnd].');
        end
        tvec = options.t(:);
        K    = size(options.PreselectedCluster,1);
        cStarts = zeros(K,1);
        cEnds   = zeros(K,1);
        for k = 1:K
            t0 = options.PreselectedCluster(k,1);
            t1 = options.PreselectedCluster(k,2);
            if t1 < t0, [t0,t1] = deal(t1,t0); end
            [~, s] = min(abs(tvec - t0));
            [~, e] = min(abs(tvec - t1));
            cStarts(k) = s;
            cEnds(k)   = e;
        end
    else
        % Threshold-defined clusters (from sigMask)
        [cStarts, cEnds] = clustme.mask_to_clusters(sigMask);

        keep = (cEnds - cStarts + 1) >= minClusterSamples;
        cStarts = cStarts(keep);
        cEnds   = cEnds(keep);
        K       = numel(cStarts);
    end


    %% 9) Observed Cluster Statistics (Inference Only) --------------------------

    obsClusterMass = zeros(1,K);

    for k = 1:K
        cols = cStarts(k):cEnds(k);
        tseg = tGLS_obs(cols);
        switch options.clusterMassMethod  % 'sum' | 'mean'
            case 'sum',  obsClusterMass(k) = sum(tseg.^2);
            case 'mean', obsClusterMass(k) = mean(tseg.^2);
        end
    end

    %% 10) Null distribution (static V; no LME refits) ----------------------

    % --- Build permutation null for maximum cluster mass statistic ---
    % (Matrix was already built in Section 6b)
    
    numPerms  = numPermsTotal;
    nullStats = zeros(numPerms,1);
    
    if options.fullLME || options.parallel    % fullLME mode is always parallel
        % Broadcast constants to workers (including precomputed signs)
        constPerms  = parallel.pool.Constant(permMatrix);
        constTcrit  = parallel.pool.Constant(tcritVec);
        constStarts = parallel.pool.Constant(cStarts);
        constEnds   = parallel.pool.Constant(cEnds);
        constK      = parallel.pool.Constant(K);
    end

    if options.fullLME
        % =========================================================================
        % PATH A: GOLD STANDARD (Full Refit Loop)
        % =========================================================================
        % This manually reconstructs y* and refits fitlme() at every step.
        % WARNING: Extremely slow. For benchmarking/validation only.

        constTbl      = parallel.pool.Constant(tblTemplate);
        constYHat0    = parallel.pool.Constant(yHat0_all);
        constE0       = parallel.pool.Constant(e0_all);

        if options.whitening
            constL = parallel.pool.Constant(L);
        end

        parfor sh = 1:numPerms
            % 1. Reconstruct Permuted Data (Freedman-Lane: y* = yHat0 + P*e0)
            pRow = constPerms.Value(sh, :);

            if strcmp(options.permutationMethod, 'withinSubject')
                % Shuffle: e0 is reordered
                e_star = constE0.Value(pRow, :);
            elseif strcmp(options.permutationMethod, 'signFlip')
                % SignFlip: e0 is flipped
                e_star = constE0.Value .* double(pRow');
            else
                % GroupLabel: We would shuffle the table column, but for one-sample
                % benchmarking we generally use signFlip. Fallback to e0.
                e_star = constE0.Value;
            end

            y_perm = constYHat0.Value + e_star;

            % 2. Fit LME at every timepoint for this shuffle
            tPerm = zeros(1, dur);
            currentTbl = constTbl.Value;

            for t = 1:dur
                if options.whitening
                    currentTbl.response = constL.Value * y_perm(:, t);
                else
                    currentTbl.response = y_perm(:, t);
                end
                try
                    % We assume 'FitMethod' matches the main run
                    m = fitlme(currentTbl, lmeFormula, 'FitMethod', options.FitMethod);
                    tPerm(t) = m.Coefficients.tStat(idxCoef);
                catch
                    tPerm(t) = 0;
                end
            end

            % 3. Calculate Max Cluster Statistic (Shared Logic)
            maxStat = 0;
            if hasPreselected
                % ROI Mode
                for kk = 1:constK.Value
                    cols = constStarts.Value(kk):constEnds.Value(kk);
                    if strcmp(options.clusterMassMethod, 'sum') 
                        cm = sum(tPerm(cols).^2);
                    else 
                        cm = mean(tPerm(cols).^2); 
                    end
                    if cm > maxStat, maxStat = cm; end
                end
            else
                % Threshold Mode
                mask = abs(tPerm(:)) >= constTcrit.Value(:);
                if any(mask)
                    idx = find(mask);
                    isStart = [true; diff(idx) > 1];
                    isEnd   = [diff(idx) > 1; true];
                    starts  = idx(isStart);
                    ends    = idx(isEnd);
                    keep    = (ends - starts + 1) >= minClusterSamples;
                    starts  = starts(keep); ends = ends(keep);

                    for kk = 1:numel(starts)
                        cols = starts(kk):ends(kk);
                        if strcmp(options.clusterMassMethod, 'sum') 
                            cm = sum(tPerm(cols).^2);
                        else 
                            cm = mean(tPerm(cols).^2); 
                        end
                        if cm > maxStat, maxStat = cm; end
                    end
                end
            end
            nullStats(sh) = maxStat;
        end

    else

        % =========================================================================
        % PATH B: FAST STATIC-V (Matrix Algebra)
        % =========================================================================
        % Standard efficient execution path.

        if options.parallel

            constE0     = parallel.pool.Constant(e0_all);      % n×T residuals from reduced model
            constT0     = parallel.pool.Constant(t0_base);     % 1×T projected reduced fit
            constWRow   = parallel.pool.Constant(w_row);
            constScale  = parallel.pool.Constant(RelScale);
            constX      = parallel.pool.Constant(X);
            constIdx    = parallel.pool.Constant(idxCoef);
            constURows  = parallel.pool.Constant(unitRows);
            constUVals  = parallel.pool.Constant(unitVals);
            constUBasis = parallel.pool.Constant(unitBasis);

            parfor sh = 1:numPerms
                % Take the precomputed sign pattern for this shuffle
                pRow = double(constPerms.Value(sh,:)).'; % nTrials×1
                tPerm = compute_t_map(pRow, options.permutationMethod, ...
                    constE0.Value, constT0.Value, constWRow.Value, ...
                    constScale.Value, constX.Value, constIdx.Value, ...
                    constURows.Value, constUVals.Value, constUBasis.Value);

                maxStat = 0;
                if hasPreselected
                    % --- Preselected windows (ROI mode) ---
                    % Use the broadcasted fixed windows (constStarts/Ends)

                    for kk = 1:constK.Value
                        cols = constStarts.Value(kk):constEnds.Value(kk);
                        cmROI = 0;
                        switch options.clusterMassMethod
                            case 'sum',  cmROI = sum(tPerm(cols).^2);
                            case 'mean', cmROI = mean(tPerm(cols).^2);
                        end
                        if cmROI > maxStat, maxStat = cmROI; end
                    end

                else
                    % --- Threshold-defined clusters (Data-driven mode) ---
                    % Ignore constStarts/constK here and detect fresh clusters on tPerm
                    mask = abs(tPerm(:)) >= constTcrit.Value(:);

                    if any(mask)
                        idx     = find(mask);
                        isStart = [true;  diff(idx) > 1];
                        isEnd   = [diff(idx) > 1; true];
                        starts  = idx(isStart);
                        ends    = idx(isEnd);

                        % apply minClusterSize (in samples)
                        keep    = (ends - starts + 1) >= minClusterSamples;
                        starts  = starts(keep);  ends = ends(keep);

                        if ~isempty(starts)
                            for kk = 1:numel(starts)
                                cols = starts(kk):ends(kk);
                                cm = 0;
                                switch options.clusterMassMethod
                                    case 'sum',  cm = sum(tPerm(cols).^2);
                                    case 'mean', cm = mean(tPerm(cols).^2);
                                end
                                if cm > maxStat, maxStat = cm; end
                            end
                        end
                    end
                end
                nullStats(sh) = maxStat;
            end
        else
            for sh = 1:numPerms
                % Take the precomputed sign pattern for this shuffle
                pRow = double(permMatrix(sh,:)).'; % nTrials×1
                tPerm = compute_t_map(pRow, options.permutationMethod, ...
                    e0_all, t0_base, w_row, RelScale, X, idxCoef, unitRows, unitVals, unitBasis);

                maxStat = 0;
                if hasPreselected
                    % --- Preselected windows (ROI mode) ---
                    % We use the broadcasted fixed windows (constStarts/Ends)

                    for kk = 1:K
                        cols = cStarts(kk):cEnds(kk);
                        cmROI = 0;
                        switch options.clusterMassMethod
                            case 'sum',  cmROI = sum(tPerm(cols).^2);
                            case 'mean', cmROI = mean(tPerm(cols).^2);
                        end
                        if cmROI > maxStat, maxStat = cmROI; end
                    end

                else
                    % --- Threshold-defined clusters (Data-driven mode) ---
                    % Ignore constStarts/constK here and detect fresh clusters on tPerm
                    mask = abs(tPerm(:)) >= tcritVec(:);

                    if any(mask)
                        idx     = find(mask);
                        isStart = [true;  diff(idx) > 1];
                        isEnd   = [diff(idx) > 1; true];
                        starts  = idx(isStart);
                        ends    = idx(isEnd);

                        % apply minClusterSize (in samples)
                        keep    = (ends - starts + 1) >= minClusterSamples;
                        starts  = starts(keep);  ends = ends(keep);

                        if ~isempty(starts)
                            for kk = 1:numel(starts)
                                cols = starts(kk):ends(kk);
                                cm = 0;
                                switch options.clusterMassMethod
                                    case 'sum',  cm = sum(tPerm(cols).^2);
                                    case 'mean', cm = mean(tPerm(cols).^2);
                                end
                                if cm > maxStat, maxStat = cm; end
                            end
                        end
                    end
                end
                nullStats(sh) = maxStat;
            end
        end
    end


    %% 11) p-values 

    % Each p-value is the proportion of permutations where the max cluster mass
    % exceeds the observed cluster mass.
    if K > 0
        pVals = zeros(1, K);
        B = numel(nullStats);
        for k = 1:K
            % Finite-permutation estimate: (b+1)/(B+1)
            b = sum(nullStats >= obsClusterMass(k));
            pVals(k) = (b + 1) / (B + 1);
        end
    else
        pVals = NaN(1, K);
    end

    %% 12) Post-hoc Cluster Characterization (Descriptive) -----------------------
    % Fits descriptive LMEs on cluster-collapsed data. 
    % Wrapped in try-catch so LME convergence failures do not discard P-values.
    tCvals          = zeros(1,K);
    coefPvals       = zeros(1,K);   % LME p-value of the chosen fixed effect per cluster
    clusterCovVars  = cell(1,K);    % random-effect variances per cluster
    residualVars    = zeros(1,K);   % residual variance per cluster
    AICs            = zeros(1,K);   % per-cluster AIC 
    clusterMeanLevels = nan(1,K);   % mean across trials of the chosen cluster measure
    varianceRatios = cell(1,K);

    for k = 1:K
        seg  = allResponses_preWhite(:, cStarts(k):cEnds(k));
        
        % 1. Compute scalar measure per trial
        switch options.clusterSummaryMetric
            case 'mean'
                vals = mean(seg, 2);                          % typical amplitude (length-invariant)
            case 'sum'
                vals = sum(seg, 2);                           % total signed mass (sample units)
            case 'median'
                vals = median(seg, 2);                        % robust typical amplitude (length-invariant)
            case 'signedPeak'
                % Robust "pre-smoothed" peak: sign(mean) * 95th percentile of |window|
                vals = sign(mean(seg, 2)) .* prctile(abs(seg), 95, 2);                           
            otherwise
                error('ClustME:BadClusterMetric','Unknown clusterSummaryMetric: %s', options.clusterSummaryMetric);
        end

        clusterMeanLevels(k) = mean(vals, 'omitnan');          % used only for plotting right-axis lines

        % 2. Fit Descriptive LME
        tblCluster = tblTemplate;
        tblCluster.response = vals;
        
        try
            m = fitlme(tblCluster, lmeFormula, 'FitMethod', options.FitMethod);
            
            % Extract stats
            AICs(k)        = m.ModelCriterion.AIC;
            tCvals(k)      = m.Coefficients.tStat(idxCoef);
            coefPvals(k)   = m.Coefficients.pValue(idxCoef);
            residualVars(k)= m.MSE;
            
            cp = covarianceParameters(m);
            vars = cellfun(@(C) diag(C), cp, 'UniformOutput', false);
            clusterCovVars{k} = vertcat(vars{:});
            varianceRatios{k} = clusterCovVars{k} ./ (clusterCovVars{k} + residualVars(k));
            
        catch ME
            warning('ClustME:ClusterFitFail', 'Descriptive fit failed for cluster %d: %s', k, ME.message);
            tCvals(k)         = NaN; 
            AICs(k)           = NaN;  
            coefPvals(k)      = NaN;   
            residualVars(k)   = NaN; 
            clusterCovVars{k} = NaN; 
            varianceRatios{k} = NaN;
        end
    end

    %% 13) Output Packaging 
    
    % ---  Visualization Data (Context for Plotting) ---
    vis_data = struct();
    
    % Core T-map Data (matches 'stats' input of plot_tmap)
    vis_data.Tmap         = tGLS_obs;
    vis_data.Fs           = options.Fs;
    vis_data.coefName     = options.testCoefficient;
    if isempty(vis_data.coefName), vis_data.coefName = 'Intercept'; end
    vis_data.alpha        = options.alphaValue;
    vis_data.tcrit        = tcritVec;     % The exact time-varying threshold
    vis_data.sigMask      = sigMask;      % The significance mask
    vis_data.cStarts      = cStarts;
    vis_data.cEnds        = cEnds;
    vis_data.clusterLevel = clusterMeanLevels; % For the right-hand axis
    
    % Null Distribution Data (matches inputs of plot_null_hist)
    vis_data.nullStats    = nullStats;
    vis_data.obsClusterMass = obsClusterMass;
    vis_data.pVals        = pVals;
    vis_data.alphaValue   = options.alphaValue; 
   
    % Metadata
    vis_data.t            = options.t;
    vis_data.clusterMassMethod = options.clusterMassMethod;

    % Diagnostic data
    vis_data.vstat = vstat;
    vis_data.idxAnchor = idxAnchor;


    clusters = repmat(struct( ...
        'type','', 'start',[], 'end',[], 'mass',[], 'measure', [], ...
        'p_value',[], ...
        'lmeTStat',[], ...
        'lmePValue',[], ...
        'covVars',[], 'resVar',[], 'varianceRatios',[], 'AIC',[]), 1, K);

    for k = 1:K
        clusters(k).start          = cStarts(k);
        clusters(k).end            = cEnds(k);
        clusters(k).mass           = obsClusterMass(k);   % Σ t^2 (or mean t^2) over the cluster
        clusters(k).measure        = clusterMeanLevels(k);   % descriptive cluster measure (mean/median/sum/peak)
        clusters(k).p_value        = pVals(k);          % permutation-based cluster p
        clusters(k).lmeTStat       = tCvals(k);         % chosen coefficient t
        clusters(k).lmePValue      = coefPvals(k);      % chosen coefficient p
        clusters(k).covVars        = clusterCovVars{k};
        clusters(k).resVar         = residualVars(k);
        clusters(k).varianceRatios = varianceRatios{k};
        clusters(k).AIC            = AICs(k);
        clusters(k).type           = ternary(tCvals(k)>0, 'positive', 'negative');
    end

    if options.verbose
        fprintf('LME on %d trials → %d clusters (%d with p<%.2f) | model: %s | time = %.2fs\n', ...
            nTrials, K, sum(pVals < options.alphaValue), options.alphaValue, lmeFormula, toc);
    end
end

%---------------------------------------------------------------------
%%    HELPER FUNCTIONS    %%
%---------------------------------------------------------------------

function out = ternary(cond,a,b)
    if cond, out = a; else, out = b; end
end

function [solveV, usedLDL] = make_V_solver(V)
% make_V_solver - Generates a linear solver closure for the marginal covariance matrix
%
% Internal Context:
%   Provides a reusable closure for Steps 5 and 6 of the ClustME pipeline to 
%   apply the inverse of the Static-V matrix without explicit matrix inversion.
%
% Inputs:
%   V        - [n × n double] Static marginal covariance matrix at the time anchor.
%
% Outputs:
%   solveV   - [function_handle] Closure accepting a matrix B to compute V \ B.
%   usedLDL  - [1×1 logical] Diagnostic flag (true if LDL fallback was triggered).
%
% Algorithmic & Exactness Notes:
%   * Symmetry & Memory: Enforces matrix symmetry prior to decomposition to 
%     prevent floating-point errors. Returning a closure encapsulates the sparse 
%     decomposition matrices, preventing RAM duplication across parallel workers.
%   * Solver Hierarchy: Prioritises sparse Cholesky decomposition. If near-singular, 
%     it injects a dynamic ridge scaled to the median diagonal variance.
%   * Indefinite Fallback: If the ridge fails, the solver transitions to an LDLT 
%     decomposition, ensuring the GLS projection completes on artefacted data.

    usedLDL = false;
    Vs = (V + V') * 0.5;                 % enforce symmetry
    n  = size(Vs,1);

    % --- Fast path: Cholesky if (near) PD ---
    [R,pdef] = chol(sparse(Vs));         % Vs = R' * R (R upper)
    if pdef == 0
        solveV = @(B) R \ (R' \ B);      % X = R \ (R' \ B)
        return;
    end

    % --- Tiny ridge if close to PD ---
    diagVs = full(diag(Vs));
    scale  = max(1, median(diagVs(isfinite(diagVs) & diagVs~=0)));
    epsV   = 1e-8 * scale;
    [R,pdef] = chol(sparse(Vs + epsV*speye(n)));
    if pdef == 0
        solveV = @(B) R \ (R' \ B);
        return;
    end

    % --- Robust fallback: LDL with permutation matrix ---
    % P'*Vs*P = L*D*L', works for SPD/PSD/indefinite. MATLAB handles 1x1/2x2 D blocks in "\".
    [L,D,P] = ldl(sparse(Vs));           % P is a sparse permutation matrix
    usedLDL = true;
    solveV  = @(B) P * (L' \ (D \ (L \ (P' * B))));
end

function tMap = compute_t_map(pRow, method, e0, t0, wRow, Sigma, X, colIdx, uRows, uVals, uBasis)
% compute_t_map - Internal GLS t-map solver for a single permutation iteration
%
% Internal Context:
%   Executed during Steps 7 and 10 of the ClustME pipeline. It applies a single 
%   permutation instruction to generate a pseudo-t-statistic trace. Execution 
%   routes between a residual projection (y-permutation) and a whitened refit 
%   (X-permutation) depending on the exchangeability scheme.
%
% Inputs:
%   pRow   - [N × 1 double] Permutation vector (signs or indices).
%   method - [char] Permutation scheme ('signFlip', 'withinSubject', 'groupLabel').
%   e0     - [N × T double] Reduced-model residuals.
%   t0     - [1 × T double] Baseline reduced-model projection.
%   wRow   - [1 × N double] GLS projection weights.
%   Sigma  - [T × 1 double] Relative scaling factor for studentisation.
%   X      - [N × P double] Full static design matrix.
%   colIdx - [1 × 1 double] Column index of the tested fixed effect.
%   uRows  - [C × 1 cell] Row indices for each exchangeability unit.
%   uVals  - [C × 1 double] Raw label magnitudes for each unit.
%   uBasis - [C × 1 cell] Whitened basis shapes for each unit.
%
% Outputs:
%   tMap   - [1 × T double] Permuted Generalised Least Squares t-statistic trace.
%
% Algorithmic & Exactness Notes:
%   * Execution Routing (Fast Path): Applies Freedman-Lane pseudo-data projection 
%     for 'signFlip', 'withinSubject', and 'wildBootstrap' schemes. 
%   * Whitened Refit (Slow Path): For the 'groupLabel' scheme (balanced designs), 
%     it reconstructs the whitened regressor column by multiplying the permuted 
%     label magnitude by the subject's exact whitened basis shape. 
%   * OLS Refit Assumption: The 'groupLabel' path executes an Ordinary Least 
%     Squares (OLS) refit on the strict mathematical assumption that the data 
%     was whitened upstream, rendering V=I.
    
    if strcmp(method, 'groupLabel')
        % --- SLOW PATH: X-permutation for group-label shuffling ---
        nTrials = size(X, 1);
        nUnits  = numel(pRow);
        
        % 1. Reconstruct Whitened Regressor Column
        X_new_col = zeros(nTrials, 1);
        permVals  = uVals(pRow); % The RAW labels (0 or 1) from the shuffled units

        for k = 1:nUnits
            % The new whitened block is: (Label Magnitude) * (Subject's Whitening Shape)
            X_new_col(uRows{k}) = permVals(k) * uBasis{k};
        end        
        
        % 2. Update Design Matrix (X*)
        X_star = X;
        X_star(:, colIdx) = X_new_col;

        % 3. OLS Refit on whitened residuals (Assuming V=I after whitening)
        % (X'*X) \ (X'*Y)
        XtX = X_star' * X_star;
        betas = XtX \ (X_star' * e0);
        
        % 4. Compute t-stat with updated SE
        invXtX = inv(XtX);
        se_j   = sqrt(invXtX(colIdx, colIdx));
        tMap   = betas(colIdx, :) ./ (se_j * Sigma.');
        
    else
        % --- FAST PATH: Sign Flip or Within-Block Shuffle (y-permutation) ---
        if strcmp(method, 'withinSubject')
            % pRow contains trial indices
            term = e0(pRow, :);
        else
            % pRow contains signs
            term = pRow .* e0;
        end

        % Fast GLS t for all time points under Freedman–Lane pseudo-data:
        %   y* = ŷ0 + signs ⊙ e0  ⇒  T* = (a' y*) / Σ_t
        %   t = (t0 + wRow * term) / Sigma
        tMap = (t0 + (wRow * term)) ./ Sigma.';
    end
end