function [V, mdlStatic, idxStatic, vstat] = build_static_V(tblTemplate, allResponses, lmeFormula, SigmaVec, options)
%
%   build_static_V constructs the marginal covariance matrix (Static-V) required 
%   for the fast GLS projection pathway. It extracts variance components from 
%   a representative Linear Mixed-Effects (LME) model and stabilizes them via 
%   temporal pooling windows.
%
% Syntax:
%   [V, mdlStatic, idxStatic, vstat] = clustme.build_static_V( ...
%       tblTemplate, allResponses, lmeFormula, SigmaVec, options)
%
% Inputs:
%   tblTemplate  - [N×V table] Master design table (trials × variables). The number 
%                  of rows must exactly match the N dimension of allResponses.
%   allResponses - [N×T double] Response matrix (trials × time samples).
%   lmeFormula   - [char] Wilkinson string defining the LME structure.
%   SigmaVec     - [T×1 double] Vector of per-time residual standard deviations.
%
% Options (Name-Value arguments):
%   .Vmode        [char]       Pooling strategy: 'adaptiveLocal', 'local', 'global', 
%                              or 'identity'. (Default: 'adaptiveLocal')
%   .t            [1×T double] Optional time vector. (Default: [])
%   .Z_thresh     [double]     Target median |z|-score for adaptive window stability. (Default: 3.0)
%   .TimeAnchor   [double]     Scalar time for anchoring the window. (Default: [])
%   .FitMethod    [char]       LME estimation method: 'REML', 'ML'. (Default: 'REML')
%   .verbose      [logical]    Print internal stability diagnostics. (Default: false)
%
% Outputs:
%   V         - [N×N sparse double] Static marginal covariance matrix.
%   mdlStatic - [LinearMixedModel] The fitted model object at the chosen anchor point.
%   idxStatic - [1×1 double] Sample index of the representative time anchor.
%   vstat     - [struct] Diagnostics for the Static-V assumption with fields:
%       .posdefOk            [logical] True if V is numerically positive-definite.
%       .cond_XtVinvX        [double] 1-norm condition estimate of the static normal matrix.
%       .sigmaCV             [double] Robust coefficient of variation of SigmaVec.
%       .anchorZ             [double] Robust z-score of SigmaVec at the chosen anchor.
%       .windowMedianAbsZ    [double] Median absolute z-score of the chosen pooling window.
%       .diagV_CV            [double] Coefficient of variation of the diagonal of V.
%       .stableFlag          [logical] Master flag indicating overall numerical stability.
%       .lambdaV             [double] Ridge penalty applied if V required regularisation.
%       .barSigma            [double] Window-averaged residual standard deviation.
%       .fellBackToMaxWindow [logical] True if the adaptive window hit the maximum limit.
%       .windowSize          [double] Number of samples in the final pooling window.
%       .windowIdx           [1×W double] Sample indices defining the pooling window.
%       .barPsi              [1×R cell] Window-averaged random-effect covariance matrices.
%       .barMSE              [double] Window-averaged residual variance (barSigma^2).
%
% POOLING BEHAVIOUR (Vmode)
% -------------------------
% • 'adaptiveLocal' (Default): Identifies an anchor timepoint and expands a 
%   symmetric window until the local variance stabilizes (median |z| <= Z_thresh). 
%   If stabilization isn't reached within 25% of the epoch, it clamps to the 
%   maximum local width (Soft Fallback).
% • 'local': Uses a fixed 5-sample window centered on the anchor.
% • 'global': Pools variance components across the entire analysis window.
% • 'identity': Forces V = I. This creates an OLS-equivalent estimator that 
%   ignores hierarchical dependencies and heteroscedasticity.
%
% Dependencies:
%   - MATLAB R2019b or newer. (Tested on R2025b)
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

    arguments
        tblTemplate table
        allResponses double
        lmeFormula (1,:) char
        SigmaVec double
        
        options.Vmode                  (1,:) char {mustBeMember(options.Vmode,{'adaptiveLocal','local','global','identity'})} = 'adaptiveLocal'
        options.t                      (1,:) double = []       % optional time vector (seconds or custom)
        options.Z_thresh               (1,1) double {mustBePositive} = 3.0   % Target median |z|-score for adaptive window
        options.TimeAnchor             double = []             % optional scalar time (same units as options.t)
        options.FitMethod              (1,:) char {mustBeMember(options.FitMethod,{'REML','ML'})} = 'REML'
        options.verbose                (1,1) logical = false   % print internal diagnostics (V stability, etc.)
    end

    % ---- Local constants ---------------------------------------------------
    LOCAL_V_WINDOW        = 5;      % minimum/target window width (in samples)
    MAX_LOCAL_FRACTION    = 0.25;   % maximum adaptiveLocal window size as a fraction of T
    MAX_COND_XTVINVX      = 1e8;    % soft upper bound on cond(X'V^{-1}X)
    MAX_SIGMA_CV          = 0.7;    % rough threshold for strong time-varying residual scale
    % -------------------------------------------------------------------------

    n   = size(tblTemplate,1);         % number of rows / trials
    dur = size(allResponses,2);        % number of time samples T

    % Robust z-scores of per-time residual SDs (SigmaVec)
    SigmaVec = SigmaVec(:);
    SigmaValid = SigmaVec(isfinite(SigmaVec));
    medSigma   = median(SigmaValid);
    madSigma   = max(mad(SigmaValid, 1), 0.01 * medSigma); % enforce a floor of 1% of the median to prevent exploding Z-scores
    if madSigma == 0 || ~isfinite(madSigma)
        zAll = zeros(size(SigmaVec));
    else
        zAll = (SigmaVec - medSigma) / madSigma;
    end

    % ---- Determine anchor index (sample) -------------------------------------
    if isfield(options,'TimeAnchor') && ~isempty(options.TimeAnchor) && ...
       isfield(options,'t') && ~isempty(options.t)
        tvec = options.t(:);
        [~, idxAnchor] = min(abs(tvec - options.TimeAnchor));
    else
        [~, idxAnchor] = min(abs(SigmaVec - median(SigmaVec,'omitnan')));
    end
    idxAnchor = max(1, min(dur, idxAnchor));  % guard

    % ---- Decide window indices winIdx according to Vmode ---------------------
    if isfield(options,'Vmode')
        Vmode = options.Vmode;
    else
        Vmode = 'adaptiveLocal';
    end

    MAX_LOCAL_SAMPLES = max(1, floor(MAX_LOCAL_FRACTION * dur));
    useGlobalV        = strcmp(Vmode,'global');

    winStart = idxAnchor;
    winEnd   = idxAnchor;
    expandIter = 0;  % counts symmetric expansions in adaptiveLocal
    UseMaxWindow = false;   % true if adaptiveLocal reaches the maximum local window before meeting the stability criterion

    if ~useGlobalV
        switch Vmode
            case 'local'
                % Fixed local window of width LOCAL_V_WINDOW (clipped at edges)
                halfL    = floor((LOCAL_V_WINDOW-1)/2);
                winStart = max(1, idxAnchor - halfL);
                winEnd   = min(dur, idxAnchor + halfL);

            case 'adaptiveLocal'
                % Expand symmetrically until mean |z|-score in window ≤ Z_THRESH,
                % or until the maximum local window is reached, in which case the 
                % widest evaluated local window is retained.
                while true
                    expandIter = expandIter + 1;
                    winIdx = winStart:winEnd;
                    zWin   = zAll(winIdx);
                    zWin   = zWin(isfinite(zWin));
                    if isempty(zWin)
                        medianAbsZ = 0;
                    else
                        medianAbsZ = median(abs(zWin));
                    end

                    if medianAbsZ <= options.Z_thresh
                        break;
                    end

                    % Stop expansion at the maximum allowed local window and 
                    % record that the adaptive criterion was not reached.
                    if numel(winIdx) >= MAX_LOCAL_SAMPLES || (winStart == 1 && winEnd == dur)
                        if options.verbose
                            fprintf('[build_static_V] Window hit max limit (%d samples) without stabilizing (Median|Z|=%.2f). Using max local window.\n', ...
                                numel(winIdx), medianAbsZ);
                        end
                        UseMaxWindow = true;  
                        break;
                    end

                    % Symmetric expansion where possible
                    winStart = max(1, winStart - 1);
                    winEnd   = min(dur, winEnd   + 1);
                end

            case 'identity'
                % Identity mode: No window search needed.
                % We just set a dummy window at the anchor to allow mdlStatic
                % to be extracted later. V will be overwritten with Identity.
                winStart = idxAnchor;
                winEnd   = idxAnchor;

            otherwise
                error('clustme:build_static_V:BadVmode', ...
                    'Unknown options.Vmode="%s". Expected ''adaptiveLocal'', ''local'' or ''global''.', Vmode);
        end

        % Enforce a minimum width for local / adaptiveLocal windows
        if ~useGlobalV
            winIdx = winStart:winEnd;
            if numel(winIdx) < LOCAL_V_WINDOW
                extra = LOCAL_V_WINDOW - numel(winIdx);
                growL = floor(extra/2);
                growR = extra - growL;
                winStart = max(1, winStart - growL);
                winEnd   = min(dur, winEnd   + growR);
                winIdx   = winStart:winEnd;
            end
        end
    end

    %  Global V mode: pool covariance components over all timepoints
    if useGlobalV
        winIdx = 1:dur;
    end

    % --- Final window mean |z| (used for reporting & stability) ------------
    zWin_final = zAll(winIdx);
    zWin_final = zWin_final(isfinite(zWin_final));
    if isempty(zWin_final)
        medianAbsZ_final = 0;
    else
        medianAbsZ_final = median(abs(zWin_final));
    end

    % --- Diagnostics: final window & expansion count -------------------------
    if options.verbose
        if strcmp(Vmode,'adaptiveLocal')
            fprintf('[build_static_V] mode=%s | anchor=%d | window=[%d,%d] (width=%d) | expansions=%d | useGlobal=%d\n', ...
                Vmode, idxAnchor, winIdx(1), winIdx(end), numel(winIdx), expandIter, useGlobalV);
        elseif strcmp(Vmode,'local')
            fprintf('[build_static_V] mode=%s | anchor=%d | window=[%d,%d] (width=%d)\n', ...
                Vmode, idxAnchor, winIdx(1), winIdx(end), numel(winIdx));
        else % 'global'
            fprintf('[build_static_V] mode=%s | using all T=%d samples\n', Vmode, dur);
        end
    end

    % Representative index for X/AIC/etc.: prefer the anchor if inside window
    if idxAnchor >= winIdx(1) && idxAnchor <= winIdx(end)
        centerIdx = idxAnchor;
    else
        centerIdx = round(median(winIdx));
    end

    % ---- Component-average covariance across window winIdx -------------------
    % Representative model at centerIdx
    tblRep = tblTemplate;
    tblRep.response = allResponses(:, centerIdx);
    mdlRep = fitlme(tblRep, lmeFormula, 'FitMethod', options.FitMethod);

    % Precompute Z_r, q_r, m_r from representative model (do not depend on response)
    psi_rep = covarianceParameters(mdlRep);
    R = numel(psi_rep);
    Zrs = cell(1,R); qrs = zeros(1,R); mrs = zeros(1,R);
    for r = 1:R
        Zrs{r} = sparse(designMatrix(mdlRep,'Random', r));   % n × (m_r * q_r)
        qrs(r) = size(psi_rep{r},1);
        pZ     = size(Zrs{r},2);
        assert(mod(pZ, qrs(r))==0, 'Z_r width (%d) not divisible by q_r (%d).', pZ, qrs(r));
        mrs(r) = pZ / qrs(r);
    end

    % Accumulate means of Psi_r and sigma^2 across the chosen window
    sumMSE = 0;
    sumPsi = cell(1,R);
    for r = 1:R
        sumPsi{r} = zeros(qrs(r), qrs(r));
    end

    for ii = 1:numel(winIdx)
        tblTmp = tblTemplate;
        tblTmp.response = allResponses(:, winIdx(ii));
        mtmp = fitlme(tblTmp, lmeFormula, 'FitMethod', options.FitMethod);
        [psi_tmp, mse_tmp] = covarianceParameters(mtmp);
        for r = 1:R
            sumPsi{r} = sumPsi{r} + psi_tmp{r};
        end
        sumMSE = sumMSE + mse_tmp;
    end

    barMSE = sumMSE / numel(winIdx);
    barPsi = cellfun(@(S) S/numel(winIdx), sumPsi, 'UniformOutput', false);

    % Rebuild V once from averaged components (same as mean(V_i) because Z_r are constant)
    V = barMSE * speye(n);
    for r = 1:R
        Dr_bar = sparse(kron(speye(mrs(r)), barPsi{r}));
        V = V + Zrs{r} * (Dr_bar * Zrs{r}');
    end
    V = (V + V') * 0.5;

    % If identity mode is requested, discard the calculated covariance and force I.
    if strcmp(Vmode, 'identity')
        V = speye(n);
    end

    % Keep representative model and index
    mdlStatic = mdlRep;
    idxStatic = centerIdx;

     % ---- Static-V diagnostics ---------
    X0 = designMatrix(mdlStatic,'Fixed');

    % Minimal V^{-1}X for diagnostics (self-contained; no dependency on make_V_solver)
    Vs         = (V + V') * 0.5;
    lambdaUsed = 0;                    % 0 means "no regularisation applied"

    % First attempt: Cholesky on the unregularised Vs
    try
        [Rchol, pdef] = chol(sparse(Vs));
    catch
        pdef = 1;                      % treat any error as a failure
    end

    % If V is not PD, try minimal shrinkage towards barMSE * I
    if pdef ~= 0
        Vs_orig      = Vs;
        I_n          = speye(size(Vs_orig,1));
        lambdasToTry = [1e-4, 1e-3, 1e-2, 1e-1, 0.5];

        for lam = lambdasToTry
            V_try = (1 - lam) * Vs_orig + lam * (barMSE * I_n);
            V_try = (V_try + V_try') * 0.5;

            try
                [Rchol_try, p_try] = chol(sparse(V_try));
            catch
                p_try = 1;
            end

            if p_try == 0
                Vs         = V_try;        % adopt regularised V
                Rchol      = Rchol_try;
                lambdaUsed = lam;
                pdef       = 0;

                fprintf(['[build_static_V] NOTE: V was not positive-definite; ', ...
                         'applied shrinkage towards barMSE*I with lambda = %.3g.\n'], ...
                         lambdaUsed);
                break
            end
        end
    end

    if pdef == 0
        % Successful chol (original or regularised)
        M1_0 = Rchol \ (Rchol' \ X0);     % V^{-1} X
    else
        % Shrinkage attempts failed – fall back to LDL as before
        [L,D,P] = ldl(sparse(Vs));
        M1_0    = P * (L' \ (D \ (L \ (P' * X0))));
    end

    XtVinvX_0 = X0' * M1_0;

    vstat = assess_V_stability(Vs, XtVinvX_0, SigmaVec, idxStatic, medianAbsZ_final, options.Z_thresh);
    vstat.lambdaV = lambdaUsed;
    vstat.barSigma = sqrt(barMSE);
    vstat.fellBackToMaxWindow = UseMaxWindow;
    vstat.windowSize = numel(winIdx);
    vstat.windowIdx  = winIdx;
    vstat.barPsi = barPsi;
    vstat.barMSE = barMSE;

    if options.verbose
        fprintf('[V-stability] PD=%d | cond(X''V^{-1}X)=%.2g | CV(Sigma)=%.2g | med|z|=%.2f | CV(diagV)=%.2g | stable=%d\n', ...
            vstat.posdefOk, vstat.cond_XtVinvX, vstat.sigmaCV, vstat.windowMedianAbsZ, vstat.diagV_CV, vstat.stableFlag);
    end

    % Warn the user if static-V diagnostics are poor
    if ~vstat.posdefOk
        warning('clustme:build_static_V:VNotPD', ...
            'Static V is not numerically positive definite; permutation results may be unstable. Consider options.Vmode="global" or revising TimeAnchor.');
    end

    if vstat.cond_XtVinvX > MAX_COND_XTVINVX
        warning('clustme:build_static_V:VIllConditioned', ...
            'cond(X''V^{-1}X)=%.2g; GLS contrast may be numerically unstable.', vstat.cond_XtVinvX);
    end

    if ~strcmp(Vmode, 'identity') && vstat.sigmaCV > MAX_SIGMA_CV
        warning('clustme:build_static_V:SigmaHeterogeneous', ...
            'CV(SigmaVec)≈%.2f suggests strong time-varying residual scale; check the static-V assumption.', vstat.sigmaCV);
    end
end


% -------------------------------------------------------------------------
function vstat = assess_V_stability(Vs, XtVinvX, SigmaVec, idxStatic, medianAbsZ, zThresh)
% assess_V_stability - Evaluates diagnostic metrics for the static-V assumption
%
% Internal Context:
%   A local helper function executed at the conclusion of build_static_V.m. It 
%   calculates numerical stability and statistical validity metrics for the 
%   marginal covariance matrix. These metrics are used upstream to trigger 
%   diagnostic warnings and are exported via the vis_data structure in the main 
%   ClustME API for post-hoc evaluation of the 'adaptiveLocal' stabilisation 
%   heuristic.
%
% Inputs:
%   Vs         - [n × n sparse double] Symmetrised static marginal covariance matrix.
%   XtVinvX    - [p × p double] Normal equations matrix from the static LME fit.
%   SigmaVec   - [T × 1 double] Vector of per-time residual standard deviations.
%   idxStatic  - [double] Sample index of the representative time anchor.
%   medianAbsZ - [double] Median absolute z-score of the final pooling window.
%   zThresh    - [double] Stability threshold targeted by the adaptive window.
%
% Outputs:
%   vstat      - [struct] Diagnostic metrics containing the following fields:
%       .posdefOk         [logical] True if Vs is strictly positive definite.
%       .cond_XtVinvX     [double] 1-norm condition estimate of the normal matrix.
%       .sigmaCV          [double] Robust coefficient of variation of SigmaVec.
%       .anchorZ          [double] Robust z-score of the residual scale at the anchor.
%       .windowMedianAbsZ [double] Preserved median absolute z-score.
%       .diagV_CV         [double] Coefficient of variation of the Vs diagonal.
%       .stableFlag       [logical] Master flag indicating overall stability.
%
% Algorithmic & Exactness Notes:
%   * Adaptive Window Validation: The stableFlag validates the success of the 
%     'adaptiveLocal' pathway by checking if medianAbsZ <= zThresh. If the 
%     parent function clamped to a maximum window size before variance 
%     stabilised, this flag captures the heuristic failure.
%   * Independent Verification: Executes a secondary Cholesky decomposition on 
%     Vs to verify its final state, accounting for any ridge shrinkage 
%     regularisation applied upstream.
%   * Division Guards: Calculates dispersion using the interquartile range (IQR) 
%     and unscaled median absolute deviation (MAD). It enforces a floor of 
%     machine epsilon (eps) on denominators to prevent division-by-zero errors 
%     on perfectly uniform data.
%   * Sparse Fallbacks: Casts the normal equations matrix to sparse format 
%     prior to condition estimation, mathematically forcing the condition 
%     number to infinity if the estimation solver fails.

    SigmaVec = SigmaVec(:);
    if idxStatic < 1 || idxStatic > numel(SigmaVec) || ~isfinite(SigmaVec(idxStatic))
        error('clustme:assess_V_stability:BadAnchor', ...
              'idxStatic out of range or SigmaVec(anchor) is NaN.');
    end

    % 1) Positive definiteness of V
    [~, pdef] = chol(Vs);
    posdefOk = (pdef == 0);

    % 2) Conditioning of XtVinvX (static GLS normal matrix)
    try
        cond_XtVinvX = condest(sparse(XtVinvX));
    catch
        cond_XtVinvX = inf;
    end

    % 3) Robust dispersion of per-time residual scales
    s = SigmaVec(isfinite(SigmaVec));
    medS = median(s);
    iqrS = iqr(s);
    madS = mad(s, 1);  % unscaled MAD
    sigmaCV = iqrS / max(eps, medS);
    anchorZ = abs(SigmaVec(idxStatic) - medS) / max(eps, madS);

    % 4) Heteroskedasticity proxy from diag(V)
    dv = full(diag(Vs));
    dv = dv(isfinite(dv) & dv > 0);
    if isempty(dv)
        diagV_CV = NaN;
    else
        diagV_CV = iqr(dv) / max(eps, median(dv));
    end

    % 5) Coarse stability flag:
    stableFlag = (posdefOk) ...
        & (cond_XtVinvX <= 1e8) ...
        & (sigmaCV       <= 0.5) ...
        & (medianAbsZ    <= zThresh);

    vstat = struct('posdefOk',posdefOk, ...
        'cond_XtVinvX',cond_XtVinvX, ...
        'sigmaCV',sigmaCV, ...
        'anchorZ',anchorZ, ...
        'windowMedianAbsZ',medianAbsZ, ...
        'diagV_CV',diagV_CV, ...
        'stableFlag',stableFlag);
end
