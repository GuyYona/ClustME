# ClustME: Fast Cluster-based Permutation Testing with Linear Mixed-Effects Models
Version 1.0.0

## Overview

ClustME implements a cluster-based permutation test for hierarchical 
time-series data with repeated observations, subject- or unit-level structure, and trial-level predictors. 
The test statistic at each time sample is a Generalized Least Squares (GLS) contrast derived from a linear mixed-effects (LME) model. 

To make time-resolved mixed-effects inference computationally feasible, ClustME estimates a static marginal covariance matrix V at a reference timepoint and reuses it during null generation. This avoids exhaustive mixed-model refitting inside the randomisation loop, while preserving cluster-level inference when the selected randomisation method matches the design's exchangeability structure.

## Installation

### Requirements
   - MATLAB R2019b or newer (Validated on R2025b).
   - Statistics and Machine Learning Toolbox
   - Parallel Computing Toolbox (optional, only required for parallel execution)

### Add ClustME to the MATLAB path
Download or clone the repository, then add the repository root (not the `+clustme` folder) to the MATLAB path from the MATLAB Command Window: 

```matlab 
addpath('/path/ClustME');
```
This adds ClustME for the current MATLAB session. To make the path persistent, you may then run:

```matlab 
savepath;
```

### Confirm installation
Verify that MATLAB can find both the main function and the package namespace:

```matlab 
which ClustME 
clustme.version 
``` 
You should see the path to `ClustME.m` and the installed version number.

## Quick start

### Build a minimal design table

ClustME expects the response data and metadata separately:

- `responses` is an `N × T` numeric matrix.
  - `N` = observations, such as trials, epochs, or repeated measurements.
  - `T` = time samples.
- `design` is an `N`-row table.
  - Each row describes the corresponding row of `responses`.
  - Variables used in the model formula must appear in this table.

For a minimal one-sample repeated-measures design, the only required design variable is the subject or experimental-unit identifier:

```matlab
nSubjects = 12;
nTrialsPerSubject = 20;
nTime = 100;

Subject = repelem((1:nSubjects)', nTrialsPerSubject);
Subject = categorical(Subject);

design = table(Subject);

rng(1042, 'twister'); % used for reproducability
responses = randn(height(design), nTime) + 0.4*exp(-((1:nTime)-55).^2/(2*8^2));
```


Here, each row of `responses` is one trial-level observation, and `design.Subject` identifies the subject that trial came from. This dataset contains a small simulated positive deflection around sample 55, added to every observation. It is intended only to demonstrate the input format and a minimal ClustME call.

### Minimal ClustME call

For a one-sample test of whether the population-level response differs from zero:

```matlab
lmeFormula = 'response ~ 1 + (1|Subject)';

clusters = ClustME(responses, design, lmeFormula);
```

This uses the default settings, including subject-level sign flipping for an intercept test.

### Inspect the output

`clusters` contains observed candidate clusters and their cluster-level p-values.

```matlab
clusters

sig = find([clusters.p_value] < 0.05);

fprintf('Significant cluster: samples %d-%d, mass = %.3f, p = %.4f\n', ...
    clusters(sig).start, clusters(sig).end, ...
    clusters(sig).mass, clusters(sig).p_value);

fprintf('Non-significant candidate clusters: %d\n', numel(clusters) - 1);
```

If no candidate clusters are detected, `clusters` will be empty.

To also return model statistics and visualisation data, request the optional outputs:

```matlab
[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula);
```

`mstats` contains model information and run provenance. `vis_data` contains the arrays used for plotting the observed t-map, cluster-forming threshold, detected clusters, and null distribution.


### Specify analysis options

The minimal call uses default settings. To control analysis settings, append name-value options after the three required inputs:

```matlab
[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula, ...
    'numPerms', 10000, ...     % <-- this is higher than the default 5000. Increase for more complex designs or noisy data. 
    'Fs', 100, ...
    'permutationMethod', 'signFlip', ...
    'permuteUnit', 'Subject');
```

This has the same structure as the minimal call, but specifies selected options explicitly. See [Options](#options) for the full list of available name-value arguments.

The randomisation method must match the exchangeability structure of the design. See [Choosing the randomisation method](#choosing-the-randomisation-method) before applying ClustME to a new experimental design.

### Run the synthetic demo
The minimal example above shows the required input structure. To see a complete worked analysis with generated data, plotting, and example outputs, run:
```matlab
run(fullfile('examples', 'demo_one_sample.m'))
```

The demo uses `clustme.bench_generator` to create example data, runs ClustME on a one-sample hierarchical time-series dataset, prints any detected clusters, and opens example visualisations. For real analyses, users provide their own `responses` matrix and matching `design` table, as described in [Input data format](#input-data-format).

## Input data format

ClustME expects three required inputs:

```matlab
clusters = ClustME(responses, design, lmeFormula);
```

### `responses`

`responses` is an `N × T` numeric matrix.

- `N` is the number of observations, such as trials, epochs, cells, or repeated measurements.
- `T` is the number of time samples.
- Each row is one observation.
- Each column is one time sample.
- All values must be finite. Remove or impute `NaN` and `Inf` values before calling ClustME.

Do not reshape the data into long format with one row per observation-timepoint pair. Time is represented by the columns of `responses`.

### `design`

`design` is an `N`-row MATLAB table containing the metadata used by the model formula.

Each row of `design` must describe the corresponding row of `responses`. For example, if row 25 of `responses` is a trial from subject 3 in condition `Oddball`, then row 25 of `design` should contain that subject and condition information.

The height of `design` must match the number of rows in `responses`:

```matlab
height(design) == size(responses, 1)
```

Variables used in `lmeFormula` must appear as columns in `design`.

Grouping variables used in random-effects terms should be categorical, string, or cell-string variables. For example:

```matlab
design.Subject = categorical(design.Subject);
```

Categorical predictors should also be explicitly coded as categorical variables. Numeric predictors are treated as continuous slopes. If a numeric variable contains group labels such as `0`, `1`, or `2`, convert it to categorical before running ClustME:

```matlab
design.Group = categorical(design.Group);
```

### `lmeFormula`

`lmeFormula` is a Wilkinson-style mixed-effects model formula compatible with MATLAB's `fitlme`.

The response variable in the formula must be named `response`. The `response` column does not need to be present in `design`; ClustME creates it internally at each time sample.

Examples:

```matlab
% One-sample repeated-measures test
lmeFormula = 'response ~ 1 + (1|Subject)';

% Within-subject condition effect
lmeFormula = 'response ~ 1 + Condition + (1|Subject)';

% Within-subject condition effect with subject-specific slopes
lmeFormula = 'response ~ 1 + Condition + (1 + Condition|Subject)';

% Between-subject group effect
lmeFormula = 'response ~ 1 + Group + (1|Subject)';

% Continuous subject-level predictor
lmeFormula = 'response ~ 1 + Score + (1|Subject)';
```

For v1.0.0, ClustME supports one-dimensional temporal clustering, static predictors, and a single random-effects grouping structure, including nested grouping terms such as Group:Subject where appropriate. Continuous time-varying covariates and crossed random-effects structures are not currently supported.

### Nested or grouped experimental units

In some designs, the independent exchangeability unit is nested within another factor. For example, subject labels may be unique only within group, animal labels may be unique only within treatment arm, or cells may be nested within animals.

In these cases, the random-effects structure can use a nested grouping term:

```matlab
lmeFormula = 'response ~ 1 + Group + (1|Group:Subject)';
```

Here, `Group:Subject` identifies subjects nested within groups. This is different from using `Subject` alone when the same subject labels may occur in more than one group.

Both variables used in the nested term should be present in `design`:

```matlab
design.Group = categorical(design.Group);
design.Subject = categorical(design.Subject);
```

For clarity, users may also create an explicit experimental-unit variable and use it consistently in both the model and randomisation settings:

```matlab
design.Unit = categorical(strcat(string(design.Group), "_", string(design.Subject)));

lmeFormula = 'response ~ 1 + Group + (1|Unit)';

[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula, ...
    'permutationMethod', 'wildBootstrap', ...
    'permuteUnit', 'Unit');
```

Use an explicit unit variable when it makes the exchangeability structure clearer, especially in datasets where lower-level identifiers are reused across higher-level groups.

### Optional time vector

If you want outputs and cluster boundaries to be reported against a physical time axis, provide a time vector using the `t` option:

```matlab
Fs = 100;
t = (-20:79) / Fs;

[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula, ...
    'Fs', Fs, ...
    't', t);
```

The time vector must have one value per column of `responses`:

```matlab
numel(t) == size(responses, 2)
```

If no time vector is provided, clusters are still valid, but their boundaries are interpreted as sample indices.

### Example design tables

#### One-sample repeated-measures design

```matlab
nSubjects = 12;
nTrialsPerSubject = 20;
nTime = 100;

Subject = repelem((1:nSubjects)', nTrialsPerSubject);
Subject = categorical(Subject);

design = table(Subject);
responses = randn(height(design), nTime);

lmeFormula = 'response ~ 1 + (1|Subject)';
clusters = ClustME(responses, design, lmeFormula);
```

#### Within-subject condition design

```matlab
nSubjects = 12;
nTrialsPerCondition = 20;
nTime = 100;

Subject = repelem((1:nSubjects)', 2 * nTrialsPerCondition);
Condition = repmat([repmat("Standard", nTrialsPerCondition, 1); ...
                    repmat("Oddball",  nTrialsPerCondition, 1)], ...
                    nSubjects, 1);

Subject = categorical(Subject);
Condition = categorical(Condition);

design = table(Subject, Condition);
responses = randn(height(design), nTime);

lmeFormula = 'response ~ 1 + Condition + (1|Subject)';

[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula, ...
    'testCoefficient', 'Condition_Oddball', ...
    'permutationMethod', 'withinSubject', ...
    'permuteUnit', 'Subject');
```

The exact coefficient name depends on MATLAB's coding of categorical predictors. If ClustME reports that the requested coefficient was not found, inspect the available coefficient names from a representative `fitlme` model.

#### Between-subject group design

```matlab
nSubjectsPerGroup = [10 10];
nTrialsPerSubject = 20;
nTime = 100;

Subject = [];
Group = [];

for g = 1:2
    for s = 1:nSubjectsPerGroup(g)
        Subject = [Subject; repmat("S" + g + "_" + s, nTrialsPerSubject, 1)];
        Group = [Group; repmat("G" + g, nTrialsPerSubject, 1)];
    end
end

Subject = categorical(Subject);
Group = categorical(Group);

design = table(Subject, Group);
responses = randn(height(design), nTime);

lmeFormula = 'response ~ 1 + Group + (1|Subject)';
```

For balanced and approximately homoscedastic groups, this type of design may be suitable for subject-level label permutation. For unbalanced or heteroscedastic groups, use subject-level wild bootstrap instead.

## Choosing the randomisation method

The randomisation method determines how ClustME generates the empirical null distribution. This choice is part of the statistical design, not a cosmetic option.

The randomisation method must match the exchangeability structure of the data. In other words, ClustME should only randomise labels or residuals among units that are exchangeable under the null hypothesis.

### Recommended choices

| Design | Recommended `permutationMethod` | Example `permuteUnit` | Notes |
|---|---|---|---|
| One-sample or intercept-only test | `'signFlip'` | `'Subject'`, `'Group:Subject'`, `'Unit'`, or `'auto'` | Appropriate when the null distribution is symmetric around zero. Randomisation should occur at the independent experimental-unit level. |
| Within-subject or within-unit condition effect | `'withinSubject'` | `'Subject'`, `'Unit'`, or `'auto'` | Shuffles residuals within each exchangeability block. |
| Balanced, approximately homoscedastic between-unit group effect | `'groupLabel'` | `'Subject'`, `'Group:Subject'`, or `'Unit'` | Permutes tested labels across independent units. The tested group label should be constant within each unit. |
| Unbalanced or heteroscedastic between-unit group effect | `'wildBootstrap'` | `'Subject'`, `'Group:Subject'`, or `'Unit'` | Uses unit-level sign multipliers on reduced-model residuals. |
| Continuous unit-level predictor | `'wildBootstrap'` | `'Subject'`, `'Group:Subject'`, or `'Unit'` | Usually preferable because there are no exchangeable categorical labels to permute. |

### `signFlip`

Use `signFlip` for one-sample or intercept-only tests where the null hypothesis is centred on zero.

```matlab
clusters = ClustME(responses, design, ...
    'response ~ 1 + (1|Subject)', ...
    'permutationMethod', 'signFlip', ...
    'permuteUnit', 'Subject');
```

For this setting, ClustME randomises the sign of the response or reduced-model residuals at the exchangeability-unit level.

### `withinSubject`

Use `withinSubject` for within-subject condition contrasts, where observations should only be shuffled within the relevant subject or unit.

```matlab
[clusters, mstats, vis_data] = ClustME(responses, design, ...
    'response ~ 1 + Condition + (1|Subject)', ...
    'testCoefficient', 'Condition_Oddball', ...
    'permutationMethod', 'withinSubject', ...
    'permuteUnit', 'Subject');
```

This avoids treating repeated observations from the same subject as independent observations.

### `groupLabel`

Use `groupLabel` for between-subject categorical predictors only when the independent units are exchangeable across groups. This usually requires balanced or near-balanced groups and no strong evidence of group-specific variance.

```matlab
[clusters, mstats, vis_data] = ClustME(responses, design, ...
    'response ~ 1 + Group + (1|Subject)', ...
    'testCoefficient', 'Group_G2', ...
    'permutationMethod', 'groupLabel', ...
    'permuteUnit', 'Subject');
```

`groupLabel` permutes the tested labels across independent units. The tested group variable should be constant within each exchangeability unit.

Do not use `groupLabel` merely because the model contains a group predictor. If groups are unbalanced or heteroscedastic, label exchangeability may fail.

### `wildBootstrap`

Use `wildBootstrap` when between-subject labels are not safely exchangeable, especially in unbalanced or heteroscedastic designs.

```matlab
[clusters, mstats, vis_data] = ClustME(responses, design, ...
    'response ~ 1 + Group + (1|Subject)', ...
    'testCoefficient', 'Group_G2', ...
    'permutationMethod', 'wildBootstrap', ...
    'permuteUnit', 'Subject');
```

The wild-bootstrap pathway randomises reduced-model residuals using subject-level sign multipliers. This preserves the fitted nuisance structure while avoiding invalid label permutation in heteroscedastic between-subject settings.

### Choosing `permuteUnit`

By default, ClustME uses:

```matlab
permuteUnit='auto'
```

With `auto`, ClustME attempts to infer the exchangeability unit from the random-effects grouping structure in the model formula. For example:

```matlab
'response ~ 1 + Condition + (1|Subject)'
```

will resolve `Subject` as the exchangeability unit.

The exchangeability unit does not have to be named `Subject`. It should identify the independent unit at which labels, signs, or residual multipliers can validly be randomised under the null. Depending on the design, this may be:

```matlab
'permuteUnit', 'Subject'
'permuteUnit', 'Group:Subject'
'permuteUnit', 'Unit'
```

For nested designs, use the nested grouping term or an explicit unit variable. For example:

```matlab
lmeFormula = 'response ~ 1 + Group + (1|Group:Subject)';

[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula, ...
    'permutationMethod', 'wildBootstrap', ...
    'permuteUnit', 'Group:Subject');
```

Use trial-level randomisation only when trial rows are genuinely exchangeable and are not nested within a higher-level dependency structure:

```matlab
'permuteUnit', 'trial'
```

Do not use trial-level randomisation for repeated-measures data simply to increase the number of possible randomisations.

## Options

Options are supplied as name-value arguments after the three required inputs:

```matlab
[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula, ...
    'optionName', optionValue);
```

### Common analysis options

| Option | Values | Default | Description |
|---|---:|---:|---|
| `testCoefficient` | character vector | `''` | Fixed-effect coefficient to test. Empty means the intercept. The name must match MATLAB's fitted coefficient name. |
| `permutationMethod` | `'signFlip'`, `'withinSubject'`, `'groupLabel'`, `'wildBootstrap'` | `'signFlip'` | Null-generation method. Must match the exchangeability structure of the design. |
| `permuteUnit` | `'auto'`, `'trial'`, or a design-table variable | `'auto'` | Exchangeability unit used for randomisation. Usually the subject or experimental unit. |
| `numPerms` | non-negative integer | `5000` | Number of randomisations used for the max-cluster null distribution. |
| `BqTarget` | non-negative integer | `2000` | Number of randomisations used to estimate the empirical cluster-forming threshold when `tcritMode` is `'permutation'`. |
| `alphaValue` | numeric value in `(0, 1]` | `0.05` | Nominal alpha level used for thresholding and cluster-level inference. |
| `Fs` | positive numeric value | `100` | Sampling frequency in Hz. Used to convert duration-based options such as `minClusterSize` into samples. |
| `t` | numeric vector | `[]` | Optional time vector with one value per time sample. Used for time-labelled outputs and time-based options. |

### Cluster options

| Option | Values | Default | Description |
|---|---:|---:|---|
| `minClusterSize` | non-negative number | `0` | Minimum cluster duration in milliseconds. A value of `0` applies no additional duration filter beyond temporal contiguity. |
| `clusterMassMethod` | `'mean'`, `'sum'` | `'mean'` | Statistic used for formal cluster inference. `'mean'` uses the mean squared t-value within the cluster; `'sum'` uses summed squared t-values. |
| `clusterSummaryMetric` | `'signedPeak'`, `'mean'`, `'sum'`, `'median'` | `'signedPeak'` | Descriptive metric extracted from each detected cluster for post-hoc summaries. This is not the inferential cluster statistic. |
| `PreselectedCluster` | `K × 2` numeric array | `[]` | Optional predefined time windows, given as `[tStart tEnd]` rows in the same units as `t`. Requires `t`. Advanced use only. |

Keep `clusterMassMethod` and `clusterSummaryMetric` conceptually separate. The first defines the statistic used for FWER-controlled cluster inference. The second controls how detected clusters are summarised descriptively.

### Static covariance and threshold options

| Option | Values | Default | Description |
|---|---:|---:|---|
| `Vmode` | `'adaptiveLocal'`, `'local'`, `'global'`, `'identity'` | `'adaptiveLocal'` | Strategy for estimating the static covariance matrix. |
| `TimeAnchor` | scalar | `[]` | Optional anchor time for static covariance estimation, in the same units as `t`. If empty, ClustME selects an anchor automatically. |
| `tcritMode` | `'permutation'`, `'parametric'` | `'permutation'` | Method for setting the pointwise cluster-forming threshold. |
| `whitening` | logical | `true` | Applies static-covariance whitening in the fast GLS pathway. Usually leave as `true`. |
| `FitMethod` | `'REML'`, `'ML'` | `'REML'` | Fitting method passed to MATLAB's LME fitting. Use `ML` mainly for model comparison workflows; use `REML` for final estimation unless there is a specific reason not to. |

#### Choosing `Vmode`

Use:

- `'adaptiveLocal'` for most event-related or non-stationary time-series analyses.
- `'global'` when the covariance and noise structure are expected to be broadly stable across the analysis window.
- `'local'` for a fixed narrow local covariance estimate around the anchor.
- `'identity'` for an OLS-equivalent diagnostic or validation baseline, not as the default hierarchical analysis.

If the expected effect has a known approximate latency, provide `t` and `TimeAnchor`:

```matlab
[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula, ...
    't', t, ...
    'TimeAnchor', 0.300, ...
    'Vmode', 'adaptiveLocal');
```

#### Choosing `tcritMode`

The default is:

```matlab
'tcritMode', 'permutation'
```

This estimates a time-varying empirical cluster-forming threshold from randomised statistics.

Use:

```matlab
'tcritMode', 'parametric'
```

only when there is a practical reason to avoid empirical thresholding, such as too few stable randomisations or very small numbers of exchangeability units. It should not be used simply because the empirical threshold does not produce candidate clusters.

### Computation and diagnostic options

| Option | Values | Default | Description |
|---|---:|---:|---|
| `parallel` | logical | `false` | Enables parallel execution where supported. Requires the Parallel Computing Toolbox. |
| `verbose` | logical | `false` | Prints additional diagnostics. Useful during development, validation, or troubleshooting. |
| `fullLME` | logical | `false` | Bypasses the fast static-covariance pathway and performs exhaustive LME refits. Mainly useful for validation or small benchmarking analyses. |
| `wbLeverage` | logical | `true` | Applies the wild-bootstrap leverage adjustment where applicable. Usually leave as `true`. |

### Example with selected options

```matlab
[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula, ...
    'testCoefficient', 'Condition_Oddball', ...
    'permutationMethod', 'withinSubject', ...
    'permuteUnit', 'Subject', ...
    'numPerms', 5000, ...
    'BqTarget', 2000, ...
    'Fs', 100, ...
    'Vmode', 'adaptiveLocal', ...
    'tcritMode', 'permutation');
```

Only specify options that are relevant to the analysis. The key choices are usually the tested coefficient, the randomisation method, the exchangeability unit, the number of randomisations, and the static-covariance pooling strategy.

## Outputs

ClustME can return one, two, or three outputs:

```matlab
clusters = ClustME(responses, design, lmeFormula);

[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula);
```

Most users should start with `clusters`. The additional outputs, `mstats` and `vis_data`, provide model information, run metadata, diagnostic arrays, and data for custom visualisation.

### `clusters`

`clusters` is a struct array containing the observed candidate clusters detected in the analysis window.

Each candidate cluster is evaluated against the empirical max-cluster null distribution and assigned a cluster-level p-value. A returned cluster is not necessarily below the chosen alpha level. To identify clusters meeting the analysis threshold, inspect `p_value` and compare it with `alphaValue`.

```matlab
clusters
```

For example, to select clusters with `p_value < 0.05`:

```matlab
sigClusters = clusters([clusters.p_value] < 0.05);
```

If no candidate clusters are detected, `clusters` will be empty.

#### Fields in `clusters`

| Field | Description |
|---|---|
| `type` | Descriptive direction of the cluster-level effect, reported as `'positive'` or `'negative'`. |
| `start` | Start sample index of the cluster. |
| `end` | End sample index of the cluster. |
| `mass` | Observed cluster statistic used for max-cluster inference. |
| `measure` | Descriptive cluster-level response summary, computed using `clusterSummaryMetric`. |
| `p_value` | Cluster-level p-value obtained by comparing the observed cluster statistic with the empirical max-cluster null distribution. |
| `lmeTStat` | Descriptive t-statistic from a post-hoc LME fitted to the cluster-collapsed response. |
| `lmePValue` | Descriptive p-value from the post-hoc cluster-level LME. This is not the cluster-level permutation p-value. |
| `covVars` | Random-effect variance estimates from the post-hoc cluster-level LME. |
| `resVar` | Residual variance estimate from the post-hoc cluster-level LME. |
| `varianceRatios` | Descriptive variance ratios from the post-hoc cluster-level LME. |
| `AIC` | AIC of the post-hoc cluster-level LME. |

The main inferential fields are:

```matlab
clusters(k).start
clusters(k).end
clusters(k).mass
clusters(k).p_value
```

The post-hoc LME fields are provided to describe detected clusters. They are separate from the cluster-level permutation inference and should not replace `p_value`.

### Cluster boundaries and time

The cluster boundaries `start` and `end` are sample indices. If a time vector was supplied using the `t` option, convert cluster indices to time using `vis_data.t`:

```matlab
k = 1;

clusterStartTime = vis_data.t(clusters(k).start);
clusterEndTime   = vis_data.t(clusters(k).end);
```

If no time vector was supplied, cluster boundaries should be interpreted as sample indices.

### `mstats`

`mstats` contains model-level information from the static-covariance fit and run metadata.

```matlab
[clusters, mstats] = ClustME(responses, design, lmeFormula);
```

#### Fields in `mstats`

| Field | Description |
|---|---|
| `AIC` | AIC of the LME used for the static covariance estimate. |
| `VarRatios` | Variance-ratio summary from the static covariance model. |
| `Model` | MATLAB `LinearMixedModel` object from the static covariance fit. |
| `Provenance` | Metadata describing the ClustME and MATLAB environment used for the run. |

`mstats.Model` is not a separate time-resolved LME fit for every sample. It is the model used to estimate the static covariance structure.

The run metadata can be inspected with:

```matlab
mstats.Provenance
```

### `vis_data`

`vis_data` contains arrays used for diagnostics, custom plotting, and validation.

```matlab
[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula);
```

This output is useful when users want to inspect the observed t-map, the cluster-forming threshold, the empirical max-cluster null distribution, or static-covariance diagnostics. It is not required for ordinary use of the cluster-level results.

#### Main fields in `vis_data`

| Field | Description |
|---|---|
| `Tmap` | Observed GLS t-statistic at each time sample. |
| `Fs` | Sampling frequency used for the analysis. |
| `t` | Optional user-supplied time vector. Empty if no time vector was provided. |
| `coefName` | Name of the tested coefficient. |
| `alpha` | Alpha level used in the analysis. |
| `tcrit` | Cluster-forming threshold at each time sample. This may be time-varying when `tcritMode` is `'permutation'`. |
| `sigMask` | Logical mask identifying samples that exceed the cluster-forming threshold. |
| `cStarts` | Start sample indices for candidate clusters. |
| `cEnds` | End sample indices for candidate clusters. |
| `clusterLevel` | Descriptive cluster-level response summaries. |
| `nullStats` | Empirical max-cluster null distribution. |
| `obsClusterMass` | Observed cluster statistic for each candidate cluster. |
| `pVals` | Cluster-level p-values corresponding to `obsClusterMass`. |
| `clusterMassMethod` | Cluster statistic used for inference, such as `'mean'` or `'sum'`. |
| `vstat` | Static-covariance diagnostics. |
| `idxAnchor` | Sample index used as the static-covariance anchor. |

### Basic inspection from `vis_data`

The observed t-map and cluster-forming threshold can be inspected using standard MATLAB plotting commands:

```matlab
if isempty(vis_data.t)
    x = 1:numel(vis_data.Tmap);
    xLabelText = 'Sample';
else
    x = vis_data.t;
    xLabelText = 'Time';
end

figure
plot(x, vis_data.Tmap)
hold on
plot(x,  vis_data.tcrit, '--')
plot(x, -vis_data.tcrit, '--')
xlabel(xLabelText)
ylabel('t-statistic')
title('Observed t-map and cluster-forming threshold')
```

This plot is a diagnostic view of the time-resolved statistic and cluster-forming threshold. Cluster-level inference is given by the `p_value` field in `clusters`.

### Static-covariance diagnostics

Diagnostics for the static covariance estimate are stored in:

```matlab
vis_data.vstat
```

This structure is mainly intended for validation, troubleshooting, and advanced users. For routine analyses, users usually do not need to inspect `vstat`.

## Toolbox utilities

### `clustme.select_best_lmm`

`clustme.select_best_lmm` evaluates a set of candidate LME formulas on the same dataset and ranks them by the AIC of the static-covariance model used by ClustME.

This utility is intended for **design-led model screening**, for example comparing a small number of pre-specified fixed-effect or random-effect structures. It should not be used as an unconstrained exploratory search over many formulas.

```matlab
results = clustme.select_best_lmm(responses, candidateFormulas, design, ...
                                  t, PreselectedCluster, ...
                                  Name, Value);
```

#### Inputs

| Input | Description |
|---|---|
| `responses` | `N × T` numeric response matrix, as used by `ClustME`. |
| `candidateFormulas` | Cell array of Wilkinson-style LME formulas to evaluate. |
| `design` | `N`-row design table matching `responses`. |
| `t` | Time vector with one value per column of `responses`. |
| `PreselectedCluster` | Predefined time window or windows, given as `K × 2` `[tStart tEnd]` rows in the same units as `t`. |
| `Name, Value` | Optional ClustME name-value arguments passed to the underlying `ClustME` calls. |

`select_best_lmm` requires both `t` and `PreselectedCluster`. The preselected window is used to evaluate each candidate model over the same temporal region. The midpoint of the first preselected window is used as the static-covariance anchor.

#### Example

```matlab
candidateFormulas = {
    'response ~ 1 + Condition + (1|Subject)'
    'response ~ 1 + Condition + Age + (1|Subject)'
    'response ~ 1 + Condition + Age + (1 + Condition|Subject)'
};

roi = [0.25 0.45];   % example time window, in the same units as t

results = clustme.select_best_lmm(responses, candidateFormulas, design, ...
                                  t, roi, ...
                                  'testCoefficient', 'Condition_Oddball', ...
                                  'permutationMethod', 'withinSubject', ...
                                  'permuteUnit', 'Subject', ...
                                  'numPerms', 5000);
```

The returned `results` struct array is sorted from best to worst by `StaticAIC`, with lower values preferred.

```matlab
{results.formula}'
[results.StaticAIC]'
```

#### Output fields

| Field | Description |
|---|---|
| `formula` | Candidate formula evaluated. |
| `StaticAIC` | AIC of the static-covariance LME. This is the primary ranking metric. |
| `StaticVarRatios` | Variance-ratio summary from the static-covariance model. |
| `clusters` | Standard `ClustME` cluster output for that formula. |
| `nClusters` | Number of observed candidate clusters. |
| `nSig` | Number of clusters with `p_value < alphaValue`. |
| `pickedIdx` | Index of the selected cluster used for prominence summaries. |
| `pickedP` | Cluster-level p-value of the selected cluster. |
| `promAvgAbs` | Average absolute grand-mean signal within the selected cluster. |
| `promPeakAbs` | Peak absolute grand-mean signal within the selected cluster. |
| `AIC` | Post-hoc cluster-level LME AIC values for detected clusters. |
| `AIC_selected` | Post-hoc cluster-level LME AIC for the selected cluster. |
| `varianceRatios` | Post-hoc cluster-level variance-ratio summaries. |

#### Interpretation

The main ranking field is:

```matlab
results(k).StaticAIC
```

This is the AIC of the static-covariance model and is used to rank candidate formulas. The utility internally uses ML estimation for these comparisons, because AIC comparisons across different fixed-effect structures should not be based on REML fits.

The cluster-related fields are provided to help inspect how each candidate model behaves over the preselected region. They should not be treated as a substitute for the primary model-ranking criterion.

#### Recommended use

Use `select_best_lmm` to compare a small set of models that are justified by the experimental design, for example:

- a baseline model against models adding one planned covariate;
- alternative random-effect structures while holding fixed effects constant;
- candidate nuisance structures before running the final inferential model.

Avoid using it to search opportunistically across many formulas until a preferred cluster-level result appears.

### `clustme.Visualizer`

`clustme.Visualizer` is a basic inspection helper for the `vis_data` output returned by `ClustME`.

It can plot the observed t-map, the cluster-forming threshold, and the empirical max-cluster null distribution. It is optional: users can also plot the arrays in `vis_data` directly with standard MATLAB commands.

```matlab
[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula);

clustme.Visualizer(vis_data);
```

By default, this opens both available views. A specific view can be requested with:

```matlab
clustme.Visualizer(vis_data, 'tmap');  % observed t-map and threshold
clustme.Visualizer(vis_data, 'hist');  % max-cluster null distribution
```

Common options include:

```matlab
clustme.Visualizer(vis_data, 'tmap', ...
    'showClusterLevel', false, ...
    'showDirectionText', false, ...
    'shadeClusters', true);
```

`Visualizer` is intended for quick diagnostic inspection rather than publication figure generation. For custom or publication figures, use the arrays in `vis_data`, such as `Tmap`, `tcrit`, `nullStats`, `obsClusterMass`, and `pVals`.

### `clustme.plotClusterLines`

`clustme.plotClusterLines` overlays cluster markers on an existing time-domain plot, such as an average response, ERP, or condition-difference trace.

```matlab
[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula);

figure
plot(vis_data.t, mean(responses, 1))
xlabel('Time')
ylabel('Response')

clustme.plotClusterLines(gca, clusters);
```

The target axes must already contain a plotted line. `plotClusterLines` uses the x-values of that line to convert `clusters.start` and `clusters.end` from sample indices into plotted x-axis positions.

By default, only clusters with `p_value < 0.05` are shown. To change this behaviour:

```matlab
clustme.plotClusterLines(gca, clusters, ...
    'onlyShowSignificant', false);
```

Cluster p-values can be shown as stars or numeric values:

```matlab
clustme.plotClusterLines(gca, clusters, ...
    'pvals', 'stars');
```

or:

```matlab
clustme.plotClusterLines(gca, clusters, ...
    'pvals', 'values');
```

Common options include `color`, `alpha`, `onlyShowSignificant`, `pvals`, and `fontSize`.

`plotClusterLines` does not perform statistical inference. It only displays cluster markers from an existing `clusters` output, so the plotted data should use the same time samples as the ClustME analysis.

### `clustme.bench_generator`

`clustme.bench_generator` creates synthetic hierarchical time-series datasets for examples, validation, and benchmarking.

It returns data in the same format expected by `ClustME`: an `N × T` response matrix, a matching design table, and a `groundTruth` structure describing the injected signal and simulation settings.

```matlab
[responses, design, groundTruth] = clustme.bench_generator(nSubjects, config);
```

This utility is mainly intended for demos, validation scripts, and method development. It is not required for analysing real data.

#### Basic example

```matlab
config = struct();
config.designType = 'one-sample';
config.nTrials = 25;
config.targetSNR = 1.25;

[responses, design, groundTruth] = clustme.bench_generator(18, config, ...
    'Fs', 100, ...
    'TimeRange', [-0.2 0.8], ...
    'noiseMode', 'complex', ...
    'RandomSeed', 42);
```

The returned objects can be passed directly to `ClustME`:

```matlab
lmeFormula = 'response ~ 1 + (1|Subject)';

clusters = ClustME(responses, design, lmeFormula, ...
    'permutationMethod', 'signFlip', ...
    'permuteUnit', 'Subject', ...
    'Fs', groundTruth.Fs, ...
    't', groundTruth.tVec);
```

#### Required inputs

| Input | Description |
|---|---|
| `nSubjects` | Number of subjects or experimental units. Use a scalar for one-sample and within-subject designs. Use a two-element vector, such as `[20 10]`, for between-group designs. |
| `config` | Struct defining the main simulation design. |

The `config` struct should contain at least:

| Field | Description |
|---|---|
| `designType` | Simulation design: `'one-sample'`, `'within'`, or `'between'`. |
| `nTrials` | Number of trials per subject or condition. If omitted, the default is 30. |
| `effectSize` | Peak amplitude of the injected signal. If omitted, the default is 0. |
| `targetSNR` | Target local signal-to-noise ratio. If supplied, this overrides `effectSize`. |

#### Supported design types

| `config.designType` | `nSubjects` format | Design table columns | Signal injection |
|---|---|---|---|
| `'one-sample'` | Scalar, for example `18` | `Condition`, `Subject`, `Trial` | Signal is added to all observations. |
| `'within'` | Scalar, for example `18` | `Condition`, `Subject`, `Trial` | Signal is added to condition `B`; condition `A` is the comparison condition. |
| `'between'` | Two-element vector, for example `[20 10]` | `Group`, `Subject`, `Trial` | Signal is added to the `Patient` group; `Control` is the comparison group. |

#### Common options

Only a small subset of commonly used options is listed here. The function contains additional options for more specialised validation scenarios.

| Option | Default | Description |
|---|---:|---|
| `Fs` | `100` | Sampling frequency in Hz. |
| `TimeRange` | `[-0.2 0.8]` | Epoch range in seconds. |
| `RandomSeed` | `[]` | Optional random seed for repeatable synthetic datasets. |
| `noiseMode` | `'gaussian'` | Noise type. Use `'complex'` for 1/f-like noise with optional non-stationarity. |
| `SubjectVar` | `2.0` | Subject-level random-intercept variance. |
| `signalTime` | `0.4` | Peak latency of the injected signal, in seconds. |
| `signalWidth` | `0.05` | Full width at half maximum of the injected Gaussian signal, in seconds. |

For advanced simulations, inspect the function header in:

```matlab
+clustme/bench_generator.m
```

#### Output fields

| Output | Description |
|---|---|
| `responses` | `N × T` numeric matrix containing the generated observations. |
| `design` | `N`-row table matching `responses`. Contains `Subject`, `Trial`, and either `Condition` or `Group`. |
| `groundTruth` | Struct containing the injected signal, time vector, signal masks, peak location, SNR information, and other simulation metadata. |


## Reporting and reproducibility

When reporting a ClustME analysis, include the model, the tested effect, and the null-generation settings. These are the details needed to understand and reproduce the inference.

At minimum, report:

- the ClustME version and MATLAB version;
- the model formula, for example `response ~ 1 + Condition + (1|Subject)`;
- the tested coefficient, or state that the intercept was tested;
- the randomisation method, for example `signFlip`, `withinSubject`, `groupLabel`, or `wildBootstrap`;
- the exchangeability unit, for example `Subject`, `Animal`, `Group:Subject`, or an explicit `Unit` variable;
- the number of randomisations, `numPerms`;
- the cluster-forming threshold method, `tcritMode`;
- the static covariance strategy, `Vmode`;
- the cluster statistic, `clusterMassMethod`;
- the sampling frequency or time vector used to interpret cluster boundaries.

The ClustME and MATLAB versions are stored in:

```matlab
mstats.Provenance
```

For Monte Carlo randomisations, users should manage their own random seed by setting the MATLAB random stream before calling ClustME. Record the seed if exact reruns are required:

```matlab
rngSeed = 1042;
rng(rngSeed, 'twister');

[clusters, mstats, vis_data] = ClustME(responses, design, lmeFormula, ...);
```

ClustME uses MATLAB's active random stream. It does not set or store a seed automatically.

Cluster-level results should be reported using the cluster boundaries, cluster statistic, and cluster-level p-value:

```matlab
clusters(k).start
clusters(k).end
clusters(k).mass
clusters(k).p_value
```

Use wording such as:

> Cluster-level inference was performed using the empirical max-cluster null distribution, controlling FWER over the analysed time window.

Avoid wording such as:

> FWER-corrected p-values.

FWER is controlled by the max-cluster procedure; it is not applied afterwards as a separate correction.

Cluster boundaries should be interpreted as observed cluster extents. Cluster-level inference does not imply that each individual time sample within the cluster is separately significant.

## Validation suite

The `validation/` folder contains two types of validation files: MATLAB unit tests for software integrity, and simulation benchmarks used for release-level and manuscript-level validation. These files are not required for ordinary ClustME analyses.

### Unit-test files

| File | Purpose |
|---|---|
| `TestClustME.m` | Tests the main ClustME engine, including core null models, known-signal detection, output structure, configuration guards, missing-data handling, exact-permutation checks, and selected edge cases. |
| `TestBenchGenerator.m` | Tests `clustme.bench_generator`, including default settings, design-table construction, signal injection, target-SNR scaling, group boundaries, heteroscedastic noise, complex noise, FWHM bounds, and input guards. |
| `TestSelectBestLmm.m` | Tests `clustme.select_best_lmm` on minimal synthetic data and verifies that the returned structure contains the expected AIC and cluster-related fields. |

Example:

```matlab
results = runtests('validation/TestClustME.m');
```

### Benchmark scripts

| File | Purpose |
|---|---|
| `run_accuracy_speed_validation.m` | Benchmarks the static covariance approximation against full timepoint-wise LME refitting, including agreement of t-statistics and runtime speedup. |
| `run_fwer_validation.m` | Estimates empirical FWER under the null across one-sample, within-subject, balanced between-subject, and heteroscedastic unbalanced between-subject settings. |
| `run_sensitivity_validation.m` | Evaluates detection sensitivity across signal-to-noise ratios and compares the GLS adaptive strategy with OLS-equivalent and fixed-window baselines. |

Benchmark behaviour is controlled by:

```text
validation/validation_settings.json
```

As full benchmark runs can require substantial runtime and parallel resources, `runMode` can be set to `debug`, `fast`, and `publication` modes. 

## Known limitations

ClustME v1.0.0 is designed for one-dimensional time-series inference with static predictors. It does not currently support time-varying covariates, multi-sensor spatial clustering, image-level clustering, or arbitrary graph-based clustering.

The current implementation supports one random-effects grouping structure, including nested terms such as `Group:Subject`, but not crossed random effects.

The static covariance approximation assumes that the main covariance structure is reasonably stable over the analysis window. If the dependence structure changes substantially over time, local power may be reduced.

Statistical validity depends on choosing a randomisation method that matches the exchangeability structure of the design. Group-label permutation should not be used when between-unit labels are not exchangeable, such as in strongly unbalanced or heteroscedastic designs.

Very small numbers of exchangeability units can make empirical thresholds unstable.

Cluster boundaries are descriptive. Cluster-level inference controls FWER over the analysed time window, but does not imply that every individual time sample within a cluster is separately significant.

## Versioning and compatibility

ClustME follows semantic versioning from v1.0.0 onward.

Major versions may introduce breaking changes to the public API or output structures.
Minor versions may add features, options, validation diagnostics, or non-breaking outputs.
Patch versions fix bugs, documentation, numerical safeguards, or validation scripts without intentionally changing the public API.

Deprecated public API will normally remain available for at least one minor release after deprecation, with a warning identifying the replacement. They may be removed in the next major release.
Exceptions may be made for behaviours that are statistically invalid, numerically unsafe, or likely to produce misleading inference. In such cases, the change will be documented clearly in the release notes.

## Citation

Before publication, please cite the GitHub/Zenodo software release and the preprint when available. Once the peer-reviewed manuscript is published, citation instructions will be updated to include the article DOI. Archived software releases will retain their own DOI so analyses can cite the exact code version used.

## Licence

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Support and bug reports

The toolbox is maintained for the documented v1.0 feature set. Bug reports should include the ClustME version, MATLAB version, operating system, model formula, relevant options, random seed if applicable, and a minimal reproducible example where possible.
The authors cannot verify the statistical validity of arbitrary user designs. Users are responsible for selecting a randomisation scheme appropriate to their exchangeability structure.

Email: guy.yona@ndcn.ox.ac.uk