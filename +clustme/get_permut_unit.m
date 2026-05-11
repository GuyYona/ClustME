function [groupingIdx, nUnits, permuteUnitOut, info] = get_permut_unit(lmeFormula, designTable, permuteUnitIn)
% get_permut_unit - Resolve permutation unit and grouping indices from LME formula
%
%   get_permut_unit parses the Wilkinson formula to identify the random-effect 
%   grouping factor. It resolves the appropriate exchangeability block structure 
%   for the permutation loop and strictly guards against unsupported crossed 
%   random effects or numeric grouping variables.
%
% Syntax:
%   [groupingIdx, nUnits, permuteUnitOut, info] = clustme.get_permut_unit(lmeFormula, designTable, permuteUnitIn)
%
% Inputs:
%   lmeFormula    - [char] Wilkinson string defining the LME structure.
%   designTable   - [N×V table] Master design table containing the grouping variables. (Default: table())
%   permuteUnitIn - [char] Requested exchangeability unit. Use 'auto' to infer 
%                   from the formula, or 'trial' for no grouping. (Default: 'auto')
%
% Outputs:
%   groupingIdx    - [N×1 double] Vector of integer indices (1..K) defining the 
%                    exchangeability blocks. Returns [] if the unit is 'trial'.
%   nUnits         - [1×1 double] Total number of independent exchangeable units (K).
%   permuteUnitOut - [char] The resolved name of the grouping variable.
%   info           - [struct] Internal resolution metadata with fields:
%       .formula       [char] The original formula.
%       .requested     [char] The originally requested unit.
%       .randomGroups  [1×R cell] Cell array of random-effect groups extracted from the formula.
%       .status        [char] Resolution path: 'inferred', 'user', or 'noRandom'.
%
% BEHAVIOUR & VALIDATION
% ----------------------
% • Auto-Resolution: If permuteUnitIn is 'auto', the function parses the formula 
%   for terms like `(1|Subject)` and automatically sets 'Subject' as the unit.
% • Crossed-Effects Gatekeeper: The ClustME architecture currently strictly forbids 
%   crossed or multiple random effects. The function will hard-fail if multiple 
%   grouping variables are detected.
% • Numeric Categorical Gatekeeper: Hard-fails if the resolved grouping variable 
%   is numeric, as LMEs require categorical factors for proper random intercepts.
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
    lmeFormula    (1,:) char
    designTable   table       = table()
    permuteUnitIn (1,:) char  = 'auto'
end

info = struct();
info.formula      = lmeFormula;
info.requested    = permuteUnitIn;

% 1. Extract Grouping Variables using Regex
% Looks for text after '|' inside parentheses, e.g., (1|Subject) -> Subject
% Handles nested terms like (1|Subject:Trial) -> Subject:Trial
tokens = regexp(lmeFormula, '\([^()]*\|([^()]+)\)', 'tokens');

foundGroups = {};
for i = 1:numel(tokens)
    gExpr = strtrim(tokens{i}{1});      % e.g. 'Group2', 'Group2:Group1'
    % Split interaction-style grouping terms into candidate design-table variables.
    vars = regexp(gExpr, ':', 'split');
    foundGroups = [foundGroups, vars]; %#ok<AGROW>
end

info.randomGroups = unique(foundGroups, 'stable');

% --- Diagnostic Error: Numeric Random Effect Numeric ---
if ~isempty(designTable)
    for i = 1:numel(info.randomGroups)
        vName = info.randomGroups{i};
        if ismember(vName, designTable.Properties.VariableNames)
            if isnumeric(designTable.(vName))
                error('ClustME:NumericCategorical', ...
                    ['Random grouping factor "%s" is numeric. Linear mixed-effects models \n' ...
                     '      require categorical variables for grouping. Please cast to categorical \n' ...
                     '      to ensure correct random-intercept estimation.'], vName);
            end
        end
    end
end

if numel(info.randomGroups) > 1
    error('ClustME:UnsupportedCrossedEffects', ...
          'Crossed or multiple random effects are not currently supported. Please aggregate your data or specify a single grouping factor.');
end

% 2. Resolve Permutation Unit Name
    if strcmp(permuteUnitIn, 'auto')
        if isempty(info.randomGroups)
            permuteUnitOut = 'trial';
            info.status = 'noRandom';
        else
            permuteUnitOut = info.randomGroups{1}; 
            info.status = 'inferred';
        end
    else
        permuteUnitOut = permuteUnitIn;
        info.status = 'user';
    end

    % 3. Consistency Checks & Warnings
    if strcmp(permuteUnitOut, 'trial')
        % Warning: User requested trial-level shuffling despite random effects
        if ~isempty(info.randomGroups)
             fprintf('ClustME WARNING: "trial" permutation requested, but model contains random effects (%s).\n', ...
                      strjoin(info.randomGroups, ', '));
        end
        groupingIdx = []; % Signal for trial-level independence
        
        if ~isempty(designTable)
            nUnits = height(designTable);
        else
            nUnits = 0; % Edge case: will be fixed when nTrials is known in main
        end
    else
        % Validation: Ensure column exists
        if ~isempty(designTable)
            if ~ismember(permuteUnitOut, designTable.Properties.VariableNames)
                 error('clustme:get_permut_unit:MissingColumn', ...
                    'Permutation unit "%s" is not a variable in the design table.', permuteUnitOut);
            end
            
            % 4. Extract and Convert to Indices
            rawGroup = designTable.(permuteUnitOut);
            if iscategorical(rawGroup) || isstring(rawGroup) || iscellstr(rawGroup)
                groupingIdx = grp2idx(rawGroup);
            else
                % Ensure compact 1..K indices for efficiency
                [~, ~, groupingIdx] = unique(rawGroup);
            end
        else
            % Fallback if no table provided (edge case)
            groupingIdx = [];
        end
        nUnits = max(groupingIdx);
    end
end
