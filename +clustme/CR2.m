classdef CR2
    % CR2 - Mathematical engine for Cluster-Robust Leverage and Standard Errors
    %
    %   The CR2 class provides static methods to compute bias-reduced,
    %   cluster-robust standard errors and leverage adjustments. It corrects
    %   for the downward bias inherent in standard sandwich estimators when
    %   the number of independent exchangeable clusters is small.
    %
    % Methods:
    %   compute_SE       - Computes CR2 standard errors for a specific fixed effect.
    %   adjust_residuals - Applies CR2 leverage correction to reduced-model residuals.
    %
    % References:
    %   [Toolbox] Yona, G., & Magill, P. J. Fast Cluster-Based Permutation Testing with
    %   Linear Mixed-Effects Models.
    %   [Methodology] Bell, R. M., & McCaffrey, D. F. (2002). Bias Reduction 
    %   in Standard Errors for Linear Regression with Multi-Stage Samples. 
    %   Survey Methodology.
    %   [Methodology] Pustejovsky, J. E., & Tipton, E. (2018). Small-Sample 
    %   Methods for Cluster-Robust Variance Estimation and Hypothesis Testing 
    %   in Fixed Effects Models. Journal of Business & Economic Statistics.
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

    methods (Static)
        
        function SeVec = compute_SE(Y, X, XtX, idxCoef, clusterIdx)
            % compute_SE - Compute Cluster-Robust (CR2) Standard Errors
            %
            % Syntax:
            %   SeVec = clustme.CR2.compute_SE(Y, X, XtX, idxCoef, clusterIdx)
            %
            % Inputs:
            %   Y          - [N×T double] Globally whitened response matrix.
            %   X          - [N×P double] Globally whitened full design matrix.
            %   XtX        - [P×P double] Pre-computed normal matrix (X'*X).
            %   idxCoef    - [1×1 double] Column index of the tested fixed effect in X.
            %   clusterIdx - [N×1 double] Vector of exchangeability block IDs.
            %
            % Outputs:
            %   SeVec      - [T×1 double] Time-varying CR2 standard errors for the
            %                specified coefficient.

            invXtX = XtX \ eye(size(X, 2));
            c = invXtX(:, idxCoef);
            
            beta_all = XtX \ (X' * Y);   
            E = Y - X * beta_all;        
            
            u = unique(clusterIdx(:));
            var_j = zeros(1, size(Y, 2));
            
            for k = 1:numel(u)
                rows = find(clusterIdx == u(k));
                Xg = X(rows, :);
                
                Ag = clustme.CR2.get_matrix(Xg, invXtX);
                
                % Variance contribution for coefficient j
                Sg = Xg' * (Ag * E(rows, :)); 
                var_j = var_j + (c' * Sg).^2;            
            end
            
            SeVec = sqrt(max(var_j, 0)).';
        end
        
        function Eadj = adjust_residuals(E0, X0, XtX0, clusterIdx)
            % adjust_residuals - Apply CR2 leverage correction to reduced-model residuals
            %
            % Syntax:
            %   Eadj = clustme.CR2.adjust_residuals(E0, X0, XtX0, clusterIdx)
            %
            % Inputs:
            %   E0         - [N×T double] Raw residuals from the reduced model.
            %   X0         - [N×P0 double] Globally whitened reduced design matrix.
            %   XtX0       - [P0×P0 double] Pre-computed reduced normal matrix (X0'*X0).
            %   clusterIdx - [N×1 double] Vector of exchangeability block IDs.
            %
            % Outputs:
            %   Eadj       - [N×T double] Leverage-adjusted residuals (base for the
            %                Wild Bootstrap and Freedman-Lane permutations).

            Eadj = E0;
            if isempty(clusterIdx) || isempty(X0)
                return; 
            end
            
            invXtX0 = XtX0 \ eye(size(X0, 2));
            u  = unique(clusterIdx(:));
            
            for k = 1:numel(u)
                rows = find(clusterIdx == u(k));
                Ag = clustme.CR2.get_matrix(X0(rows, :), invXtX0);
                Eadj(rows, :) = Ag * E0(rows, :);
            end
        end
        
    end
    
    methods (Static, Access = private)
        
        function Ag = get_matrix(Xg, invXtX)
            % GET_MATRIX Shared internal math for the symmetric inverse square root
            % of the residual maker matrix: Ag = (I - Hg)^(-1/2)
            
            Hg = Xg * invXtX * Xg';
            Hg = (Hg + Hg') * 0.5;
            
            Rg = eye(size(Hg)) - Hg;
            Rg = (Rg + Rg') * 0.5;
            
            [U, D] = eig(Rg);
            lam = max(real(diag(D)), 1e-10);
            Ag = U * diag(1 ./ sqrt(lam)) * U';
        end
        
    end
end