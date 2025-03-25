function [sGamma, sAlpha, sBeta, JumpPost] = m20140708_ForwardBackwardBayes_BernoulliJump_fn(s, pJ, pgrid, Alpha0, tMax)
% Forward-Backward algorithm to solve a Hidden Markov Model, in which the 
% hidden states are Bernoulli parameters controlling the observed outcome s.
%
% Adapted from m20140624_ForwardBackwardBayes_BernoulliJump_fn to make the
% computation for each time step and saving time (instead of calling the
% function several times).
%
% [sGamma, sAlpha, sBeta, JumpPost] = ...
%   m20140708_ForwardBackwardBayes_BernoulliJump_fn(s, pJ, pgrid, Alpha0, tMax)
% 
% Input:
%      s: sequence of numeric values (including 1s and another value)
%     pJ: prior on jump occurence at any given location
%  pgrid: vector of state, pgrid(i) is p(s=1|x(i)), a Bernoulli parameter
% Alpha0: prior on states (i.e. prior on the Bernoulli parameter, provided 
%         the grid pgrid).
%   tMax: s(tMax) is the 1st element of posterior, feedback inference (i.e.
%         it is not computed for s(1:tMax), which is potentially not
%         interesting, and thus not computed to save time).
%
% Output:
%   sGamma: Gamma(:,tMax+t) is the posterior distribution for states 
%           x(tMax+t) (i.e. hidden Bernoulli parameters) given observation 
%           s(tMax:t) [= FORWARD BACKWARD INFERENCE]
%   sAlpha: Alpha(:,t) is the posterior distribution for states 
%           x(t) (i.e. hidden Bernoulli parameters) given observation 
%           s(1:t)    [= FORWARD INFERENCE]
%    sBeta: Beta(:,tMax+t) is the posterior distribution for states 
%           x(tMax+t) (i.e. hidden Bernoulli parameters) given observation 
%           s(tMax:t)    [= FEEDBACK INFERENCE]
% JumpPost: JumpPost{t} Posterior on jump probability given s(tMax:t).

seqL   = length(s);
n      = length(pgrid);
pgrid  = (pgrid(:))'; % make sure that pgrid is a row vector
Alpha0 = Alpha0(:); % make sure that Alpha0 is a column vector

% FORWARD PASS: p(x(i) | s(1:i))
% ==============================

% Alpha(i,t) = p(x(i) | s(1:t))
% Actually, because we want the to distinguish jump vs. no jump. we split 
% the values of Alpha in 2 columns, corresponding to jump or no jump. 
% Alpha(i, 1, t) = p(x(i), J=0 | s(1:t))
% Alpha(i, 2, t) = p(x(i), J=1 | s(1:t))
% Initialize alpha of the forward algorithm
Alpha = zeros(n, 2, seqL);

% get the matrix of non diagonal elements
NonDiag = ones(n);
NonDiag(logical(eye(n))) = 0;

% Compute the transition matrix (jump = non diagonal transitions).
% NB: the prior on jump occurence pJ is NOT included here. 
% Trans(i,j) is the probability to jump FROM i TO J
% Hence, sum(Trans(i,:)) = 1
% Likelihood of the target location reflect the prior on states.
Trans = (1 ./ (NonDiag * Alpha0)) * Alpha0';
Trans = Trans .* NonDiag;

% Compute alpha iteratively (forward pass)
for k = 1:seqL;    
    
    % Specify likelihood of current observation
    LL = eye(n);
    if s(k) == 1
        LL(logical(LL)) = pgrid;
    else
        LL(logical(LL)) = 1-pgrid;
    end
    
    % Compute the new alpha, based on the former alpha, the prior on
    % transition between states and the likelihood. See for instance
    % Wikipedia entry for 'Forward algorithm'. 
    if k == 1        
        Alpha(:,1, k) = (1-pJ)   * LL * Alpha0;
        Alpha(:,2, k) = pJ/(n-1) * LL * Alpha0;
    else
        
        % No Jump at k:
        % - take the prior on 'no jump': (1-pJ)
        % - take the current observation likelihood under x_i (LL)
        % - take the posterior on x_i(t-1) (summed over whether there was a
        % jump of not at t-1)
        Alpha(:,1,k) = (1-pJ)   * LL * (Alpha(:,1,k-1) + Alpha(:,2,k-1));
        
        % Jump at k:
        % - take the prior on 'jump': (1-pJ)
        % - take the current observation likelihood under x_i (LL)
        % - take the posterior on all the other state, excluding x_i(t-1)
        % (summed over whether there was a jump or not at i-1)
        % - sum over the likelihood of the ORIGINS of such state
        % (hence the transpose on Trans, to sum over the ORIGIN)
        Alpha(:,2,k) = pJ * LL * (Trans' * (Alpha(:,1,k-1) + Alpha(:,2,k-1)));
        
    end
    
    % scale alpha as a posterior (which we will do eventually) to avoid
    % numeric overflow
    NormalizationCst = sum(sum(Alpha(:,:,k),2),1);
    Alpha(:,1, k) = Alpha(:,1,k) / NormalizationCst;
    Alpha(:,2, k) = Alpha(:,2,k) / NormalizationCst;
end

% BACKWARD PASS: p(y(i+1:N | x(i))
% ================================
% COMPUTE THE BACKWARD PASS VARYING THE MAXIMAL NUMBER OF ITEMS TAKEN INTO
% ACCOUNT

% Beta(i,t) = p(s(t+1:N) | x(t))
% Actually, we compute p(x(t) | s(t+1:N)) because what we want in the end
% is a distribution over x, and the scaling of the distribution does not 
% matter
% In addition, because we want the to distinguish jump vs. no jump. we split 
% the values of Alpha in 2 columns, corresponding to jump or no jump. 

AllBeta = cell(1, seqL-tMax+1);

% loop over the maximal time horizon
for tEnd = tMax:seqL
    
    Beta = zeros(n, 2, tEnd);
    
    % Compute beta iteratively (backward pass)
    for k = tEnd:-1:tMax;
        
        % Specify likelihood of current observation
        LL = eye(n);
        if s(k) == 1
            LL(logical(LL)) = pgrid;
        else
            LL(logical(LL)) = 1-pgrid;
        end
        
        if k == tEnd
            Beta(:,1,k) = Alpha0;
            Beta(:,2,k) = Alpha0;
        else
            % No Jump from k to k+1
            % take only diagonal elements
            Beta(:,1, k) = (1-pJ) * (LL * (Beta(:,1,k+1) + Beta(:,2,k+1)) );
            
            Beta(:,2, k) = (pJ/(n-1)*NonDiag) * (LL * (Beta(:,1,k+1) + Beta(:,2,k+1)));
            % Jump from k to k+1
            % sum over non diagonal elements
            % NB: there is no transpose here on Trans because we sum over
            % TARGET location (not ORIGIN)
            Beta(:,2, k) = (pJ*Trans) * (LL * (Beta(:,1,k+1) + Beta(:,2,k+1)));
        end
        
        % scale beta to sum = 1. This normalization is only for convinience,
        % since we don't need this scaling factor in the end.
        NormalizationCst = sum(sum(Beta(:,:,k),2),1);
        Beta(:,1, k) = Beta(:,1,k) / NormalizationCst;
        Beta(:,2, k) = Beta(:,2,k) / NormalizationCst;
    end
    
    % Shift Beta so that Beta(:,:,k) is the posterior given s(k+1:N)
    newBeta = zeros(size(Beta));
    newBeta(:,:,1) = 1;
    newBeta(:,:,2:end) = Beta(:,:,1:end-1);
    clear Beta; Beta = newBeta; clear newBeta;
    
    AllBeta{tEnd-tMax+1} = Beta;
end

% COMBINE FORWARD AND BACKWARD PASS
% =================================
% COMPUTE VARYING THE MAXIMAL NUMBER OF ITEMS TAKEN INTO ACCOUNT
sGamma = cell(1, seqL-tMax+1);
sBeta = cell(1, seqL-tMax+1);
JumpPost = cell(1, seqL-tMax+1);

for tEnd = tMax:seqL
    % p(x(i) | y(1:N)) ~ p(y(i+1:N) | x(i)) p(x(i) | y(1:i))
    Gamma = squeeze(sum(Alpha(:, :, tMax:tEnd),2)) .* ...
            squeeze(sum(AllBeta{tEnd-tMax+1}(:, :, tMax:tEnd), 2));
    
    % Scale gamma as a posterior on observations
    cst = repmat(sum(Gamma, 1), [n, 1]);
    Gamma = Gamma ./ cst;
    
    % Compute the posterior on states, summed over jump or no jump
    sGamma{tEnd-tMax+1}   = Gamma;
    
    % Compute the forward & backword posterior
    sBeta{tEnd-tMax+1}    = squeeze(sum(AllBeta{tEnd-tMax+1}, 2));
    
    % Compute the posterior on jump, summed over the states
    % GammaJ = p(x(i), J=1 | y(1:N)) ~ p(y(i+1:N) | x(i), J=1) p(x(i), J=1 | y(1:i))
    %                                ~ [1/p(J=1) * p(y(i+1:N), J=1 | x(i))] p(x(i), J=1 | y(1:i))
    %                                ~ [1/p(J=1) * Beta2] Alpha2
    GammaJ   = squeeze(Alpha(:,2,tMax:tEnd)) .* ...
        ((1/pJ) *  squeeze(AllBeta{tEnd-tMax+1}(:, 2, tMax:tEnd)));
    GammaJ = GammaJ ./ cst;
    JumpPost{tEnd-tMax+1} = sum(GammaJ, 1);
end

sAlpha   = squeeze(sum(Alpha, 2));


