function [cp,DP,CL,pc_hat,ps,Record,hyper_c_rec,RTcpRprt]...
    = BernCPKLfun(Data,alpha_p,beta_p,alpha_c,beta_c,KLcrit,BFcrit)
% Analyses binary data sequences by looking first for evidence of
% divergence from currently assumed p, and, when divergence is detected,
% testing the sequence up to the detection point to determine whether the
% divergence arose because there was a change in p or because earlier
% estimate was erroneous. Second thoughts are implemented: When a new
% divergence is detected, it checks whether in the light of subsequent data,
% the most recent change point was or was not valid. If it is not valid on
% second thought, it is expunged and the origin of the current data segment
% is extended back to the previous change point.
%
% Syntax
% [cp,DP,CL,pc_hat,ps,Record,hyper_c_rec] =...
%     BernCPKLfun(Data,alpha_p,beta_p,alpha_c,beta_c,KLcrit,BFcrit)
%
% All 7 input arguments are obligatory:
%  1) Vector of binary data (ouput of a Bernoulli process)
%  2) alpha hyperparameter for initial beta prior on p_g (suggest .5 or 1)
%  3) beta hyperparameter for initial beta prior on p_g (suggest .5 or 1)
%  4) alpha hyperparameter for initial beta prior on p_c
%     (values from .01 to 4 are generally recommended for this parameter).
%     The smaller the value you choose, the weaker the prior, which is to
%     say that the sooner the data (the "observed" changes) will dominate.
%     'Observed' is in scare quotes because in the early portions of a
%     sequence, whether a change is "observed" or not may depend on this prior 
%  5) The expectation of the initial beta prior on p_c should
%     be roughly equal to half the inverse of one's a priori guess as to
%     the expected number of trials between changes. The a piori
%     expectation is (alpha_c + beta_c/alpha_c). The current estimate of
%     the expected number of trials between changes has a strong influence
%     on the prior odds that there is a change in a sequence of a given
%     length. These prior odds eventually dominate change detection; that is,
%     when the prior odds are 100:1 that a change has occurred in a
%     sequence of a given length, then the algorithm will almost always
%     find a change in it. However, in making your guesstimate, it is
%     better to err on the conservative side,that is, with high values for
%     (alpha_c +beta_c)/alpha_c, because when there is clear evidence of a
%     change, the algorithm will find that change no matter the prior odds.
%     If you grossly underestimate the expected number of trials between
%     changes, the algorithm will begin by finding more changes than
%     there are. However, if you make the prior weak,it will rapidly
%     converge on an approximately correct estimate. alpha_c = .01, beta_c
%     = 1 is an example of a very weak prior, with and expected number of
%     trials between changes of 1.01/.01 = 101; alpha_c = 2; beta_c = 200
%     is a strong prior with the same expectation.
%  6) KL decision criterion (suggest in range .23 to 3.32): .23 = even odds
%     (alpha level .5); .82 = 4:1 odds (alpha level .2); 1.35 = 9:1 odds
%     (alpha level .1); 1.92 = 19:1 odds (alpha .05); 3.32 = 99:1 odds (alpha .01)
%  7) BF decision criterion = the decision threshold on the posterior odds 
%     in favor of a change. The posterior odds are the product of the prior
%     odds and the Bayes Factor (likelihood ratio). Values of 1 to 8 are
%     suggested
%
% Output arguments:
%  cp = vector of final (post 2nd thoughts) change points
%  DP = vector of the detection points, the points at which the algorithm
%       detected a problem with the assumed value of p
%  CL = cell array containing relative likelihood limits on the CPs; has
%        same number of rows as cp array
%  pc_hat = final estimate of the probability of a change (mean of final
%           prior on p_c)
%  ps = the array of p values, one value for each of the segments
%        delimited by the change points in cp
%  Record = the record of successive detection points and change points;
%     some of which will probably not appear in cp because they were
%     expunged by second thoughts.
%     Record =
%     [alpha_p beta_p p_hat alpha_c beta_c pc_hat cp(end) DP(end) Nc Det];
%  hyper_c_rec = record of successive values of hyperparameters for the
%     prior on p_c. The algorithm adjusts this prior as it finds change
%     points but if the prior is set initially to make change points very
%     frequent when they are in fact infrequent, then it will take a good
%     bit of data before the algorithm as adjusted the prior to a more
%     appropriate value
%  RTcpRprt = apparent cps, model's real-time cps

Data = reshape(Data,length(Data),1); % making sure Data is a column vector

Data = Data+0; % converts logic vectors to 0's & 1's

if ~isempty(setdiff(Data,[0 1]))
    disp('Error: Input is not a binary vector')
    return
end

if nargin < 7
    display({'Not enough inputs; 7 required; in the following order:';...
        '1) Vector of binary data';...
        '2) alpha hyperparameter for initial beta prior on p_g (.5 or 1)';...
        '3) beta hyperparameter for initial beta prior on p_g (.5 or 1)';...
        '4) alpha hyperparameter for initial beta prior on p_c';...
        '5) beta hyperparameter for initial beta prior on p_c;';...
        '6) KL decision criterion (suggest in range .18 to 2.5)';...
        '7) BF decision criterion (suggest range .5 to 16)'})
    
     cp = [];
     DP = [];
     CL = [];
     pc_hat = [];
     ps = [];
     Record = [];
     hyper_c_rec = [];
     RTcpRprt = [];
        
    return
end
%%
p_hat = alpha_p/(alpha_p + beta_p); % intializing current estimate of p_hat
% NB these are the means of the prior probability distributions.

alpha_c_h = alpha_c; % initial hyperparameter for beta dist on p_c

beta_c_h = beta_c; % initial hyperparameter for beta dist on p_c

pc_hat = alpha_c/(alpha_c + beta_c);

hyper_c_rec = [alpha_c beta_c]; % record of values of hyperparameters on p_c

alpha_a = alpha_p; beta_a = beta_p; % These are the hyperparameters on p_hat
% after a change. These do not change as the computation evolves, whereas
% alpha_p and beta_p do change because they are the hyperparameters on p
% before the change

hyper_p_rec = [alpha_p beta_p]; % record of the values of the hyperparameters 
% on p_hat, the current estimate of p

Nc = alpha_c; % Initializing the vector that keeps track of the number of
% change points. It will be incremented by 1 each time a change point is
% detected. This count always includes the # of successes assumed a priori,
% which may well be smaller than 1 (but, of course, not less than 0)

NcInit = Nc;

cp = 0; % initializing cp vector

DP = 0; % Successive detection points, specified in terms of the row
% indices for the original (unaugmented & untruncated) data vector.
% Det is the detection point in the D vector, which is augmented by 4
% "prior" observations and then truncated as change points are found (and
% re-augmented after each truncation with 4 "prior" observations)

D = Data;

Det = 0; % initializing

CL = {[] [] [] [] [] [] [] [] [] []}; % initializing cell array that will
% contain the confidence limits on the cp's

Record = [alpha_p beta_p p_hat alpha_c beta_c pc_hat cp(end) DP(end) Nc Det];

Drec{1} = D; % record of successive D vectors

RTcpRprt = []; % DPs at which model believes there has been a change (model's
% real-time change reports

%%
while length(D) > 20
%%    
    N = (1:length(D))';

    p_o = cumsum(D)./N; % the value in row n of this vector is the observed
    % probability of success as of the nth draw

    KL = berndkl(p_o,repmat(p_hat,N(end),1)); % a vector giving KL divergences
    % of the successive observed p values from the current estimate.
    % repmat(p_hat,N(end),1) replicates the current estimate to make a column
    % vector of it that is the same length as p_o, the vector of observed
    % p's
 
    E = N.*KL; % E is the product of the number of observations and the KL
    % divergence. When p_hat p_true, this product is distributed
    % gamma(.5,1). It is called E for [strength of] evidence against p_hat


%%    
    if isempty(find(E(Det+1:end)>KLcrit,1)) % at no point in D does the
        % evidence of error exceed the decision criterion
        
        break % break out of while loop
        
    else % evidence of error exceeds the decision criterion at some point
        
        Det = Det + find(E(Det+1:end)>KLcrit,1); % Det is the detection
        % point, the observation upon which the evidence that something is
        % wrong with p_hat exceeds the detection criterion. Det is the row
        % index of the data relative to the most recent change point.
        % In other words, it is the row number in the D vector. Det was
        % initialized to 0 before entering the while loop; further down in
        % the loop, when a change point is found, it gets reinitialized,
        % but NOT TO 0! It is set to the number of rows(stimuli) between
        % the most recent change point and the detection point, forcing the
        % computation to begin beyond this most recent detection point
    end

    DP(end+1) = cp(end) + Det; % recording latest detection point in 
    % "absolute" row indices (the row indices of the original untruncated
    %  data vector).
    
    alpha_c = Nc + alpha_c_h; % updating parameters of beta distribution
    % on p_c
    
    beta_c = DP(end)-Nc + beta_c_h; % ditto
    
    pc_hat = alpha_c/(alpha_c + beta_c); % current estimate of p_c

    [CP,PstO,LF,Lmts]=BernCP(D(1:Det),alpha_p,beta_p,alpha_a,beta_a,pc_hat);
    %{
    Looking for a change point in the sequence up to Det. CP is the
    estimated change point; PstO is the posterior odds favoring the change
    model over the no-change model, that is, the odds after taking into
    account the prior probability of a change in a sequence of length Det,
    given pc_hat, the estimated probability of a change after any given
    tria; LF is the likelihood function (likelihood of a change as a
    function of where the change is assumed to have occurred)--not used;
    Lmts gives critical values of relative likelihood. The beta prior on
    the pre-change p is determined by the parameters alpha_p and beta_p,
    which evolve during the computation. This prior specifies the uncertainty
    regarding the true value of the current estimate of p. alpha_a & beta_a
    are the hyperparameters of the prior on p_g after a change. This prior
    does not evolve. The LF output is not used. In later versions of Matlab,
    replacing it with a ~ will make the code run faster, but the tilde 
    output argument makes earlier versions crash
    %}
    
    % THERE IS AN APPARENT CHANGE POINT
    if (~isempty(PstO))&&(PstO > BFcrit) % there is an apparent change
        % point. [I don't think that the (~isempty(PstO)) is necessary,
        % because I think BernCP never returns an empty value for this, but
        % I put it in just to be safe.]
                  
        RTcpRprt(end+1) = DP(end);

        cp(end+1,1) = cp(end)+CP; % recording CP
        
        CL(end+1,:) = Lmts; % recording confidence limits on cp

        alpha_p = 1 + sum(D(CP+1:Det)); % updating parameters of beta prior
        % on p_g. When a change point has been found, the new parameters
        % should be the number of successes and failures from the first trial after
        % the change point up to and including the trial on which the change was
        % detected PLUS(!) the initial (maximally weak) hyperparameters.

        beta_p = 1 + length(D(CP+1:Det))-sum(D(CP+1:Det));
        % see above

        hyper_p_rec(end+1,:) = [alpha_p beta_p];

        hyper_c_rec(end+1,:) = [alpha_c beta_c];

        D(1:CP) = [];
        % truncating data vector at change point
        
        Det = 1 + DP(end) - cp(end); % resetting Det to begin one datum
        % after the most recent detection point. This forces computation to
        % go on from the change point

        Drec{end+1} = D; % adding new D vector to the record of D vectors

        p_hat = alpha_p/(alpha_p + beta_p); % recomputing current estimate
        % of p_g. The new estimate is the mean of the updated beta prior

        Nc = Nc+1; % increment number of observed changes


    % NO APPARENT CHANGE POINT IN LATEST STRETCH OF DATA
    % AND AT LEAST ONE PREVIOUS CHANGE POINT HAS BEEN FOUND

    elseif ((isempty(PstO)||(PstO < BFcrit)) && size(cp,1) > 1)
        % when there isn't an apparent change point in the new stretch of
        % data covering the interval from the last apparent change point to
        % the latest detection point, then either:

        % 1) the previous change point is not justified in the light of
        % subsequent data

        % or

        % 2) the estimate of p_hat for the current segment was wrong;
        % and should be revised in the light of the further data we now
        % have

        TestPrevCP_data = Data(cp(end-1)+1:DP(end)); % take into
        % consideration the data going back to the penultimate change point
        % in order to test whether in the light of the data observed after
        % the most recent change point we still believe that the most
        % recent change point was real

        [CPr,Or,LF,Lmtsr] = BernCP(TestPrevCP_data,alpha_a,beta_a,alpha_a,beta_a,pc_hat);
        % Or gives the posterior odds that there is a change point in the
        % data from the penultimate change point up to the current
        % detection point. Notice that we use agnostic prior on p_g because 
        % it is unknown whether the most recent data or the older data provide
        % the appropriate prior. That is, we use alpha_a & beta_a as the
        % parameters of the beta distribution on p_g befor the change,
        % rather than alpha-P and beta_p. The LF output is not used. In later
        % versions of Matlab, replacing it with a ~ will make the code run
        % faster, but tildes as output arguments make earlier versions crash

     % 1) Previous change point was in retrospect not justified 

        if isempty(CPr)||(Or < BFcrit) % In this case, we do five things:
        % i) Delete most recent cp from the record of cps;
        % ii)update parameters for prior on p_g, hence also the estimate of
             % p_g
        % iii) update parameters for prior on pc, hence, also the estimate of pc
        % iv) put back into the D vector the data going back to the last cp
        % before the false alarm
        % v) adjust Det, the number of rows from the now most recent cp to
        % the latest detection point
        
        % i) Deleting most recent estimate of cp & parameters on its prior
            cp(end) = []; % delete most recent cp
            
            CL(end,:) = []; % delete the conf limits on the just deleted cp

            hyper_p_rec(end,:) = [];

            hyper_c_rec(end,:) =[];

            Nc = Nc-1; % reducing change-point count

        % ii) updating parameters of prior on p_g & estimate of p_g
            alpha_p = hyper_p_rec(end,1) + sum(TestPrevCP_data); % update prior
            % on p-g; sum(TestPrevCP_data) is the number of successes since the
            % last cp (after deletion of the bogus more recent cp)

            beta_p = hyper_p_rec(end,2) + length(TestPrevCP_data)...
                - sum(TestPrevCP_data);
            % length(TestPrevCP_data) - sum(TestPrevCP_data) is the number
            % of failures since the last valid cp

            p_hat = alpha_p/(alpha_p + beta_p); % update p_hat

        % iii) updating parameters of prior on pc
            alpha_c = Nc + alpha_c_h; % recalculating alpha param on p_c

            beta_c = DP(end) - Nc + beta_c_h; % ditto for beta param

         % iv) putting back in the data back to the last cp before the
         %   false alarm   
            D = Data(cp(end)+1:end); % now the D vector has the
            % data going back to the penultimate cp
            
         % v) Adjusting Det
            Det = 1 + DP(end) - cp(end);

    % 2) previous estimate of p_g was wrong

        else % in this case we will do five things:
            % i) revise the estimate of the location of that cp in
            %    the light of the additional data;
            % ii) update parameters of prior on p_g and estimate (p_hat) of
            %     p_g, using data from the new estimate of the most recent
            %     cp up to the latest detection point;
            % iii) adjust data in D in light of revised location for latest
            %     cp
            % iv) Adjust Det to be # of rows from revised cp to latest
            %    detection point

         % i) Revised estimate of location of most recent cp  
            cp(end) = cp(end-1,1) + CPr;
            % replacing previous estimate of
            % most recent cp with the new estimate of that cp
            
            CL(end,:) = Lmtsr; % replacing previous confidence limits on cp

         % ii) Updating hyperparameters and estimate of p_hat  
            alpha_p = 1 + sum(Data(cp(end)+1:DP(end)));

            beta_p = 1 + length(Data(cp(end)+1:DP(end))) -...
                sum(Data(cp(end)+1:DP(end)));

            p_hat = alpha_p/(alpha_p + beta_p); % update p_hat (est of p_g)

         % iii) Adjusting data in D
            D = Data(cp(end)+1:end); % Is this necessary?
            
         % iv) Adjusting Det
            Det = 1+ DP(end) - cp(end);

        end % of updating after 2nd thoughts

    else % there was no apparent change point and no change point has so far
        % been found, in which case, it must be that the initial estimate of
        % p_g was wrong. We update parameters of distribution on p_g
        % and p_hat, the the estimate of p_g
       

        alpha_p = 1 + sum(D(1:Det));

        beta_p = 1 + length(D(1:Det)) - sum(D(1:Det));

        p_hat = alpha_p/(alpha_p + beta_p); % update p_hat

    end % actions taken after detecting something wrong with p_hat
           
    Record(end+1,:) =...
        [alpha_p beta_p p_hat alpha_c beta_c pc_hat cp(end) DP(end) Nc-NcInit Det];

end % of while loop

CmRec = cumsum(Data);

cp(end+1)=length(CmRec); % makes final trial a nominal change point

Succ = [CmRec(cp(2));diff(CmRec([cp(2:end);length(Data)]))]; % number of successes from one change point
% to the next

Trls = [diff(cp);length(Data)-cp(end)];

ps = Succ./Trls;

end

function Dkl = berndkl(p1,p2)
% computes the Kullback-Leibler divergence of Bernoulli(p2) from
% Bernoulli(p1) in nats. Allows either or both p1 & p2 to be 0 or 1. Both 
% may be vectors, provided they are of same length

p1 = reshape(p1,length(p1),1); % insure that it's a column vector

p2 = reshape(p2,length(p2),1); % ditto

if length(p1) ~= length(p2)
    disp('Error in call to berndkl: vectors not the same length')
    return
end

if any(p1>1) || any(p1<0) || any(p2>1) || any(p2<0)
    disp('Error in call to berndkl: some probabilities >1 or <0')
    return
end

q1 = 1 - p1; % complementary values

q2 = 1 - p2; % complementary values

LVp1_0 = p1==0;

LVp1_1 = p1==1;

LVp2 = (p2==0) | (p2==1);

LVeq = p1==p2;

Dkl = nan(length(p1),1);

Dkl(LVeq) = 0; % whenever both p's are equal, the divergence is 0

Dkl(~LVeq&LVp2) = Inf; % whenever p2 is 0 or 1 and not equal to p1,
% divergence is infinite
    
Dkl(~LVeq&LVp1_0) = -log(q2(~LVeq&LVp1_0)); %whenever p1 is 0 and not equal to p2
    
Dkl(~LVeq&LVp1_1) = -log(p2(~LVeq&LVp1_1)); % whenever p1 is 1 & not equal to p2   

Dkl(~LVeq&~LVp1_0&~LVp1_1&~LVp2) = p1(~LVeq&~LVp1_0&~LVp1_1&~LVp2)...
    .*log(p1(~LVeq&~LVp1_0&~LVp1_1&~LVp2)./p2(~LVeq&~LVp1_0&~LVp1_1&~LVp2))...
    + q1(~LVeq&~LVp1_0&~LVp1_1&~LVp2).*log(q1(~LVeq&~LVp1_0&~LVp1_1&~LVp2)...
    ./q2(~LVeq&~LVp1_0&~LVp1_1&~LVp2)); % the simple case where both p's
end

function [CP,Odds,LkFun,CL,CPmean] = BernCP(Data,AlphaB,BetaB,AlphaA,BetaA,pc)
% Computes maximally likely change point (CP) and odds in favor of a change
% given a binary data string, Data, the parameters of the beta prior
% on the value of p before the change (AlphaB & BetaB), the parameters
% of the beta prior on the p after the change (AlphaA,BetaA), and the
% prior probability of a change after any given trial (pc). This is called
% from Line 515 with AlphaB & BetaB evolving and from Line 597 with them
% set to their initial agnostic values (.5 or 1)
%
%Syntax [CP,Odds,LkFun,CL] = BernCP(Data,AlphaB,BetaB,AlphaA,BetaA,pc)
%
% CL is a cell array giving lower (L) and upper (U) relative likelihood
% limits as follows:
% CL = {[.01L] [.05L] [.1L] [.2L] [.5L] [.5U] [.2U] [.1U] [.05U] [.01U]}
% CPmean is the mean of the marginal likelihood distribution on CP. The CP
% output is this value rounded to the nearest integer

Data = reshape(Data,length(Data),1); % making sure Data is a column vector

Data = Data+0; % converts logic vectors to 0's & 1's

if ~isempty(setdiff(Data,[0 1]))
    disp('Error: Input is not a binary vector')
    return
end

CP = nan; % intializing

Odds = nan;% intializing

LkFun = zeros(length(Data),1);% intializing

CL(1,1:10) = {[]};% intializing

CPmean = nan;

if nargin<6
    display({['Not enough input arguments;'];['6 required: '];...
        ['Vector of binary data'];...
        ['Hyperparameter alpha for prior on p before change'];...
        ['Hyperparameter beta for prior on p before change'];...
        ['Hyperparameter alpha for prior on p after change'];...
        ['Hyperparameter beta for prior on p after change'];...
        ['Probability of a change after any one trial']})
    return
end
    
if ~(beta(AlphaB,BetaB)>0)||~(beta(AlphaA,BetaA)>0)
    
    display('Prior too strong; makes beta(a,b)=0')
    
    return
    
end
%%

p = .01:.01:.99; % row vector of p values

dp = p(2)-p(1); % delta p

L = (1:length(Data))'; % cumulative number of observations (col vector)

Sf = cumsum(Data); % cumulative successes up to and including the nth datum
% (col vector w length = length(Data))

Ff = L - Sf; % cumulative failures up to and including the nth datum
% (col vector w length = the length of the data)

Sr = sum(Data) - Sf; % succeses after nth datum (col vector w length =
% the length of the data))

Lr = ((L(end)-1):-1:0)'; % number of observations after the nth datum
% (length = the length of the data)

Fr = Lr - Sr; % failures after the nth datum (col vector, w length =
% the length of the data)

%% The following relies on the analytic result that
% given a beta prior with parameters, alpha & beta, and observed
% numbers of successes and failures, nS & nF, the posterior marginal
% likelihood of a change hypotheses, given the data and the assumption of
% a beta prior with parameters alpha & beta) is beta(alpha+nS,beta+nF).

LgPstLkf = log(beta(AlphaB + Sf, BetaB + Ff)); % col vector
% giving the log likelihood for the data up to and including datum(n) under
% the prior distribution on p specified by hyperparameters AlphaB and BetaB

LgPstLkr = log(beta(AlphaA + Sr, BetaA + Fr)); % col vector
% giving the log likelihood for the data after datum(n) under the prior
% distribution specified by hyperparameters AlphaA & BetaA


%%
LgLkFun = LgPstLkf + LgPstLkr; % the log likelihood function giving, as a
% function of cp, the log likelihood of a model that assumes a change in p 
% after observation cp.

LgLkFunN = LgLkFun + abs(max(LgLkFun)); % setting peak value of log
% likelihood function to 0, so that when it is transformed into a
% likelihood function prior to integration its values are reasonable. This
% is equivalent to rescaling the likelihood function, which has no effect
% on the relative likelihood or the expectation, which are the two
% quantities of interest

LkFun = exp(LgLkFunN); % the likelihood function
% The last value in this vector is the likelihood of a model
% that assumes a change after the end of the data, in other words, no
% change within the span of the observations. Thus, assuming a uniform
% prior on cp (change equally likely after any trial), the marginal
% likelihood of a change is the average value of this likelihood function

CPmean = sum((1:length(Data)).*LkFun')/sum(LkFun); % expectation of the
% likelihood function (NB, not its mean value)

RelLkhd = (sum(LkFun(1:end))/(length(Data)))/LkFun(end); % the Bayes
% Factor, the ratio of the posterior likelihoods of the two models (change
% vs no-change)
%%

pNC = 1-pc; % Probabilty that p does not change after any given trial

PriorOdds = L(end)*pc/pNC; % prior odds that there is a change
% in a sequence as long as Data = Np_c/(1-p_c)

Odds = PriorOdds*RelLkhd; % Posterior odds equal relative likelihood of the
% two models (change: no-change) times the prior odds in favor of one or
% the other

BF = RelLkhd;
%% Computing CP & relative likelihood limits on its location

CP = round(CPmean);

Mx = max(LkFun(1:end-1)); % maximum of the likelihood function for
% change points before the last datum and location of this maximum
% (the estimate of CP)

CL{1} = find(LkFun>.01*Mx,1)-1;

CL{2} = find(LkFun>.025*Mx,1)-1;

CL{3} = find(LkFun>.1*Mx,1)-1;

CL{4} = find(LkFun>.2*Mx,1)-1;

CL{5} = find(LkFun>.5*Mx,1)-1;

CL{6} = find(LkFun>.5*Mx,1,'last')+1;

CL{7} = find(LkFun>.2*Mx,1,'last')+1;

CL{8} = find(LkFun>.1*Mx,1,'last')+1;

CL{9} = find(LkFun>.025*Mx,1,'last')+1;

CL{10} = find(LkFun>.01*Mx,1,'last')+1;
end