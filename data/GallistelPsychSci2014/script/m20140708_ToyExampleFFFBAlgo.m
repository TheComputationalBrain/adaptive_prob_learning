clear all
% DEFINE BERNOULLI SEQUENCE WITH JUMPS
% ====================================
p     = 0.35;
pA    = [p  1-p ];
L     = [100 200];   % length of each chunck
pJump = (length(pA)-1)/sum(L);

fullp = zeros(1, sum(L));
for k = 1:length(L)
    fullp(sum(L(1:(k-1)))+1:sum(L(1:k))) = pA(k)*ones(1, L(k));
end

% simulate a sequence
s = (rand(1,sum(L)) > fullp) + 1; % A:1; B:2

% COMPUTE THE FORWARD-BACKWARD ALGORITHM FOR THE HIDDEN MARKOV SEQUENCE
% ============================================================

% Descretization into state X: t=p(s=1|X)
% number of states
n = 10;
t = linspace(0,1,n);

% Specify transition matrix between states
% T(i,j) = p(x(i) | x(j))
T = ones(n)*pJump/(n-1);
T(logical(eye(n))) = (1-pJump);

% Alpha(k, x) = p(x, s(1:k)
% Initialize alph to uniform
Alpha0 = ones(n, 1) * (1/n);

tMax = 50;

% function version
[sGamma, sAlpha, sBeta, JumpPost] = m20140708_ForwardBackwardBayes_BernoulliJump_fn(...
    s, ...
    pJump, ...
    t, ...
    Alpha0, ...
    tMax);

tEnd = 60;
plot(tMax+(1:length(JumpPost{tEnd}(1:tEnd))), JumpPost{end}(1:tEnd))

r4mat = zeros(length(JumpPost), length(s));
for k = 2:length(JumpPost)
    r4mat(k, tMax:tMax+k-2) = JumpPost{k}(1:k-1);
end
subplot(2, 1, 1)
imagesc(r4mat, [0 0.1])

% COMPUTE GALLISTEL MODEL
% ======================

Data = rand(100, 1) > 0.5;
alpha_p = 1;
beta_p = 1;
alpha_c = 1;
beta_c = 1;
KLcrit = 1.35; 
BFcrit = 1;

[cp, DP, CL, pc_hat, ps, Record, hyper_c_rec, RTcpRprt]= BernCPKLfun(...
    s(:) == 2, ...                  % sequence
    1, ...                  % prior on p (beta(1))
    1, ...                  % prior on p (beta(2))
    33, ...                 % prior on pc (beta(1))
    10000, ...              % prior on pc (beta(2))
    1.35, ...               % KL criterion
    1);                     % BF criterion

subplot(2, 1, 2)
bar(cp(2:end-1), 1)
xlim([1 length(s)])
