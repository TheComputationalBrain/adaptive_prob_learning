L1          = 200;
L2          = 200;
thd         = 0.0001;
KLCrit      = 1.35;
nSim        = 1000;
p           = [0.2 0.35];
LpreTest    = [0 40 80 120 160];

nP = length(p);
nL = length(LpreTest);
FirstDetect_FB = nan(nSim, nP, nL);
FirstDetect_Ga = nan(nSim, nP, nL);
for iSim = 1:nSim
    disp(iSim)
    
    for iP = 1:nP
        % DEFINE BERNOULLI SEQUENCE WITH JUMPS
        % ====================================
        pA    = [p(iP)  1-p(iP) ];
        L     = [L1 L2];   % length of each chunck
        pJump = (length(pA)-1)/sum(L);
        
        fullp = zeros(1, sum(L));
        for k = 1:length(L)
            fullp(sum(L(1:(k-1)))+1:sum(L(1:k))) = pA(k)*ones(1, L(k));
        end
        
        % simulate a sequence
        s = (rand(1,sum(L)) > fullp) + 1; % A:1; B:2
        
        for iLpre = 1:length(LpreTest)
            seq = s(LpreTest(iLpre)+1:end);
            
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
            
            tMax = 1;
            
            % function version
            [sGamma, sAlpha, sBeta, JumpPost] = m20140708_ForwardBackwardBayes_BernoulliJump_fn(...
                seq, ...
                pJump, ...
                t, ...
                Alpha0, ...
                tMax);
            
            catJump = zeros(length(JumpPost), length(s));
            for k = 2:length(JumpPost)
                % catJump(k, 2:k) = JumpPost{k}(2:k);
                catJump(tMax+k, (tMax+1):tMax+k-1) = JumpPost{k}(2:k);
                
                % r4mat(k, tMax:tMax+k-2) = JumpPost{k}(1:k-1);
            end
            
            subcatJump = catJump(L1-LpreTest(iLpre):end, L1-LpreTest(iLpre):end);
            
            % binarize detection
            binJ = subcatJump > thd;
            
            % get the cumuluted number of detection per row
            cs = cumsum(binJ,2);
            
            % take the 1st row with at least one significant detection
            minval = find(cs(:,end), 1, 'first');
            if ~isempty(minval)
                FirstDetect_FB(iSim, iP, iLpre) = minval;
            end
            
            % COMPUTE GALLISTEL MODEL
            % ======================
            
            [cp, DP, CL, pc_hat, ps, Record, hyper_c_rec, RTcpRprt]= BernCPKLfun(...
                seq(:) == 1, ...        % sequence
                1, ...                  % prior on p (beta(1))
                1, ...                  % prior on p (beta(2))
                round(pJump*10000), ... % prior on pc (beta(1))
                10000, ...              % prior on pc (beta(2))
                KLCrit, ...               % KL criterion
                1);                     % BF criterion
            
            ind = find(DP > (L1-LpreTest(iLpre)), 1, 'first');
            if ~isempty(ind)
                FirstDetect_Ga(iSim, iP, iLpre) = DP(ind)-(L1-LpreTest(iLpre));
            end
        end
    end
end

TimeStamp = clock;
TimeStamp = TimeStamp(2:end-1);
fID = sprintf('%d_%d_%d_%d', TimeStamp);
save(strcat(['simGallistel_', fID, '.mat']), ...
    'L1', ...
    'L2', ...
    'thd', ...
    'KLCrit', ...
    'nSim', ...
    'p', ...
    'LpreTest', ...
    'FirstDetect_FB', ...
    'FirstDetect_Ga')

