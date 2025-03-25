%--------------------------------------------------------------------------
% Code file for Data in Brief article 
% Khaw, Stevens, Woodford, August 2017 
%--------------------------------------------------------------------------

clear
clc
format compact 

%--------------------------------------------------------------------------
% Get data 
%--------------------------------------------------------------------------

nSubs = 11;                         % number of subjects
nses  = 10;                         % number of sessions per subject 
T     = 999;                        % number of observations per session 

% Data is in matrix form (T x (nSubs x nses))
pMat  = dlmread('probs.txt') ;      % hidden probability 
phMat = dlmread('phats.txt') ;      % subjects' estimates
stMat = dlmread('obs.txt') ;        % ring realizations / outcome


rtMat = dlmread('rtimes.txt') ;     % subjects' reaction times 


% --------------------------------------------------------------------------
% Sample series (Figure 2)
%--------------------------------------------------------------------------

% Subject 10, session 8 
select = 9*10+8;  
pSel   = pMat(:,select); 
phtSel = phMat(:,select); 

f2 = figure; 
plot(1:length(pSel),pSel,'-b','LineWidth',1.5);
hold on
plot(1:numel(pSel),phtSel,'-m','LineWidth',2);
ylim([-0.05 1.01]);
set(gca, 'XTick', [0:200:1000])
set(gca, 'YTick', [0:0.2:1.0])
%xticks(1:200:1000)
%yticks([0:0.200:1.000])
xlabel('Time (# of Rings)')
ylabel('Probability');
legend({'Hidden Probability','Subject''s Estimate', 'Location', 'Southeast'});
box off
hold off 


%% --------------------------------------------------------------------------
% Construct step_height and step_width 
% --------------------------------------------------------------------------

latsVec = []; % slider adjustment latency

for S = 1:11 % stepping through the subjects
    for s = 1:10  % stepping through the sessions 
        
        select    = (S-1)*10 + s;  % subject x session indexing
        curPht    = phMat(:,select);         
        delays = zeros(size(curPht,1)-1,1);
        changeCounter = 1;
        for i = 2:size(curPht,1) % step width count loop
            if  curPht(i) ~= curPht(i-1) % if change detected
                delays(i) = changeCounter;
                changeCounter = 1;
            elseif curPht(i) == curPht(i-1) % no change detected
                delays(i) = 0;
                changeCounter = changeCounter + 1;
            end
        end
        
        % concatenate data
        latsVec = [latsVec; delays(2:(end))];
    end
end



%--------------------------------------------------------------------------
% Distribution of Values, Adjustment Lags, and RT (Figure 3)
%--------------------------------------------------------------------------
f3 = figure; 
subplot(1,3,1)
[Y,X] = hist(phMat(:),0:0.01:1);
bar(X,Y,'FaceColor','k');
box off
xlim([0 1]);
xlabel('Slider Settings');
ylabel('Observations');

subplot(1,3,2)
semilogxhist(latsVec(latsVec~=0))
xlabel('Adjustment Lag')
box off

subplot(1,3,3)
rtUse = rtMat(:); 
[Y,X] = hist(rtUse(rtUse < mean(rtUse)+3*std(rtUse)));
bar(X,Y,'FaceColor','k');
box off
xlabel('Reaction Times (s)');

% --------------------------------------------------------------------------
% Repeated sessions (Figure 4)
% Subject 3, sessions 4 and 6 
%--------------------------------------------------------------------------

S        = 3;
sessions = [4; 6]; 
ns       = length(sessions);  


rvals = [0 0.5 ];
bvals = [0 0.5 ];
gvals = [0 0.5 ];


f4=figure; hold on 
for ii = 1:ns  
    sel = (S-1)*10 + sessions(ii) ;
    plot(phMat(:,sel), 'Color', [rvals(ii),bvals(ii),gvals(ii)], 'LineWidth',2);    
end

ylim([0 1.01]);
legend('Session 4','Session 6','Location', 'Southeast');
xlabel('Time (# of Rings)')
ylabel('Probability');
box off
hold off 


