% script m-file IterateModelRuns.m makes repeated calls to the modeling
% function, feeding different parameters on each call. Stimuli.mat,
% Data.mat,SubRws1.mat, SesRws1.mat and ChangePointModel.m must be in
% Matlab's current directory when these cells are run.

PgHyp = [.5 .5]; % parameters of uniform beta prior on p_g
% To compute results using Jeffreys prior on p_g, comment out above and
% uncomment line below. Do likewise for Lines 39 & 41 in loop in next cell
% PgHyp = [.5 .5];

PChyp = [.5 .5]; % parameters of beta prior on p_c
%
T1 = [.18 .23 .29 .35 .44 .54 .66 .82 1.04 1.35 1.92]; % T1 is decision 
% criterion on nDkl (the evidence that there is a problem). The loop in the
% next cell steps through these values

T2 = [.5 1 2 4 8 16]; % T2 is the decision criterion on the posterior odds;
% the loop in the next cell steps through these values


%% loading files and running model repeatedly; these files must be in the
% current directory
load Data.mat

load SubRws1.mat

load SesRws1.mat

load Stimuli.mat

% Iterated calls to ChangePointModel
for i = 1: length(T2)
    
    for j = 1:length(T1)
        
        Rslts_T2(i).T1(j) = ChangePointModel(T1(j),T2(i),PChyp,PgHyp);
         
    end  
end


