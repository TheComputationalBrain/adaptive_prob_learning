load ../Data.mat

% column identification
% 1: Experimenter Code (1=Monika Krishan; 2 = Juliette Ye)
% 2: Subject ID number (1-10)
% 3: Session number (for a subject)
% 4: Stimulus (0 = red, 1 = green)
% 5: Subject's slider setting
% 6: Change Key (1 = clicked on to report a perceived change)
% 7: 2nd Thought (1 = clicked on to report a 2nd thought); NaN for first 5 Subjects
% 8: Reaction time in seconds (not analyzed)
% 9: Observed pg, as determined by running the ideal free real-time observer
%       algorithm
% 10: Objective change point; 1 = change in true (hidden) pg
% 11: Flags changes in  true p (not counting starts of sessions); sign indicates
%       direction of change
% 12 = signed step heights (changes in slider position)
% 13 = step widths


%% plot example session
iSub = 1;
iSess = 2;
ind = Data1(:,2) == iSub & Data1(:,3) == iSess;
subdat = Data1(ind,:);

clf;
hold on
plot(subdat(:,9), 'r')
plot(subdat(:,5), 'b')
plot(find(subdat(:, 6)), 0.5, 'go')

%% Plot average result

% Get charatecteristics of jump & jump detection
nSub =  max(Data1(:,2));
allJchar = cell(1, nSub);
for iSub = 1:nSub
    
    catOverSess = [];
    nSess = max(Data1(Data1(:,2) == iSub,3));
    
    for iSess = 1:nSess
        
        % get data for this subject & session
        ind = Data1(:,2) == iSub & Data1(:,3) == iSess;
        subdat = Data1(ind,:);
        
        % get true jump position
        TrueJpos = find(subdat(:, 10));
        if TrueJpos(1) == 1;
            % remove 1st Jump if it is at session initiation.
            TrueJpos = TrueJpos(2:end);
        end
        if TrueJpos(end) == size(subdat, 1)
            % remove last Jump if tis is at session end.
            TrueJpos = TrueJpos(1:end-1);
        end
        nJump = length(TrueJpos);
        
        % Get subject's detection
        SubJdec  = find(subdat(:, 6));
        
        % Get characteristics
        Jchar = zeros(nJump, 4);
        for k = 1:nJump
            Jchar(k, 1) = subdat(TrueJpos(k)-1, 9);         % pre-jump pg
            Jchar(k, 2) = subdat(TrueJpos(k)+1, 9);         % post-jump pg
            if k == 1
                Jchar(k, 3) = TrueJpos(k);                  % pre-jump length
            else
                Jchar(k, 3) = TrueJpos(k) - TrueJpos(k-1);
            end
            
            % compute latency of subject detection
            
            % get subject 1st detection after this jump
            ind = find(SubJdec > TrueJpos(k), 1, 'first');
            if ~isempty(ind) % i.e. if there is such a subjective detection
                
                % check there is not another, new jump before the subject's
                % detection. If so, the jump is considered as missed.
                
                if k < nJump % i.e. if not the last jump
                    
                    if SubJdec(ind) > TrueJpos(k+1) % i.e. if not detected
                        Jchar(k, 4) = 0;
                    else
                        Jchar(k, 4) = SubJdec(ind) - TrueJpos(k);
                    end
                else
                    Jchar(k, 4) = SubJdec(ind) - TrueJpos(k);
                end
            end
        end
        
        % concatenate data over sessions
        catOverSess = [catOverSess; Jchar];
    end
    
    % save subject's data
    allJchar{iSub} = catOverSess;
end

% concatenate over subjects
catdat = [];
for iSub = 1:nSub
    catdat = [catdat; allJchar{iSub}];
end

% remove jump dat were not detected
subcatdat = catdat(catdat(:, end) ~= 0,:);

changemag = abs(subcatdat(:,1) - subcatdat(:,2));
meddiff = median(changemag);
dat_easy = subcatdat(changemag > meddiff, :);
dat_hard = subcatdat(changemag < meddiff, :);

% Average data with bins
nBin = 5;
binEdge = prctile(subcatdat(:,3), linspace(0,100, nBin+1));

m_easy = zeros(1, nBin);
sem_easy = zeros(1, nBin);
m_hard = zeros(1, nBin);
sem_hard = zeros(1, nBin);
binc_easy = zeros(1, nBin);
binc_hard = zeros(1, nBin);

for iBin = 1:nBin
    % easy
    ind = (dat_easy(:,3) >= binEdge(iBin)) ...
        & (dat_easy(:,3) < binEdge(iBin+1));
    
    m_easy(iBin) = mean(dat_easy(ind, 4));
    sem_easy(iBin) = stderror(dat_easy(ind, 4));
    binc_easy(iBin) = mean(dat_easy(ind, 3));
    
    % hard
    ind = (dat_hard(:,3) >= binEdge(iBin)) ...
        & (dat_hard(:,3) < binEdge(iBin+1));
    
    m_hard(iBin) = mean(dat_hard(ind, 4));
    sem_hard(iBin) = stderror(dat_hard(ind, 4));
    binc_hard(iBin) = mean(dat_hard(ind, 3));
end

% plot results

figure(1); clf; set(gcf, 'Color', [1 1 1])
errorbar(binc_easy, m_easy, sem_easy, 'k')
hold on
errorbar(binc_hard, m_hard, sem_hard, 'k--')
legend({'large delta pg', 'small delta pg'}, ...
    'Location', 'NorthEastOutside')
 
xlabel('Duration of pre-jump period (in samples)')
ylabel('Subject Latency of 1st detection')

hgexport(1, '~/ResultGalistel_Observations.eps')





