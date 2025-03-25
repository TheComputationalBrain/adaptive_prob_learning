function ProbEstExperiment

% figure;
% Ones=find(dataSequence(1:999)>0);
% Zeros=find(dataSequence(1:999)<1);
% plot(Ones,.1*ones(length(Ones)),'+r', 'MarkerSize',3)
% hold on
% plot(Zeros,.095*ones(length(Zeros)),'g+','MarkerSize',3)
% hold on
% ylim([0 0.15]);
%
% num = length(index_CP);
% y1 = zeros(1,num); y2 = .12*ones(1,num);
% aa = [index_CP(1:num) index_CP(1:num)];
% bb = [y1(1:num) y2(1:num)];
% for i=1:num
%     line([index_CP(i) index_CP(i)], [y1(i) y2(i)])
% end
%
%
% text(10, 0.13, num2str(pList(2:length(pList))), 'FontSize', 7);
% text(10, 0.12, num2str(index_CP), 'FontSize', 7);
%
% hold off
%%%%%%%%%%%%
% This script runs experiment 3-main
%create main gui figure
parentFigHandle = figure('Visible', 'on', 'Name', 'EXPERIMENT_3: ESTIMATION of PROPORTIONS w/ CHANGE DETECTION',...
    'Position', [200, 100, 800, 550], 'Color', [0 0 0], 'MenuBar', 'none');
%whitebg(parentFigHandle, [0.8 0.66 1]);

%create toolbar -  NOT required
%toolBarHandle = uitoolbar(parentFigHandle);
%add pushtools

%create gui data structure
subjData.Id = 0;
subjData.date = clock;
subjData.name = 'Monika';
guidata(parentFigHandle,subjData);
% ******** DOOOOOOOOO Toolbar entries: Task description, subj #, initial,
% and later plot data, analyze etc
%************ DOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
introText = sprintf(strcat('\n\n This is the main experiment.\n\n\n Enter your initials, your subject number and your session number in the space provided.\n\n\n',...
    'Then click "BEGIN" to start the experiment\n\n\n\n',...
    'GOOD LUCK !!!'));

% create uipanel object as bg for instructions
introPanelHandle = uipanel('Title','EXPERIMENT 3',...
    'FontSize',12, 'FontWeight', 'bold',...
    'BackgroundColor', [0.9 0.9 1],...
    'Position',[.12 .02 .87 .95], 'FontName','Arial');

%create intro-instruction static text field - size of main gui minus subj field?
%introText = 'Your task is to estimate the proporion of green marbles present in the jar on the basis of evidence';
introTextHandle= uicontrol(introPanelHandle,'Style','text',...
    'String', introText, 'HorizontalAlignment', 'left',...
    'FontSize',10,...
    'ForegroundColor', [0 0 0],...
    'BackgroundColor', [0.9 0.9 0.95], 'Units', 'normalized',...  %[0.5 0.7 1]
    'Position', [0.02 0.02 0.95 0.95]); %[10 10 675 490]); %, 'FontName','Arial'); %[110, 100, 680, 400] pos.


%DOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
%create subject Info Edit_textboxes  XXXXXXXXXX change, no value assoc.w/ edittext
subjInitialHandle = uicontrol(parentFigHandle, 'Style','edit',...
    'BackgroundColor',[0.2 0.5 0.7],...
    'String','Subj. Initials', 'Position',[10, 600, 65, 20]);

subjNumberHandle = uicontrol(parentFigHandle, 'Style','edit',...
    'BackgroundColor',[0.2 0.5 0.7],...
    'String','Subj # ', 'Position',[10, 550, 50, 20]);

sessionNumberHandle = uicontrol(parentFigHandle, 'Style','edit',...
    'BackgroundColor',[0.2 0.5 0.7],...
    'String','Session #', 'Position',[10, 500, 70, 20]);
 

beginFlag = 'off';

%create subject info GUIdata structure
mySubjectData.name = get(subjInitialHandle, 'String');
mySubjectData.IdNumber = get(subjNumberHandle, 'String');
mySubjectData.sessionNumber = get(sessionNumberHandle, 'String');
mySubjectData.dateTime = date;
mySubjectData.blockNumber = 1;
mySubjectData.inputColorSequence = [];
mySubjectData.outputEstmProportionVector = [];
mySubjectData.currPValue = [];
mySubjectData.ResponseTimeVector = [];
mySubjectData.objectiveChangeCount = 0;
mySubjectData.objectiveChangeIndices = 0;
mySubjectData.objectivePValues = [];
mySubjectData.subjectiveChangeCount = 0;
mySubjectData.subjectiveChangeIndices = [];
mySubjectData.changeFlag = 'false';

mySubjectData.subjectiveWithdrawChangeCount = 0;
mySubjectData.subjectiveWithdrawChangeIndices = [];
mySubjectData.verbalChangeRecord = [];
mySubjectData.withdrawChangeFlag = 'false';


guidata(parentFigHandle, mySubjectData);
% Use clock to get date and time. USE tic and toc to measure RT.
disp(mySubjectData)

% create ring path
xx = linspace(0.15,0.20,5);
yy(1:5) = 0.38;
ringCoords=[xx;yy];
% try roll  0.06 0.4 0.15 0.26]
 
%create begin experiment practice button
beginPracticeText = {'BEGIN', 'PRACTICE', 'SESSION'};
beginBtnHandle = uicontrol(parentFigHandle, 'String', 'BEGIN',...
    'BackgroundColor',[0.93 0.9 0.5],...
    'Position', [10, 400, 70, 40], 'Tag', 'beginButton', 'Callback', @beginButton_Callback);
%wrap button label text and reposition  NOT SHOWING UP :-(
% [outstring, newpos] = textwrap(beginBtnHandle,beginPracticeText)
% pos(4) = newpos(4); % this assigns zeroes to the first 3 elements of pos!!
% set(beginBtnHandle,'String',outstring,'Position', [pos(1),pos(2),pos(3)+100,pos(4)+20]);

    function beginFlag = beginButton_Callback(beginBtnHandle, eventdata, handles)
        %make intro text and intoPanel invisible
        set(introTextHandle,'Visible','off');
        set(introPanelHandle,'Visible','off');
        % hide subj info fields
        set(subjNumberHandle,'Visible','off');
        set(subjInitialHandle,'Visible','off');
        set(sessionNumberHandle,'Visible','off');
        
        %save Subject info
        mySubjectData.IdNumber = str2num(get(subjNumberHandle, 'String'));
        mySubjectData.sessionNumber = str2num(get(sessionNumberHandle, 'String'));
        mySubjectData.name = str2num(get(subjInitialHandle, 'String'));
        
        %hide begin button
        set(beginBtnHandle,'Visible','off');
        beginFlag ='on';
        % refresh guidata Note: both get and save of guidata struct values via
        % func guidata
        %subjData = guidata(beginBtnHandle); %getting the struct
        mySubjectData.name = get(subjInitialHandle, 'String'); % modify
        % REMOVE to test array length:  mySubjectData.inputColorSequence = [mySubjectData.inputColorSequence 9];
        guidata(parentFigHandle,mySubjectData); %save modified struct as guidata
        % VERIFY initial subject data
        disp(mySubjectData)

        if (beginFlag=='on')

            %create slider ui
            sliderHandle = uicontrol(parentFigHandle, 'Style', 'slider',...
                'Max', 1, 'Min',0, 'Value',0,...
                'SliderStep',[0.05 0.1], 'Units', 'normalized',... %[80 150 400 20]
                'Position',[0.06 0.3 0.5 0.05], 'BackgroundColor', [0.2 0.3 1],...
                'Callback',@slider_callback);

            %uicontrol(sliderHandle); %focus
            %create slider label static text box
            statictextHandle= uicontrol(parentFigHandle,'Style','text',... %Label Slider
                'String', sprintf('0%%                                                                 100%%\n NO GREEN                                                         ALL GREEN'),...
                'FontSize', 13, 'FontWeight', 'bold', 'Units', 'normalized',... %[60 110 415 30]
                'Position', [0.01 0.2 0.6 0.05],'BackgroundColor', [0 0 0], 'ForegroundColor', [0 0.8 0],'FontName','FixedWidth');

            %create NEXT button to trigger next data element
            submitBtnHandle = uicontrol('Style', 'pushbutton', 'String', 'NEXT',...
                'FontSize', 13, 'FontWeight', 'bold',...
                'Units', 'normalized',... %[490 130 60 30]
                'Position', [0.6 0.3 0.1 0.07], 'Callback', @pushbutton1_Callback);

            changeBtnHandle = uicontrol('Style', 'pushbutton', 'String', 'I think the box has changed',...
                'FontSize', 12, 'FontWeight', 'bold',...
                'Units', 'normalized','BackgroundColor', [0.2 0.5 0.9], 'ForegroundColor', [0 0 0],... %[500 170 30 30]
                'Position', [0.6 0.24 0.12 0.05], 'Callback', @pushbuttonChange_Callback);            
            
            withdrawChangeBtnHandle = uicontrol('Style', 'pushbutton', 'String', 'I take that back!',...
                'FontSize', 12, 'FontWeight', 'bold',...
                'Units', 'normalized','BackgroundColor', [0.8 0.8 0.5], 'ForegroundColor', [0 0 0],... %[500 170 30 30]
                'Position', [0.66 0.2 0.062 0.03], 'Callback', @pushbuttonWithdrawChange_Callback);    

            % create proportion textbox and set value to curr slider value
            % pVal=0;  **************************************SKIP for now
            %pVal= get(sliderHandle,'Value');
            %propDisplayTextHandle=uicontrol(parentFigHandle,'Style','text',...
            % 'String', num2str(pVal),'FontWeight', 'bold',...
            % 'Position', [300 160 40 20]);

            % % TRY rectangle object
            % dataRectHandle= rectangle('Position',[0.3 0.6 0.55,1.38],'Curvature',[1,1],... %[0,0,0.55,1.38]
            % 'FaceColor','b');

            %create Jar
            jarN = 1000;
            jarHandle = axes('Parent',parentFigHandle,'Position',[.60 .40 .35 .45], 'Color', 'none','Box', 'on');
            set(jarHandle,'XTickLabel','');
            set(jarHandle,'YTickLabel','');
            set(jarHandle,'XTick',0);
            set(jarHandle,'YTick',0);

 
            %prepare data sequence
            pList=[]; index_CP=[]; pChange = 0.005; numTrials=1000;
            [pList, index_CP] = getPList(pChange, numTrials);
            mySubjectData.objectiveChangeCount = length(pList);
            mySubjectData.pValues = pList;
            mySubjectData.objectiveChangeIndices = index_CP;
            guidata(parentFigHandle,mySubjectData);
            dataSequenceArray = [];
            dataSequenceArray = prepareDataSequence(pList, index_CP);
            disp('The p vals are');            
            mySubjectData.pValues
            mySubjectData.objectivePValues = pList(2:end);

            %currentDataElement_X=1; % dont need this here. might later if doing mult sessions

            currentDataElement_Y=0;
            %create static text obj to represent data element
            dataRectHandle= uicontrol(parentFigHandle,'Style','text','FontWeight', 'bold',...
            'String', 'O','FontSize', 50, 'ForegroundColor','black','BackgroundColor', 'black', 'Units', 'normalized',...
            'Position', [0.125 0.6 0.07 0.07]); %[100 250 50 50]);
 
        end
        
        %create bin to hold input dataRect
        dataBinTextBoxHandle = uicontrol(parentFigHandle,'Style','pushbutton',...
            'String','Box of RINGS','FontSize', 16,...
            'BackgroundColor', [0.2 0.3 0.7],'Units', 'normalized',... %[0.2 0.3 1]
            'Position', [0.06 0.4 0.15 0.3], 'Visible', 'on');
          
                     
        % w = waitforbuttonpress;
        % if w == 0
        %     disp('Button click')
        % else
        %     disp('   Key press')
        % end


        % Start Timer to measure RT in the very first trial ONLY
        tic;               % corresponding toc called in SUBMIT callback
        % Start Timer to measure RT in the very first trial ONLY

        function slider_callback(hSlider,eventdata)
            rx = 1 + (jarN-1)*rand(jarN,1);
            ry = 1 + (jarN-1)*rand(jarN,1);
            propValue=0;
            propValue=get(hSlider,'Value');


            %updating proportion text box lable to display current slider value
            % SKIP for NOW *** set(propDisplayTextHandle,'String',num2str(propValue));
            % Refresh JAR
            gNum=ceil(propValue*jarN); rNum=jarN-gNum;
            gYPoints=ry(1:gNum); rYPoints=ry(gNum+1:jarN);
            gXPoints = rx(1:gNum); rXPoints = rx(1+gNum:jarN);
            set(jarHandle,'NextPlot','replace');
            newplot; %required to force a refresh when gXPoints becomes [] at p=0
            plot(gca, gYPoints, gXPoints, 'Linestyle', 'none', 'Marker', 'o', 'MarkerSize', 5, 'Color', [0 0.8 0]); %'* g');

            %create title
            titleTextHandle = text('String','MY CURRENT ESTIMATE:  % of GREEN RINGS','Color','b','FontSize',14);
            set(jarHandle,'Title',titleTextHandle);

            set(gca,'XTickLabel','');
            set(gca,'YTickLabel',''); %reqd to hide ticks and coords
            set(gca,'XTick',0);
            set(gca,'YTick',0);
            set(gca,'Xlim',[-2,1000]);
            set(gca,'Ylim',[-2,1000]);
            hold on
            plot(gca, rXPoints, rYPoints, 'Linestyle', 'none', 'Marker', 'o', 'MarkerSize', 5,'Color', [1 0 0]);  % r');
            set(jarHandle,'Color', 'none');
            hold off
        end

        % code for key press to move slider position


        % change call back leads to submit calback
        function pushbuttonChange_Callback(hChangeBtn, eventdata, handles)
            set(changeBtnHandle,'Enable','off');
            mySubjectData.changeFlag = 'true';
            mySubjectData.subjectiveChangeCount = mySubjectData.subjectiveChangeCount+1;
            mySubjectData.subjectiveChangeIndices = [mySubjectData.subjectiveChangeIndices currentDataElement_Y];
            guidata(parentFigHandle,mySubjectData);
            dummy2 = []; dummy3=[];
            pushbutton1_Callback(submitBtnHandle, dummy2, dummy3);
        end
        
        
        % WITHDRAW PREV CHANGE call back leads to submit callback
        function pushbuttonWithdrawChange_Callback(hWithdrawChangeBtn, eventdata, handles)
            set(withdrawChangeBtnHandle,'Enable','off');
            mySubjectData.withdrawChangeFlag = 'true';            
            mySubjectData.subjectiveWithdrawChangeCount = mySubjectData.subjectiveWithdrawChangeCount + 1;
            mySubjectData.subjectiveWithdrawChangeIndices = [mySubjectData.subjectiveWithdrawChangeIndices currentDataElement_Y];
            guidata(parentFigHandle,mySubjectData); %save guidata
            dummy2 = []; dummy3=[];
            pushbutton1_Callback(submitBtnHandle, dummy2, dummy3);
        end

        function pushbutton1_Callback(hSubmitBtn, eventdata, handles)

            set(submitBtnHandle,'Enable', 'off');
            set(changeBtnHandle,'Enable','off');
            set(withdrawChangeBtnHandle,'Enable','off');
            %create vertical DOWN path for dataRect
                    
            if currentDataElement_Y > 0                     
                numOfPoints = size(ringCoords,2);            
                for i=4:(-1):1
                    set(dataRectHandle, 'Position', [ringCoords(1,i) ringCoords(2,i) 0.07 0.07]);
                    pause(0.005);
                end

            else 
                currentDataElement_Y=1;  
            end
            
            trialDuration = toc;
            mySubjectData.ResponseTimeVector = [mySubjectData.ResponseTimeVector trialDuration];

            if strcmp(mySubjectData.changeFlag,'false')==1 && strcmp(mySubjectData.withdrawChangeFlag,'false')==1
                newPropEstimate=get(sliderHandle,'Value');
                mySubjectData.outputEstmProportionVector = [mySubjectData.outputEstmProportionVector newPropEstimate];
                mySubjectData.verbalChangeRecord = [mySubjectData.verbalChangeRecord 0]; %0-no change, 1-change, 2-change withdrawn
                
            elseif strcmp(mySubjectData.changeFlag,'true')==1
                 
                mySubjectData.verbalChangeRecord = [mySubjectData.verbalChangeRecord 1]; %0-no change, 1-change, 2-change withdrawn                
                newPropEstimate=get(sliderHandle,'Value');
                mySubjectData.outputEstmProportionVector = [mySubjectData.outputEstmProportionVector newPropEstimate]; %instead of 2
                 
                mySubjectData.changeFlag = 'false';
                
            elseif strcmp(mySubjectData.withdrawChangeFlag,'true')==1
                mySubjectData.verbalChangeRecord = [mySubjectData.verbalChangeRecord 2];  %0-no change, 1-change, 2-change withdrawn                
                newPropEstimate=get(sliderHandle,'Value');
                mySubjectData.outputEstmProportionVector = [mySubjectData.outputEstmProportionVector newPropEstimate]; %instead of estm=2
                 
                mySubjectData.withdrawChangeFlag = 'false';
                
            end
                                 
            if currentDataElement_Y > numTrials
                %
                pause(2.5)
                set(dataBinTextBoxHandle,'Visible','off');
                set(dataRectHandle,'ForegroundColor','black','BackgroundColor', 'black');
                pause(1.5)
                % save guidata corresp to current block
                guidata(parentFigHandle, mySubjectData);
                
                mySubjectData.dateTime = [mySubjectData.dateTime clock];
                set(submitBtnHandle,'Enable', 'off');
                set(changeBtnHandle,'Enable', 'off');
                set(withdrawChangeBtnHandle,'Enable','off');
 
                subjOpFileName = strcat('CPexper_Subj_', num2str(mySubjectData.IdNumber), '_', num2str(mySubjectData.sessionNumber), '.mat');
                save(subjOpFileName, '-struct', 'mySubjectData',  'IdNumber', 'sessionNumber', 'objectivePValues','inputColorSequence',...
                    'objectiveChangeCount', 'objectiveChangeIndices', 'outputEstmProportionVector', 'ResponseTimeVector', ...
                    'subjectiveChangeCount', 'subjectiveChangeIndices', 'name',...
                    'subjectiveWithdrawChangeCount', 'subjectiveWithdrawChangeIndices', 'verbalChangeRecord');


                h = msgbox('********* PHEW!!! End of Experiment!!! *********', 'STOP!','help','modal');

                %HIDE task related objects and SHOW End panel.
                set(submitBtnHandle,'Visible','off');
                set(changeBtnHandle,'Visible','off');
                set(withdrawChangeBtnHandle,'Visible','off');
                set(statictextHandle,'Visible','off');
                set(jarHandle,'Visible','off');
                set(sliderHandle,'Visible','off');
                set(dataRectHandle,'Visible','off');
                % SHOW end panel after setting text string
                byeText = sprintf('\n\n\n\nTHANK YOU FOR PARTICIPATING IN THIS EXPERIMENT')
                set(introTextHandle,'String',byeText, 'FontSize',15);
                set(introPanelHandle,'Title','THE END');
                set(introPanelHandle,'Visible','on');
                set(introTextHandle,'Visible','on');
                
                %CLOSE ALL OPENFILES HERE
            else
                currentInput = dataSequenceArray(currentDataElement_Y);
                if currentInput == 1
                    dataColor = 'g'; mySubjectData.inputColorSequence=[mySubjectData.inputColorSequence 1];
                else
                    dataColor = 'r'; mySubjectData.inputColorSequence=[mySubjectData.inputColorSequence 0];
                end

                % inputcolorseq being updated twice?? To save data -- USE load on .mat file - gen
                % matrix - write to xls
                %mySubjectData.inputColorSequence = [mySubjectData.inputColorSequence currentInput];
                % ditch for now objectInfo = get(hSubmitBtn,'handles')
                set(dataRectHandle,'ForegroundColor','black','BackgroundColor', 'black');
                pause(0.6);
                set(dataRectHandle,'ForegroundColor',dataColor);%,'BackgroundColor', dataColor);

                %create vertical UP path for dataRect  
                 % add random jitt to last coord's y and x
                newCoords = ringCoords;
                newCoords(1,4)= ringCoords(1,4)+0.01*rand();
                newCoords(2,4)=ringCoords(2,4)+0.02*rand();
                if currentDataElement_Y <= numTrials
                    pause(0.005);
                    for i=1:4
                        set(dataRectHandle, 'Position', [newCoords(1,i) newCoords(2,i) 0.07 0.07]);
                        pause(0.007);
                    end
                end

                currentDataElement_Y=currentDataElement_Y+1; 
                pause(0.2)
                set(submitBtnHandle,'Enable', 'on');
                set(changeBtnHandle,'Enable','on');
                set(withdrawChangeBtnHandle,'Enable','on');
            end
        end

    end
end

function [pList index_CP] = getPList(pChange, N) % computes the locations
% (index_CP) and the p values (pList) of the successive values of p in a
% given session. N is sequence length (=1,000in Psych Rev experiment).
% pChange is probability of a change (=.005 in Psych Rev exper)

obj_CP = binornd(1, pChange, 1, N); % a 1xN vector of 0 and 1, 1 represents
% "change" in p
index_CP = find(obj_CP); % finds the locations within the above vector where
% the 1's are, that is, the change point locations

% append index=1 to index_CP
if index_CP(1) ~= 1
    index_CP = [1 index_CP]; % notional change point at start of sequnce.
    % in other words, every sequence begins with a change point
end
% append index=1000 to index_CP  N = 30 in the practice session, 1000 else.
if index_CP(length(index_CP)) ~= N
    index_CP = [index_CP N]; % notional change point at end of sequence
    % In other words, every sequence of locations starts and ends with a
    % notional change point
end

num_CP = length(index_CP); % computing the p value for each step, that is,
% for each successive location in the sequence of steps
p1 = rand; % first p value, the value w which sequence begins
pList = [p1]; % building the vector of p values
for i = 1:(num_CP-1) % stepping through the locations
    p2 = rand; % choosing a random p
    diff = abs(log10(p2/(1-p2))-log10(p1/(1-p1))); % computing how much the
    % odds for this p differ from the odds of the preceding p

    while abs(log10(p2/(1-p2))-log10(p1/(1-p1))) < 0.6 % .6 is the log10(4)
        % while the odds ratio between the previous p and the new p is less
        % than 4
        p2 = rand; % choose again
    end
    pList = [pList p2]; % enter new p in the vector of p values
    p1= p2;
end
end
% index_CP is the vector of change point locations
% plist is the vector of corresponding p values.
% Matt will apply his smooth function (moll) to the sequence of steps
% generated by the above function


function dataSequenceArray = prepareDataSequence(pList, index_CP)
% This generates the binary sequence seen by the subject. This original
% code assumes that p values are constant between change points. We will
% need to modify this when we use instead the smoothed sequences of p
% values that result when moll is applied to the above step function. The
% following command should give us what we want:
% O = binornd(ones(1,N),[Vector of smoothed p values])
dataSequenceArray = [];
num_CP = length(index_CP);
N  =1;
for i=2:num_CP
    %generate binomial seq of length determined by index_CP
    seqLength = index_CP(i)-index_CP(i-1);
    seq = binornd(N, pList(i), 1, seqLength);
    dataSequenceArray = [dataSequenceArray seq];
end
dataSequenceArray = [dataSequenceArray 0];
end
%%  

function getPlots
subjId = input('Enter subject number  ');

sessionNum = input('Enter session number  ');
filename = ['C:\toolbox\Psychtoolbox\Exp3CDwUndo_Subj_' num2str(subjId) '_' num2str(sessionNum) '.mat'];
load(filename);
whos

num_CP = length(objectiveChangeIndices);
for i=2:num_CP  
    p(objectiveChangeIndices(i-1):objectiveChangeIndices(i))=objectivePValues(i-1); 
end

figure
plot(p,'k')
hold on
plot(outputEstmProportionVector,'+','MarkerSize',3,'Color',[0 0.9 0])
hold on
%plot(inputColorSequence,'+')
Ones=find(inputColorSequence(1:1000)>0);
Zeros=find(inputColorSequence(1:1000)<1);
plot(Ones,1.5*ones(length(Ones)),'o', 'MarkerSize',1.5,'Color',[0 0.8 0])
hold on
plot(Zeros,1.52*ones(length(Zeros)),'or','MarkerSize',1.5)
plot(verbalChangeRecord, 'm*')

% plot data average between subjectiveChangeIndices
% plot data average between subjective changes in proportion

%-------
figureName = strcat('Exp2CD_Subj_', num2str(subjId), '_', num2str(sessionNum));
saveas(gcf,figureName,'fig');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
