%--------------------------------------------------------------------------
% Data Contents for Data in Brief article
% Khaw, Stevens, Woodford, September 2017 
%--------------------------------------------------------------------------

This zip archive should include the following files:

1) ‘props.txt’      - Data file for the hidden probability
2) ‘phats.txt’      - Data file for subjects’ estimates
3) ‘obs.txt’        - Data file for ring realizations
4) ‘rtimes.txt’     - Data file for subjects’ reaction times
5) ‘DIB.m’          - MATLAB code file to generate figures used in article
6) ‘semilogxhist.m’ - Helper MATLAB function to produce histogram used in article

Within each data file, there are 109,890 observations for each variable corresponding to 11 subjects, 10 sessions per subject (999 observations per session). 

Each column represents observations from a single session from a single subject. 
Hence, data from a specific session can be accessed by the following column index:

(S-1)*10 + s

where ’S’ is the subject number and ’s’ is the session number.  