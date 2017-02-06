%% Artificial Neural Networks and other Learning Systems - Lab 2

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
addpath('info');

%% 5 RBF Placement by Self Organization
% <html><h3>Competitive learning - single winner</h3></html>
clear;
plotinit;
data=read('cluster');
units=5;
vqinit;
singlewinner=1;
vqiter;

% // TODO fix drifting trajectories of the centers

%%
% <html><h3>Competitive learning - shared updates</h3></html>
clear;
plotinit;
data=read('cluster');
units=5;
vqinit;
singlewinner=0;
vqiter;

%%
% _*Question 9,10*_
% Using a single winner strategy:
%
% * *CON* Sometimes some units never get updated because at the very beginning they
% were placed in un unfortunate position, far from any datapoint.
% * *PRO* Once the fitting is complete we can check through the validation
% data whether there are units that are never highly active and delete
% them.

%% 
% <html><h3>Expectation maximization - single winner</h3></html>
%
% We see how the central big unit is "dead", meaning that it will no be
% useful for a latter clustering of the data due to the fact that its
% center is too far from any of the inputs and its spread is too high to
% have an high activation value.
clear;
rng(123);
plotinit;
data=read('cluster');
units=5;
vqinit;
singlewinner=1;
emiterb;

%%
% <html><h3>Expectation maximization - shared updates - 7 units</h3></html>
clear;
rng(12);
plotinit;
data=read('cluster');
units=7;
vqinit;
singlewinner=0;
emiterb;

%%
% <html><h3>Expectation maximization - shared updates - 30 units</h3></html>
%
% We can see how allowing the spread of the RBF units to vary yields an
% even worse overfitting because the, each input point has a corresponding RBF centered on it, 
% some of them even more than one)
clear;
rng(12);
plotinit;
data=read('cluster');
units=30;
vqinit;
singlewinner=0;
emiterb;

%% 
close all;