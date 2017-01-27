clear all; 
close all;

%% Delta
SEPARABLE_DATA = 0;
NONSEPARABLE_DAT = 1;

% Resolution
res = [30, 100]; % res(1) is frame-rate and res(2) is quality

disp('delta - separable data...')
delta_video(SEPARABLE_DATA, 'delta_sep.avi', res);

disp('delta - nonseparable data...')
delta_video(NONSEPARABLE_DAT,'delta_nonsep.avi', res);

%% Backpropagation
mode = SEPARABLE_DATA;
disp('backpropagation error - separable data...')
backpropagation_video(mode, 'back_prop_sep.avi', res);

mode = NONSEPARABLE_DAT;
disp('backpropagation error - nonseparable data...')
backpropagation_video(mode, 'back_prop_nonsep.avi', res);

%% Function Aproximation
% Experiments:
% - number of samples (15 is bad, 40 is good)
n = 25;

% - number of hidden neurons (with n=40: 2 is bad, 
%                                        6 is already good, 
%                                        1000 is too much because 
%                                        many weights just go to zero)
hidden = 3;
disp('Function Approximation')
function_aprox_video(hidden, n,'fct_aprox_n_25_hid_3', res);

% Experiments:
% - number of samples (15 is bad, 40 is good)
n = 40;

% - number of hidden neurons (with n=40: 2 is bad, 
%                                        6 is already good, 
%                                        1000 is too much because 
%                                        many weights just go to zero)
hidden = 3;
disp('Function Approximation')
function_aprox_video(hidden, n,'fct_aprox_n_40_hid_3', res);
% Experiments:
% - number of samples (15 is bad, 40 is good)
% n = 25;

% - number of hidden neurons (with n=40: 2 is bad, 
%                                        6 is already good, 
%                                        1000 is too much because 
%                                        many weights just go to zero)
% hidden = 6;
% function_aprox_video(hidden, n,'Function_aprox_n_25_hid_6,');
% Experiments:
% - number of samples (15 is bad, 40 is good)
n = 40;

% - number of hidden neurons (with n=40: 2 is bad, 
%                                        6 is already good, 
%                                        1000 is too much because 
%                                        many weights just go to zero)
hidden = 6;
disp('Function Approximation')
function_aprox_video(hidden, n,'fct_aprox_n_40_hid_6,');

hidden = 100;
disp('Function Approximation')
function_aprox_video(hidden, n,'fct_aprox_n_40_hid_10,');