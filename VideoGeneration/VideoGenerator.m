clear all; 
close all;

%% Delta
SEPARABLE_DATA = 1;
NONSEPARABLE_DAT = 0;

mode = SEPARABLE_DATA;
Delta_video(mode);

mode = NONSEPARABLE_DAT;
Delta_video(mode);

%% Backpropagation
mode = SEPARABLE_DATA;
backpropagation_video(mode);

mode = NONSEPARABLE_DAT;
backpropagation_video(mode);

%% Function Aproximation
% Experiments:
% - number of samples (15 is bad, 40 is good)
n = 25;

% - number of hidden neurons (with n=40: 2 is bad, 
%                                        6 is already good, 
%                                        1000 is too much because 
%                                        many weights just go to zero)
hidden = 3;
function_aprox_video(hidden, n,'Function_aprox_n_25_hid_3,');

% Experiments:
% - number of samples (15 is bad, 40 is good)
n = 40;

% - number of hidden neurons (with n=40: 2 is bad, 
%                                        6 is already good, 
%                                        1000 is too much because 
%                                        many weights just go to zero)
hidden = 3;
function_aprox_video(hidden, n,'Function_aprox_n_40_hid_3,');
% Experiments:
% - number of samples (15 is bad, 40 is good)
n = 25;

% - number of hidden neurons (with n=40: 2 is bad, 
%                                        6 is already good, 
%                                        1000 is too much because 
%                                        many weights just go to zero)
hidden = 6;
% function_aprox_video(hidden, n,'Function_aprox_n_25_hid_6,');
% Experiments:
% - number of samples (15 is bad, 40 is good)
n = 40;

% - number of hidden neurons (with n=40: 2 is bad, 
%                                        6 is already good, 
%                                        1000 is too much because 
%                                        many weights just go to zero)
hidden = 6;
function_aprox_video(hidden, n,'Function_aprox_n_40_hid_6,');