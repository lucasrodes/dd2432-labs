%% Artificial Neural Networks and other Learning Systems - Lab 2

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
addpath('info');

%% 6 Function Approximation for Noisy Data
%

% Setup
clear;
rng(3); % 3 or 4 don't lead to a big dead unit
plotinit;

% Load data
[xtrain, ytrain]=readxy('ballist',2,2);
[xtest, ytest]=readxy('balltest',2,2);

allx = [xtrain, xtest];
ally = [ytrain, ytest];
scatter3(allx(:,1), allx(:,2), ally(:,1),'.');
scatter3(allx(:,1), allx(:,2), ally(:,2)),'.';

% Use e.g. 20 units and train the unsupervised part.
data=xtrain;
units=20;
vqinit;
singlewinner=1;
emiterb

% In this case we have a two dimensional input space and a two dimensional
% output space. The two output values are coded with one unit each in the output
% layer. Here we will train the weight vectors to these units separately. Start as
% before by calculating the matrix $\phi$ of RBF responses:
Phi=calcPhi(xtrain,m,sigma2);

% Extract the two desired y vectors for train and test
d1=ytrain(:,1);
d2=ytrain(:,2);
dtest1=ytest(:,1);
dtest2=ytest(:,2);

% and calculate the weight vectors by the pseudo inverse method
w1=Phi\d1;
w2=Phi\d2;

% Now we can calculate approximations of training data
y1=Phi*w1;
y2=Phi*w2;

% as well as approximations of test data
Phitest=calcPhi(xtest,m,sigma2);
ytest1=Phitest*w1;
ytest2=Phitest*w2;

% Finally we plot these
figure;
subplot(2,2,1); xyplot(d1,y1,'train angle'); grid on; axis('image');
subplot(2,2,2); xyplot(d2,y2,'train velocity');  grid on; axis('image');
subplot(2,2,3); xyplot(dtest1,ytest1,'test angle'); grid on; axis('image');
subplot(2,2,4); xyplot(dtest2,ytest2,'test velocity');  grid on; axis('image');

x1_full= linspace(min(allx(:,1)), max(allx(:,1)), 100);
x2_full= linspace(min(allx(:,2)), max(allx(:,2)), 100);
[x1_full, x2_full] = meshgrid(x1_full, x2_full);
data = [x1_full(:), x2_full(:)];

Phifull=calcPhi(data,m,sigma2);
y1_full=Phifull*w1;
y2_full=Phifull*w2;

figure;
subplot(1,2,1);
hold on;
surf(x1_full,x2_full,reshape(y1_full,[100,100]), 'EdgeColor','none','LineStyle','none','FaceLighting','phong');
scatter3(allx(:,1), allx(:,2), ally(:,1),'.k');
colormap cool;
alpha(.2);

subplot(1,2,2);
hold on;
surf(x1_full,x2_full,reshape(y2_full,[100,100]), 'EdgeColor','none','LineStyle','none','FaceLighting','phong');
scatter3(allx(:,1), allx(:,2), ally(:,2),'.k');
colormap cool;
alpha(.2);

%% 6 Bonus - Smoothing with Gaussian kernel
% Smoothing with a Gaussian kernel:
%
% $$\tilde{y}^1_i = \sum_k e^{-\frac{||\mathbf{x}_i-\mathbf{x}_k||^2}{\sigma}}y^1_k$$
%
% $$\tilde{y}^2_i = \sum_k e^{-\frac{||\mathbf{x}_i-\mathbf{x}_k||^2}{\sigma}}y^2_k$$
%
% In matrix form:
%
% $$ K_{i,j} = e^{-\frac{||\mathbf{x}_i-\mathbf{x}_k||^2}{\sigma}} $$
%
% $$\tilde{y} = K y$$
%

% Just to plot the gram matrix
K = exp(-squareform(pdist(sort(xtrain), 'squaredeuclidean')./0.05)); 
HeatMap(K);
% Actual useful gram matrix
K = exp(-squareform(pdist(xtrain, 'squaredeuclidean')./0.05));
ytrain_smooth = K*ytrain ./ [sum(K, 2), sum(K, 2)];

% Plotting just ytrain vs ytrain smoothed (dimension 1 only)
figure;
hold on;
scatter3(xtrain(:,1), xtrain(:,2), ytrain(:,1),'.b');
scatter3(xtrain(:,1), xtrain(:,2), ytrain_smooth(:,1),'.r');

% Plotting surfaces from ytrain vs ytrain smoothed (dimension 1 only)
figure;
hold on;

% non smoothed y1
x1_full= linspace(min(xtrain(:,1)), max(xtrain(:,1)), 100);
x2_full= linspace(min(xtrain(:,2)), max(xtrain(:,2)), 100);
[x1_full_mesh, x2_full_mesh] = meshgrid(x1_full, x2_full);
z_train = griddata(xtrain(:,1), xtrain(:,2), ytrain(:,1) ,x1_full_mesh, x2_full_mesh);
scatter3(xtrain(:,1),xtrain(:,2),ytrain(:,1),'.r');
surf(x1_full_mesh,x2_full_mesh,z_train, 'EdgeColor','red','FaceLighting','phong');
alpha(0.2);

% smoothed y1
x1_full= linspace(min(xtrain(:,1)), max(xtrain(:,1)), 100);
x2_full= linspace(min(xtrain(:,2)), max(xtrain(:,2)), 100);
[x1_full_mesh, x2_full_mesh] = meshgrid(x1_full, x2_full);
z_train = griddata(xtrain(:,1), xtrain(:,2), ytrain_smooth(:,1) ,x1_full_mesh, x2_full_mesh);
surf(x1_full_mesh,x2_full_mesh,z_train, 'EdgeColor','blue','FaceLighting','phong');
scatter3(xtrain(:,1),xtrain(:,2),ytrain_smooth(:,1),'.b');
alpha(0.2);

Phi=calcPhi(xtrain,m,sigma2);

% Extract the two desired y vectors for train and test
d1=ytrain_smooth(:,1);
d2=ytrain_smooth(:,2);
dtest1=ytest(:,1);
dtest2=ytest(:,2);

% and calculate the weight vectors by the pseudo inverse method
w1=Phi\d1;
w2=Phi\d2;

% Now we can calculate approximations of training data
y1=Phi*w1;
y2=Phi*w2;

% as well as approximations of test data
Phitest=calcPhi(xtest,m,sigma2);
ytest1=Phitest*w1;
ytest2=Phitest*w2;

% Finally we plot these
figure;
subplot(2,2,1); xyplot(d1,y1,'train angle'); grid on; axis('image');
subplot(2,2,2); xyplot(d2,y2,'train velocity');  grid on; axis('image');
subplot(2,2,3); xyplot(dtest1,ytest1,'test angle'); grid on; axis('image');
subplot(2,2,4); xyplot(dtest2,ytest2,'test velocity');  grid on; axis('image');

x1_full= linspace(min(allx(:,1)), max(allx(:,1)), 100);
x2_full= linspace(min(allx(:,2)), max(allx(:,2)), 100);
[x1_full, x2_full] = meshgrid(x1_full, x2_full);
data = [x1_full(:), x2_full(:)];

Phifull=calcPhi(data,m,sigma2);
y1_full=Phifull*w1;
y2_full=Phifull*w2;

figure;
subplot(1,2,1);
hold on;
surf(x1_full,x2_full,reshape(y1_full,[100,100]), 'EdgeColor','none','LineStyle','none','FaceLighting','phong');
scatter3(allx(:,1), allx(:,2), ally(:,1),'.k');
colormap cool;
alpha(.2);

subplot(1,2,2);
hold on;
surf(x1_full,x2_full,reshape(y2_full,[100,100]), 'EdgeColor','none','LineStyle','none','FaceLighting','phong');
scatter3(allx(:,1), allx(:,2), ally(:,2),'.k');
colormap cool;
alpha(.2);

%% 
close all;