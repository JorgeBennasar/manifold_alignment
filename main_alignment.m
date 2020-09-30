clear;
clc;

%%

addpath('C:\Users\jorge\OneDrive\Escritorio\thesis upload\Analysis')
addpath('C:\Users\jorge\OneDrive\Escritorio\thesis upload\DataProcessing')
addpath('C:\Users\jorge\OneDrive\Escritorio\thesis upload\Plotting')
addpath('C:\Users\jorge\OneDrive\Escritorio\thesis upload\td_limblab')
addpath('C:\Users\jorge\OneDrive\Escritorio\thesis upload\td_limblab\td_dpca')
addpath('C:\Users\jorge\OneDrive\Escritorio\thesis upload\td_limblab\td_gpfa')
addpath('C:\Users\jorge\OneDrive\Escritorio\thesis upload\Tools')
addpath('C:\Users\jorge\OneDrive\Escritorio\thesis upload\util')
addpath('C:\Users\jorge\OneDrive\Escritorio\thesis upload\util\subfcn')
addpath('C:\Users\jorge\OneDrive\Escritorio\thesis upload\my_LSTM_alignment')

%%

monkey_1 = 'Chewie_CO_CS_2015-03-19'; 
monkey_2 = 'Chewie_CO_CS_2015-03-12';
input_type = 'pca'; % 'pca' or 'isomap'
do_iso = 'no';
trials_per_target = 40;
selection = 'M1';
has_PMd_1 = 'no';
has_PMd_2 = 'no';
num_bins = 3; 
min_firing_rate = 0.05;
width_smooth = 0.1;
bins = [-10, 14];
time = bins(2)-bins(1)+1;
target_sel = 1:8;

% Monkey 1:

if strcmp(monkey_1,'Mihili_CO_VR_2014-03-03')
    load('Mihili_CO_VR_2014-03-03.mat', 'trial_data');
elseif strcmp(monkey_1,'Mihili_CO_FF_2014-02-18')
    load('Mihili_CO_FF_2014-02-18.mat', 'trial_data');
elseif strcmp(monkey_1,'Mihili_CO_FF_2014-02-17')
    load('Mihili_CO_FF_2014-02-17.mat', 'trial_data');
elseif strcmp(monkey_1,'Mihili_CO_VR_2014-03-04')
    load('Mihili_CO_VR_2014-03-04.mat', 'trial_data');
elseif strcmp(monkey_1,'Chewie_CO_FF_2016-10-07')    
    load('Chewie_CO_FF_2016-10-07.mat', 'trial_data');
elseif strcmp(monkey_1,'Chewie_CO_CS_2016-10-21')
    load('Chewie_CO_CS_2016-10-21.mat', 'trial_data');
elseif strcmp(monkey_1,'Chewie_CO_VR_2016-10-06')
    load('Chewie_CO_VR_2016-10-06.mat', 'trial_data');
elseif strcmp(monkey_1,'Chewie_CO_VR_2016-09-29')
    load('Chewie_CO_VR_2016-09-29.mat', 'trial_data');
elseif strcmp(monkey_1,'Chewie_CO_CS_2015-03-19')
    load('Chewie_CO_CS_2015-03-19.mat', 'trial_data');
elseif strcmp(monkey_1,'Chewie_CO_CS_2015-03-11')
    load('Chewie_CO_CS_2015-03-11.mat', 'trial_data');
elseif strcmp(monkey_1,'Chewie_CO_CS_2015-03-12')
    load('Chewie_CO_CS_2015-03-12.mat', 'trial_data');
elseif strcmp(monkey_1,'Chewie_CO_CS_2015-03-13')
    load('Chewie_CO_CS_2015-03-13.mat', 'trial_data');
end

td = removeBadTrials(trial_data);
td = binTD(td,num_bins);
td = removeBadNeurons(td,struct('min_fr',min_firing_rate));
td = sqrtTransform(td,struct('signals',{'M1_spikes'}));
td = smoothSignals(td,struct('signals',{'M1_spikes'},'width', ...
    width_smooth));  
if strcmp(has_PMd_1,'yes') == 1
    td = sqrtTransform(td,struct('signals',{'PMd_spikes'}));
    td = smoothSignals(td,struct('signals',{'PMd_spikes'},'width', ...
        width_smooth));
end

l = length(td);
t_1 = [];
t_2 = [];
t_3 = [];
counter = 0;
for i = 1:l
    t_1_now = td(i-counter).idx_target_on;
    t_2_now = td(i-counter).idx_movement_on;
    if (t_1_now < 3000) == 0 || (t_2_now < 3000) == 0
        td(i-counter) = [];
        counter = counter + 1;
    else
        t_1 = [t_1 td(i-counter).idx_target_on];
        t_2 = [t_2 td(i-counter).idx_movement_on];
    end
end

counter = 0;
for i = 1:length(td)
    threshold_x = 0.125*max(abs(td(i-counter).vel(:,1)));
    threshold_y = 0.125*max(abs(td(i-counter).vel(:,2)));
    a = abs(td(i-counter).vel(t_2(i-counter)+1,1) - ...
        td(i-counter).vel(t_2(i-counter),1));
    b = abs(td(i-counter).vel(t_2(i-counter)+1,2) - ...
        td(i-counter).vel(t_2(i-counter),2));
    if (a < threshold_x && b < threshold_y) || (t_2(i-counter) - ...
            t_1(i-counter) < -bins(1)-1) || ...
            (t_2(i-counter) <= -bins(1)) || ...
            size(td(i-counter).vel,1) - t_2(i-counter) < ...
            bins(2)+1 || strcmp(td(i-counter).epoch,'BL') == 0 || ...
            strcmp(td(i-counter).task,'CO') == 0
        td(i-counter) = [];
        t_1(i-counter) = [];
        t_2(i-counter) = [];
        counter = counter + 1;
    end
end

disp(['Removed ' num2str(counter) ' trials']);
td = trimTD(td,{'idx_movement_on',bins(1)},{'idx_movement_on',bins(2)}); 

target_index_1 = ones(1,8);
target_counter_1 = zeros(1,8);
targets_1 = zeros(1,length(td));
counter = 1;
for i = 1:length(td)
    if ismember(td(i).target_direction,target_index_1) == 0
        target_index_1(counter) = td(i).target_direction;
        counter = counter + 1;
    end
    idx = find(target_index_1 == td(i).target_direction);
    target_counter_1(idx) = target_counter_1(idx) + 1;
    targets_1(i) = idx;
end

if strcmp(selection,'M1')
    neurons = size(td(1).M1_spikes,2);
elseif strcmp(selection,'PMd')
    neurons = size(td(1).PMd_spikes,2);
end

N_1 = zeros(neurons,length(td),time);
Y_1 = zeros(2,length(td),time);
for i = 1:length(td)
    for j = 1:time
        if strcmp(selection,'M1')
            N_1(:,i,j) = td(i).M1_spikes(j,1:neurons);
        elseif strcmp(selection,'PMd')
            N_1(:,i,j) = td(i).PMd_spikes(j,1:neurons);
        end
        Y_1(:,i,j) = td(i).vel(j,:);
    end
end

% Monkey 2:

if strcmp(monkey_2,'Mihili_CO_VR_2014-03-03')
    load('Mihili_CO_VR_2014-03-03.mat', 'trial_data');
elseif strcmp(monkey_2,'Mihili_CO_FF_2014-02-18')
    load('Mihili_CO_FF_2014-02-18.mat', 'trial_data');
elseif strcmp(monkey_2,'Mihili_CO_FF_2014-02-17')
    load('Mihili_CO_FF_2014-02-17.mat', 'trial_data');
elseif strcmp(monkey_2,'Mihili_CO_VR_2014-03-04')
    load('Mihili_CO_VR_2014-03-04.mat', 'trial_data');
elseif strcmp(monkey_2,'Chewie_CO_FF_2016-10-07')    
    load('Chewie_CO_FF_2016-10-07.mat', 'trial_data');
elseif strcmp(monkey_2,'Chewie_CO_CS_2016-10-21')
    load('Chewie_CO_CS_2016-10-21.mat', 'trial_data');
elseif strcmp(monkey_2,'Chewie_CO_VR_2016-10-06')
    load('Chewie_CO_VR_2016-10-06.mat', 'trial_data');
elseif strcmp(monkey_2,'Chewie_CO_VR_2016-09-29')
    load('Chewie_CO_VR_2016-09-29.mat', 'trial_data');
elseif strcmp(monkey_2,'Chewie_CO_CS_2015-03-19')
    load('Chewie_CO_CS_2015-03-19.mat', 'trial_data');
elseif strcmp(monkey_2,'Chewie_CO_CS_2015-03-11')
    load('Chewie_CO_CS_2015-03-11.mat', 'trial_data');
elseif strcmp(monkey_2,'Chewie_CO_CS_2015-03-12')
    load('Chewie_CO_CS_2015-03-12.mat', 'trial_data');
elseif strcmp(monkey_2,'Chewie_CO_CS_2015-03-13')
    load('Chewie_CO_CS_2015-03-13.mat', 'trial_data');
end
 
td = removeBadTrials(trial_data);
td = binTD(td,num_bins);
td = removeBadNeurons(td,struct('min_fr',min_firing_rate));
td = sqrtTransform(td,struct('signals',{'M1_spikes'}));
td = smoothSignals(td,struct('signals',{'M1_spikes'},'width', ...
    width_smooth));  
if strcmp(has_PMd_1,'yes') == 1
    td = sqrtTransform(td,struct('signals',{'PMd_spikes'}));
    td = smoothSignals(td,struct('signals',{'PMd_spikes'},'width', ...
        width_smooth));
end

l = length(td);
t_1 = [];
t_2 = [];
t_3 = [];
counter = 0;
for i = 1:l
    t_1_now = td(i-counter).idx_target_on;
    t_2_now = td(i-counter).idx_movement_on;
    if (t_1_now < 3000) == 0 || (t_2_now < 3000) == 0
        td(i-counter) = [];
        counter = counter + 1;
    else
        t_1 = [t_1 td(i-counter).idx_target_on];
        t_2 = [t_2 td(i-counter).idx_movement_on];
    end
end

counter = 0;
for i = 1:length(td)
    threshold_x = 0.125*max(abs(td(i-counter).vel(:,1)));
    threshold_y = 0.125*max(abs(td(i-counter).vel(:,2)));
    a = abs(td(i-counter).vel(t_2(i-counter)+1,1) - ...
        td(i-counter).vel(t_2(i-counter),1));
    b = abs(td(i-counter).vel(t_2(i-counter)+1,2) - ...
        td(i-counter).vel(t_2(i-counter),2));
    if (a < threshold_x && b < threshold_y) || (t_2(i-counter) - ...
            t_1(i-counter) < -bins(1)-1) || ...
            (t_2(i-counter) <= -bins(1)) || ...
            size(td(i-counter).vel,1) - t_2(i-counter) < ...
            bins(2)+1 || strcmp(td(i-counter).epoch,'BL') == 0 || ...
            strcmp(td(i-counter).task,'CO') == 0
        td(i-counter) = [];
        t_1(i-counter) = [];
        t_2(i-counter) = [];
        counter = counter + 1;
    end
end

disp(['Removed ' num2str(counter) ' trials']);
td = trimTD(td,{'idx_movement_on',bins(1)},{'idx_movement_on',bins(2)}); 

target_index_2 = ones(1,8);
target_counter_2 = zeros(1,8);
targets_2 = zeros(1,length(td));
counter = 1;
for i = 1:length(td)
    if ismember(td(i).target_direction,target_index_2) == 0
        target_index_2(counter) = td(i).target_direction;
        counter = counter + 1;
    end
    idx = find(target_index_2 == td(i).target_direction);
    target_counter_2(idx) = target_counter_2(idx) + 1;
    targets_2(i) = idx;
end

if strcmp(selection,'M1')
    neurons = size(td(1).M1_spikes,2);
elseif strcmp(selection,'PMd')
    neurons = size(td(1).PMd_spikes,2);
end

N_2 = zeros(neurons,length(td),time);
Y_2 = zeros(2,length(td),time);
for i = 1:length(td)
    for j = 1:time
        if strcmp(selection,'M1')
            N_2(:,i,j) = td(i).M1_spikes(j,1:neurons);
        elseif strcmp(selection,'PMd')
            N_2(:,i,j) = td(i).PMd_spikes(j,1:neurons);
        end
        Y_2(:,i,j) = td(i).vel(j,:);
    end
end

% Target alignment:

[targets_1,targets_2] = ...
    align_targets(targets_1,target_index_1,targets_2,target_index_2);

% Dynamics alignment:

N_1_aux = zeros(size(N_1,1),8*trials_per_target,size(N_1,3));
N_2_aux = zeros(size(N_2,1),8*trials_per_target,size(N_2,3));
Y_1_aux = zeros(size(Y_1,1),8*trials_per_target,size(Y_1,3));
Y_2_aux = zeros(size(Y_2,1),8*trials_per_target,size(Y_2,3));
targets_1_aux = zeros(1,8*trials_per_target);
targets_2_aux = zeros(1,8*trials_per_target);
for i = 1:8*trials_per_target
    target_now = mod(i,8)+1;
    aux_n_1 = N_1(:,find(targets_1 == target_now),:);
    aux_y_1 = Y_1(:,find(targets_1 == target_now),:);
    idx = randperm(size(aux_n_1,2));
    N_1_aux(:,i,:) = aux_n_1(:,idx(1),:);
    Y_1_aux(:,i,:) = aux_y_1(:,idx(1),:);
    targets_1_aux(i) = target_now;
    aux_n_2 = N_2(:,find(targets_2 == target_now),:);
    aux_y_2 = Y_2(:,find(targets_2 == target_now),:);
    idx = randperm(size(aux_n_2,2));
    N_2_aux(:,i,:) = aux_n_2(:,idx(1),:);
    Y_2_aux(:,i,:) = aux_y_2(:,idx(1),:);
    targets_2_aux(i) = target_now;
end
    
idx = randperm(8*trials_per_target);
N_1 = N_1_aux(:,idx,:);
Y_1 = Y_1_aux(:,idx,:);
targets_1 = targets_1_aux(idx);
N_2 = N_2_aux(:,idx,:);
Y_2 = Y_2_aux(:,idx,:);
targets_2 = targets_2_aux(idx);

%% Inputs:

dims = 3;
train_percentage = 0.25;

%t_train = fix(min(size(N_1,2),size(N_2,2))/2);
%t_test = fix(min(size(N_1,2),size(N_2,2))/2);

t_train = fix(min(size(N_1,2),size(N_2,2))*train_percentage);
t_test = fix(min(size(N_1,2),size(N_2,2))*(1-train_percentage));

[X_1_train,~,~,c_1_train] = get_pca(N_1(:,1:t_train,:),dims);
[X_2_train,~,~,c_2_train] = get_pca(N_2(:,1:t_train,:),dims);

% ALIGN:

[~,X_2_aligned_train,transform] = procrustes(transpose(A_to_AA(X_1_train)),transpose(A_to_AA(X_2_train)));
X_2_aligned_train = AA_to_A(transpose(X_2_aligned_train),25);

c = transform.c;
T = transform.T;
b = transform.b;

% TEST:

X_1_test = transpose(A_to_AA(N_1(:,t_train+1:t_train+t_test,:)))*c_1_train;
X_2_test = transpose(A_to_AA(N_2(:,t_train+1:t_train+t_test,:)))*c_2_train;
X_1_test = AA_to_A(transpose(X_1_test(:,1:dims)),25);
X_2_test = AA_to_A(transpose(X_2_test(:,1:dims)),25);

X_2_aligned_test = b*transpose(A_to_AA(X_2_test))*T + transpose(repmat(transpose(c(1,:)),1,size(X_2_test,2)*size(X_2_test,3)));
X_2_aligned_test = AA_to_A(transpose(X_2_aligned_test),25);

disp('#######################');
x = A_to_AA(X_1_train);
y = A_to_AA(X_2_aligned_train);
r = corrcoef(x,y);
disp(['Train aligned: ', num2str(r(1,2))]);
x = A_to_AA(X_1_train);
y = A_to_AA(X_2_train);
r = corrcoef(x,y);
disp(['Train not aligned: ', num2str(r(1,2))]);
x = A_to_AA(X_1_test);
y = A_to_AA(X_2_aligned_test);
r = corrcoef(x,y);
disp(['Test aligned: ', num2str(r(1,2))]);
x = A_to_AA(X_1_test);
y = A_to_AA(X_2_test);
r = corrcoef(x,y);
disp(['Test not aligned: ', num2str(r(1,2))]);
disp('#######################');

%{
figure;
subplot(3,1,1);
plot(squeeze(A_to_AA(X_1_train(1,:,:))),'b','LineWidth',2);
hold on;
plot(squeeze(A_to_AA(X_2_aligned_train(1,:,:))),'r','LineWidth',2);
subplot(3,1,2);
plot(squeeze(A_to_AA(X_1_train(2,:,:))),'b','LineWidth',2);
hold on;
plot(squeeze(A_to_AA(X_2_aligned_train(2,:,:))),'r','LineWidth',2);
subplot(3,1,3);
plot(squeeze(A_to_AA(X_1_train(3,:,:))),'b','LineWidth',2);
hold on;
plot(squeeze(A_to_AA(X_2_aligned_train(3,:,:))),'r','LineWidth',2);

figure;
subplot(3,1,1);
plot(squeeze(A_to_AA(X_1_test(1,:,:))),'b','LineWidth',2);
hold on;
plot(squeeze(A_to_AA(X_2_test(1,:,:))),'r','LineWidth',2);
subplot(3,1,2);
plot(squeeze(A_to_AA(X_1_test(2,:,:))),'b','LineWidth',2);
hold on;
plot(squeeze(A_to_AA(X_2_test(2,:,:))),'r','LineWidth',2);
subplot(3,1,3);
plot(squeeze(A_to_AA(X_1_test(3,:,:))),'b','LineWidth',2);
hold on;
plot(squeeze(A_to_AA(X_2_test(3,:,:))),'r','LineWidth',2);

figure;
subplot(3,1,1);
plot(squeeze(A_to_AA(X_1_test(1,:,:))),'b','LineWidth',2);
hold on;
plot(squeeze(A_to_AA(X_2_aligned_test(1,:,:))),'r','LineWidth',2);
subplot(3,1,2);
plot(squeeze(A_to_AA(X_1_test(2,:,:))),'b','LineWidth',2);
hold on;
plot(squeeze(A_to_AA(X_2_aligned_test(2,:,:))),'r','LineWidth',2);
subplot(3,1,3);
plot(squeeze(A_to_AA(X_1_test(3,:,:))),'b','LineWidth',2);
hold on;
plot(squeeze(A_to_AA(X_2_aligned_test(3,:,:))),'r','LineWidth',2);
%}

%% Alignment visualization:

PSTH_1_train = zeros(dims,8,25);
PSTH_2_train = zeros(dims,8,25);
PSTH_2_aligned_train = zeros(dims,8,25);
for i = 1:8
    counter_psth = 0;
    for j = 1:t_train
        if targets_1(j) == i
            for k = 1:dims
                PSTH_1_train(k,i,:) = PSTH_1_train(k,i,:) + X_1_train(k,j,:);
                PSTH_2_train(k,i,:) = PSTH_2_train(k,i,:) + X_2_train(k,j,:);
                PSTH_2_aligned_train(k,i,:) = PSTH_2_aligned_train(k,i,:) + X_2_aligned_train(k,j,:);
            end
            counter_psth = counter_psth + 1;
        end
    end
    PSTH_1_train(:,i,:) = PSTH_1_train(:,i,:)/counter_psth;
    PSTH_2_train(:,i,:) = PSTH_2_train(:,i,:)/counter_psth;
    PSTH_2_aligned_train(:,i,:) = PSTH_2_aligned_train(:,i,:)/counter_psth;
end

figure;
subplot(1,2,1);
plot3(squeeze(PSTH_1_train(1,1,:)),squeeze(PSTH_1_train(2,1,:)),squeeze(PSTH_1_train(3,1,:)),'Color',[1 0 0],'LineWidth',3);
hold on;
plot3(squeeze(PSTH_1_train(1,2,:)),squeeze(PSTH_1_train(2,2,:)),squeeze(PSTH_1_train(3,2,:)),'Color',[1 0.7 0],'LineWidth',3);
plot3(squeeze(PSTH_1_train(1,3,:)),squeeze(PSTH_1_train(2,3,:)),squeeze(PSTH_1_train(3,3,:)),'Color',[1 1 0],'LineWidth',3);
plot3(squeeze(PSTH_1_train(1,4,:)),squeeze(PSTH_1_train(2,4,:)),squeeze(PSTH_1_train(3,4,:)),'Color',[0.7 1 0],'LineWidth',3);
plot3(squeeze(PSTH_1_train(1,5,:)),squeeze(PSTH_1_train(2,5,:)),squeeze(PSTH_1_train(3,5,:)),'Color',[0 1 0],'LineWidth',3);
plot3(squeeze(PSTH_1_train(1,6,:)),squeeze(PSTH_1_train(2,6,:)),squeeze(PSTH_1_train(3,6,:)),'Color',[0 1 0.7],'LineWidth',3);
plot3(squeeze(PSTH_1_train(1,7,:)),squeeze(PSTH_1_train(2,7,:)),squeeze(PSTH_1_train(3,7,:)),'Color',[0 1 1],'LineWidth',3);
plot3(squeeze(PSTH_1_train(1,8,:)),squeeze(PSTH_1_train(2,8,:)),squeeze(PSTH_1_train(3,8,:)),'Color',[0 0.7 1],'LineWidth',3);
plot3(squeeze(PSTH_2_train(1,1,:)),squeeze(PSTH_2_train(2,1,:)),squeeze(PSTH_2_train(3,1,:)),'Color',[0.8 0 0],'LineWidth',3);
plot3(squeeze(PSTH_2_train(1,2,:)),squeeze(PSTH_2_train(2,2,:)),squeeze(PSTH_2_train(3,2,:)),'Color',[0.8 0.5 0],'LineWidth',3);
plot3(squeeze(PSTH_2_train(1,3,:)),squeeze(PSTH_2_train(2,3,:)),squeeze(PSTH_2_train(3,3,:)),'Color',[0.8 0.8 0],'LineWidth',3);
plot3(squeeze(PSTH_2_train(1,4,:)),squeeze(PSTH_2_train(2,4,:)),squeeze(PSTH_2_train(3,4,:)),'Color',[0.5 0.8 0],'LineWidth',3);
plot3(squeeze(PSTH_2_train(1,5,:)),squeeze(PSTH_2_train(2,5,:)),squeeze(PSTH_2_train(3,5,:)),'Color',[0 0.8 0],'LineWidth',3);
plot3(squeeze(PSTH_2_train(1,6,:)),squeeze(PSTH_2_train(2,6,:)),squeeze(PSTH_2_train(3,6,:)),'Color',[0 0.8 0.5],'LineWidth',3);
plot3(squeeze(PSTH_2_train(1,7,:)),squeeze(PSTH_2_train(2,7,:)),squeeze(PSTH_2_train(3,7,:)),'Color',[0 0.8 0.8],'LineWidth',3);
plot3(squeeze(PSTH_2_train(1,8,:)),squeeze(PSTH_2_train(2,8,:)),squeeze(PSTH_2_train(3,8,:)),'Color',[0 0.5 0.8],'LineWidth',3);
title('Not aligned');
subplot(1,2,2);
plot3(squeeze(PSTH_1_train(1,1,:)),squeeze(PSTH_1_train(2,1,:)),squeeze(PSTH_1_train(3,1,:)),'Color',[0.8 0 0],'LineWidth',3);
hold on;
plot3(squeeze(PSTH_1_train(1,2,:)),squeeze(PSTH_1_train(2,2,:)),squeeze(PSTH_1_train(3,2,:)),'Color',[1 0.7 0],'LineWidth',3);
plot3(squeeze(PSTH_1_train(1,3,:)),squeeze(PSTH_1_train(2,3,:)),squeeze(PSTH_1_train(3,3,:)),'Color',[1 1 0],'LineWidth',3);
plot3(squeeze(PSTH_1_train(1,4,:)),squeeze(PSTH_1_train(2,4,:)),squeeze(PSTH_1_train(3,4,:)),'Color',[0.7 1 0],'LineWidth',3);
plot3(squeeze(PSTH_1_train(1,5,:)),squeeze(PSTH_1_train(2,5,:)),squeeze(PSTH_1_train(3,5,:)),'Color',[0 1 0],'LineWidth',3);
plot3(squeeze(PSTH_1_train(1,6,:)),squeeze(PSTH_1_train(2,6,:)),squeeze(PSTH_1_train(3,6,:)),'Color',[0 1 0.7],'LineWidth',3);
plot3(squeeze(PSTH_1_train(1,7,:)),squeeze(PSTH_1_train(2,7,:)),squeeze(PSTH_1_train(3,7,:)),'Color',[0 1 1],'LineWidth',3);
plot3(squeeze(PSTH_1_train(1,8,:)),squeeze(PSTH_1_train(2,8,:)),squeeze(PSTH_1_train(3,8,:)),'Color',[0 0.7 1],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_train(1,1,:)),squeeze(PSTH_2_aligned_train(2,1,:)),squeeze(PSTH_2_aligned_train(3,1,:)),'Color',[1 0 0],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_train(1,2,:)),squeeze(PSTH_2_aligned_train(2,2,:)),squeeze(PSTH_2_aligned_train(3,2,:)),'Color',[0.8 0.5 0],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_train(1,3,:)),squeeze(PSTH_2_aligned_train(2,3,:)),squeeze(PSTH_2_aligned_train(3,3,:)),'Color',[0.8 0.8 0],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_train(1,4,:)),squeeze(PSTH_2_aligned_train(2,4,:)),squeeze(PSTH_2_aligned_train(3,4,:)),'Color',[0.5 0.8 0],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_train(1,5,:)),squeeze(PSTH_2_aligned_train(2,5,:)),squeeze(PSTH_2_aligned_train(3,5,:)),'Color',[0 0.8 0],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_train(1,6,:)),squeeze(PSTH_2_aligned_train(2,6,:)),squeeze(PSTH_2_aligned_train(3,6,:)),'Color',[0 0.8 0.5],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_train(1,7,:)),squeeze(PSTH_2_aligned_train(2,7,:)),squeeze(PSTH_2_aligned_train(3,7,:)),'Color',[0 0.8 0.8],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_train(1,8,:)),squeeze(PSTH_2_aligned_train(2,8,:)),squeeze(PSTH_2_aligned_train(3,8,:)),'Color',[0 0.5 0.8],'LineWidth',3);
title('Aligned');
suptitle('Train data');

PSTH_1_test = zeros(dims,8,25);
PSTH_2_test = zeros(dims,8,25);
PSTH_2_aligned_test = zeros(dims,8,25);
for i = 1:8
    counter_psth = 0;
    for j = 1:t_test
        if targets_1(t_train+j) == i
            for k = 1:dims
                PSTH_1_test(k,i,:) = PSTH_1_test(k,i,:) + X_1_test(k,j,:);
                PSTH_2_test(k,i,:) = PSTH_2_test(k,i,:) + X_2_test(k,j,:);
                PSTH_2_aligned_test(k,i,:) = PSTH_2_aligned_test(k,i,:) + X_2_aligned_test(k,j,:);
            end
            counter_psth = counter_psth + 1;
        end
    end
    PSTH_1_test(:,i,:) = PSTH_1_test(:,i,:)/counter_psth;
    PSTH_2_test(:,i,:) = PSTH_2_test(:,i,:)/counter_psth;
    PSTH_2_aligned_test(:,i,:) = PSTH_2_aligned_test(:,i,:)/counter_psth;
end

figure;
subplot(1,2,1);
plot3(squeeze(PSTH_1_test(1,1,:)),squeeze(PSTH_1_test(2,1,:)),squeeze(PSTH_1_test(3,1,:)),'Color',[1 0 0],'LineWidth',3);
hold on;
plot3(squeeze(PSTH_1_test(1,2,:)),squeeze(PSTH_1_test(2,2,:)),squeeze(PSTH_1_test(3,2,:)),'Color',[1 0.7 0],'LineWidth',3);
plot3(squeeze(PSTH_1_test(1,3,:)),squeeze(PSTH_1_test(2,3,:)),squeeze(PSTH_1_test(3,3,:)),'Color',[1 1 0],'LineWidth',3);
plot3(squeeze(PSTH_1_test(1,4,:)),squeeze(PSTH_1_test(2,4,:)),squeeze(PSTH_1_test(3,4,:)),'Color',[0.7 1 0],'LineWidth',3);
plot3(squeeze(PSTH_1_test(1,5,:)),squeeze(PSTH_1_test(2,5,:)),squeeze(PSTH_1_test(3,5,:)),'Color',[0 1 0],'LineWidth',3);
plot3(squeeze(PSTH_1_test(1,6,:)),squeeze(PSTH_1_test(2,6,:)),squeeze(PSTH_1_test(3,6,:)),'Color',[0 1 0.7],'LineWidth',3);
plot3(squeeze(PSTH_1_test(1,7,:)),squeeze(PSTH_1_test(2,7,:)),squeeze(PSTH_1_test(3,7,:)),'Color',[0 1 1],'LineWidth',3);
plot3(squeeze(PSTH_1_test(1,8,:)),squeeze(PSTH_1_test(2,8,:)),squeeze(PSTH_1_test(3,8,:)),'Color',[0 0.7 1],'LineWidth',3);
plot3(squeeze(PSTH_2_test(1,1,:)),squeeze(PSTH_2_test(2,1,:)),squeeze(PSTH_2_test(3,1,:)),'Color',[0.9 0 0],'LineWidth',3);
plot3(squeeze(PSTH_2_test(1,2,:)),squeeze(PSTH_2_test(2,2,:)),squeeze(PSTH_2_test(3,2,:)),'Color',[0.9 0.6 0],'LineWidth',3);
plot3(squeeze(PSTH_2_test(1,3,:)),squeeze(PSTH_2_test(2,3,:)),squeeze(PSTH_2_test(3,3,:)),'Color',[0.9 0.9 0],'LineWidth',3);
plot3(squeeze(PSTH_2_test(1,4,:)),squeeze(PSTH_2_test(2,4,:)),squeeze(PSTH_2_test(3,4,:)),'Color',[0.6 0.9 0],'LineWidth',3);
plot3(squeeze(PSTH_2_test(1,5,:)),squeeze(PSTH_2_test(2,5,:)),squeeze(PSTH_2_test(3,5,:)),'Color',[0 0.9 0],'LineWidth',3);
plot3(squeeze(PSTH_2_test(1,6,:)),squeeze(PSTH_2_test(2,6,:)),squeeze(PSTH_2_test(3,6,:)),'Color',[0 0.9 0.6],'LineWidth',3);
plot3(squeeze(PSTH_2_test(1,7,:)),squeeze(PSTH_2_test(2,7,:)),squeeze(PSTH_2_test(3,7,:)),'Color',[0 0.9 0.9],'LineWidth',3);
plot3(squeeze(PSTH_2_test(1,8,:)),squeeze(PSTH_2_test(2,8,:)),squeeze(PSTH_2_test(3,8,:)),'Color',[0 0.6 0.9],'LineWidth',3);
title('Not aligned');
subplot(1,2,2);
plot3(squeeze(PSTH_1_test(1,1,:)),squeeze(PSTH_1_test(2,1,:)),squeeze(PSTH_1_test(3,1,:)),'Color',[1 0 0],'LineWidth',3);
hold on;
plot3(squeeze(PSTH_1_test(1,2,:)),squeeze(PSTH_1_test(2,2,:)),squeeze(PSTH_1_test(3,2,:)),'Color',[1 0.7 0],'LineWidth',3);
plot3(squeeze(PSTH_1_test(1,3,:)),squeeze(PSTH_1_test(2,3,:)),squeeze(PSTH_1_test(3,3,:)),'Color',[1 1 0],'LineWidth',3);
plot3(squeeze(PSTH_1_test(1,4,:)),squeeze(PSTH_1_test(2,4,:)),squeeze(PSTH_1_test(3,4,:)),'Color',[0.7 1 0],'LineWidth',3);
plot3(squeeze(PSTH_1_test(1,5,:)),squeeze(PSTH_1_test(2,5,:)),squeeze(PSTH_1_test(3,5,:)),'Color',[0 1 0],'LineWidth',3);
plot3(squeeze(PSTH_1_test(1,6,:)),squeeze(PSTH_1_test(2,6,:)),squeeze(PSTH_1_test(3,6,:)),'Color',[0 1 0.7],'LineWidth',3);
plot3(squeeze(PSTH_1_test(1,7,:)),squeeze(PSTH_1_test(2,7,:)),squeeze(PSTH_1_test(3,7,:)),'Color',[0 1 1],'LineWidth',3);
plot3(squeeze(PSTH_1_test(1,8,:)),squeeze(PSTH_1_test(2,8,:)),squeeze(PSTH_1_test(3,8,:)),'Color',[0 0.7 1],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_test(1,1,:)),squeeze(PSTH_2_aligned_test(2,1,:)),squeeze(PSTH_2_aligned_test(3,1,:)),'Color',[0.9 0 0],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_test(1,2,:)),squeeze(PSTH_2_aligned_test(2,2,:)),squeeze(PSTH_2_aligned_test(3,2,:)),'Color',[0.9 0.6 0],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_test(1,3,:)),squeeze(PSTH_2_aligned_test(2,3,:)),squeeze(PSTH_2_aligned_test(3,3,:)),'Color',[0.9 0.9 0],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_test(1,4,:)),squeeze(PSTH_2_aligned_test(2,4,:)),squeeze(PSTH_2_aligned_test(3,4,:)),'Color',[0.6 0.9 0],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_test(1,5,:)),squeeze(PSTH_2_aligned_test(2,5,:)),squeeze(PSTH_2_aligned_test(3,5,:)),'Color',[0 0.9 0],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_test(1,6,:)),squeeze(PSTH_2_aligned_test(2,6,:)),squeeze(PSTH_2_aligned_test(3,6,:)),'Color',[0 0.9 0.6],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_test(1,7,:)),squeeze(PSTH_2_aligned_test(2,7,:)),squeeze(PSTH_2_aligned_test(3,7,:)),'Color',[0 0.9 0.9],'LineWidth',3);
plot3(squeeze(PSTH_2_aligned_test(1,8,:)),squeeze(PSTH_2_aligned_test(2,8,:)),squeeze(PSTH_2_aligned_test(3,8,:)),'Color',[0 0.6 0.9],'LineWidth',3);
title('Aligned');
suptitle('Test data');

%% Train:

data_1.X = X_1_train;
data_1.Y = Y_1(:,1:t_train,:);
data_1.targets = targets_1(:,1:t_train,:);

clear param;
percentage_train = 1;
param.m_train_per_target = 4;
param.mini_batch_size = 8;
param.num_epochs = 10000;
param.stop_condition = 10; 
param.n_hidden = 30;
param.beta_1 = 0.9;
param.beta_2 = 0.999;
param.epsilon = 1e-8;
param.learning_rate = 0.005;
param.optimization = 'adam';
param.lambda = 0; 
param.connectivity = 1;
param.mode = 1; 
param.network_model = 'ER';
param.links = 1;
param.correlation_reg = 0;

if param.mode == 1
    [net_1,cost_train_1,subsets_1] = run_my_LSTM_alignment(data_1,param);
    disp('###########');
    disp(['Cost train 1:' ' ' num2str(cost_train_1)]);
    disp('###########');
elseif param.mode == 2
    data_1.x_train = subsets_1.train.X;
    data_1.y_train = subsets_1.train.Y;
    [net_1,cost_train_1,~] = run_my_LSTM_alignment(data_1,param); 
    disp('###########');
    disp(['Cost train 1:' ' ' num2str(cost_train_1)]);
    disp('###########');
end

%% Velocity prediction:

trials_test = size(X_2_aligned_test,2);
Y_pred = zeros(2,trials_test,25);
idx_trials = randperm(trials_test);
for i = 1:trials_test
    [Y_pred_t,~] = LSTM_predict_alignment(X_2_aligned_test(:,idx_trials(i),:),net_1);
    Y_pred(:,i,:) = Y_pred_t;
end

aux = Y_2(:,t_train+1:t_train+t_test,:);
Y_test = aux(:,idx_trials,:);

Y_test = A_to_AA(Y_test);
Y_pred = A_to_AA(Y_pred);

vaf_x = compute_vaf(transpose(Y_test(1,:)), ...
    transpose(A_to_AA(Y_pred(1,:))));
vaf_y = compute_vaf(transpose(Y_test(2,:)), ...
    transpose(A_to_AA(Y_pred(2,:))));

figure;
ax(1) = subplot(2,1,1); hold all;
plot(Y_test(1,:),'Color',[1 0.4 0],'LineWidth',1.5);
plot(Y_pred(1,:),'Color',[0 0.4 1],'LineWidth',1.5);
title(['X (VAF = ' num2str(vaf_x,3) ')']);
xlabel('time');
ylabel('velocity');
axis tight;
ax(2) = subplot(2,1,2); hold all;
plot(Y_test(2,:),'Color',[1 0.4 0],'LineWidth',1.5);
plot(Y_pred(2,:),'Color',[0 0.4 1],'LineWidth',1.5);
title(['Y (VAF = ' num2str(vaf_y,3) ')']);
xlabel('time');
ylabel('velocity');
axis tight;
h = legend({'Actual','Predicted'},'Location','southoutside');
set(h,'Box','off');
linkaxes(ax,'x');
suptitle('LSTM Velocity Prediction (Test): OTHER ALIGNED');

trials_test = size(X_2_test,2);
Y_pred = zeros(2,trials_test,25);
idx_trials = randperm(trials_test);
for i = 1:trials_test
    [Y_pred_t,~] = LSTM_predict_alignment(X_2_test(:,idx_trials(i),:),net_1);
    Y_pred(:,i,:) = Y_pred_t;
end

aux = Y_2(:,t_train+1:t_train+t_test,:);
Y_test = aux(:,idx_trials,:);

Y_test = A_to_AA(Y_test);
Y_pred = A_to_AA(Y_pred);

vaf_x = compute_vaf(transpose(Y_test(1,:)), ...
    transpose(A_to_AA(Y_pred(1,:))));
vaf_y = compute_vaf(transpose(Y_test(2,:)), ...
    transpose(A_to_AA(Y_pred(2,:))));

figure;
ax(1) = subplot(2,1,1); hold all;
plot(Y_test(1,:),'Color',[1 0.4 0],'LineWidth',1.5);
plot(Y_pred(1,:),'Color',[0 0.4 1],'LineWidth',1.5);
title(['X (VAF = ' num2str(vaf_x,3) ')']);
xlabel('time');
ylabel('velocity');
axis tight;
ax(2) = subplot(2,1,2); hold all;
plot(Y_test(2,:),'Color',[1 0.4 0],'LineWidth',1.5);
plot(Y_pred(2,:),'Color',[0 0.4 1],'LineWidth',1.5);
title(['Y (VAF = ' num2str(vaf_y,3) ')']);
xlabel('time');
ylabel('velocity');
axis tight;
h = legend({'Actual','Predicted'},'Location','southoutside');
set(h,'Box','off');
linkaxes(ax,'x');
suptitle('LSTM Velocity Prediction (Test): OTHER NOT ALIGNED');

trials_test = size(X_1_test,2);
Y_pred = zeros(2,trials_test,25);
idx_trials = randperm(trials_test);
for i = 1:trials_test
    [Y_pred_t,~] = LSTM_predict_alignment(X_1_test(:,idx_trials(i),:),net_1);
    Y_pred(:,i,:) = Y_pred_t;
end

aux = Y_1(:,t_train+1:t_train+t_test,:);
Y_test = aux(:,idx_trials,:);

Y_test = A_to_AA(Y_test);
Y_pred = A_to_AA(Y_pred);

vaf_x = compute_vaf(transpose(Y_test(1,:)), ...
    transpose(A_to_AA(Y_pred(1,:))));
vaf_y = compute_vaf(transpose(Y_test(2,:)), ...
    transpose(A_to_AA(Y_pred(2,:))));

figure;
ax(1) = subplot(2,1,1); hold all;
plot(Y_test(1,:),'Color',[1 0.4 0],'LineWidth',1.5);
plot(Y_pred(1,:),'Color',[0 0.4 1],'LineWidth',1.5);
title(['X (VAF = ' num2str(vaf_x,3) ')']);
xlabel('time');
ylabel('velocity');
axis tight;
ax(2) = subplot(2,1,2); hold all;
plot(Y_test(2,:),'Color',[1 0.4 0],'LineWidth',1.5);
plot(Y_pred(2,:),'Color',[0 0.4 1],'LineWidth',1.5);
title(['Y (VAF = ' num2str(vaf_y,3) ')']);
xlabel('time');
ylabel('velocity');
axis tight;
h = legend({'Actual','Predicted'},'Location','southoutside');
set(h,'Box','off');
linkaxes(ax,'x');
suptitle('LSTM Velocity Prediction (Test): SAME');
