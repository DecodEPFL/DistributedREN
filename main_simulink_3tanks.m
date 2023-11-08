clear all
close all
clc


% model parameters
param.nx = 3;
param.nu = 1;
param.A1 = 38;
param.A2 = 32;
param.A3 = 21;
param.a1 = 0.05;
param.a2 = 0.03;
param.a3 = 0.06;
param.g = 981;
param.ku = 50;
param.k1 = 0.32;
param.k2 = 0.23;
param.k3 = 0.52;
param.noisePower = 0.001;

% simulations parameters
Ts = 0.1;
simtime = 200;
num_exp = 6;

% simulations loop

for exp = 1:num_exp+1
    
    % make validation dataset longer than experiments
    if exp == num_exp+1
        simtime = 600;
    end
    % initial condition
    mininitialstate = 1;
    maxinitialstate = 4;
    initial_state = (maxinitialstate-mininitialstate).*rand(param.nx,1) + mininitialstate;
    % simulation simulink
    out = sim('sim_3tank.slx',simtime);
    XX = out.x';
    input_signal =  reshape(out.u,size(XX,2),1)';
    tspan = size(XX,2);
    if exp <= num_exp
            % collection input for training dataset
            dExp{1,exp}= [zeros(1,(tspan));
                input_signal(1,:);
                zeros(2,(tspan))];
            % collection state for training dataset
            yExp{1,exp} = XX;
    else
        % collection input for validation dataset
        dExp_val{1,1}= [zeros(1,(tspan));
                input_signal(1,:);
                zeros(2,(tspan))];
        % collection state for validation dataset
        yExp_val{1,1} = XX;
    end

end 

% save the data
save('dataset_sysID_3tanks','Ts','dExp','dExp_val','yExp','yExp_val')


figure;plot(XX(1,:));title('h1')
figure;plot(XX(2,:));title('h2')
figure;plot(XX(3,:));title('h3')
figure;plot(input_signal(1,:));title('input')