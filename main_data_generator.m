clear all
close all
clc


% model parameters
param.nx = 3;
param.nu = 1;

% Areas (cm²)
param.A1 = 32.5;
param.A2 = 29.3;
param.A3 = 27.8;

% Outlet orifice areas (cm²)
param.a1 = 10.9;   
param.a2 = 4.87;    
param.a3 = 9.5;  

% Interconnection coefficients
param.k1 = 0.2;
param.k2 = 0.28;
param.k3 = 1.3;
param.ku = 34.2;   

param.g = 981;

% Noise power
param.noisePower = 0.001;


% simulations parameters
param.Ts = 0.1;
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
    initial_state = ((maxinitialstate-mininitialstate).*rand(param.nx,1) + mininitialstate)/10;
    % simulation simulink
    out = sim('tank_simulator.slx',simtime);
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
Ts = param.Ts;
save('dataset_sysID_3tanks','Ts','dExp','dExp_val','yExp','yExp_val')

time = linspace(0,length(XX(1,:))*Ts,length(XX(1,:)));

figure;plot(time,XX(1,:));title('h1')
figure;plot(time,XX(2,:));title('h2')
figure;plot(time,XX(3,:));title('h3')
figure;plot(input_signal(1,:));title('input')

