clear all
close all
clc


Ts = 0.1;
simtime = 100;
param.nx = 3;
num_exp = 4;

for exp = 1:num_exp+1
    
    if exp == num_exp+1
        simtime = 600;
    end
    a = 1;
    b = 4;
    initial_state = (b-a).*rand(param.nx,1) + a;

    out = sim('sim_3tank.slx',simtime);
    XX = out.x';
    input_signal =  reshape(out.u,size(XX,2),1)';
    tspan = size(XX,2);
    if exp <= num_exp
            dExp{1,exp}= [zeros(1,(tspan));
                input_signal(1,:);
                zeros(2,(tspan))];
            yExp{1,exp} = XX;
    else
        dExp_val{1,1}= [zeros(1,(tspan));
                input_signal(1,:);
                zeros(2,(tspan))];
        yExp_val{1,1} = XX;
    end

end 

save('dataset_sysID_3tanks_final','Ts','dExp','dExp_val','yExp','yExp_val')


figure;plot(XX(1,:));title('h1')
figure;plot(XX(2,:));title('h2')
figure;plot(XX(3,:));title('h3')
figure;plot(input_signal(1,:));title('input')