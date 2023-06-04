warning("off")

bf1 = {@theta1, @theta2, @theta3, @theta4};%, @theta5};%, @theta6, @theta7, @theta8, @theta9, @theta10};
bf1d = {@theta1d, @theta2d, @theta3d, @theta4d};%, @theta5d};
bf2 = {@theta1, @theta2, @theta3, @theta4};%, @theta5};%, @theta6, @theta7, @theta8, @theta9, @theta10};
bf2d = {@theta1d, @theta2d, @theta3d, @theta4d};%, @theta5d};%, @theta6, @theta7, @theta8, @theta9, @theta10};

r1 = 100;
r2 = 40;

tic
[We, D2e, Vte, D1e, Zte, cD1e, cD2e] = PARATUCK2_CTD(Jac, bf1, bf1d, bf2d, r1, r2, inputs);
toc

%% Bias correction

nb_per_class = 5;
nb_of_classes = 36;

nb_outputs = size(We,1);
nb_samples = nb_per_class * nb_of_classes;
biasVec = zeros(nb_samples * nb_outputs,1);
Wvec = zeros(nb_samples * nb_outputs, r2);
for i=1:nb_samples
    biasVec((i-1) * nb_outputs + 1: i * nb_outputs) = ...
        F(:,i) - fbar(inputs(:,i), We, Vte, Zte, cD1e, cD2e, r1, r2);

    Wvec((i-1) * size(We,1) + 1: i * size(We,1), :) = We;
end

cBias = Wvec \ biasVec;

cD2e_2 = zeros(size(cD2e,1) + r2,1);
d = size(bf2d,2) + 1;
biasIdx = 1;
for i=1:r2
    cD2e_2((i-1) * d + 1) = cBias(biasIdx);
    biasIdx = biasIdx + 1;

    cD2e_2((i-1)* d + 2: i * d) = cD2e((i-1) * (d-1) + 1: i * (d-1));
end

%size(cD2e_2)
cD2e = cD2e_2;
%% Save parameters

save("Parameters_results/CTD_BIAS_SEED/Third_row_of_seeds/CTD_BIAS_100_40", "We", "D2e", "Vte", "D1e", "Zte", "cD1e", "cD2e")
%% Error on F

Fapprox = zeros(size(F,1), size(F,2));

model = {};
model{end+1} = 'models/cov_first_part.mat';
nn = cudaconvnet_to_mconvnet(model{1});

ims = [];
for j=1:nb_samples
    ims = cat(4, ims, random_samples(:,:,j));
end
ims = single(ims);
% preprocess
data = reshape(ims, [], size(ims,4));
mu = mean(data, 1);
data = data - repmat(mu, size(data,1), 1);
v = std(data, 0, 1);
data = data ./ (0.0000001 + repmat(v, size(data,1), 1));
ims = reshape(data, size(ims));
clear data;

size(ims)
nn = nn.forward(nn, struct('data', single(ims)));

new_input = nn.Xout(:,:,:,:);

size(new_input)

for i=1:size(F,2)
    d = single(tens2vec(new_input(:,:,:,i), [1,2,3]));
    Fapprox(:,i) = flexible4(d, We, Vte, Zte, cD1e, cD2e, r1, r2);
end

Ferror = frob(F - Fapprox)^2 / frob(F)^2

%% Functions

function [f] = theta1(x)
    f = x;
end

function [f] = theta1d(x)
    f = 1;
end

function [f] = theta2(x)
    f = x^2;
end

function [f] = theta2d(x)
    f = 2*x;
end

function [f] = theta3(x)
    f = x^3;
end

function [f] = theta3d(x)
    f = 3*x^2;
end

function [f] = theta4(x)
    f = x^4;
end

function [f] = theta4d(x)
    f = 4*x^3;
end

function [f] = theta5(x)
    f = x^5;
end

function [f] = theta5d(x)
    f = 5*x^4;
end

function [f] = fbar(input, W, Vt, Zt, cD1, cD2, r1, r2)
    P1 = zeros(r1,1);
    x = Zt * input;
    for i=1:r1
        P1(i,1) = cD1((4* (i - 1)) + 1) * x(i) + ...
                 cD1((4* (i - 1)) + 2) * x(i)^2 + ...
                 cD1((4* (i - 1)) + 3) * x(i)^3 + ...
                 cD1((4* (i - 1)) + 4) * x(i)^4; % + ...
%                  cD1((10* (i - 1)) + 5) * x(i)^5 + ...
%                  cD1((10* (i - 1)) + 6) * x(i)^6 + ...
%                  cD1((10* (i - 1)) + 7) * x(i)^7 + ...
%                  cD1((10* (i - 1)) + 8) * x(i)^8 + ...
%                  cD1((10* (i - 1)) + 9) * x(i)^9 + ...
%                  cD1((10* (i - 1)) + 10) * x(i)^10;
    end

    P2 = zeros(r2,1);
    x = Vt * P1;
    for i=1:r2
        P2(i,1) = cD2((4* (i - 1)) + 1) * x(i) + ...
                 cD2((4* (i - 1)) + 2) * x(i)^2 + ...
                 cD2((4* (i - 1)) + 3) * x(i)^3 + ...
                 cD2((4* (i - 1)) + 4) * x(i)^4; %+ ...
%                  cD2((10* (i - 1)) + 5) * x(i)^5 + ...
%                  cD2((10* (i - 1)) + 6) * x(i)^6 + ...
%                  cD2((10* (i - 1)) + 7) * x(i)^7 + ...
%                  cD2((10* (i - 1)) + 8) * x(i)^8 + ...
%                  cD2((10* (i - 1)) + 9) * x(i)^9 + ...
%                  cD2((10* (i - 1)) + 10) * x(i)^10;
    end

    f = W * P2;
end

function [f] = flexible4(input, W, Vt, Zt, cD1, cD2, r1, r2)
    P1 = zeros(r1,1);
    x = Zt * input;
    for i=1:r1
        P1(i,1) = cD1((4* (i - 1)) + 1) * x(i) + ...
                 cD1((4* (i - 1)) + 2) * x(i)^2 + ...
                 cD1((4* (i - 1)) + 3) * x(i)^3 + ...
                 cD1((4* (i - 1)) + 4) * x(i)^4; % + ...
                 %cD1((5* (i - 1)) + 5) * x(i)^5; %+ ...
                 %cD1((7* (i - 1)) + 6) * x(i)^6 + ...
                 %cD1((7* (i - 1)) + 7) * x(i)^7;
    end

    P2 = zeros(r2,1);
    x = Vt * P1;
    for i=1:r2
        P2(i,1) = cD2((5* (i - 1)) + 1) + ...
                 cD2((5* (i - 1)) + 2) * x(i) + ...
                 cD2((5* (i - 1)) + 3) * x(i)^2 + ...
                 cD2((5* (i - 1)) + 4) * x(i)^3 + ...
                 cD2((5* (i - 1)) + 5) * x(i)^4;% + ...
                 %cD2((6* (i - 1)) + 6) * x(i)^5; % + ...
                 %cD2((8* (i - 1)) + 7) * x(i)^6 + ...
                 %cD2((8* (i - 1)) + 8) * x(i)^7;
    end

    f = W * P2;
end