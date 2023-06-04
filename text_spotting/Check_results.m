% Check results of computed compression

%clear all
addpath(genpath('./'));

dset = {};
model = {};
classchans = {};
%% 

%We = Wres;
%Vte = Vtres;
%Zte = Ztres;
%cD1e = cD1res;
%cD2e = cD2res;
%"done"
%% Load test set and first part of network
dset{end+1} = 'data/icdar2003-chars-test.mat';
model{end+1} = 'models/cov_first_part.mat';
fprintf("Data and model loaded")
%% Run compressed network

%r1 = 40;
%r2 = 40;
for i=1:numel(dset)
    fprintf('Testing %s ...\n', model{i});
    % load model
    nn = cudaconvnet_to_mconvnet(model{i});
    % load data
    s = load(dset{i});
    ims = [];
    labels = [];
    for j=1:numel(s.gt.labels)
        ims = cat(4, ims, s.gt.images{j}{:});
        labels = cat(2, labels, (j)*ones(1,numel(s.gt.images{j})));
    end
    size(ims)
    ims = single(ims);
    labels = single(labels);
    % preprocess
    data = reshape(ims, [], size(ims,4));
    mu = mean(data, 1);
    data = data - repmat(mu, size(data,1), 1);
    v = std(data, 0, 1);
    data = data ./ (0.0000001 + repmat(v, size(data,1), 1));
    ims = reshape(data, size(ims));
    clear data;
    
    nn = nn.forward(nn, struct('data', single(ims)));

    new_data = nn.Xout(:,:,:,:);
    
    size(new_data)
    
    outputs = zeros(36, size(new_data, 4));
    for j=1:size(new_data, 4)
        d = single(tens2vec(new_data(:,:,:,j), [1,2,3]));
        outputs(:,j) = flexible4(d, We, Vte, Zte, cD1e, cD2e, r1, r2);
    end

    %% go
    [~,pred] = max(outputs, [], 1);

    err = sum(labels == pred) / numel(pred);
    fprintf('\taccuracy: %.2f percent\n', err*100);
end

%% Functions

function [f] = flexible6(input, W, Vt, Zt, cD1, cD2, r1, r2)
    P1 = zeros(r1,1);
    x = Zt * input;
    for i=1:r1
        P1(i,1) = cD1((6* (i - 1)) + 1) * x(i) + ...
                 cD1((6* (i - 1)) + 2) * x(i)^2 + ...
                 cD1((6* (i - 1)) + 3) * x(i)^3 + ...
                 cD1((6* (i - 1)) + 4) * x(i)^4 + ...
                 cD1((6* (i - 1)) + 5) * x(i)^5 + ...
                 cD1((6* (i - 1)) + 6) * x(i)^6;
    end

    P2 = zeros(r2,1);
    x = Vt * P1;
    for i=1:r2
        P2(i,1) = cD2((7* (i - 1)) + 1) + ...
                 cD2((7* (i - 1)) + 2) * x(i) + ...
                 cD2((7* (i - 1)) + 3) * x(i)^2 + ...
                 cD2((7* (i - 1)) + 4) * x(i)^3 + ...
                 cD2((7* (i - 1)) + 5) * x(i)^4 + ...
                 cD2((7* (i - 1)) + 6) * x(i)^5 + ...
                 cD2((7* (i - 1)) + 7) * x(i)^6;
    end

    f = W * P2;
end

function [f] = flexible5(input, W, Vt, Zt, cD1, cD2, r1, r2)
    P1 = zeros(r1,1);
    x = Zt * input;
    for i=1:r1
        P1(i,1) = cD1((5* (i - 1)) + 1) * x(i) + ...
                 cD1((5* (i - 1)) + 2) * x(i)^2 + ...
                 cD1((5* (i - 1)) + 3) * x(i)^3 + ...
                 cD1((5* (i - 1)) + 4) * x(i)^4 + ...
                 cD1((5* (i - 1)) + 5) * x(i)^5;
    end

    P2 = zeros(r2,1);
    x = Vt * P1;
    for i=1:r2
        P2(i,1) = cD2((6* (i - 1)) + 1) + ...
                 cD2((6* (i - 1)) + 2) * x(i) + ...
                 cD2((6* (i - 1)) + 3) * x(i)^2 + ...
                 cD2((6* (i - 1)) + 4) * x(i)^3 + ...
                 cD2((6* (i - 1)) + 5) * x(i)^4 + ...
                 cD2((6* (i - 1)) + 6) * x(i)^5;
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

function [f] = flexible3(input, W, Vt, Zt, cD1, cD2, r1, r2)
    P1 = zeros(r1,1);
    x = Zt * input;
    for i=1:r1
        P1(i,1) = cD1((3* (i - 1)) + 1) * x(i) + ...
                 cD1((3* (i - 1)) + 2) * x(i)^2 + ...
                 cD1((3* (i - 1)) + 3) * x(i)^3;
    end

    P2 = zeros(r2,1);
    x = Vt * P1;
    for i=1:r2
        P2(i,1) = cD2((4* (i - 1)) + 1) + ...
                 cD2((4* (i - 1)) + 2) * x(i) + ...
                 cD2((4* (i - 1)) + 3) * x(i)^2 + ...
                 cD2((4* (i - 1)) + 4) * x(i)^3;
    end

    f = W * P2;
end


function [f] = flexible2(input, W, Vt, Zt, cD1, cD2, r1, r2)
    P1 = zeros(r1,1);
    x = Zt * input;
    for i=1:r1
        P1(i,1) = cD1((2* (i - 1)) + 1) * x(i) + ...
                 cD1((2* (i - 1)) + 2) * x(i)^2;
    end

    P2 = zeros(r2,1);
    x = Vt * P1;
    for i=1:r2
        P2(i,1) = cD2((3* (i - 1)) + 1) + ...
                 cD2((3* (i - 1)) + 2) * x(i) + ...
                 cD2((3* (i - 1)) + 3) * x(i)^2;
    end

    f = W * P2;
end