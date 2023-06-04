% Builds jacobian tensor for splitted network.

clear all
addpath(genpath('./'));

dset = {};
model = {};
classchans = {};

%% Case-insensitive
dset{end+1} = 'eccv2014_textspotting/data/case-insensitive-train.mat';
model{end+1} = 'eccv2014_textspotting/models/cov_first_part.mat';
model{end+1} = 'eccv2014_textspotting/models/cov_second_part.mat';
classchans{end+1} = 2:37;  % ignore background class

%% Load 10 data items from each class

% load data
s = load(dset{1});
images = s.gt.images;

nb_per_class = 10;
nb_of_classes = size(images,2);
random_samples = zeros(24,24,nb_per_class * nb_of_classes);
for i=1:nb_of_classes
    cc = images{i}; % Current Class
    
    msize = numel(cc);
    idx = randperm(msize);
    for j=1:nb_per_class
        random_samples(:,:,(i-1) * nb_per_class + j) = cc{idx(1,j)};
    end
end

%% Send random_samples through first part of the network

nn = cudaconvnet_to_mconvnet(model{1});

input_size = 8 * 8 * 64;

inputs = zeros(input_size,size(random_samples,3));
for i=1:size(random_samples, 3)
    ims = [];
    ims = cat(4, ims, random_samples(:,:,i));
    ims = single(ims);

    % preprocess
    data = reshape(ims, [], size(ims,4));
    mu = mean(data, 1);
    data = data - repmat(mu, size(data,1), 1);
    v = std(data, 0, 1);
    data = data ./ (0.0000001 + repmat(v, size(data,1), 1));
    ims = reshape(data, size(ims));
    
    clear data;

     nn = nn.forward(nn, struct('data', single(ims)));
     
     inputs(:,i) = reshape(squeeze(nn.Xout(:,:,:,:)), input_size, 1);
end

size(inputs)

%% Compute Jacobian tensor and zeroth-order information matrix

tic
[F, ~] = compute_Jac_and_F(inputs);
toc

%% Functions

function [F, Jac] = compute_Jac_and_F(inputs)
    [nb_inputs, nb_samples] = size(inputs);
    nb_outputs = 36;

    F = zeros(nb_outputs, nb_samples);
    Jac = zeros(nb_outputs, nb_inputs, nb_samples);
    for i=1:nb_samples
        [F(:,i), Jac(:,:,i)] = dlfeval(@network, dlarray(inputs(:,i)));
    end
end

function [f, grad] = network(input)
    load('eccv2014_textspotting/models/cov_second_part.mat');
    
    % Parameters of first hidden layer
    W8 = get_weights(layer8);
    b8 = layer8.biases;
    
    W9 = get_weights(layer9);
    b9 = layer9.biases;
    
    W10 = get_weights(layer10);
    b10 = layer10.biases;
    
    W11 = get_weights(layer11);
    b11 = layer11.biases;
    
    % Parameters of second hidden layer
    W13 = get_weights(layer13);
    b13 = layer13.biases;
    
    W14 = get_weights(layer14);
    b14 = layer14.biases;
    
    W15 = get_weights(layer15);
    b15 = layer15.biases;
    
    W16 = get_weights(layer16);
    b16 = layer16.biases;

    d = input;
    
    % Hidden layer 1
    vecs = cat(3, ...
        single(W8.' * d + b8), ...
        single(W9.' * d + b9), ...
        single(W10.' * d + b10), ...
        single(W11.' * d + b11));
    
    % Maxout
    x1 = max(vecs, [], 3);
    
    % Hidden layer 2
    vecs = cat(3, ...
        single(W13.' * x1 + b13), ...
        single(W14.' * x1 + b14), ...
        single(W15.' * x1 + b15), ...
        single(W16.' * x1 + b16));
    
    % Final maxout
    f = max(vecs(2:37,:,:), [], 3);

    outputs = 36;
    grad = zeros(outputs, size(input,1));

    for i = 1:outputs
        grad(i,:) = dlgradient(f(i), d);
    end
end

function weights = get_weights(layer)
    % weights is rows x cols x chans x filters

    weights = layer.weights;

    orig1 = size(weights,2);
    orig2 = size(weights,3);

    weights = reshape(weights, [size(weights,2) size(weights,3)]);
    sz = layer.filterSize;
    chans = layer.filterChannels;
    weights = reshape(weights, [sz sz chans size(weights,2)]);
    weights = permute(weights, [2 1 3 4]);
    
    % reshape back into normal form
    weights = reshape(weights, [orig1 orig2]);

end

