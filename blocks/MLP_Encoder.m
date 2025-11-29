function [studentTaps, dlnet] = MLP_Encoder(teacherTaps, stageIdx, latent_dim, resetFlag, lr)
%#codegen
persistent dlnet_prev stageIdx_prev

% ========================
% 1. Initialize / Expand
% ========================
if isempty(dlnet_prev) || resetFlag || stageIdx ~= stageIdx_prev
    fprintf('Initializing/Expanding MLP for Stage %d...\n', stageIdx);

    if isempty(dlnet_prev)
        % ---- Stage 1: create base network ----
        layers = [
            featureInputLayer(length(teacherTaps), 'Name', 'input')
            fullyConnectedLayer(64, 'Name', 'fc1')
            reluLayer('Name', 'relu1')
            fullyConnectedLayer(latent_dim, 'Name', 'latent_stage1')
        ];
        lgraph = layerGraph(layers);
        dlnet_prev = dlnetwork(lgraph);

    else
        % ---- Later stage: transform + expand ----
        netLayers = layerGraph(dlnet_prev);

        % Step 1: Get last latent layer name
        prevLatentName = sprintf('latent_stage%d', stageIdx_prev);

        % Step 2: Rename old latent layer to FC_hidden_stageN
        newHiddenName = sprintf('fc_hidden_stage%d', stageIdx_prev);
        netLayers = replaceLayer(netLayers, prevLatentName, ...
            fullyConnectedLayer(latent_dim, 'Name', newHiddenName));

        % Step 3: Add ReLU + new Latent layer
        reluName = sprintf('relu_stage%d', stageIdx);
        newLatentName = sprintf('latent_stage%d', stageIdx);

        newLayers = [
            reluLayer('Name', reluName)
            fullyConnectedLayer(latent_dim, 'Name', newLatentName)
        ];

        % Step 4: Connect old hidden → ReLU
        netLayers = addLayers(netLayers, newLayers);
        netLayers = connectLayers(netLayers, newHiddenName, reluName);

        % Step 5: Build new dlnetwork
        dlnet_prev = dlnetwork(netLayers);
    end

    stageIdx_prev = stageIdx;
end

% ========================
% 2. Forward Pass
% ========================
dlX = dlarray(single(teacherTaps(:)), "CB"); % column batch
dlY = forward(dlnet_prev, dlX);

studentTaps = extractdata(dlY);
studentTaps = studentTaps(:)';

% ========================
% 3. Return
% ========================
dlnet = dlnet_prev;

end