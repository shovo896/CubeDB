clear; clc; close all;
rng(20261010,'twister');

%% DATASET
root = fullfile('CubeDN_dataset','CubeDN_3000_RAD_RED');
outDir = fullfile('CubeDN_dataset','GA_CubeFormer_V2_Results');
if ~exist(outDir,'dir'), mkdir(outDir); end
assert(isfolder(root),'RAD/RED dataset folder missing.');
trainFiles = dir(fullfile(root,'train','scene_*.mat'));
valFiles = dir(fullfile(root,'val','scene_*.mat'));
testFiles = dir(fullfile(root,'test','scene_*.mat'));
assert(numel(trainFiles)==2100 && numel(valFiles)==600 && numel(testFiles)==300,'Expected train/val/test = 2100/600/300.');
assert(isfile(fullfile(root,'feature_info.mat')),'feature_info.mat missing.');
infoFile = load(fullfile(root,'feature_info.mat'));
info = infoFile.featureInfo;
assert(all(isfield(info,{'range_m','radialVelocity_mps','directionCosine','radarHPosition_m','radarVPosition_m'})),'Missing physical axes in featureInfo.');

%% TRAINING SETTINGS
numEpochs = 100;
learnRate = 0.0003;
locWeight = 0.2;
weightDecay = 1e-5;
embedDim = 32;
numHeads = 4;
headDim = embedDim/numHeads;
numTokens = 64;
positionScale = 25;
assert(mod(embedDim,numHeads)==0,'Embedding dimension must be divisible by heads.');
fprintf('\nGA-CubeFormer V2 | Train=%d | Val=%d | Test=%d\n',numel(trainFiles),numel(valFiles),numel(testFiles));

%% TRAIN-ONLY NORMALIZATION
normCount = min(300,numel(trainFiles));
normalization = zeros(normCount,4);
for k = 1:normCount
    S = load(fullfile(trainFiles(k).folder,trainFiles(k).name),'RAD','RED');
    assert(isequal(size(S.RAD),[64 64 16]) && isequal(size(S.RED),[64 64 16]),'Unexpected RAD/RED dimensions.');
    normalization(k,:) = [mean(double(S.RAD(:))),std(double(S.RAD(:))),mean(double(S.RED(:))),std(double(S.RED(:)))];
end
stats.meanH = single(mean(normalization(:,1)));
stats.stdH = single(max(mean(normalization(:,2)),1));
stats.meanV = single(mean(normalization(:,3)));
stats.stdV = single(max(mean(normalization(:,4)),1));
fprintf('Train normalization: RAD %.2f +/- %.2f | RED %.2f +/- %.2f\n',stats.meanH,stats.stdH,stats.meanV,stats.stdV);

%% TOKEN GEOMETRY
rAxis = double(info.range_m(:));
vAxis = double(info.radialVelocity_mps(:));
uAxis = double(info.directionCosine(:));
assert(numel(rAxis)==64 && numel(vAxis)==64 && numel(uAxis)==16,'Physical axis lengths do not match feature dimensions.');
rCenters = mean(reshape(rAxis,16,4),1);
vCenters = mean(reshape(vAxis,16,4),1);
uCenters = mean(reshape(uAxis,4,4),1);
[R,V,U] = ndgrid(rCenters,vCenters,uCenters);
tokenR = R(:);
tokenV = V(:);
tokenU = U(:);
pH = double(info.radarHPosition_m(:));
pV = double(info.radarVPosition_m(:));
assert(numel(pH)==3 && numel(pV)==3,'Radar positions must be 3D vectors.');
baseline = norm(pH-pV);
velocityScale = max(max(abs(vAxis)),0.1);
geoFeatures = single([tokenR(:)'/max(max(rAxis),1);tokenV(:)'/velocityScale;tokenU(:)']);
geometryBias = zeros(numTokens,numTokens,'single');

for i = 1:numTokens
    for j = 1:numTokens
        rangeViolation = max(0,abs(tokenR(i)-tokenR(j))-baseline);
        velocityMismatch = abs(tokenV(i)-tokenV(j))/velocityScale;
        directionViolation = max(0,tokenU(i)^2+tokenU(j)^2-1);
        geometryBias(i,j) = single(-0.5*rangeViolation-0.3*velocityMismatch-2*directionViolation);
    end
end

fprintf('Geometry-aware attention: %d tokens, %d heads, baseline %.3f m\n',numTokens,numHeads,baseline);

%% PAIRED FEATURE SEPARABILITY CHECK
pairCount = min(100,floor(numel(trainFiles)/2));
pairDifferences = zeros(pairCount,1);

for k = 1:pairCount
    nameP = sprintf('scene_%05d_present.mat',k);
    nameA = sprintf('scene_%05d_absent.mat',k);
    assert(isfile(fullfile(root,'train',nameP)) && isfile(fullfile(root,'train',nameA)),'Expected paired scene filenames missing.');
    P = load(fullfile(root,'train',nameP),'RAD','RED');
    A = load(fullfile(root,'train',nameA),'RAD','RED');
    [hp,vp] = makeTokensV2(P.RAD,P.RED,stats);
    [ha,va] = makeTokensV2(A.RAD,A.RED,stats);
    pairDifferences(k) = mean(abs(double([hp(:);vp(:)])-double([ha(:);va(:)])));
end

fprintf('Mean paired token difference (first %d training scenes): %.6g\n',pairCount,mean(pairDifferences));
if mean(pairDifferences)<1e-4, warning('Present/absent pooled features are almost identical. Detection may stay near chance.'); end

%% MODEL INITIALIZATION
model = struct();
model.WH = dlarray(single(randn(embedDim,5)*sqrt(2/5)));
model.BH = dlarray(zeros(embedDim,1,'single'));
model.WV = dlarray(single(randn(embedDim,5)*sqrt(2/5)));
model.BV = dlarray(zeros(embedDim,1,'single'));
model.WQ = dlarray(single(randn(embedDim,embedDim)*sqrt(1/embedDim)));
model.WK = dlarray(single(randn(embedDim,embedDim)*sqrt(1/embedDim)));
model.WValue = dlarray(single(randn(embedDim,embedDim)*sqrt(1/embedDim)));
model.WFusion = dlarray(single(randn(embedDim,2*embedDim)*sqrt(1/embedDim)));
model.BFusion = dlarray(zeros(embedDim,1,'single'));
model.WClass = dlarray(single(randn(1,2*embedDim)*0.05));
model.BClass = dlarray(single(0));
model.WPosition = dlarray(single(randn(3,2*embedDim)*0.05));
model.BPosition = dlarray(zeros(3,1,'single'));

names = fieldnames(model);
avgGrad = struct();
avgSqGrad = struct();
iteration = 0;
bestValLoss = inf;
bestModel = model;
history = zeros(numEpochs,6);

%% TRAINING
for epoch = 1:numEpochs
    order = randperm(numel(trainFiles));
    trainLossSum = 0;
    trainCorrect = 0;
    for k = 1:numel(order)
        S = load(fullfile(trainFiles(order(k)).folder,trainFiles(order(k)).name),'RAD','RED','label','groundTruth');
        [hTokens,vTokens] = makeTokensV2(S.RAD,S.RED,stats);
        y = single(S.label);
        target = zeros(3,1,'single');
        if y==1, target = single(S.groundTruth.position_m(:)/positionScale); end
        [loss,grads,p] = dlfeval(@gradientsV2,model,hTokens,vTokens,geoFeatures,geometryBias,numHeads,headDim,y,target,locWeight,weightDecay);
        iteration = iteration+1;
        for j = 1:numel(names)
            name = names{j};
            g = max(min(grads.(name),5),-5);
            if ~isfield(avgGrad,name), avgGrad.(name) = []; avgSqGrad.(name) = []; end
            [model.(name),avgGrad.(name),avgSqGrad.(name)] = adamupdate(model.(name),g,avgGrad.(name),avgSqGrad.(name),iteration,learnRate);
        end
        trainLossSum = trainLossSum+double(gather(extractdata(loss)));
        predicted = double(gather(extractdata(p)))>=0.5;
        trainCorrect = trainCorrect+double(predicted==double(y));
        if mod(k,300)==0, fprintf('Epoch %d/%d | %d/%d\n',epoch,numEpochs,k,numel(order)); end
    end
    trainLoss = trainLossSum/numel(trainFiles);
    trainAcc = trainCorrect/numel(trainFiles);
    [valLoss,valAcc,valError,valConfusion] = evaluateV2(model,valFiles,stats,geoFeatures,geometryBias,numHeads,headDim,locWeight,positionScale);
    history(epoch,:) = [epoch,trainLoss,trainAcc,valLoss,valAcc,valError];
    fprintf('Epoch %02d | Train loss %.4f acc %.2f%% | Val loss %.4f acc %.2f%% | 3D error %.3f m\n',epoch,trainLoss,100*trainAcc,valLoss,100*valAcc,valError);
    if valLoss<bestValLoss
        bestValLoss = valLoss;
        bestModel = model;
        save(fullfile(outDir,'best_GA_CubeFormer_V2.mat'),'bestModel','stats','info','geometryBias','geoFeatures','history','numHeads','headDim','positionScale','-v7.3');
        fprintf('Best validation model saved.\n');
    end
end

%% TEST EVALUATION
[testLoss,testAcc,testError,confusion] = evaluateV2(bestModel,testFiles,stats,geoFeatures,geometryBias,numHeads,headDim,locWeight,positionScale);
TP = confusion(1);
TN = confusion(2);
FP = confusion(3);
FN = confusion(4);
precision = TP/max(TP+FP,1);
recall = TP/max(TP+FN,1);
F1 = 2*precision*recall/max(precision+recall,eps);

fprintf('\nTEST | Accuracy %.2f%% | Precision %.4f | Recall %.4f | F1 %.4f | 3D error %.3f m\n',100*testAcc,precision,recall,F1,testError);
fprintf('Confusion | TP=%d TN=%d FP=%d FN=%d\n',TP,TN,FP,FN);

results = struct('testLoss',testLoss,'testAccuracy',testAcc,'positionEuclideanError_m',testError,'precision',precision,'recall',recall,'F1',F1,'confusion',confusion,'pairedTokenDifference',mean(pairDifferences));
save(fullfile(outDir,'training_results.mat'),'results','history','stats','-v7');

figure;
plot(history(:,1),history(:,2),'-o','LineWidth',1.5);
hold on;
plot(history(:,1),history(:,4),'-s','LineWidth',1.5);
grid on;
xlabel('Epoch');
ylabel('Loss');
legend('Train','Validation');
title('GA-CubeFormer V2 Loss');
saveas(gcf,fullfile(outDir,'training_history.png'));

fprintf('Completed. Results in: %s\n',outDir);

%% LOCAL FUNCTIONS
function [H,V] = makeTokensV2(RAD,RED,stats)
    H = (single(RAD)-stats.meanH)/stats.stdH;
    V = (single(RED)-stats.meanV)/stats.stdV;
    H = max(min(H,10),-10);
    V = max(min(V,10),-10);
    H = reshape(H,16,4,16,4,4,4);
    V = reshape(V,16,4,16,4,4,4);
    hMean = squeeze(mean(mean(mean(H,1),3),5));
    vMean = squeeze(mean(mean(mean(V,1),3),5));
    hMax = squeeze(max(max(max(H,[],1),[],3),[],5));
    vMax = squeeze(max(max(max(V,[],1),[],3),[],5));
    H = single([reshape(hMean,1,[]);reshape(hMax,1,[])]);
    V = single([reshape(vMean,1,[]);reshape(vMax,1,[])]);
end

function [p,xyz] = forwardV2(model,hTokens,vTokens,geoFeatures,B,numHeads,headDim)
    H = tanh(model.WH*dlarray([hTokens;geoFeatures])+model.BH);
    V = tanh(model.WV*dlarray([vTokens;geoFeatures])+model.BV);
    QH = model.WQ*H;
    KH = model.WK*H;
    VH = model.WValue*H;
    QV = model.WQ*V;
    KV = model.WK*V;
    VV = model.WValue*V;
    fusedH = QH*0;
    fusedV = QV*0;
    for head = 1:numHeads
        ix = (head-1)*headDim+(1:headDim);
        scoreHV = (QH(ix,:).'*KV(ix,:))/sqrt(headDim)+dlarray(B);
        scoreVH = (QV(ix,:).'*KH(ix,:))/sqrt(headDim)+dlarray(B.');
        scoreHV = scoreHV-max(scoreHV,[],2);
        scoreVH = scoreVH-max(scoreVH,[],2);
        attentionHV = exp(scoreHV)./(sum(exp(scoreHV),2)+1e-8);
        attentionVH = exp(scoreVH)./(sum(exp(scoreVH),2)+1e-8);
        fusedH(ix,:) = VV(ix,:)*attentionHV.';
        fusedV(ix,:) = VH(ix,:)*attentionVH.';
    end
    fused = tanh(model.WFusion*[H+fusedH;V+fusedV]+model.BFusion);
    pooled = [mean(fused,2);max(fused,[],2)];
    logit = model.WClass*pooled+model.BClass;
    p = 1./(1+exp(-logit));
    xyz = model.WPosition*pooled+model.BPosition;
end

function [loss,grads,p] = gradientsV2(model,H,V,geoFeatures,B,numHeads,headDim,y,target,locWeight,weightDecay)
    [p,xyz] = forwardV2(model,H,V,geoFeatures,B,numHeads,headDim);
    pSafe = min(max(p,1e-6),1-1e-6);
    detectionLoss = -y*log(pSafe)-(1-y)*log(1-pSafe);
    localizationLoss = y*mean((xyz-dlarray(target)).^2,'all');
    loss = detectionLoss+locWeight*localizationLoss;
    names = fieldnames(model);
    for j = 1:numel(names), loss = loss+weightDecay*sum(model.(names{j}).^2,'all'); end
    grads = dlgradient(loss,model);
end

function [avgLoss,accuracy,positionError,confusion] = evaluateV2(model,files,stats,geoFeatures,B,numHeads,headDim,locWeight,positionScale)
    totalLoss = 0;
    correct = 0;
    totalPositionError = 0;
    presentCount = 0;
    TP = 0; TN = 0; FP = 0; FN = 0;
    for k = 1:numel(files)
        S = load(fullfile(files(k).folder,files(k).name),'RAD','RED','label','groundTruth');
        [H,V] = makeTokensV2(S.RAD,S.RED,stats);
        [p,xyz] = forwardV2(model,H,V,geoFeatures,B,numHeads,headDim);
        p = double(gather(extractdata(p)));
        xyz = double(gather(extractdata(xyz)))*positionScale;
        y = double(S.label);
        predicted = double(p>=0.5);
        correct = correct+double(predicted==y);
        totalLoss = totalLoss-y*log(max(p,1e-6))-(1-y)*log(max(1-p,1e-6));
        if y==1
            trueXYZ = double(S.groundTruth.position_m(:));
            totalPositionError = totalPositionError+norm(xyz(:)-trueXYZ);
            totalLoss = totalLoss+locWeight*mean((xyz(:)/positionScale-trueXYZ/positionScale).^2);
            presentCount = presentCount+1;
        end
        if y==1 && predicted==1, TP = TP+1; end
        if y==0 && predicted==0, TN = TN+1; end
        if y==0 && predicted==1, FP = FP+1; end
        if y==1 && predicted==0, FN = FN+1; end
    end
    avgLoss = totalLoss/numel(files);
    accuracy = correct/numel(files);
    positionError = totalPositionError/max(presentCount,1);
    confusion = [TP,TN,FP,FN];
end