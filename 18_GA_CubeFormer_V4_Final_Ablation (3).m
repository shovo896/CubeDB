%% 18_GA_CubeFormer_V4_Final_Ablation.m
% MATLAB R2026a, Deep Learning Toolbox.
% 4 controlled ablations x 3 seeds; NO smoke test.
% Same train/validation/test split, preprocessing, optimization and output heads.
% Test set is evaluated only after validation checkpoint and threshold selection.
% Outputs PNG (300 DPI), vector PDF, editable FIG, CSV, MAT.
% NOTE: Research code, not runtime-verified in MATLAB. No accuracy guarantee.
clear; clc; close all;

%% 01. PATHS AND FINAL SETTINGS
root = fullfile('CubeDN_dataset','CubeDN_3000_RAD_RED');
outDir = fullfile('CubeDN_dataset','GA_CubeFormer_V4_Final_Ablation');
figDir = fullfile(outDir,'Figures');
ckptDir = fullfile(outDir,'Checkpoints');
if ~isfolder(outDir), mkdir(outDir); end
if ~isfolder(figDir), mkdir(figDir); end
if ~isfolder(ckptDir), mkdir(ckptDir); end
assert(isfolder(root),'Dataset folder not found.');
seeds = [11 23 37];
maxEpochs = 35;
batchSize = 16;
learningRate = 3e-4;
patience = 10;
embeddingDim = 16;
numHeads = 4;
xyzWeight = 1.0;
physicsWeight = 0.5;
clipValue = 5;
modelNames = {'A1 CNN only','A2 CNN + cross-attention','A3 + geometry bias','A4 + physics loss'};
modelTags = {'A1_CNN','A2_CrossAttention','A3_Geometry','A4_Full'};
numModels = 4;
numSeeds = numel(seeds);
assert(mod(embeddingDim,numHeads)==0,'Embedding/head mismatch.');

%% 02. DATASET AND METADATA
trainFiles = dir(fullfile(root,'train','scene_*.mat'));
valFiles = dir(fullfile(root,'val','scene_*.mat'));
testFiles = dir(fullfile(root,'test','scene_*.mat'));
assert(numel(trainFiles)==2100,'Expected 2100 training samples.');
assert(numel(valFiles)==600,'Expected 600 validation samples.');
assert(numel(testFiles)==300,'Expected 300 test samples.');
meta = load(fullfile(root,'feature_info.mat'),'featureInfo');
featureInfo = meta.featureInfo;
r = double(featureInfo.range_m(:));
v = double(featureInfo.radialVelocity_mps(:));
u = double(featureInfo.directionCosine(:));
assert(numel(r)==64 && numel(v)==64 && numel(u)==16,'Unexpected radar axes.');
radarH = single(featureInfo.radarHPosition_m(:));
radarV = single(featureInfo.radarVPosition_m(:));
assert(numel(radarH)==3 && numel(radarV)==3,'Radar positions must be 3D.');
r32 = mean(reshape(r,2,32),1).';
v16 = mean(reshape(v,4,16),1).';
r8 = mean(reshape(r32,4,8),1).';
v4 = mean(reshape(v16,4,4),1).';
u4 = mean(reshape(u,4,4),1).';
geomBias = makeGeometryBias(r8,v4,u4,radarH,radarV);

%% 03. SHARED TRAIN-ONLY NORMALIZATION
fprintf('Computing shared train-only normalization...\n');
sumH = 0;
sumV = 0;
sumH2 = 0;
sumV2 = 0;
nValues = 0;
for k = 1:numel(trainFiles)
    D = load(fullfile(trainFiles(k).folder,trainFiles(k).name),'RAD','RED');
    H = reduceCube(D.RAD);
    V = reduceCube(D.RED);
    sumH = sumH+sum(double(H(:)));
    sumV = sumV+sum(double(V(:)));
    sumH2 = sumH2+sum(double(H(:)).^2);
    sumV2 = sumV2+sum(double(V(:)).^2);
    nValues = nValues+numel(H);
end
stats.meanH = single(sumH/nValues);
stats.meanV = single(sumV/nValues);
stats.stdH = single(sqrt(max(sumH2/nValues-double(stats.meanH)^2,1e-2)));
stats.stdV = single(sqrt(max(sumV2/nValues-double(stats.meanV)^2,1e-2)));

%% 04. LOAD SPLITS ONCE
fprintf('Loading train split...\n');
[HTrain,VTrain,LTrain,PTrain] = loadSplit(trainFiles,stats);
fprintf('Loading validation split...\n');
[HVal,VVal,LVal,PVal] = loadSplit(valFiles,stats);
fprintf('Loading held-out test split...\n');
[HTest,VTest,LTest,PTest] = loadSplit(testFiles,stats);
assert(sum(LTrain)==1050 && sum(LVal)==300 && sum(LTest)==150,'Unexpected class counts.');
positionScale = single(max(1,1.2*max(abs(PTrain(:)))));
fprintf('Position normalization scale: %.3f m\n',positionScale);

%% 05. RUN ARRAYS
nRuns = numModels*numSeeds;
runModel = strings(nRuns,1);
runSeed = zeros(nRuns,1);
accuracy = nan(nRuns,1);
precision = nan(nRuns,1);
recall = nan(nRuns,1);
specificity = nan(nRuns,1);
F1 = nan(nRuns,1);
balancedAccuracy = nan(nRuns,1);
locErrorAllPositive = nan(nRuns,1);
locErrorDetectedPositive = nan(nRuns,1);
TP = zeros(nRuns,1);
TN = zeros(nRuns,1);
FP = zeros(nRuns,1);
FN = zeros(nRuns,1);
bestEpoch = zeros(nRuns,1);
threshold = nan(nRuns,1);
trainCurves = nan(maxEpochs,nRuns);
valCurves = nan(maxEpochs,nRuns);
testScoresAll = nan(numel(LTest),nRuns);
testXYZAll = nan(3,numel(LTest),nRuns);
runNumber = 0;

%% 06. TRAIN 4 MODELS WITH IDENTICAL SEEDS AND BUDGET
for s = 1:numSeeds
    for m = 1:numModels
        runNumber = runNumber+1;
        seed = seeds(s);
        rng(seed,'twister');
        fprintf('\n=== RUN %d/%d | %s | seed %d ===\n',runNumber,nRuns,modelNames{m},seed);
        netH = makeEncoder('H',embeddingDim);
        netV = makeEncoder('V',embeddingDim);
        theta = makeParameters(embeddingDim);
        avgH = [];
        avgSqH = [];
        avgV = [];
        avgSqV = [];
        avgT = [];
        avgSqT = [];
        iter = 0;
        highestValBA = -inf;
        lowestValLoss = inf;
        stale = 0;
        bestEpochThisRun = 0;
        bestThreshold = 0.5;
        for epoch = 1:maxEpochs
            order = randperm(numel(LTrain));
            sumLoss = 0;
            nBatches = 0;
            for first = 1:batchSize:numel(order)
                ids = order(first:min(first+batchSize-1,numel(order)));
                XH = dlarray(HTrain(:,:,:,:,ids),'SSSCB');
                XV = dlarray(VTrain(:,:,:,:,ids),'SSSCB');
                Y = dlarray(reshape(single(LTrain(ids)),1,[]));
                XYZ = dlarray(single(PTrain(:,ids))/positionScale);
                [loss,gH,gV,gT] = dlfeval(@getGradients,netH,netV,theta,XH,XV,Y,XYZ,m,geomBias,numHeads,radarH,radarV,positionScale,xyzWeight,physicsWeight);
                gH = dlupdate(@(g) clipGradient(g,clipValue),gH);
                gV = dlupdate(@(g) clipGradient(g,clipValue),gV);
                gT = dlupdate(@(g) clipGradient(g,clipValue),gT);
                iter = iter+1;
                [netH,avgH,avgSqH] = adamupdate(netH,gH,avgH,avgSqH,iter,learningRate);
                [netV,avgV,avgSqV] = adamupdate(netV,gV,avgV,avgSqV,iter,learningRate);
                [theta,avgT,avgSqT] = adamupdate(theta,gT,avgT,avgSqT,iter,learningRate);
                sumLoss = sumLoss+double(gather(extractdata(loss)));
                nBatches = nBatches+1;
            end
            trainCurves(epoch,runNumber) = sumLoss/nBatches;
            [valLoss,valScores] = evaluateModel(netH,netV,theta,HVal,VVal,LVal,PVal,m,geomBias,numHeads,radarH,radarV,positionScale,xyzWeight,physicsWeight,batchSize);
            valCurves(epoch,runNumber) = valLoss;
            [valThreshold,valBA] = chooseThreshold(valScores,LVal);
            fprintf('Epoch %02d | Train %.4f | Val %.4f | Val balanced accuracy %.2f%%\n',epoch,trainCurves(epoch,runNumber),valLoss,100*valBA);
            isBetter = valBA>highestValBA+1e-6 || (abs(valBA-highestValBA)<=1e-6 && valLoss<lowestValLoss);
            if isBetter
                highestValBA = valBA;
                lowestValLoss = valLoss;
                bestEpochThisRun = epoch;
                bestThreshold = valThreshold;
                stale = 0;
                checkpoint = fullfile(ckptDir,sprintf('%s_seed_%d.mat',modelTags{m},seed));
                save(checkpoint,'netH','netV','theta','bestThreshold','bestEpochThisRun','positionScale','stats','featureInfo','-v7.3');
            else
                stale = stale+1;
            end
            if stale>=patience
                fprintf('Early stopping at epoch %d.\n',epoch);
                break;
            end
        end
        saved = load(checkpoint,'netH','netV','theta');
        netH = saved.netH;
        netV = saved.netV;
        theta = saved.theta;
        [~,testScores,testXYZ] = evaluateModel(netH,netV,theta,HTest,VTest,LTest,PTest,m,geomBias,numHeads,radarH,radarV,positionScale,xyzWeight,physicsWeight,batchSize);
        pred = double(testScores>=bestThreshold);
        tp = sum(pred==1 & LTest==1);
        tn = sum(pred==0 & LTest==0);
        fp = sum(pred==1 & LTest==0);
        fn = sum(pred==0 & LTest==1);
        runModel(runNumber) = string(modelTags{m});
        runSeed(runNumber) = seed;
        accuracy(runNumber) = (tp+tn)/numel(LTest);
        precision(runNumber) = tp/max(tp+fp,1);
        recall(runNumber) = tp/max(tp+fn,1);
        specificity(runNumber) = tn/max(tn+fp,1);
        F1(runNumber) = 2*precision(runNumber)*recall(runNumber)/max(precision(runNumber)+recall(runNumber),eps);
        balancedAccuracy(runNumber) = (recall(runNumber)+specificity(runNumber))/2;
        errors = sqrt(sum((testXYZ-double(PTest)).^2,1)).';
        locErrorAllPositive(runNumber) = mean(errors(LTest==1));
        locErrorDetectedPositive(runNumber) = mean(errors(LTest==1 & pred==1),'omitnan');
        TP(runNumber) = tp;
        TN(runNumber) = tn;
        FP(runNumber) = fp;
        FN(runNumber) = fn;
        bestEpoch(runNumber) = bestEpochThisRun;
        threshold(runNumber) = bestThreshold;
        testScoresAll(:,runNumber) = testScores;
        testXYZAll(:,:,runNumber) = testXYZ;
        fprintf('TEST | Acc %.2f%% | F1 %.2f%% | Recall %.2f%% | 3D error %.3f m\n',100*accuracy(runNumber),100*F1(runNumber),100*recall(runNumber),locErrorAllPositive(runNumber));
        save(fullfile(outDir,'Intermediate_Results.mat'),'runModel','runSeed','accuracy','F1','recall','locErrorAllPositive','testScoresAll','trainCurves','valCurves','-v7.3');
    end
end

%% 07. TABLES FOR PAPER
runTable = table(runModel,runSeed,accuracy,precision,recall,specificity,F1,balancedAccuracy,locErrorAllPositive,locErrorDetectedPositive,TP,TN,FP,FN,bestEpoch,threshold);
writetable(runTable,fullfile(outDir,'Table_01_Per_Seed_Results.csv'));
means = nan(numModels,4);
sds = nan(numModels,4);
for m = 1:numModels
    idx = find(runModel==string(modelTags{m}));
    values = [100*accuracy(idx),100*F1(idx),100*recall(idx),locErrorAllPositive(idx)];
    means(m,:) = mean(values,1,'omitnan');
    sds(m,:) = std(values,0,1,'omitnan');
end
model = string(modelTags(:));
summaryTable = table(model,means(:,1),sds(:,1),means(:,2),sds(:,2),means(:,3),sds(:,3),means(:,4),sds(:,4),'VariableNames',{'Model','AccuracyMeanPct','AccuracySDPct','F1MeanPct','F1SDPct','RecallMeanPct','RecallSDPct','XYZErrorMeanM','XYZErrorSDM'});
writetable(summaryTable,fullfile(outDir,'Table_02_Ablation_Mean_SD.csv'));
disp(summaryTable);

%% 08. PAPER FIGURE 01: ACCURACY
f = paperFigure('Ablation accuracy');
bar(1:4,means(:,1),0.62);
hold on;
errorbar(1:4,means(:,1),sds(:,1),'k.','LineWidth',1.5,'CapSize',12);
formatModelAxis('Test accuracy (%)','Controlled ablation: detection accuracy');
ylim([0 100]);
savePaperFigure(f,figDir,'Fig01_Accuracy_Mean_SD');

%% 09. PAPER FIGURE 02: F1
f = paperFigure('Ablation F1');
bar(1:4,means(:,2),0.62);
hold on;
errorbar(1:4,means(:,2),sds(:,2),'k.','LineWidth',1.5,'CapSize',12);
formatModelAxis('Test F1-score (%)','Controlled ablation: F1-score');
ylim([0 100]);
savePaperFigure(f,figDir,'Fig02_F1_Mean_SD');

%% 10. PAPER FIGURE 03: RECALL
f = paperFigure('Ablation recall');
bar(1:4,means(:,3),0.62);
hold on;
errorbar(1:4,means(:,3),sds(:,3),'k.','LineWidth',1.5,'CapSize',12);
formatModelAxis('Test recall (%)','Controlled ablation: recall');
ylim([0 100]);
savePaperFigure(f,figDir,'Fig03_Recall_Mean_SD');

%% 11. PAPER FIGURE 04: XYZ ERROR
f = paperFigure('Ablation localization');
bar(1:4,means(:,4),0.62);
hold on;
errorbar(1:4,means(:,4),sds(:,4),'k.','LineWidth',1.5,'CapSize',12);
formatModelAxis('Mean 3D error (m)','Localization error on all drone-present test samples');
savePaperFigure(f,figDir,'Fig04_Localization_Mean_SD');

%% 12. PAPER FIGURE 05: TRAINING CURVES
f = paperFigure('Training convergence');
hold on;
for m = 1:4
    ids = find(runModel==string(modelTags{m}));
    plot(1:maxEpochs,mean(trainCurves(:,ids),2,'omitnan'),'LineWidth',1.8,'DisplayName',modelTags{m});
end
grid on;
xlabel('Epoch');
ylabel('Training objective');
title('Training convergence (mean across seeds)');
legend('Location','best','Interpreter','none');
savePaperFigure(f,figDir,'Fig05_Training_Convergence');

%% 13. PAPER FIGURE 06: VALIDATION CURVES
f = paperFigure('Validation convergence');
hold on;
for m = 1:4
    ids = find(runModel==string(modelTags{m}));
    plot(1:maxEpochs,mean(valCurves(:,ids),2,'omitnan'),'LineWidth',1.8,'DisplayName',modelTags{m});
end
grid on;
xlabel('Epoch');
ylabel('Validation objective');
title('Validation convergence (mean across seeds)');
legend('Location','best','Interpreter','none');
savePaperFigure(f,figDir,'Fig06_Validation_Convergence');

%% 14. PAPER FIGURE 07: CONFUSION MATRICES, FIRST SEED
f = figure('Color','w','Position',[80 80 1200 900],'Name','Confusion matrices');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
for m = 1:4
    idx = find(runModel==string(modelTags{m}) & runSeed==seeds(1),1);
    C = [TN(idx),FP(idx);FN(idx),TP(idx)];
    nexttile;
    imagesc(C);
    axis image;
    colormap(parula);
    colorbar;
    set(gca,'XTick',1:2,'XTickLabel',{'Absent','Present'},'YTick',1:2,'YTickLabel',{'Absent','Present'});
    xlabel('Predicted');
    ylabel('Actual');
    title(sprintf('%s | seed %d',modelTags{m},seeds(1)),'Interpreter','none');
    for row = 1:2
        for col = 1:2
            text(col,row,sprintf('%d',C(row,col)),'HorizontalAlignment','center','FontWeight','bold','Color','w','BackgroundColor','k');
        end
    end
end
savePaperFigure(f,figDir,'Fig07_Confusion_Matrices');

%% 15. PAPER FIGURE 08: CONFIDENCE DISTRIBUTIONS
f = figure('Color','w','Position',[80 80 1200 900],'Name','Confidence distributions');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
for m = 1:4
    idx = find(runModel==string(modelTags{m}) & runSeed==seeds(1),1);
    nexttile;
    histogram(testScoresAll(LTest==0,idx),0:0.05:1,'Normalization','probability','DisplayName','Absent');
    hold on;
    histogram(testScoresAll(LTest==1,idx),0:0.05:1,'Normalization','probability','DisplayName','Present');
    xline(threshold(idx),'k--','LineWidth',1.4,'DisplayName','Val threshold');
    xlim([0 1]);
    grid on;
    xlabel('Detection probability');
    ylabel('Fraction');
    title(modelTags{m},'Interpreter','none');
    legend('Location','best');
end
savePaperFigure(f,figDir,'Fig08_Confidence_Distributions');

%% 16. PAPER FIGURE 09: PER-SEED ACCURACY
f = paperFigure('Per-seed accuracy');
hold on;
for s = 1:numSeeds
    ids = find(runSeed==seeds(s));
    plot(1:4,100*accuracy(ids),'-o','LineWidth',1.6,'MarkerSize',7,'DisplayName',sprintf('Seed %d',seeds(s)));
end
formatModelAxis('Test accuracy (%)','Run-to-run stability');
ylim([0 100]);
legend('Location','best');
savePaperFigure(f,figDir,'Fig09_Per_Seed_Accuracy');

%% 17. PAPER FIGURE 10: XYZ SCATTER, FULL MODEL FIRST SEED
idx = find(runModel==string(modelTags{4}) & runSeed==seeds(1),1);
present = LTest==1;
xyzPred = testXYZAll(:,:,idx);
f = figure('Color','w','Position',[80 80 1350 450],'Name','Full model XYZ');
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
axisNames = {'X','Y','Z'};
for d = 1:3
    nexttile;
    scatter(double(PTest(d,present)),xyzPred(d,present),22,'filled');
    hold on;
    limits = [min([double(PTest(d,present)),xyzPred(d,present)]),max([double(PTest(d,present)),xyzPred(d,present)])];
    plot(limits,limits,'k--','LineWidth',1.3);
    grid on;
    axis equal;
    xlabel(['Ground truth ' axisNames{d} ' (m)']);
    ylabel(['Predicted ' axisNames{d} ' (m)']);
    title([axisNames{d} ' coordinate']);
end
savePaperFigure(f,figDir,'Fig10_Full_Model_XYZ');

%% 18. PAPER FIGURE 11: GEOMETRY BIAS
f = paperFigure('Geometry prior');
imagesc(geomBias);
axis image;
colorbar;
xlabel('Vertical radar token');
ylabel('Horizontal radar token');
title('Fixed geometry compatibility prior (not learned attention)');
savePaperFigure(f,figDir,'Fig11_Geometry_Prior');

%% 19. PAIRED SEED DIFFERENCES, NOT A SIGNIFICANCE CLAIM
pairedDifferences = nan(numSeeds,3);
for s = 1:numSeeds
    i1 = find(runModel==string(modelTags{1}) & runSeed==seeds(s),1);
    i2 = find(runModel==string(modelTags{2}) & runSeed==seeds(s),1);
    i3 = find(runModel==string(modelTags{3}) & runSeed==seeds(s),1);
    i4 = find(runModel==string(modelTags{4}) & runSeed==seeds(s),1);
    pairedDifferences(s,:) = 100*[accuracy(i2)-accuracy(i1),accuracy(i3)-accuracy(i2),accuracy(i4)-accuracy(i3)];
end
pairedTable = table(seeds(:),pairedDifferences(:,1),pairedDifferences(:,2),pairedDifferences(:,3),'VariableNames',{'Seed','CrossAttentionGain_pp','GeometryGain_pp','PhysicsGain_pp'});
writetable(pairedTable,fullfile(outDir,'Table_03_Paired_Accuracy_Differences.csv'));

%% 20. SAVE COMPLETE RESEARCH ARTIFACTS
save(fullfile(outDir,'All_Ablation_Results.mat'),'runTable','summaryTable','pairedTable','testScoresAll','testXYZAll','trainCurves','valCurves','LTest','PTest','stats','featureInfo','seeds','-v7.3');
fid = fopen(fullfile(outDir,'README_Experiment_Protocol.txt'),'w');
assert(fid>0,'Could not write protocol.');
fprintf(fid,'GA-CubeFormer V4 controlled ablation\n');
fprintf(fid,'Synthetic RAD/RED dataset, fixed train/val/test split: 2100/600/300.\n');
fprintf(fid,'Four variants: CNN; CNN+cross-attention; +geometry bias; +physics loss.\n');
fprintf(fid,'Seeds: %s. Max epochs: %d. Batch size: %d. Adam LR: %.6f.\n',mat2str(seeds),maxEpochs,batchSize,learningRate);
fprintf(fid,'Validation selects checkpoint and classification threshold. Test is held out.\n');
fprintf(fid,'3D error computed for all ground-truth positive test samples; detected-only separately.\n');
fprintf(fid,'Geometry prior uses range/Doppler proximity and angular feasibility approximation.\n');
fprintf(fid,'Physics loss uses radar-range and direction-cosine consistency with true 3D position.\n');
fprintf(fid,'No guarantee of outperforming any baseline. No claim of real-world generalization.\n');
fprintf(fid,'Training objectives are not directly comparable between A3 and A4 due to extra loss.\n');
fclose(fid);
fprintf('\nALL RUNS FINISHED. Results: %s\n',outDir);
fprintf('Figures: %s\n',figDir);

%% LOCAL FUNCTION 01: REDUCE WITHOUT COLLAPSING DOPPLER
function X = reduceCube(A)
    A = single(A);
    assert(isequal(size(A),[64 64 16]),'Expected 64x64x16 cube.');
    A = reshape(A,2,32,4,16,16);
    A = mean(A,1);
    A = mean(A,3);
    X = reshape(A,32,16,16);
end

%% LOCAL FUNCTION 02: LOAD SPLIT
function [H,V,L,P] = loadSplit(files,stats)
    n = numel(files);
    H = zeros(32,16,16,1,n,'single');
    V = zeros(32,16,16,1,n,'single');
    L = zeros(n,1);
    P = zeros(3,n,'single');
    for k = 1:n
        D = load(fullfile(files(k).folder,files(k).name),'RAD','RED','label','groundTruth');
        h = reduceCube(D.RAD);
        v = reduceCube(D.RED);
        H(:,:,:,:,k) = reshape(max(-8,min(8,(h-stats.meanH)/stats.stdH)),32,16,16,1);
        V(:,:,:,:,k) = reshape(max(-8,min(8,(v-stats.meanV)/stats.stdV)),32,16,16,1);
        L(k) = double(D.label);
        if L(k)==1, P(:,k) = single(D.groundTruth.position_m(:)); end
    end
end

%% LOCAL FUNCTION 03: FIXED GEOMETRY PRIOR
function bias = makeGeometryBias(r,v,u,radarH,radarV)
    [R,D,U] = ndgrid(r,v,u);
    rr = R(:);
    dd = D(:);
    uu = U(:);
    baseline = norm(double(radarH-radarV));
    rangeTol = max(2*median(abs(diff(r))),baseline+0.01);
    dopplerTol = max(2*median(abs(diff(v))),0.01);
    rangePenalty = (rr-rr.').^2/(2*rangeTol^2);
    dopplerPenalty = (dd-dd.').^2/(2*dopplerTol^2);
    feasibilityPenalty = 4*max(0,uu.^2+(uu.').^2-1);
    bias = single(max(-8,-rangePenalty-dopplerPenalty-feasibilityPenalty));
end

%% LOCAL FUNCTION 04: SAME CNN ENCODER IN ALL VARIANTS
function net = makeEncoder(prefix,embeddingDim)
    layers = [
        image3dInputLayer([32 16 16 1],Normalization='none',Name=[prefix '_input'])
        convolution3dLayer(3,8,Stride=2,Padding='same',Name=[prefix '_conv1'])
        reluLayer(Name=[prefix '_relu1'])
        convolution3dLayer(3,embeddingDim,Stride=2,Padding='same',Name=[prefix '_conv2'])
        reluLayer(Name=[prefix '_relu2'])
        convolution3dLayer(3,embeddingDim,Padding='same',Name=[prefix '_conv3'])
        reluLayer(Name=[prefix '_relu3'])
    ];
    net = dlnetwork(layers);
end

%% LOCAL FUNCTION 05: TRAINABLE PARAMETERS
function t = makeParameters(F)
    t.WqH = dlarray(single(randn(F,F)*sqrt(1/F)));
    t.WkH = dlarray(single(randn(F,F)*sqrt(1/F)));
    t.WvH = dlarray(single(randn(F,F)*sqrt(1/F)));
    t.WqV = dlarray(single(randn(F,F)*sqrt(1/F)));
    t.WkV = dlarray(single(randn(F,F)*sqrt(1/F)));
    t.WvV = dlarray(single(randn(F,F)*sqrt(1/F)));
    t.geomRaw = dlarray(single(0.5));
    t.Wf = dlarray(single(randn(F,2*F)*sqrt(1/(2*F))));
    t.bf = dlarray(zeros(F,1,'single'));
    t.Wcls = dlarray(single(randn(1,F)*0.05));
    t.bcls = dlarray(zeros(1,1,'single'));
    t.Wxyz = dlarray(single(randn(3,F)*0.05));
    t.bxyz = dlarray(zeros(3,1,'single'));
end

%% LOCAL FUNCTION 06: FORWARD
function [prob,xyz] = modelForward(netH,netV,t,XH,XV,modelIndex,geomBias,numHeads,positionScale)
    FH = stripdims(forward(netH,XH));
    FV = stripdims(forward(netV,XV));
    shape = size(FH);
    B = shape(5);
    F = shape(4);
    T = prod(shape(1:3));
    H = permute(reshape(FH,T,F,B),[2 1 3]);
    V = permute(reshape(FV,T,F,B),[2 1 3]);
    qH = reshape(t.WqH*reshape(H,F,[]),F,T,B);
    kH = reshape(t.WkH*reshape(H,F,[]),F,T,B);
    vH = reshape(t.WvH*reshape(H,F,[]),F,T,B);
    qV = reshape(t.WqV*reshape(V,F,[]),F,T,B);
    kV = reshape(t.WkV*reshape(V,F,[]),F,T,B);
    vV = reshape(t.WvV*reshape(V,F,[]),F,T,B);
    headDim = F/numHeads;
    outH = [];
    outV = [];
    for head = 1:numHeads
        rows = (head-1)*headDim+1:head*headDim;
        logitsH = pagemtimes(permute(qH(rows,:,:),[2 1 3]),kV(rows,:,:))/sqrt(headDim);
        logitsV = pagemtimes(permute(qV(rows,:,:),[2 1 3]),kH(rows,:,:))/sqrt(headDim);
        if modelIndex>=3
            strength = 0.5+0.5*tanh(t.geomRaw);
            logitsH = logitsH+strength*geomBias;
            logitsV = logitsV+strength*geomBias.';
        end
        weightsH = exp(logitsH-max(logitsH,[],2));
        weightsV = exp(logitsV-max(logitsV,[],2));
        weightsH = weightsH./(sum(weightsH,2)+1e-8);
        weightsV = weightsV./(sum(weightsV,2)+1e-8);
        outH = cat(1,outH,pagemtimes(vV(rows,:,:),permute(weightsH,[2 1 3])));
        outV = cat(1,outV,pagemtimes(vH(rows,:,:),permute(weightsV,[2 1 3])));
    end
    if modelIndex==1
        fusedH = H;
        fusedV = V;
    else
        fusedH = H+outH;
        fusedV = V+outV;
    end
    poolH = 0.5*(mean(fusedH,2)+max(fusedH,[],2));
    poolV = 0.5*(mean(fusedV,2)+max(fusedV,[],2));
    combined = cat(1,reshape(poolH,F,B),reshape(poolV,F,B));
    hidden = tanh(t.Wf*combined+t.bf);
    logits = t.Wcls*hidden+t.bcls;
    prob = 1./(1+exp(-logits));
    xyz = positionScale*(t.Wxyz*hidden+t.bxyz);
end

%% LOCAL FUNCTION 07: PHYSICS-CONSISTENCY LOSS
function physicsLoss = physicsConsistency(xyz,xyzTarget,labels,radarH,radarV)
    target = xyzTarget;
    dh = xyz-radarH;
    dv = xyz-radarV;
    th = target-radarH;
    tv = target-radarV;
    rh = sqrt(sum(dh.^2,1)+1e-6);
    rv = sqrt(sum(dv.^2,1)+1e-6);
    rth = sqrt(sum(th.^2,1)+1e-6);
    rtv = sqrt(sum(tv.^2,1)+1e-6);
    uh = dh(1,:)./rh;
    uv = dv(3,:)./rv;
    uth = th(1,:)./rth;
    utv = tv(3,:)./rtv;
    perSample = ((rh-rth)/25).^2+((rv-rtv)/25).^2+(uh-uth).^2+(uv-utv).^2;
    physicsLoss = sum(perSample.*labels,'all')/(sum(labels,'all')+1e-6);
end

%% LOCAL FUNCTION 08: DIFFERENTIABLE LOSS
function [loss,gH,gV,gT] = getGradients(netH,netV,t,XH,XV,labels,target,modelIndex,geomBias,numHeads,radarH,radarV,positionScale,xyzWeight,physicsWeight)
    [prob,xyz] = modelForward(netH,netV,t,XH,XV,modelIndex,geomBias,numHeads,positionScale);
    prob = min(max(prob,1e-6),1-1e-6);
    clsLoss = -mean(labels.*log(prob)+(1-labels).*log(1-prob),'all');
    err = sum((xyz/positionScale-target).^2,1);
    xyzLoss = sum(err.*labels,'all')/(sum(labels,'all')+1e-6);
    loss = clsLoss+xyzWeight*xyzLoss;
    if modelIndex==4
        phys = physicsConsistency(xyz,target*positionScale,labels,radarH,radarV);
        loss = loss+physicsWeight*phys;
    end
    % Keep gradients defined for common parameter struct in every ablation.
    zeroLink = 0*(sum(t.WqH,'all')+sum(t.WkH,'all')+sum(t.WvH,'all')+sum(t.WqV,'all')+sum(t.WkV,'all')+sum(t.WvV,'all')+sum(t.geomRaw,'all'));
    loss = loss+zeroLink;
    [gH,gV,gT] = dlgradient(loss,netH.Learnables,netV.Learnables,t);
end

%% LOCAL FUNCTION 09: EVALUATION, NO TEST TUNING
function [meanLoss,scores,xyzAll] = evaluateModel(netH,netV,t,H,V,L,P,modelIndex,geomBias,numHeads,radarH,radarV,positionScale,xyzWeight,physicsWeight,batchSize)
    N = numel(L);
    scores = zeros(N,1);
    xyzAll = zeros(3,N);
    lossSum = 0;
    for first = 1:batchSize:N
        ids = first:min(first+batchSize-1,N);
        XH = dlarray(H(:,:,:,:,ids),'SSSCB');
        XV = dlarray(V(:,:,:,:,ids),'SSSCB');
        [p,xyz] = modelForward(netH,netV,t,XH,XV,modelIndex,geomBias,numHeads,positionScale);
        p = reshape(double(gather(extractdata(p))),1,[]);
        xyz = double(gather(extractdata(xyz)));
        scores(ids) = p(:);
        xyzAll(:,ids) = xyz;
        labels = reshape(double(L(ids)),1,[]);
        p = max(1e-6,min(1-1e-6,p));
        cls = -mean(labels.*log(p)+(1-labels).*log(1-p));
        e = sum(((xyz-double(P(:,ids)))/double(positionScale)).^2,1);
        loc = sum(e.*labels)/max(sum(labels),1);
        batchLoss = cls+xyzWeight*loc;
        if modelIndex==4
            ph = double(gather(extractdata(physicsConsistency(dlarray(single(xyz)),dlarray(single(P(:,ids))),dlarray(single(labels)),radarH,radarV))));
            batchLoss = batchLoss+physicsWeight*ph;
        end
        lossSum = lossSum+batchLoss*numel(ids);
    end
    meanLoss = lossSum/N;
end

%% LOCAL FUNCTION 10: VALIDATION-ONLY THRESHOLD
function [bestThreshold,bestBA] = chooseThreshold(scores,labels)
    grid = linspace(0.02,0.98,97);
    bestThreshold = 0.5;
    bestBA = -inf;
    for k = 1:numel(grid)
        t = grid(k);
        tpr = mean(scores(labels==1)>=t);
        tnr = mean(scores(labels==0)<t);
        ba = 0.5*(tpr+tnr);
        if ba>bestBA
            bestBA = ba;
            bestThreshold = t;
        end
    end
end

%% LOCAL FUNCTION 11: GRADIENT CLIP
function g = clipGradient(g,limit)
    if isempty(g), return; end
    normG = sqrt(sum(g.^2,'all')+1e-12);
    g = g.*min(1,limit/normG);
end

%% LOCAL FUNCTION 12: PAPER FIGURE HELPERS
function f = paperFigure(name)
    f = figure('Color','w','Position',[100 100 1050 620],'Name',name);
end
function formatModelAxis(ylabelText,titleText)
    grid on;
    set(gca,'XTick',1:4,'XTickLabel',{'A1 CNN','A2 Attention','A3 Geometry','A4 Full'},'FontSize',12,'LineWidth',1);
    xlim([0.5 4.5]);
    ylabel(ylabelText);
    title(titleText);
end
function savePaperFigure(f,folder,name)
    savefig(f,fullfile(folder,[name '.fig']));
    exportgraphics(f,fullfile(folder,[name '.png']),'Resolution',300);
    exportgraphics(f,fullfile(folder,[name '.pdf']),'ContentType','vector');
    fprintf('Saved %s (.fig, .png, .pdf)\n',name);
end
