
clear; clc; close all;
rng(11,'twister');
root = fullfile('CubeDN_dataset','CubeDN_3000_RAD_RED');
resultDir = fullfile('CubeDN_dataset','GA_CubeFormer_V4_Single_Results');
xaiDir = fullfile(resultDir,'V4_XAI_Paper_Analysis');
figDir = fullfile(xaiDir,'Figures');
dataDir = fullfile(xaiDir,'Data');
if ~isfolder(figDir), mkdir(figDir); end
if ~isfolder(dataDir), mkdir(dataDir); end
checkpointFile = fullfile(resultDir,'GA_CubeFormer_V4_Best_Checkpoint.mat');
resultFile = fullfile(resultDir,'GA_CubeFormer_V4_Test_Results.mat');
assert(isfile(checkpointFile),'V4 checkpoint missing. Set resultDir correctly.');
assert(isfile(resultFile),'V4 saved test results missing.');
C = load(checkpointFile);
R = load(resultFile);
netH = C.netH;
netV = C.netV;
theta = C.theta;
G = single(C.geomBias);
stats = C.stats;
positionScale = single(C.positionScale);
threshold = double(C.bestThreshold);
numHeads = 4;
labels = double(R.LTest(:));
scores = double(R.testScores(:));
pred = double(scores>=threshold);
xyzTrue = double(R.PTest);
xyzPred = double(R.testXYZ);
files = dir(fullfile(root,'test','scene_*.mat'));
assert(numel(files)==numel(labels),'Test file count mismatch.');
N = numel(labels);
TP = sum(labels==1 & pred==1);
TN = sum(labels==0 & pred==0);
FP = sum(labels==0 & pred==1);
FN = sum(labels==1 & pred==0);
errors = sqrt(sum((xyzPred-xyzTrue).^2,1)).';
fpr = FP/max(FP+TN,1);
tpr = TP/max(TP+FN,1);
precision = TP/max(TP+FP,1);
F1 = 2*precision*tpr/max(precision+tpr,eps);
fprintf('V4 checkpoint loaded | TP %d TN %d FP %d FN %d\n',TP,TN,FP,FN);

%% 01. ROC and PR curve from fixed test scores
[sortedScores,ix] = sort(scores,'descend');
ySorted = labels(ix);
tpCum = cumsum(ySorted==1);
fpCum = cumsum(ySorted==0);
rocX = [0;fpCum/max(sum(labels==0),1);1];
rocY = [0;tpCum/max(sum(labels==1),1);1];
[rocX,ord] = sort(rocX);
rocY = rocY(ord);
aucROC = trapz(rocX,rocY);
prRecall = [0;tpCum/max(sum(labels==1),1)];
prPrecision = [1;tpCum./max(tpCum+fpCum,1)];
aucPR = sum((prRecall(2:end)-prRecall(1:end-1)).*prPrecision(2:end));
f = figure('Color','w','Position',[100 100 850 570]);
plot(rocX,rocY,'LineWidth',2);
hold on;
plot([0 1],[0 1],'k--');
axis([0 1 0 1]);
axis square;
grid on;
xlabel('False positive rate');
ylabel('True positive rate');
title(sprintf('01 | ROC curve (AUC %.3f)',aucROC));
saveXAI(f,figDir,'01_ROC_AUC');
f = figure('Color','w','Position',[100 100 850 570]);
plot(prRecall,prPrecision,'LineWidth',2);
hold on;
yline(mean(labels),'k--');
axis([0 1 0 1]);
axis square;
grid on;
xlabel('Recall');
ylabel('Precision');
title(sprintf('02 | Precision-recall (AP %.3f)',aucPR));
saveXAI(f,figDir,'02_Precision_Recall');

%% 02. Calibration and confidence distribution
edges = linspace(0,1,11);
centers = (edges(1:end-1)+edges(2:end))/2;
obs = nan(1,10);
conf = nan(1,10);
counts = zeros(1,10);
for j = 1:10
    mask = scores>=edges(j) & scores<edges(j+1);
    if j==10, mask = scores>=edges(j) & scores<=edges(j+1); end
    counts(j) = sum(mask);
    if counts(j)>0
        obs(j) = mean(labels(mask));
        conf(j) = mean(scores(mask));
    end
end
brier = mean((scores-labels).^2);
ece = sum(counts(~isnan(obs)).*abs(obs(~isnan(obs))-conf(~isnan(obs))))/N;
f = figure('Color','w','Position',[100 100 850 570]);
plot([0 1],[0 1],'k--');
hold on;
plot(conf,obs,'o-','LineWidth',2,'MarkerSize',8);
axis([0 1 0 1]);
axis square;
grid on;
xlabel('Mean predicted confidence');
ylabel('Observed positive fraction');
title(sprintf('03 | Reliability (Brier %.3f, ECE %.3f)',brier,ece));
saveXAI(f,figDir,'03_Calibration_Reliability');
f = figure('Color','w','Position',[100 100 850 570]);
histogram(scores(labels==0),20,'Normalization','probability');
hold on;
histogram(scores(labels==1),20,'Normalization','probability');
xline(threshold,'k--','LineWidth',2);
grid on;
xlabel('Drone probability');
ylabel('Relative frequency');
legend('Absent','Present','Validation threshold','Location','best');
title('04 | Class confidence distributions');
saveXAI(f,figDir,'04_Confidence_Distribution');

%% 03. Confusion matrix, threshold sensitivity (diagnostic only)
f = figure('Color','w','Position',[100 100 850 570]);
confusionchart(labels,pred);
title('05 | V4 held-out confusion matrix');
saveXAI(f,figDir,'05_Confusion_Matrix');
thresholdGrid = linspace(0,1,101);
F1grid = zeros(size(thresholdGrid));
BAgrid = zeros(size(thresholdGrid));
for j = 1:numel(thresholdGrid)
    p = scores>=thresholdGrid(j);
    a = sum(p & labels==1);
    b = sum(p & labels==0);
    c = sum(~p & labels==1);
    d = sum(~p & labels==0);
    F1grid(j) = 2*a/max(2*a+b+c,1);
    BAgrid(j) = 0.5*(a/max(a+c,1)+d/max(b+d,1));
end
f = figure('Color','w','Position',[100 100 850 570]);
plot(thresholdGrid,F1grid,'LineWidth',2);
hold on;
plot(thresholdGrid,BAgrid,'LineWidth',2);
xline(threshold,'k--','LineWidth',1.5);
grid on;
xlabel('Threshold (test diagnostic only)');
ylabel('Metric');
legend('F1','Balanced accuracy','Fixed validation threshold','Location','best');
title('06 | Threshold sensitivity (not for threshold tuning)');
saveXAI(f,figDir,'06_Threshold_Sensitivity');

%% 04. Localization diagnostics
positive = labels==1;
f = figure('Color','w','Position',[100 100 850 570]);
histogram(errors(positive),25);
grid on;
xlabel('3D localization error (m)');
ylabel('Drone-present scenes');
title(sprintf('07 | Localization error (mean %.2f m)',mean(errors(positive))));
saveXAI(f,figDir,'07_Localization_Error');
f = figure('Color','w','Position',[100 100 1200 450]);
for d = 1:3
    subplot(1,3,d);
    scatter(xyzTrue(d,positive),xyzPred(d,positive),15,'filled');
    hold on;
    lim = [min([xyzTrue(d,positive),xyzPred(d,positive)]),max([xyzTrue(d,positive),xyzPred(d,positive)])];
    plot(lim,lim,'k--');
    grid on;
    axis equal;
    xlabel('True position (m)');
    ylabel('Predicted position (m)');
    title(sprintf('%c coordinate','X'+d-1));
end
saveXAI(f,figDir,'08_XYZ_Prediction');

%% 05. Geometry prior and attention XAI
f = figure('Color','w','Position',[100 100 850 570]);
imagesc(G);
axis image;
colorbar;
xlabel('Vertical token');
ylabel('Horizontal token');
title('09 | Fixed geometry compatibility prior');
saveXAI(f,figDir,'09_Geometry_Prior');

%% 06. Representative TP / TN / FP / FN cases
categories = {'TP','TN','FP','FN'};
selection = [find(labels==1 & pred==1,1),find(labels==0 & pred==0,1),find(labels==0 & pred==1,1),find(labels==1 & pred==0,1)];
if numel(selection)~=4 || any(selection==0)
    warning('At least one confusion-matrix category has no example.');
end
for c = 1:4
    if c>numel(selection) || selection(c)<1, continue; end
    idx = selection(c);
    D = load(fullfile(files(idx).folder,files(idx).name),'RAD','RED');
    H = reduceXAI(D.RAD);
    V = reduceXAI(D.RED);
    H = max(-8,min(8,(H-stats.meanH)/stats.stdH));
    V = max(-8,min(8,(V-stats.meanV)/stats.stdV));
    XH = dlarray(reshape(single(H),32,16,16,1,1),'SSSCB');
    XV = dlarray(reshape(single(V),32,16,16,1,1),'SSSCB');
    [prob,xyz,attH,attV,CAMH,CAMV,salH,salV] = dlfeval(@explainV4,netH,netV,theta,XH,XV,G,numHeads,positionScale);
    attH = double(gather(extractdata(attH)));
    attV = double(gather(extractdata(attV)));
    CAMH = double(gather(extractdata(CAMH)));
    CAMV = double(gather(extractdata(CAMV)));
    salH = double(gather(extractdata(salH)));
    salV = double(gather(extractdata(salV)));
    f = figure('Color','w','Position',[100 100 1150 470]);
    subplot(1,2,1);
    imagesc(mean(attH,3));
    axis image;
    colorbar;
    xlabel('V key token');
    ylabel('H query token');
    title('H queries V');
    subplot(1,2,2);
    imagesc(mean(attV,3));
    axis image;
    colorbar;
    xlabel('H key token');
    ylabel('V query token');
    title('V queries H');
    sgtitle(sprintf('10 | Bidirectional attention: %s, test %d',categories{c},idx));
    saveXAI(f,figDir,sprintf('10_Attention_%s',categories{c}));
    f = figure('Color','w','Position',[100 100 1150 470]);
    subplot(1,2,1);
    imagesc(squeeze(max(CAMH,[],3)));
    axis image;
    colorbar;
    xlabel('Doppler token');
    ylabel('Range token');
    title('Horizontal encoder Grad-CAM');
    subplot(1,2,2);
    imagesc(squeeze(max(CAMV,[],3)));
    axis image;
    colorbar;
    xlabel('Doppler token');
    ylabel('Range token');
    title('Vertical encoder Grad-CAM');
    sgtitle(sprintf('11 | Last-convolution Grad-CAM: %s',categories{c}));
    saveXAI(f,figDir,sprintf('11_GradCAM_%s',categories{c}));
    f = figure('Color','w','Position',[100 100 1150 470]);
    subplot(1,2,1);
    imagesc(squeeze(max(salH,[],3)));
    axis image;
    colorbar;
    xlabel('Doppler bin');
    ylabel('Range bin');
    title('Horizontal input gradient magnitude');
    subplot(1,2,2);
    imagesc(squeeze(max(salV,[],3)));
    axis image;
    colorbar;
    xlabel('Doppler bin');
    ylabel('Range bin');
    title('Vertical input gradient magnitude');
    sgtitle(sprintf('12 | Input saliency: %s',categories{c}));
    saveXAI(f,figDir,sprintf('12_Input_Saliency_%s',categories{c}));
    entropyH = -sum(attH.*log(max(attH,1e-12)),2)/log(size(attH,2));
    entropyV = -sum(attV.*log(max(attV,1e-12)),2)/log(size(attV,2));
    f = figure('Color','w','Position',[100 100 950 550]);
    bar([squeeze(mean(entropyH,1)),squeeze(mean(entropyV,1))]);
    ylim([0 1]);
    grid on;
    xlabel('Attention head');
    ylabel('Normalized attention entropy');
    legend('H to V','V to H','Location','best');
    title(sprintf('13 | Attention concentration: %s',categories{c}));
    saveXAI(f,figDir,sprintf('13_Attention_Entropy_%s',categories{c}));
    f = figure('Color','w','Position',[100 100 950 550]);
    bar(squeeze(mean(attH,1)));
    grid on;
    xlabel('Key token');
    ylabel('Mean attention weight');
    title(sprintf('14 | Per-head token importance: %s',categories{c}));
    saveXAI(f,figDir,sprintf('14_Token_Importance_%s',categories{c}));
    [occH,occV] = occlusionV4(netH,netV,theta,single(H),single(V),G,numHeads,positionScale);
    f = figure('Color','w','Position',[100 100 1100 480]);
    subplot(1,2,1);
    imagesc(occH);
    axis image;
    colorbar;
    xlabel('Doppler block');
    ylabel('Range block');
    title('Horizontal radar occlusion');
    subplot(1,2,2);
    imagesc(occV);
    axis image;
    colorbar;
    xlabel('Doppler block');
    ylabel('Range block');
    title('Vertical radar occlusion');
    sgtitle(sprintf('15 | Probability drop on occlusion: %s',categories{c}));
    saveXAI(f,figDir,sprintf('15_Occlusion_%s',categories{c}));
    save(fullfile(dataDir,sprintf('Explanation_%s.mat',categories{c})),'idx','attH','attV','CAMH','CAMV','salH','salV','entropyH','entropyV','occH','occV','-v7.3');
    fprintf('Explained %s: test index %d\n',categories{c},idx);
end

%% 07. Paper-ready metrics and provenance
metric = {'Accuracy';'Precision';'Recall';'Specificity';'F1';'FPR';'ROC_AUC';'PR_AP';'Brier';'ECE';'Mean_XYZ_Error_m'};
value = [(TP+TN)/N;precision;tpr;TN/max(TN+FP,1);F1;fpr;aucROC;aucPR;brier;ece;mean(errors(positive))];
writetable(table(metric,value),fullfile(dataDir,'V4_XAI_Metrics.csv'));
save(fullfile(dataDir,'V4_XAI_All_Data.mat'),'labels','scores','pred','xyzTrue','xyzPred','errors','threshold','rocX','rocY','prRecall','prPrecision','aucROC','aucPR','brier','ece','selection');
fid = fopen(fullfile(xaiDir,'README_XAI.txt'),'w');
fprintf(fid,'V4 XAI | Existing trained checkpoint; no retraining or weight changes.\n');
fprintf(fid,'Figures 01-15; representative TP TN FP FN cases when available.\n');
fprintf(fid,'ROC/PR and threshold sensitivity are test-set descriptive diagnostics only.\n');
fprintf(fid,'The decision threshold is the ORIGINAL validation-selected threshold.\n');
fprintf(fid,'Grad-CAM uses gradients of detection logit wrt final CNN encoder activations.\n');
fprintf(fid,'Input saliency is absolute gradient wrt normalized input cube.\n');
fprintf(fid,'Attention heatmaps are association visualizations, not causal proof.\n');
fprintf(fid,'Occlusion uses zeroed normalized blocks, not a physical radar intervention.\n');
fprintf(fid,'The geometry prior is approximate and fixed.\n');
fprintf(fid,'Test-set example selection is qualitative; do not claim representative population statistics.\n');
fclose(fid);
fprintf('\nDONE. XAI figures: %s\n',figDir);

%% LOCAL FUNCTIONS
function X = reduceXAI(A)
    A = single(A);
    assert(isequal(size(A),[64 64 16]),'Input cube must be 64x64x16.');
    A = reshape(A,2,32,4,16,16);
    A = mean(A,1);
    A = mean(A,3);
    X = reshape(A,32,16,16);
end

function [prob,xyz,AH,AV] = forwardXAI(FH,FV,t,G,numHeads,positionScale)
    B = size(FH,5);
    F = size(FH,4);
    T = size(FH,1)*size(FH,2)*size(FH,3);
    H = permute(reshape(FH,T,F,B),[2 1 3]);
    V = permute(reshape(FV,T,F,B),[2 1 3]);
    qH = reshape(t.WqH*reshape(H,F,[]),F,T,B);
    kH = reshape(t.WkH*reshape(H,F,[]),F,T,B);
    vH = reshape(t.WvH*reshape(H,F,[]),F,T,B);
    qV = reshape(t.WqV*reshape(V,F,[]),F,T,B);
    kV = reshape(t.WkV*reshape(V,F,[]),F,T,B);
    vV = reshape(t.WvV*reshape(V,F,[]),F,T,B);
    partsH = cell(1,numHeads);
    partsV = cell(1,numHeads);
    AH = cell(1,numHeads);
    AV = cell(1,numHeads);
    headDim = F/numHeads;
    strength = 0.5+0.5*tanh(t.geomRaw);
    for head = 1:numHeads
        rows = (head-1)*headDim+1:head*headDim;
        logH = pagemtimes(permute(qH(rows,:,:),[2 1 3]),kV(rows,:,:))/sqrt(headDim)+strength*G;
        logV = pagemtimes(permute(qV(rows,:,:),[2 1 3]),kH(rows,:,:))/sqrt(headDim)+strength*G.';
        wH = exp(logH-max(logH,[],2));
        wV = exp(logV-max(logV,[],2));
        wH = wH./(sum(wH,2)+1e-8);
        wV = wV./(sum(wV,2)+1e-8);
        partsH{head} = pagemtimes(vV(rows,:,:),permute(wH,[2 1 3]));
        partsV{head} = pagemtimes(vH(rows,:,:),permute(wV,[2 1 3]));
        AH{head} = wH;
        AV{head} = wV;
    end
    fusedH = H+cat(1,partsH{:});
    fusedV = V+cat(1,partsV{:});
    poolH = 0.5*(mean(fusedH,2)+max(fusedH,[],2));
    poolV = 0.5*(mean(fusedV,2)+max(fusedV,[],2));
    combined = cat(1,reshape(poolH,F,B),reshape(poolV,F,B));
    hidden = tanh(t.Wf*combined+t.bf);
    logit = t.Wcls*hidden+t.bcls;
    prob = 1./(1+exp(-logit));
    xyz = positionScale*(t.Wxyz*hidden+t.bxyz);
    AH = cat(4,AH{:});
    AV = cat(4,AV{:});
    AH = reshape(AH,T,T,B,numHeads);
    AV = reshape(AV,T,T,B,numHeads);
    if B==1
        AH = reshape(AH,T,T,numHeads);
        AV = reshape(AV,T,T,numHeads);
    end
end

function [prob,xyz,AH,AV,CAMH,CAMV,salH,salV] = explainV4(netH,netV,t,XH,XV,G,numHeads,positionScale)
    FH = stripdims(forward(netH,XH));
    FV = stripdims(forward(netV,XV));
    [prob,xyz,AH,AV] = forwardXAI(FH,FV,t,G,numHeads,positionScale);
    logit = log(prob+1e-7)-log(1-prob+1e-7);
    [gradH,gradV,gradXH,gradXV] = dlgradient(sum(logit,'all'),FH,FV,XH,XV);
    FH = stripdims(FH);
    FV = stripdims(FV);
    gradH = stripdims(gradH);
    gradV = stripdims(gradV);
    gradXH = stripdims(gradXH);
    gradXV = stripdims(gradXV);
    alphaH = mean(gradH,[1 2 3]);
    alphaV = mean(gradV,[1 2 3]);
    CAMH = max(sum(FH.*alphaH,4),0);
    CAMV = max(sum(FV.*alphaV,4),0);
    CAMH = CAMH/(max(CAMH,[],'all')+1e-8);
    CAMV = CAMV/(max(CAMV,[],'all')+1e-8);
    salH = abs(gradXH);
    salV = abs(gradXV);
    salH = salH/(max(salH,[],'all')+1e-8);
    salV = salV/(max(salV,[],'all')+1e-8);
end

function p = predictXAI(netH,netV,t,H,V,G,numHeads,positionScale)
    XH = dlarray(reshape(single(H),32,16,16,1,1),'SSSCB');
    XV = dlarray(reshape(single(V),32,16,16,1,1),'SSSCB');
    FH = stripdims(forward(netH,XH));
    FV = stripdims(forward(netV,XV));
    [p,~] = forwardXAI(FH,FV,t,G,numHeads,positionScale);
    p = double(gather(extractdata(p)));
end

function [mapH,mapV] = occlusionV4(netH,netV,t,H,V,G,numHeads,positionScale)
    original = predictXAI(netH,netV,t,H,V,G,numHeads,positionScale);
    mapH = zeros(4,2);
    mapV = zeros(4,2);
    for r = 1:4
        for d = 1:2
            rows = (r-1)*8+1:r*8;
            cols = (d-1)*8+1:d*8;
            h = H;
            h(rows,cols,:) = 0;
            mapH(r,d) = original-predictXAI(netH,netV,t,h,V,G,numHeads,positionScale);
            v = V;
            v(rows,cols,:) = 0;
            mapV(r,d) = original-predictXAI(netH,netV,t,H,v,G,numHeads,positionScale);
        end
    end
end

function saveXAI(f,folder,name)
    savefig(f,fullfile(folder,[name '.fig']));
    exportgraphics(f,fullfile(folder,[name '.png']),'Resolution',300);
    exportgraphics(f,fullfile(folder,[name '.pdf']),'ContentType','vector');
    close(f);
    fprintf('Saved %s\n',name);
end
