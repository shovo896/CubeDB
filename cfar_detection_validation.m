
clear;
clc;
close all;

%% 1. Load Configuration

S = load('CubeDN_dataset/setup_config.mat','cfg');
cfg = S.cfg;

V = load('CubeDN_dataset/visibility_test.mat');

assert(isfield(V,'rawPresentH'),'Missing rawPresentH.');
assert(isfield(V,'rawPresentV'),'Missing rawPresentV.');
assert(isfield(V,'rawAbsentH'),'Missing rawAbsentH.');
assert(isfield(V,'rawAbsentV'),'Missing rawAbsentV.');


%% 2. Radar Parameters

c = 299792458;
lambda = c/cfg.fc;

Ns = cfg.numSamples;
Nc = cfg.numChirps;

fs = cfg.fs;
slope = cfg.slope;
Tchirp = cfg.Tchirp;

Nrange = 512;
Ndoppler = 128;

assert(Ndoppler>=Nc,'Doppler FFT must not truncate chirps.');

%% 3. CFAR Configuration

trainingRange = 10;
trainingDoppler = 8;

guardRange = 3;
guardDoppler = 2;

desiredPfa = 1e-4;

% Evaluation tolerances, not detector inputs.
rangeToleranceBins = 2;
dopplerToleranceBins = 2;

%% 4. Prepare Radar Axes

rangeAxis = (0:Nrange-1)*fs/Nrange*c/(2*slope);

velocityAxis = ((0:Ndoppler-1)-floor(Ndoppler/2))*lambda/(2*Ndoppler*Tchirp);

% Positive beat-frequency range bins.
validRange = 1:floor(Nrange/2);

rangeAxisValid = rangeAxis(validRange);

%% 5. Prepare Windows

rangeWindow = 0.5-0.5*cos(2*pi*(0:Ns-1)'/Ns);

dopplerWindow = 0.5-0.5*cos(2*pi*(0:Nc-1)/Nc);

window2D = single(rangeWindow*dopplerWindow);

%% 6. Load IQ Signals

rawPresentH = V.rawPresentH;
rawPresentV = V.rawPresentV;

rawAbsentH = V.rawAbsentH;
rawAbsentV = V.rawAbsentV;

%% 7. Range-Doppler Processing

disp('Processing Range-Doppler maps...');

% First antenna is used for this baseline detector.
% Antenna beamforming is a later development stage.

iqPH = rawPresentH(:,:,1).*window2D;
iqPV = rawPresentV(:,:,1).*window2D;

iqAH = rawAbsentH(:,:,1).*window2D;
iqAV = rawAbsentV(:,:,1).*window2D;

fftPH = fftshift(fft(fft(iqPH,Nrange,1),Ndoppler,2),2);
fftPV = fftshift(fft(fft(iqPV,Nrange,1),Ndoppler,2),2);

fftAH = fftshift(fft(fft(iqAH,Nrange,1),Ndoppler,2),2);
fftAV = fftshift(fft(fft(iqAV,Nrange,1),Ndoppler,2),2);

powerPH = abs(double(fftPH(validRange,:))).^2;
powerPV = abs(double(fftPV(validRange,:))).^2;

powerAH = abs(double(fftAH(validRange,:))).^2;
powerAV = abs(double(fftAV(validRange,:))).^2;

%% 8. Construct CFAR Training Kernel

outerRange = trainingRange+guardRange;
outerDoppler = trainingDoppler+guardDoppler;

kernelRows = 2*outerRange+1;
kernelCols = 2*outerDoppler+1;

trainingKernel = ones(kernelRows,kernelCols);

centerRow = outerRange+1;
centerCol = outerDoppler+1;

trainingKernel(centerRow-guardRange:centerRow+guardRange,centerCol-guardDoppler:centerCol+guardDoppler) = 0;

numTrainingCells = sum(trainingKernel(:));

assert(numTrainingCells>0,'Invalid CFAR training configuration.');

%% 9. CA-CFAR Threshold Multiplier

% The following alpha assumes independent exponential
% training-cell power samples.
% Correlated FFT bins and nonhomogeneous clutter
% can invalidate nominal Pfa calibration.

alpha = numTrainingCells*(desiredPfa^(-1/numTrainingCells)-1);

fprintf('Training Cells: %d\n',numTrainingCells);
fprintf('CFAR Alpha: %.4f\n',alpha);

%% 10. Calculate CFAR Thresholds

% Convolution computes sums of neighboring training cells.

noiseSumPH = conv2(powerPH,trainingKernel,'same');
noiseSumPV = conv2(powerPV,trainingKernel,'same');

noiseSumAH = conv2(powerAH,trainingKernel,'same');
noiseSumAV = conv2(powerAV,trainingKernel,'same');

thresholdPH = alpha*noiseSumPH/numTrainingCells;
thresholdPV = alpha*noiseSumPV/numTrainingCells;

thresholdAH = alpha*noiseSumAH/numTrainingCells;
thresholdAV = alpha*noiseSumAV/numTrainingCells;

%% 11. Valid CFAR Region

[numR,numD] = size(powerPH);

validMask = false(numR,numD);

rStart = outerRange+1;
rEnd = numR-outerRange;

dStart = outerDoppler+1;
dEnd = numD-outerDoppler;

assert(rStart<=rEnd,'Insufficient range cells.');
assert(dStart<=dEnd,'Insufficient Doppler cells.');

validMask(rStart:rEnd,dStart:dEnd) = true;

%% 12. Generate CFAR Detection Maps

detectPH = powerPH>thresholdPH & validMask;
detectPV = powerPV>thresholdPV & validMask;

detectAH = powerAH>thresholdAH & validMask;
detectAV = powerAV>thresholdAV & validMask;

%% 13. Extract Detection Coordinates

[rPH,dPH] = find(detectPH);
[rPV,dPV] = find(detectPV);

[rAH,dAH] = find(detectAH);
[rAV,dAV] = find(detectAV);

fprintf('\nDetection Counts\n');

fprintf('Present H: %d\n',numel(rPH));
fprintf('Present V: %d\n',numel(rPV));

fprintf('Absent H: %d\n',numel(rAH));
fprintf('Absent V: %d\n',numel(rAV));

%% 14. Ground Truth for Evaluation Only

dronePos = V.visibilityReport.dronePosition;
droneVel = V.visibilityReport.droneVelocity;

relH = dronePos-cfg.radarH.position;
relV = dronePos-cfg.radarV.position;

trueRangeH = norm(relH);
trueRangeV = norm(relV);

trueVelocityH = dot(droneVel,relH)/trueRangeH;
trueVelocityV = dot(droneVel,relV)/trueRangeV;

[~,trueRH] = min(abs(rangeAxisValid-trueRangeH));
[~,trueRV] = min(abs(rangeAxisValid-trueRangeV));

[~,trueDH] = min(abs(velocityAxis-trueVelocityH));
[~,trueDV] = min(abs(velocityAxis-trueVelocityV));

%% 15. Match Detections to Ground Truth

matchPH = abs(rPH-trueRH)<=rangeToleranceBins & abs(dPH-trueDH)<=dopplerToleranceBins;
matchPV = abs(rPV-trueRV)<=rangeToleranceBins & abs(dPV-trueDV)<=dopplerToleranceBins;

matchAH = abs(rAH-trueRH)<=rangeToleranceBins & abs(dAH-trueDH)<=dopplerToleranceBins;
matchAV = abs(rAV-trueRV)<=rangeToleranceBins & abs(dAV-trueDV)<=dopplerToleranceBins;

droneDetectedH = any(matchPH);
droneDetectedV = any(matchPV);

backgroundMatchH = any(matchAH);
backgroundMatchV = any(matchAV);

%% 16. Count Detections Outside Target Region

falseCandidatePH = sum(~matchPH);
falseCandidatePV = sum(~matchPV);

falseCandidateAH = numel(rAH);
falseCandidateAV = numel(rAV);

% Present-scene detections outside the target region
% are candidates, not confirmed false alarms,
% because background scatterers are present.

%% 17. Empirical Absent-Scene Exceedance Fraction

numTestCells = nnz(validMask);

absentExceedanceH = nnz(detectAH)/numTestCells;
absentExceedanceV = nnz(detectAV)/numTestCells;



%% 18. Print Detection Results



fprintf('Drone Detected H: %d\n',droneDetectedH);
fprintf('Drone Detected V: %d\n',droneDetectedV);

fprintf('Background Match H: %d\n',backgroundMatchH);
fprintf('Background Match V: %d\n',backgroundMatchV);

fprintf('Absent Exceedance H: %.8f\n',absentExceedanceH);
fprintf('Absent Exceedance V: %.8f\n',absentExceedanceV);

%% 19. Horizontal Radar Visualization

figure('Name','CubeDN CFAR - Horizontal Radar');

subplot(1,3,1);

imagesc(velocityAxis,rangeAxisValid,10*log10(powerPH+eps));
axis xy;
colorbar;

xlabel('Velocity (m/s)');
ylabel('Range (m)');
title('Drone Present');

hold on;
plot(trueVelocityH,trueRangeH,'wo','MarkerSize',9,'LineWidth',2);

subplot(1,3,2);

imagesc(velocityAxis,rangeAxisValid,double(detectPH));
axis xy;
colorbar;

xlabel('Velocity (m/s)');
ylabel('Range (m)');
title('CFAR Present');

hold on;
plot(trueVelocityH,trueRangeH,'ro','MarkerSize',9,'LineWidth',2);

subplot(1,3,3);

imagesc(velocityAxis,rangeAxisValid,double(detectAH));
axis xy;
colorbar;

xlabel('Velocity (m/s)');
ylabel('Range (m)');
title('CFAR Drone Absent');

%% 20. Vertical Radar Visualization

figure('Name','CubeDN CFAR - Vertical Radar');

subplot(1,3,1);

imagesc(velocityAxis,rangeAxisValid,10*log10(powerPV+eps));
axis xy;
colorbar;

xlabel('Velocity (m/s)');
ylabel('Range (m)');
title('Drone Present');

hold on;
plot(trueVelocityV,trueRangeV,'wo','MarkerSize',9,'LineWidth',2);

subplot(1,3,2);

imagesc(velocityAxis,rangeAxisValid,double(detectPV));
axis xy;
colorbar;

xlabel('Velocity (m/s)');
ylabel('Range (m)');
title('CFAR Present');

hold on;
plot(trueVelocityV,trueRangeV,'ro','MarkerSize',9,'LineWidth',2);

subplot(1,3,3);

imagesc(velocityAxis,rangeAxisValid,double(detectAV));
axis xy;
colorbar;

xlabel('Velocity (m/s)');
ylabel('Range (m)');
title('CFAR Drone Absent');

%% 21. Store Validation Report

cfarReport.frequency_Hz = cfg.fc;

cfarReport.desiredPfa = desiredPfa;
cfarReport.alpha = alpha;

cfarReport.numTrainingCells = numTrainingCells;
cfarReport.numTestCells = numTestCells;

cfarReport.trainingRange = trainingRange;
cfarReport.trainingDoppler = trainingDoppler;

cfarReport.guardRange = guardRange;
cfarReport.guardDoppler = guardDoppler;

cfarReport.droneDetectedH = droneDetectedH;
cfarReport.droneDetectedV = droneDetectedV;

cfarReport.backgroundMatchH = backgroundMatchH;
cfarReport.backgroundMatchV = backgroundMatchV;

cfarReport.presentDetectionsH = numel(rPH);
cfarReport.presentDetectionsV = numel(rPV);

cfarReport.absentDetectionsH = numel(rAH);
cfarReport.absentDetectionsV = numel(rAV);

cfarReport.falseCandidatePresentH = falseCandidatePH;
cfarReport.falseCandidatePresentV = falseCandidatePV;

cfarReport.falseCandidateAbsentH = falseCandidateAH;
cfarReport.falseCandidateAbsentV = falseCandidateAV;

cfarReport.absentExceedanceH = absentExceedanceH;
cfarReport.absentExceedanceV = absentExceedanceV;

cfarReport.trueRangeH = trueRangeH;
cfarReport.trueRangeV = trueRangeV;

cfarReport.trueVelocityH = trueVelocityH;
cfarReport.trueVelocityV = trueVelocityV;

cfarReport.validationType = 'Single-pair synthetic CFAR baseline';
cfarReport.industryValidated = false;

%% 22. Save Results

outputFolder = cfg.dataset.outputFolder;

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

outputFile = fullfile(outputFolder,'cfar_validation.mat');

save(outputFile,'cfarReport','detectPH','detectPV','detectAH','detectAV','thresholdPH','thresholdPV','thresholdAH','thresholdAV','rangeAxisValid','velocityAxis','-v7.3');

%% 23. Final Summary


fprintf('Saved: %s\n',outputFile);

disp('Single-pair CFAR validation completed.');
disp('Monte Carlo Pd/Pfa validation still required.');