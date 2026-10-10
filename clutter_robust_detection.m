clear;
clc;
close all;

%% 1. Load Configuration and IQ

S = load('CubeDN_dataset/setup_config.mat','cfg');
cfg = S.cfg;

V = load('CubeDN_dataset/visibility_test.mat');

assert(isfield(V,'rawPresentH'),'Missing rawPresentH.');
assert(isfield(V,'rawAbsentH'),'Missing rawAbsentH.');
assert(isfield(V,'rawPresentV'),'Missing rawPresentV.');
assert(isfield(V,'rawAbsentV'),'Missing rawAbsentV.');
assert(isfield(V,'visibilityReport'),'Missing visibilityReport.');

%% 2. Radar Parameters

c = 299792458;
lambda = c/cfg.fc;

Ns = cfg.numSamples;
Nc = cfg.numChirps;
Na = cfg.numVirtualAntennas;

fs = cfg.fs;
slope = cfg.slope;
Tchirp = cfg.Tchirp;
d = cfg.antennaSpacing;

Nrange = 512;
Ndoppler = 128;

Nr = floor(Nrange/2);

assert(Na>=2,'At least two antennas are required.');
assert(size(V.rawPresentH,3)==Na,'Horizontal antenna count mismatch.');
assert(size(V.rawPresentV,3)==Na,'Vertical antenna count mismatch.');

%% 3. Detection Parameters

trainingR = 10;
trainingD = 8;

guardR = 3;
guardD = 2;

osRankFraction = 0.75;
nominalPfa = 1e-4;

angleGrid = -60:2:60;

minimumSpatialGain_dB = 3;

rangeAgreementBins = 2;
velocityAgreementBins = 2;

truthRangeToleranceBins = 2;
truthVelocityToleranceBins = 2;

%% 4. Radar Axes

rangeAxis = (0:Nr-1)'*fs/Nrange*c/(2*slope);

velocityAxis = ((0:Ndoppler-1)-floor(Ndoppler/2))*lambda/(2*Ndoppler*Tchirp);

%% 5. FFT Windows

rangeWin = 0.5-0.5*cos(2*pi*(0:Ns-1)'/Ns);

dopplerWin = 0.5-0.5*cos(2*pi*(0:Nc-1)/Nc);

window2D = single(rangeWin*dopplerWin);

%% 6. OS-CFAR Training Cells

outerR = trainingR+guardR;
outerD = trainingD+guardD;

offsetR = [];
offsetD = [];

for dr = -outerR:outerR

    for dd = -outerD:outerD

        if abs(dr)<=guardR && abs(dd)<=guardD

            continue;

        end

        offsetR(end+1) = dr;

        offsetD(end+1) = dd;

    end

end

Ntrain = numel(offsetR);

osRank = ceil(osRankFraction*Ntrain);

fprintf('OS-CFAR Training Cells: %d\n',Ntrain);
fprintf('OS-CFAR Rank: %d\n',osRank);

%% 7. OS-CFAR Alpha

% Theoretical independent exponential training-cell model.

targetLogPfa = log(nominalPfa);

lowAlpha = 0;
highAlpha = 1000;

for iter = 1:70

    midAlpha = (lowAlpha+highAlpha)/2;

    logPfa = gammaln(Ntrain+1)+gammaln(Ntrain-osRank+midAlpha+1)-gammaln(Ntrain-osRank+1)-gammaln(Ntrain+midAlpha+1);

    if logPfa>targetLogPfa

        lowAlpha = midAlpha;

    else

        highAlpha = midAlpha;

    end

end

alphaOS = (lowAlpha+highAlpha)/2;

fprintf('OS-CFAR Alpha: %.4f\n',alphaOS);

%% 8. Allocate Results

detectOSPresent = cell(1,2);
detectOSAbsent = cell(1,2);

detectPresent = cell(1,2);
detectAbsent = cell(1,2);

powerPresent = cell(1,2);
powerAbsent = cell(1,2);

anglePresent = cell(1,2);
angleAbsent = cell(1,2);

spatialGainPresent = cell(1,2);
spatialGainAbsent = cell(1,2);

%% 9. Process Horizontal and Vertical Radars

for q = 1:2

    fprintf('\nProcessing Radar %d...\n',q);

    if q==1

        iqPresent = V.rawPresentH;

        iqAbsent = V.rawAbsentH;

    else

        iqPresent = V.rawPresentV;

        iqAbsent = V.rawAbsentV;

    end

    %% 9A. Range-Doppler FFT

    fftPresent = complex(zeros(Nr,Ndoppler,Na,'single'));

    fftAbsent = complex(zeros(Nr,Ndoppler,Na,'single'));

    for ant = 1:Na

        fp = fftshift(fft(fft(iqPresent(:,:,ant).*window2D,Nrange,1),Ndoppler,2),2);

        fa = fftshift(fft(fft(iqAbsent(:,:,ant).*window2D,Nrange,1),Ndoppler,2),2);

        fftPresent(:,:,ant) = single(fp(1:Nr,:));

        fftAbsent(:,:,ant) = single(fa(1:Nr,:));

    end

    %% 9B. Noncoherent Antenna Power

    pPresent = mean(abs(double(fftPresent)).^2,3);

    pAbsent = mean(abs(double(fftAbsent)).^2,3);

    powerPresent{q} = pPresent;

    powerAbsent{q} = pAbsent;

    %% 9C. OS-CFAR

    detectionP = false(Nr,Ndoppler);

    detectionA = false(Nr,Ndoppler);

    for r = outerR+1:Nr-outerR

        for v = outerD+1:Ndoppler-outerD

            valuesP = zeros(1,Ntrain);

            valuesA = zeros(1,Ntrain);

            for k = 1:Ntrain

                rr = r+offsetR(k);

                vv = v+offsetD(k);

                valuesP(k) = pPresent(rr,vv);

                valuesA(k) = pAbsent(rr,vv);

            end

            valuesP = sort(valuesP);

            valuesA = sort(valuesA);

            thresholdP = alphaOS*valuesP(osRank);

            thresholdA = alphaOS*valuesA(osRank);

            detectionP(r,v) = pPresent(r,v)>thresholdP;

            detectionA(r,v) = pAbsent(r,v)>thresholdA;

        end

    end

    detectOSPresent{q} = detectionP;
    detectOSAbsent{q} = detectionA;

    fprintf('OS-CFAR Present: %d\n',nnz(detectionP));
    fprintf('OS-CFAR Absent: %d\n',nnz(detectionA));

    %% 9D. Steering Matrix

    antennaIndex = (0:Na-1)';

    steering = exp(1i*2*pi*d/lambda*antennaIndex*sind(angleGrid));

    %% 9E. Spatial Beamforming

    angleMapP = nan(Nr,Ndoppler);

    angleMapA = nan(Nr,Ndoppler);

    gainMapP = zeros(Nr,Ndoppler);

    gainMapA = zeros(Nr,Ndoppler);

    [rP,vP] = find(detectionP);

    [rA,vA] = find(detectionA);

    for k = 1:numel(rP)

        rr = rP(k);

        vv = vP(k);

        snapshot = reshape(double(fftPresent(rr,vv,:)),Na,1);

        response = abs(steering'*snapshot).^2/Na^2;

        [peak,index] = max(response);

        incoherent = mean(abs(snapshot).^2);

        gainMapP(rr,vv) = 10*log10((Na*peak+realmin)/(incoherent+realmin));

        angleMapP(rr,vv) = angleGrid(index);

    end

    for k = 1:numel(rA)

        rr = rA(k);

        vv = vA(k);

        snapshot = reshape(double(fftAbsent(rr,vv,:)),Na,1);

        response = abs(steering'*snapshot).^2/Na^2;

        [peak,index] = max(response);

        incoherent = mean(abs(snapshot).^2);

        gainMapA(rr,vv) = 10*log10((Na*peak+realmin)/(incoherent+realmin));

        angleMapA(rr,vv) = angleGrid(index);

    end

    %% 9F. Spatial Gain Filter

    detectionP = detectionP & gainMapP>=minimumSpatialGain_dB;

    detectionA = detectionA & gainMapA>=minimumSpatialGain_dB;

    detectPresent{q} = detectionP;
    detectAbsent{q} = detectionA;

    anglePresent{q} = angleMapP;
    angleAbsent{q} = angleMapA;

    spatialGainPresent{q} = gainMapP;
    spatialGainAbsent{q} = gainMapA;

    fprintf('Spatial Filter Present: %d\n',nnz(detectionP));
    fprintf('Spatial Filter Absent: %d\n',nnz(detectionA));

end

%% 10. Dual Radar Agreement

jointPresentH = false(Nr,Ndoppler);
jointPresentV = false(Nr,Ndoppler);

jointAbsentH = false(Nr,Ndoppler);
jointAbsentV = false(Nr,Ndoppler);

for r = 1:Nr

    rLow = max(1,r-rangeAgreementBins);
    rHigh = min(Nr,r+rangeAgreementBins);

    for v = 1:Ndoppler

        vLow = max(1,v-velocityAgreementBins);
        vHigh = min(Ndoppler,v+velocityAgreementBins);

        if detectPresent{1}(r,v)

            jointPresentH(r,v) = any(detectPresent{2}(rLow:rHigh,vLow:vHigh),'all');

        end

        if detectPresent{2}(r,v)

            jointPresentV(r,v) = any(detectPresent{1}(rLow:rHigh,vLow:vHigh),'all');

        end

        if detectAbsent{1}(r,v)

            jointAbsentH(r,v) = any(detectAbsent{2}(rLow:rHigh,vLow:vHigh),'all');

        end

        if detectAbsent{2}(r,v)

            jointAbsentV(r,v) = any(detectAbsent{1}(rLow:rHigh,vLow:vHigh),'all');

        end

    end

end

%% 11. Ground Truth for Evaluation Only

dronePos = V.visibilityReport.dronePosition;
droneVel = V.visibilityReport.droneVelocity;

radarPositions = [cfg.radarH.position,cfg.radarV.position];

truthRange = zeros(1,2);
truthVelocity = zeros(1,2);

for q = 1:2

    rel = dronePos-radarPositions(:,q);

    truthRange(q) = norm(rel);

    truthVelocity(q) = dot(droneVel,rel)/truthRange(q);

end

%% 12. Ground Truth Neighborhood Test

hitPresent = false(1,2);
hitAbsent = false(1,2);

hitJointPresent = false(1,2);
hitJointAbsent = false(1,2);

for q = 1:2

    [~,rTruth] = min(abs(rangeAxis-truthRange(q)));

    [~,vTruth] = min(abs(velocityAxis-truthVelocity(q)));

    rr = max(1,rTruth-truthRangeToleranceBins):min(Nr,rTruth+truthRangeToleranceBins);

    vv = max(1,vTruth-truthVelocityToleranceBins):min(Ndoppler,vTruth+truthVelocityToleranceBins);

    hitPresent(q) = any(detectPresent{q}(rr,vv),'all');

    hitAbsent(q) = any(detectAbsent{q}(rr,vv),'all');

    if q==1

        hitJointPresent(q) = any(jointPresentH(rr,vv),'all');

        hitJointAbsent(q) = any(jointAbsentH(rr,vv),'all');

    else

        hitJointPresent(q) = any(jointPresentV(rr,vv),'all');

        hitJointAbsent(q) = any(jointAbsentV(rr,vv),'all');

    end

end

%% 13. Create Report

robustReport.frequency_Hz = cfg.fc;

robustReport.nominalPfa = nominalPfa;

robustReport.osRank = osRank;
robustReport.osAlpha = alphaOS;

robustReport.minimumSpatialGain_dB = minimumSpatialGain_dB;

robustReport.osPresentH = nnz(detectOSPresent{1});
robustReport.osPresentV = nnz(detectOSPresent{2});

robustReport.osAbsentH = nnz(detectOSAbsent{1});
robustReport.osAbsentV = nnz(detectOSAbsent{2});

robustReport.presentDetectionsH = nnz(detectPresent{1});
robustReport.presentDetectionsV = nnz(detectPresent{2});

robustReport.absentDetectionsH = nnz(detectAbsent{1});
robustReport.absentDetectionsV = nnz(detectAbsent{2});

robustReport.jointPresentH = nnz(jointPresentH);
robustReport.jointPresentV = nnz(jointPresentV);

robustReport.jointAbsentH = nnz(jointAbsentH);
robustReport.jointAbsentV = nnz(jointAbsentV);

robustReport.droneHitH = hitPresent(1);
robustReport.droneHitV = hitPresent(2);

robustReport.absentNearTruthH = hitAbsent(1);
robustReport.absentNearTruthV = hitAbsent(2);

robustReport.jointDroneHitH = hitJointPresent(1);
robustReport.jointDroneHitV = hitJointPresent(2);

robustReport.jointAbsentNearTruthH = hitJointAbsent(1);
robustReport.jointAbsentNearTruthV = hitJointAbsent(2);

robustReport.industryValidated = false;

robustReport.notes = 'Corrected spatial gain OS-CFAR single-pair diagnostic. Dual radar association is range-velocity only.';

%% 14. Print Summary



fprintf('OS Present H: %d\n',robustReport.osPresentH);
fprintf('OS Absent H: %d\n',robustReport.osAbsentH);

fprintf('OS Present V: %d\n',robustReport.osPresentV);
fprintf('OS Absent V: %d\n',robustReport.osAbsentV);

fprintf('Filtered Present H: %d\n',robustReport.presentDetectionsH);
fprintf('Filtered Absent H: %d\n',robustReport.absentDetectionsH);

fprintf('Filtered Present V: %d\n',robustReport.presentDetectionsV);
fprintf('Filtered Absent V: %d\n',robustReport.absentDetectionsV);

fprintf('Joint Present H: %d\n',robustReport.jointPresentH);
fprintf('Joint Absent H: %d\n',robustReport.jointAbsentH);

fprintf('Joint Present V: %d\n',robustReport.jointPresentV);
fprintf('Joint Absent V: %d\n',robustReport.jointAbsentV);

fprintf('Drone Hit H: %d\n',robustReport.droneHitH);
fprintf('Drone Hit V: %d\n',robustReport.droneHitV);

fprintf('Joint Drone Hit H: %d\n',robustReport.jointDroneHitH);
fprintf('Joint Drone Hit V: %d\n',robustReport.jointDroneHitV);

fprintf('Joint Absent Near Truth H: %d\n',robustReport.jointAbsentNearTruthH);
fprintf('Joint Absent Near Truth V: %d\n',robustReport.jointAbsentNearTruthV);

%% 15. Horizontal Visualization

figure('Name','CubeDN Corrected Robust CFAR Horizontal');

subplot(1,3,1);

imagesc(velocityAxis,rangeAxis,10*log10(powerPresent{1}+eps));
axis xy;
colorbar;
title('Horizontal RD Power');
xlabel('Velocity (m/s)');
ylabel('Range (m)');

subplot(1,3,2);

imagesc(velocityAxis,rangeAxis,double(detectPresent{1}));
axis xy;
colorbar;
title('OS-CFAR + Spatial');
xlabel('Velocity (m/s)');
ylabel('Range (m)');

subplot(1,3,3);

imagesc(velocityAxis,rangeAxis,double(jointPresentH));
axis xy;
colorbar;
title('Dual Radar Agreement');
xlabel('Velocity (m/s)');
ylabel('Range (m)');

%% 16. Vertical Visualization

figure('Name','CubeDN Corrected Robust CFAR Vertical');

subplot(1,3,1);

imagesc(velocityAxis,rangeAxis,10*log10(powerPresent{2}+eps));
axis xy;
colorbar;
title('Vertical RD Power');
xlabel('Velocity (m/s)');
ylabel('Range (m)');

subplot(1,3,2);

imagesc(velocityAxis,rangeAxis,double(detectPresent{2}));
axis xy;
colorbar;
title('OS-CFAR + Spatial');
xlabel('Velocity (m/s)');
ylabel('Range (m)');

subplot(1,3,3);

imagesc(velocityAxis,rangeAxis,double(jointPresentV));
axis xy;
colorbar;
title('Dual Radar Agreement');
xlabel('Velocity (m/s)');
ylabel('Range (m)');

%% 17. Save Results

outputFolder = cfg.dataset.outputFolder;

if ~exist(outputFolder,'dir')

    mkdir(outputFolder);

end

outputFile = fullfile(outputFolder,'clutter_robust_validation.mat');

save(outputFile,'robustReport','detectOSPresent','detectOSAbsent','detectPresent','detectAbsent','jointPresentH','jointPresentV','jointAbsentH','jointAbsentV','anglePresent','angleAbsent','spatialGainPresent','spatialGainAbsent','rangeAxis','velocityAxis','-v7.3');

fprintf('\nSaved: %s\n',outputFile);

disp('Corrected clutter robust detection completed.');