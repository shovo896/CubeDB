

clear;
clc;
close all;

%% 1. Load Configuration

S = load('CubeDN_dataset/setup_config.mat','cfg');
cfg = S.cfg;

C = load('CubeDN_dataset/radar_calibration.mat','calibration');
calibration = C.calibration;

assert(abs(cfg.fc-77e9)<1e8,'Expected 77 GHz radar.');
assert(cfg.drone.RCS>0,'Drone RCS must be positive.');

rng(2026);

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

tFast = (0:Ns-1)'/fs;
tSlow = (0:Nc-1)*Tchirp;

antennaOffsets = ((0:Na-1)-(Na-1)/2)*d;

%% 3. Radar Positions

radarPositions = [cfg.radarH.position,cfg.radarV.position];

arrayAxes = [1 0;0 0;0 1];

dronePos = cfg.drone.position;
droneVel = cfg.drone.velocity;

%% 4. Physical Receiver Parameters

kB = 1.380649e-23;
T0 = 290;

Pt = 10^((calibration.txPower_dBm-30)/10);
Gt = 10^(calibration.txGain_dBi/10);
Gr = 10^(calibration.rxGain_dBi/10);

NF = 10^(calibration.noiseFigure_dB/10);
L = 10^(calibration.systemLoss_dB/10);

Bnoise = calibration.noiseBandwidth_Hz;

noisePower = kB*T0*Bnoise*NF;

assert(noisePower>0,'Invalid receiver noise power.');

% IQ is expressed in sqrt(watts).
% ADC voltage/count scaling is not modeled.

noiseStd = sqrt(noisePower/2);

%% 5. Controlled Background Scene

numGround = 35;
numBuildings = 12;
numTrees = 20;
numMoving = 5;

groundPos = [-10+20*rand(1,numGround);2+18*rand(1,numGround);zeros(1,numGround)];

buildingPos = [-8+16*rand(1,numBuildings);5+15*rand(1,numBuildings);1+7*rand(1,numBuildings)];

treePos = [-10+20*rand(1,numTrees);3+17*rand(1,numTrees);0.5+5*rand(1,numTrees)];

movingPos = [-6+12*rand(1,numMoving);4+12*rand(1,numMoving);0.3+1.5*rand(1,numMoving)];

movingVel = [-0.5+rand(1,numMoving);-0.5+rand(1,numMoving);zeros(1,numMoving)];

%% 6. Background RCS

groundRCS = 10.^((-35+15*rand(1,numGround))/10);
buildingRCS = 10.^((-15+20*rand(1,numBuildings))/10);
treeRCS = 10.^((-30+15*rand(1,numTrees))/10);
movingRCS = 10.^((-25+15*rand(1,numMoving))/10);

backgroundPos = [groundPos,buildingPos,treePos,movingPos];

backgroundVel = [zeros(3,numGround),zeros(3,numBuildings),zeros(3,numTrees),movingVel];

backgroundRCS = [groundRCS,buildingRCS,treeRCS,movingRCS];

numBackground = size(backgroundPos,2);

%% 7. Initialize IQ Arrays

rawPresentH = complex(zeros(Ns,Nc,Na,'single'));
rawPresentV = complex(zeros(Ns,Nc,Na,'single'));

rawAbsentH = complex(zeros(Ns,Nc,Na,'single'));
rawAbsentV = complex(zeros(Ns,Nc,Na,'single'));

targetOnlyH = complex(zeros(Ns,Nc,Na,'single'));
targetOnlyV = complex(zeros(Ns,Nc,Na,'single'));

backgroundH = complex(zeros(Ns,Nc,Na,'single'));
backgroundV = complex(zeros(Ns,Nc,Na,'single'));

noiseOnlyH = complex(zeros(Ns,Nc,Na,'single'));
noiseOnlyV = complex(zeros(Ns,Nc,Na,'single'));

%% 8. Generate Matched Background

disp('Generating matched background...');

for radarIdx = 1:2

    radarPos = radarPositions(:,radarIdx);
    arrayAxis = arrayAxes(:,radarIdx);

    backgroundIQ = complex(zeros(Ns,Nc,Na,'single'));

    for targetIdx = 1:numBackground

        p0 = backgroundPos(:,targetIdx);
        v = backgroundVel(:,targetIdx);
        rcs = backgroundRCS(targetIdx);

        phase0 = 2*pi*rand;

        for chirpIdx = 1:Nc

            currentPos = p0+v*tSlow(chirpIdx);
            rel = currentPos-radarPos;
            R = norm(rel);

            if R<0.1
                continue;
            end

            u = rel/R;

            gainPattern = max(0.05,u(2))^2;

            Pr = Pt*Gt*Gr*lambda^2*rcs*gainPattern^2/((4*pi)^3*R^4*L);

            amplitude = sqrt(max(Pr,0));

            beat = 2*slope*R/c;

            for antIdx = 1:Na

                spatialPhase = 2*pi/lambda*antennaOffsets(antIdx)*dot(u,arrayAxis);

                phase = 2*pi*beat*tFast+4*pi*R/lambda+spatialPhase+phase0;

                echo = amplitude*exp(1j*phase);

                backgroundIQ(:,chirpIdx,antIdx) = backgroundIQ(:,chirpIdx,antIdx)+single(echo);

            end

        end

    end

    if radarIdx==1
        backgroundH = backgroundIQ;
    else
        backgroundV = backgroundIQ;
    end

end

%% 9. Generate Drone-Only Echo

disp('Generating drone-only echo...');

for radarIdx = 1:2

    radarPos = radarPositions(:,radarIdx);
    arrayAxis = arrayAxes(:,radarIdx);

    targetIQ = complex(zeros(Ns,Nc,Na,'single'));

    phase0 = pi/4;

    for chirpIdx = 1:Nc

        currentPos = dronePos+droneVel*tSlow(chirpIdx);

        rel = currentPos-radarPos;
        R = norm(rel);
        u = rel/R;

        gainPattern = max(0.05,u(2))^2;

        Pr = Pt*Gt*Gr*lambda^2*cfg.drone.RCS*gainPattern^2/((4*pi)^3*R^4*L);

        amplitude = sqrt(max(Pr,0));

        beat = 2*slope*R/c;

        for antIdx = 1:Na

            spatialPhase = 2*pi/lambda*antennaOffsets(antIdx)*dot(u,arrayAxis);

            phase = 2*pi*beat*tFast+4*pi*R/lambda+spatialPhase+phase0;

            echo = amplitude*exp(1j*phase);

            targetIQ(:,chirpIdx,antIdx) = targetIQ(:,chirpIdx,antIdx)+single(echo);

        end

    end

    if radarIdx==1
        targetOnlyH = targetIQ;
    else
        targetOnlyV = targetIQ;
    end

end

%% 10. Generate Common Receiver Noise

disp('Generating common receiver noise...');

noiseOnlyH = single(noiseStd)*complex(randn(Ns,Nc,Na,'single'),randn(Ns,Nc,Na,'single'));

noiseOnlyV = single(noiseStd)*complex(randn(Ns,Nc,Na,'single'),randn(Ns,Nc,Na,'single'));

%% 11. Create Paired IQ

rawAbsentH = backgroundH+noiseOnlyH;
rawAbsentV = backgroundV+noiseOnlyV;

rawPresentH = rawAbsentH+targetOnlyH;
rawPresentV = rawAbsentV+targetOnlyV;

%% 12. Verify Pair Consistency

differenceH = rawPresentH-rawAbsentH;
differenceV = rawPresentV-rawAbsentV;

pairErrorH = norm(double(differenceH(:))-double(targetOnlyH(:)))/max(norm(double(targetOnlyH(:))),eps);

pairErrorV = norm(double(differenceV(:))-double(targetOnlyV(:)))/max(norm(double(targetOnlyV(:))),eps);

fprintf('Pair Error H: %.6g\n',pairErrorH);
fprintf('Pair Error V: %.6g\n',pairErrorV);

%% 13. Input Power Ratios

targetPowerH = mean(abs(double(targetOnlyH(:))).^2);
targetPowerV = mean(abs(double(targetOnlyV(:))).^2);

clutterPowerH = mean(abs(double(backgroundH(:))).^2);
clutterPowerV = mean(abs(double(backgroundV(:))).^2);

measuredNoiseH = mean(abs(double(noiseOnlyH(:))).^2);
measuredNoiseV = mean(abs(double(noiseOnlyV(:))).^2);

inputSNR_H = 10*log10((targetPowerH+realmin)/(measuredNoiseH+realmin));
inputSNR_V = 10*log10((targetPowerV+realmin)/(measuredNoiseV+realmin));

inputSCR_H = 10*log10((targetPowerH+realmin)/(clutterPowerH+realmin));
inputSCR_V = 10*log10((targetPowerV+realmin)/(clutterPowerV+realmin));

inputSCNR_H = 10*log10((targetPowerH+realmin)/(clutterPowerH+measuredNoiseH+realmin));
inputSCNR_V = 10*log10((targetPowerV+realmin)/(clutterPowerV+measuredNoiseV+realmin));

%% 14. Range-Doppler Processing

Nrange = 512;
Ndoppler = 128;

rangeWindow = single(hann(Ns,'periodic'));
dopplerWindow = single(hann(Nc,'periodic')).';

window2D = rangeWindow*dopplerWindow;

RDpresentH = zeros(Nrange,Ndoppler,'single');
RDpresentV = zeros(Nrange,Ndoppler,'single');

RDabsentH = zeros(Nrange,Ndoppler,'single');
RDabsentV = zeros(Nrange,Ndoppler,'single');

RDtargetH = zeros(Nrange,Ndoppler,'single');
RDtargetV = zeros(Nrange,Ndoppler,'single');

for radarIdx = 1:2

    if radarIdx==1
        presentIQ = rawPresentH;
        absentIQ = rawAbsentH;
        targetIQ = targetOnlyH;
    else
        presentIQ = rawPresentV;
        absentIQ = rawAbsentV;
        targetIQ = targetOnlyV;
    end

    presentFFT = fftshift(fft(fft(presentIQ(:,:,1).*window2D,Nrange,1),Ndoppler,2),2);

    absentFFT = fftshift(fft(fft(absentIQ(:,:,1).*window2D,Nrange,1),Ndoppler,2),2);

    targetFFT = fftshift(fft(fft(targetIQ(:,:,1).*window2D,Nrange,1),Ndoppler,2),2);

    if radarIdx==1
        RDpresentH = abs(presentFFT).^2;
        RDabsentH = abs(absentFFT).^2;
        RDtargetH = abs(targetFFT).^2;
    else
        RDpresentV = abs(presentFFT).^2;
        RDabsentV = abs(absentFFT).^2;
        RDtargetV = abs(targetFFT).^2;
    end

end

%% 15. Radar Axes

rangeAxis = (0:Nrange-1)*fs/Nrange*c/(2*slope);

dopplerFrequency = ((0:Ndoppler-1)-floor(Ndoppler/2))/(Ndoppler*Tchirp);

velocityAxis = dopplerFrequency*lambda/2;

validRange = 1:floor(Nrange/2);

%% 16. Expected Drone Position

relH = dronePos-radarPositions(:,1);
relV = dronePos-radarPositions(:,2);

trueRangeH = norm(relH);
trueRangeV = norm(relV);

trueVelocityH = dot(droneVel,relH)/trueRangeH;
trueVelocityV = dot(droneVel,relV)/trueRangeV;

[~,rangeIdxH] = min(abs(rangeAxis-trueRangeH));
[~,rangeIdxV] = min(abs(rangeAxis-trueRangeV));

[~,dopplerIdxH] = min(abs(velocityAxis-trueVelocityH));
[~,dopplerIdxV] = min(abs(velocityAxis-trueVelocityV));

%% 17. Target-Bin Power Ratios

targetBinPowerH = RDtargetH(rangeIdxH,dopplerIdxH);
targetBinPowerV = RDtargetV(rangeIdxV,dopplerIdxV);

backgroundBinPowerH = RDabsentH(rangeIdxH,dopplerIdxH);
backgroundBinPowerV = RDabsentV(rangeIdxV,dopplerIdxV);

targetBinContrastH = 10*log10((double(targetBinPowerH)+realmin)/(double(backgroundBinPowerH)+realmin));
targetBinContrastV = 10*log10((double(targetBinPowerV)+realmin)/(double(backgroundBinPowerV)+realmin));

%% 18. Display Results

fprintf('Input SNR H: %.2f dB\n',inputSNR_H);
fprintf('Input SNR V: %.2f dB\n',inputSNR_V);

fprintf('Input SCR H: %.2f dB\n',inputSCR_H);
fprintf('Input SCR V: %.2f dB\n',inputSCR_V);

fprintf('Input SCNR H: %.2f dB\n',inputSCNR_H);
fprintf('Input SCNR V: %.2f dB\n',inputSCNR_V);

fprintf('Target Bin Contrast H: %.2f dB\n',targetBinContrastH);
fprintf('Target Bin Contrast V: %.2f dB\n',targetBinContrastV);

%% 19. Horizontal Radar Visualization

figure('Name','CubeDN Paired Visibility - Horizontal');

subplot(1,3,1);
imagesc(velocityAxis,rangeAxis(validRange),10*log10(double(RDpresentH(validRange,:))+eps));
axis xy;
colorbar;
title('Drone Present');
xlabel('Velocity (m/s)');
ylabel('Range (m)');

subplot(1,3,2);
imagesc(velocityAxis,rangeAxis(validRange),10*log10(double(RDabsentH(validRange,:))+eps));
axis xy;
colorbar;
title('Drone Absent');
xlabel('Velocity (m/s)');
ylabel('Range (m)');

subplot(1,3,3);
imagesc(velocityAxis,rangeAxis(validRange),10*log10(double(RDtargetH(validRange,:))+eps));
axis xy;
colorbar;
title('Isolated Drone');
xlabel('Velocity (m/s)');
ylabel('Range (m)');

%% 20. Vertical Radar Visualization

figure('Name','CubeDN Paired Visibility - Vertical');

subplot(1,3,1);
imagesc(velocityAxis,rangeAxis(validRange),10*log10(double(RDpresentV(validRange,:))+eps));
axis xy;
colorbar;
title('Drone Present');
xlabel('Velocity (m/s)');
ylabel('Range (m)');

subplot(1,3,2);
imagesc(velocityAxis,rangeAxis(validRange),10*log10(double(RDabsentV(validRange,:))+eps));
axis xy;
colorbar;
title('Drone Absent');
xlabel('Velocity (m/s)');
ylabel('Range (m)');

subplot(1,3,3);
imagesc(velocityAxis,rangeAxis(validRange),10*log10(double(RDtargetV(validRange,:))+eps));
axis xy;
colorbar;
title('Isolated Drone');
xlabel('Velocity (m/s)');
ylabel('Range (m)');

%% 21. Save Quality Metrics

visibilityReport.frequency_Hz = cfg.fc;
visibilityReport.dronePosition = dronePos;
visibilityReport.droneVelocity = droneVel;

visibilityReport.inputSNR_H_dB = inputSNR_H;
visibilityReport.inputSNR_V_dB = inputSNR_V;

visibilityReport.inputSCR_H_dB = inputSCR_H;
visibilityReport.inputSCR_V_dB = inputSCR_V;

visibilityReport.inputSCNR_H_dB = inputSCNR_H;
visibilityReport.inputSCNR_V_dB = inputSCNR_V;

visibilityReport.targetBinContrastH_dB = targetBinContrastH;
visibilityReport.targetBinContrastV_dB = targetBinContrastV;

visibilityReport.pairErrorH = pairErrorH;
visibilityReport.pairErrorV = pairErrorV;

visibilityReport.calibrated = false;
visibilityReport.validationType = 'Controlled synthetic paired IQ test';

%% 22. Save Data

outputFolder = cfg.dataset.outputFolder;

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

outputFile = fullfile(outputFolder,'visibility_test.mat');

save(outputFile,'visibilityReport','rawPresentH','rawPresentV','rawAbsentH','rawAbsentV','targetOnlyH','targetOnlyV','backgroundH','backgroundV','noiseOnlyH','noiseOnlyV','rangeAxis','velocityAxis','-v7.3');

%% 23. Final Summary

fprintf('\n==========================================\n');
fprintf(' Paired Visibility Test Complete\n');
fprintf('==========================================\n');

fprintf('Saved: %s\n',outputFile);

disp('Controlled paired radar visibility test completed.');