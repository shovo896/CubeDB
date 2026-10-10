clc;
close all;

%% 1. Load Configuration and Radar Data

cfgData = load('CubeDN_dataset/setup_config.mat','cfg');
cfg = cfgData.cfg;

radarData = load('CubeDN_dataset/RAD_RED_data.mat');
sceneData = load('CubeDN_dataset/dual_radar_data.mat');

RAD = double(radarData.RAD);
RED = double(radarData.RED);

rangeAxis = radarData.rangeAxis(:);
velocityAxis = radarData.velocityAxis(:);
azimuthAxis = radarData.azimuthAxis(:);
elevationAxis = radarData.elevationAxis(:);

dronePos = sceneData.dronePos;
droneVel = sceneData.droneVel;

radarHPos = sceneData.radarHPos;
radarVPos = sceneData.radarVPos;



%% 2. Check Dimensions

[D,R,A] = size(RAD);
[D2,R2,E] = size(RED);

assert(D == D2,'Doppler dimension mismatch.');
assert(R == R2,'Range dimension mismatch.');
assert(numel(rangeAxis) == R,'Range axis mismatch.');
assert(numel(velocityAxis) == D,'Velocity axis mismatch.');
assert(numel(azimuthAxis) == A,'Azimuth axis mismatch.');
assert(numel(elevationAxis) == E,'Elevation axis mismatch.');

assert(all(isfinite(RAD(:))),'RAD contains invalid values.');
assert(all(isfinite(RED(:))),'RED contains invalid values.');

%% 3. Calculate True Drone Geometry

relH = dronePos - radarHPos;
relV = dronePos - radarVPos;

trueRangeH = norm(relH);
trueRangeV = norm(relV);

trueAzimuth = asind(relH(1)/trueRangeH);
trueElevation = asind(relV(3)/trueRangeV);

trueVelocityH = dot(droneVel,relH)/trueRangeH;
trueVelocityV = dot(droneVel,relV)/trueRangeV;

fprintf('\nTrue Drone Parameters\n');
fprintf('Range H: %.4f m\n',trueRangeH);
fprintf('Range V: %.4f m\n',trueRangeV);
fprintf('Azimuth: %.4f deg\n',trueAzimuth);
fprintf('Elevation: %.4f deg\n',trueElevation);
fprintf('Radial Velocity H: %.4f m/s\n',trueVelocityH);
fprintf('Radial Velocity V: %.4f m/s\n',trueVelocityV);

%% 4. Locate Expected Radar Bins

[~,rH] = min(abs(rangeAxis-trueRangeH));
[~,rV] = min(abs(rangeAxis-trueRangeV));

[~,aH] = min(abs(azimuthAxis-trueAzimuth));
[~,eV] = min(abs(elevationAxis-trueElevation));

% Current simulator convention:
% Increasing target range gives positive slow-time phase.
% Verify the sign using a known moving target.

[~,dH] = min(abs(velocityAxis-trueVelocityH));
[~,dV] = min(abs(velocityAxis-trueVelocityV));

fprintf('\nExpected Radar Bins\n');
fprintf('Horizontal: D=%d R=%d A=%d\n',dH,rH,aH);
fprintf('Vertical: D=%d R=%d E=%d\n',dV,rV,eV);

%% 5. Convert Magnitude to Power

powerH = RAD.^2;
powerV = RED.^2;

%% 6. Define Local Search Windows

rangeRadius = 2;
dopplerRadius = 2;
angleRadius = 2;

rangeGuard = 4;
rangeBackgroundRadius = 15;

%% 7. Find Drone-Region Peak

dWindowH = max(1,dH-dopplerRadius):min(D,dH+dopplerRadius);
rWindowH = max(1,rH-rangeRadius):min(R,rH+rangeRadius);
aWindowH = max(1,aH-angleRadius):min(A,aH+angleRadius);

dWindowV = max(1,dV-dopplerRadius):min(D,dV+dopplerRadius);
rWindowV = max(1,rV-rangeRadius):min(R,rV+rangeRadius);
eWindowV = max(1,eV-angleRadius):min(E,eV+angleRadius);

regionH = powerH(dWindowH,rWindowH,aWindowH);
regionV = powerV(dWindowV,rWindowV,eWindowV);

[peakPowerH,localIndexH] = max(regionH(:));
[peakPowerV,localIndexV] = max(regionV(:));

[localDH,localRH,localAH] = ind2sub(size(regionH),localIndexH);
[localDV,localRV,localEV] = ind2sub(size(regionV),localIndexV);

detectedDH = dWindowH(localDH);
detectedRH = rWindowH(localRH);
detectedAH = aWindowH(localAH);

detectedDV = dWindowV(localDV);
detectedRV = rWindowV(localRV);
detectedEV = eWindowV(localEV);



candidateRangeH = max(1,rH-rangeBackgroundRadius):min(R,rH+rangeBackgroundRadius);
candidateRangeV = max(1,rV-rangeBackgroundRadius):min(R,rV+rangeBackgroundRadius);

backgroundRangeH = candidateRangeH(abs(candidateRangeH-rH) > rangeGuard);
backgroundRangeV = candidateRangeV(abs(candidateRangeV-rV) > rangeGuard);

assert(~isempty(backgroundRangeH),'Not enough horizontal background bins.');
assert(~isempty(backgroundRangeV),'Not enough vertical background bins.');

backgroundSamplesH = powerH(dWindowH,backgroundRangeH,aWindowH);
backgroundSamplesV = powerV(dWindowV,backgroundRangeV,eWindowV);

backgroundPowerH = median(backgroundSamplesH(:));
backgroundPowerV = median(backgroundSamplesV(:));



contrastH_dB = 10*log10((peakPowerH+eps)/(backgroundPowerH+eps));
contrastV_dB = 10*log10((peakPowerV+eps)/(backgroundPowerV+eps));


fprintf('Horizontal Radar: %.2f dB\n',contrastH_dB);
fprintf('Vertical Radar: %.2f dB\n',contrastV_dB);

% These values are NOT calibrated receiver SNR.
% They are NOT isolated target-to-clutter SCR.

%% 10. Calculate Detection Position Errors

estimatedRangeH = rangeAxis(detectedRH);
estimatedRangeV = rangeAxis(detectedRV);

estimatedVelocityH = velocityAxis(detectedDH);
estimatedVelocityV = velocityAxis(detectedDV);

estimatedAzimuth = azimuthAxis(detectedAH);
estimatedElevation = elevationAxis(detectedEV);

rangeErrorH = abs(estimatedRangeH-trueRangeH);
rangeErrorV = abs(estimatedRangeV-trueRangeV);

velocityErrorH = abs(estimatedVelocityH-trueVelocityH);
velocityErrorV = abs(estimatedVelocityV-trueVelocityV);

azimuthError = abs(estimatedAzimuth-trueAzimuth);
elevationError = abs(estimatedElevation-trueElevation);



fprintf('Range Error H: %.4f m\n',rangeErrorH);
fprintf('Range Error V: %.4f m\n',rangeErrorV);
fprintf('Velocity Error H: %.4f m/s\n',velocityErrorH);
fprintf('Velocity Error V: %.4f m/s\n',velocityErrorV);
fprintf('Azimuth Error: %.4f deg\n',azimuthError);
fprintf('Elevation Error: %.4f deg\n',elevationError);

%% 11. Plot Horizontal Range-Doppler Map

RD_H = squeeze(max(RAD,[],3));
RD_H_dB = 20*log10(RD_H+eps);
RD_H_dB = RD_H_dB-max(RD_H_dB(:));

figure('Name','CubeDN QA - Horizontal Radar');

imagesc(velocityAxis,rangeAxis,RD_H_dB.');
axis xy;
colorbar;
clim([-60 0]);

hold on;

plot(trueVelocityH,trueRangeH,'wo','MarkerSize',11,'LineWidth',2);
plot(estimatedVelocityH,estimatedRangeH,'rx','MarkerSize',11,'LineWidth',2);

xlabel('Radial Velocity (m/s)');
ylabel('Range (m)');
title('Horizontal Radar - Expected vs Local Peak');
legend('Expected Drone','Local Peak','Location','best');
grid on;

%% 12. Plot Vertical Range-Doppler Map

RD_V = squeeze(max(RED,[],3));
RD_V_dB = 20*log10(RD_V+eps);
RD_V_dB = RD_V_dB-max(RD_V_dB(:));

figure('Name','CubeDN QA - Vertical Radar');

imagesc(velocityAxis,rangeAxis,RD_V_dB.');
axis xy;
colorbar;
clim([-60 0]);

hold on;

plot(trueVelocityV,trueRangeV,'wo','MarkerSize',11,'LineWidth',2);
plot(estimatedVelocityV,estimatedRangeV,'rx','MarkerSize',11,'LineWidth',2);

xlabel('Radial Velocity (m/s)');
ylabel('Range (m)');
title('Vertical Radar - Expected vs Local Peak');
legend('Expected Drone','Local Peak','Location','best');
grid on;

%% 13. Plot Range-Angle Maps

RA_H = squeeze(max(RAD,[],1));
RE_V = squeeze(max(RED,[],1));

figure('Name','CubeDN QA - Angular Response');

subplot(1,2,1);

imagesc(azimuthAxis,rangeAxis,20*log10(RA_H+eps));
axis xy;
colorbar;
hold on;

plot(trueAzimuth,trueRangeH,'wo','MarkerSize',10,'LineWidth',2);

xlabel('Horizontal Spatial Angle (deg)');
ylabel('Range (m)');
title('Horizontal Radar');
grid on;

subplot(1,2,2);

imagesc(elevationAxis,rangeAxis,20*log10(RE_V+eps));
axis xy;
colorbar;
hold on;

plot(trueElevation,trueRangeV,'wo','MarkerSize',10,'LineWidth',2);

xlabel('Vertical Spatial Angle (deg)');
ylabel('Range (m)');
title('Vertical Radar');
grid on;

%% 14. Create Quality Report

qualityReport.trueRangeH = trueRangeH;
qualityReport.trueRangeV = trueRangeV;

qualityReport.trueVelocityH = trueVelocityH;
qualityReport.trueVelocityV = trueVelocityV;

qualityReport.trueAzimuth = trueAzimuth;
qualityReport.trueElevation = trueElevation;

qualityReport.estimatedRangeH = estimatedRangeH;
qualityReport.estimatedRangeV = estimatedRangeV;

qualityReport.estimatedVelocityH = estimatedVelocityH;
qualityReport.estimatedVelocityV = estimatedVelocityV;

qualityReport.estimatedAzimuth = estimatedAzimuth;
qualityReport.estimatedElevation = estimatedElevation;

qualityReport.rangeErrorH = rangeErrorH;
qualityReport.rangeErrorV = rangeErrorV;

qualityReport.velocityErrorH = velocityErrorH;
qualityReport.velocityErrorV = velocityErrorV;

qualityReport.azimuthError = azimuthError;
qualityReport.elevationError = elevationError;

qualityReport.localContrastH_dB = contrastH_dB;
qualityReport.localContrastV_dB = contrastV_dB;

qualityReport.peakPowerH = peakPowerH;
qualityReport.peakPowerV = peakPowerV;

qualityReport.backgroundPowerH = backgroundPowerH;
qualityReport.backgroundPowerV = backgroundPowerV;

qualityReport.calibrated = false;
qualityReport.method = 'Local background contrast using neighboring range bins';

%% 15. Save Quality Report

outputFile = fullfile(cfg.dataset.outputFolder,'radar_quality_check.mat');

save(outputFile,'qualityReport','-v7.3');

%% 16. Final Summary



fprintf('Horizontal Contrast: %.2f dB\n',contrastH_dB);
fprintf('Vertical Contrast: %.2f dB\n',contrastV_dB);
fprintf('Output: %s\n',outputFile);

disp('Radar quality report saved successfully.');