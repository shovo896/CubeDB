clc;
close all;

%% 1. Load Configuration and Data

load('CubeDN_dataset/setup_config.mat', 'cfg');
data = load('CubeDN_dataset/dual_radar_data.mat');

rawH = data.rawH;
rawV = data.rawV;


%% 2. Radar Parameters

c = cfg.c;
lambda = cfg.lambda;
fs = cfg.fs;
slope = cfg.slope;
PRI = cfg.Tchirp;

[Ns, Nc, Na] = size(rawH);

assert(isequal(size(rawH), size(rawV)), 'Horizontal and vertical IQ data sizes do not match.');
assert(Nc == cfg.numChirps, 'Number of chirps does not match the configuration.');
assert(Na == cfg.numVirtualAntennas, 'Number of antennas does not match the configuration.');

%% 3. FFT Configuration

Nrange = max(512, 2^nextpow2(Ns));
Ndoppler = max(cfg.numDopplerBins, Nc);
NangleH = max(cfg.numAzimuthBins, Na);
NangleV = max(cfg.numElevationBins, Na);

fprintf('ADC Samples       : %d\n', Ns);
fprintf('Chirps            : %d\n', Nc);
fprintf('Virtual Antennas  : %d\n', Na);
fprintf('Range FFT Size    : %d\n', Nrange);
fprintf('Doppler FFT Size  : %d\n', Ndoppler);
fprintf('Azimuth FFT Size  : %d\n', NangleH);
fprintf('Elevation FFT Size: %d\n', NangleV);

%% 4. Window Functions

rangeWindow = 0.5 - 0.5*cos(2*pi*(0:Ns-1)'/(Ns-1));
dopplerWindow = 0.5 - 0.5*cos(2*pi*(0:Nc-1)/(Nc-1));
angleWindow = 0.5 - 0.5*cos(2*pi*(0:Na-1)/(Na-1));

%% 5. Range FFT

disp('Processing Range FFT...');

rangeInputH = rawH .* reshape(rangeWindow, [], 1, 1);
rangeInputV = rawV .* reshape(rangeWindow, [], 1, 1);

rangeFFTH = fft(rangeInputH, Nrange, 1);
rangeFFTV = fft(rangeInputV, Nrange, 1);

Npositive = floor(Nrange/2);

rangeFFTH = rangeFFTH(1:Npositive, :, :);
rangeFFTV = rangeFFTV(1:Npositive, :, :);

rangeAxisFull = (0:Npositive-1)' * fs/Nrange * c/(2*slope);

validRange = rangeAxisFull <= cfg.maxRange;

rangeAxis = rangeAxisFull(validRange);

rangeFFTH = rangeFFTH(validRange, :, :);
rangeFFTV = rangeFFTV(validRange, :, :);

Nr = numel(rangeAxis);

fprintf('Valid Range Bins: %d\n', Nr);

%% 6. Doppler FFT

disp('Processing Doppler FFT...');

dopplerInputH = rangeFFTH .* reshape(dopplerWindow, 1, [], 1);
dopplerInputV = rangeFFTV .* reshape(dopplerWindow, 1, [], 1);

dopplerFFTH = fftshift(fft(dopplerInputH, Ndoppler, 2), 2);
dopplerFFTV = fftshift(fft(dopplerInputV, Ndoppler, 2), 2);

dopplerBins = -floor(Ndoppler/2):ceil(Ndoppler/2)-1;

dopplerAxis = dopplerBins / (Ndoppler*PRI);

velocityAxis = dopplerAxis * lambda/2;

%% 7. Spatial Angle FFT

disp('Processing Angle FFT...');

angleInputH = dopplerFFTH .* reshape(angleWindow, 1, 1, []);
angleInputV = dopplerFFTV .* reshape(angleWindow, 1, 1, []);

angleFFTH = fftshift(fft(angleInputH, NangleH, 3), 3);
angleFFTV = fftshift(fft(angleInputV, NangleV, 3), 3);

%% 8. Construct RAD and RED Cubes

RAD = single(permute(abs(angleFFTH), [2 1 3]));
RED = single(permute(abs(angleFFTV), [2 1 3]));

fprintf('RAD Dimension: [%s]\n', num2str(size(RAD)));
fprintf('RED Dimension: [%s]\n', num2str(size(RED)));

%% 9. Physical Angle Axes

antennaSpacing = cfg.antennaSpacing;

spatialH = (-floor(NangleH/2):ceil(NangleH/2)-1) / NangleH;
spatialV = (-floor(NangleV/2):ceil(NangleV/2)-1) / NangleV;

directionH = spatialH * lambda/antennaSpacing;
directionV = spatialV * lambda/antennaSpacing;

directionH = max(-1, min(1, directionH));
directionV = max(-1, min(1, directionV));

azimuthAxis = asind(directionH);
elevationAxis = asind(directionV);

% These are spatial-angle axes for the respective ULAs.
% Full world azimuth/elevation conversion requires joint geometry.

%% 10. Generate Range-Doppler Maps

RD_H = sum(RAD, 3);
RD_V = sum(RED, 3);

RD_H_dB = 20*log10(double(RD_H) + eps);
RD_V_dB = 20*log10(double(RD_V) + eps);

RD_H_dB = RD_H_dB - max(RD_H_dB(:));
RD_V_dB = RD_V_dB - max(RD_V_dB(:));

%% 11. Generate Range-Angle Maps

RA_H = squeeze(max(RAD, [], 1));
RE_V = squeeze(max(RED, [], 1));

RA_H_dB = 20*log10(double(RA_H) + eps);
RE_V_dB = 20*log10(double(RE_V) + eps);

RA_H_dB = RA_H_dB - max(RA_H_dB(:));
RE_V_dB = RE_V_dB - max(RE_V_dB(:));

%% 12. Plot Range-Doppler Maps

figure('Name', 'CubeDN Dual Radar Range-Doppler Maps');

subplot(1, 2, 1);
imagesc(velocityAxis, rangeAxis, RD_H_dB.');
axis xy;
colorbar;
clim([-60 0]);
xlabel('Radial Velocity (m/s)');
ylabel('Range (m)');
title('Horizontal Radar - Range Doppler');
grid on;

subplot(1, 2, 2);
imagesc(velocityAxis, rangeAxis, RD_V_dB.');
axis xy;
colorbar;
clim([-60 0]);
xlabel('Radial Velocity (m/s)');
ylabel('Range (m)');
title('Vertical Radar - Range Doppler');
grid on;

%% 13. Plot Range-Azimuth and Range-Elevation Maps

figure('Name', 'CubeDN Range-Angle Maps');

subplot(1, 2, 1);
imagesc(azimuthAxis, rangeAxis, RA_H_dB);
axis xy;
colorbar;
clim([-60 0]);
xlabel('Horizontal Spatial Angle (deg)');
ylabel('Range (m)');
title('Horizontal Radar - Range Azimuth');
grid on;

subplot(1, 2, 2);
imagesc(elevationAxis, rangeAxis, RE_V_dB);
axis xy;
colorbar;
clim([-60 0]);
xlabel('Vertical Spatial Angle (deg)');
ylabel('Range (m)');
title('Vertical Radar - Range Elevation');
grid on;

%% 14. Find Strongest Radar Responses

[~, idxH] = max(RAD(:));
[dH, rH, aH] = ind2sub(size(RAD), idxH);

[~, idxV] = max(RED(:));
[dV, rV, eV] = ind2sub(size(RED), idxV);

fprintf('\n==========================================\n');
fprintf('       Strongest Radar Responses          \n');
fprintf('==========================================\n');

fprintf('\nHorizontal Radar:\n');
fprintf('Range             : %.3f m\n', rangeAxis(rH));
fprintf('Radial Velocity   : %.3f m/s\n', velocityAxis(dH));
fprintf('Spatial Angle     : %.3f deg\n', azimuthAxis(aH));

fprintf('\nVertical Radar:\n');
fprintf('Range             : %.3f m\n', rangeAxis(rV));
fprintf('Radial Velocity   : %.3f m/s\n', velocityAxis(dV));
fprintf('Spatial Angle     : %.3f deg\n', elevationAxis(eV));

%% 15. Save RAD and RED Cubes

outputFile = fullfile(cfg.dataset.outputFolder, 'RAD_RED_data.mat');

save(outputFile, 'RAD', 'RED', 'rangeAxis', 'velocityAxis', 'azimuthAxis', 'elevationAxis', 'Nrange', 'Ndoppler', 'NangleH', 'NangleV', '-v7.3');

%% 16. Final Summary

fprintf('\n==========================================\n');
fprintf('       CubeDN Processing Complete         \n');
fprintf('==========================================\n');

fprintf('RAD Size : [%s]\n', num2str(size(RAD)));
fprintf('RED Size : [%s]\n', num2str(size(RED)));
fprintf('Output   : %s\n', outputFile);

disp('RAD and RED cubes saved successfully.');