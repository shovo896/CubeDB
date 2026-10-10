%% =========================================================
%  CubeDN - Dual mmWave Radar Simulation
%  File: 01_setup.m
%  Purpose: Initial Radar and Drone Configuration
% ==========================================================

clc;
clear;
close all;

%% 1. Physical Constants

cfg.c = physconst('LightSpeed');

%% 2. Radar Parameters

cfg.fc = 77e9;                  % Carrier frequency: 77 GHz
cfg.lambda = cfg.c / cfg.fc;

cfg.rangeRes = 0.116;           % Range resolution: 11.6 cm
cfg.maxRange = 20;              % Maximum range: 20 m

cfg.B = cfg.c / (2*cfg.rangeRes);

cfg.Tchirp = 60e-6;             % Chirp duration
cfg.slope = cfg.B / cfg.Tchirp;

cfg.fs = 5e6;                  % Sampling frequency

%% 3. FMCW Waveform

cfg.waveform = phased.FMCWWaveform( ...
    'SweepTime', cfg.Tchirp, ...
    'SweepBandwidth', cfg.B, ...
    'SampleRate', cfg.fs, ...
    'SweepDirection', 'Up');

%% 4. Radar Cube Parameters

cfg.numSamples = round(cfg.fs * cfg.Tchirp);

cfg.numChirps = 128;

cfg.numRangeBins = 256;
cfg.numDopplerBins = 128;

cfg.numAzimuthBins = 32;
cfg.numElevationBins = 32;

%% 5. Virtual Antenna Array

cfg.numTx = 3;
cfg.numRx = 4;

cfg.numVirtualAntennas = 8;  % Simplified initial model

cfg.antennaSpacing = cfg.lambda/2;

cfg.virtualArray = phased.ULA( ...
    'NumElements', cfg.numVirtualAntennas, ...
    'ElementSpacing', cfg.antennaSpacing);

%% 6. Dual Radar Configuration

% Horizontal Radar
cfg.radarH.position = [0; 0; 0];
cfg.radarH.rotation = 0;

% Vertical Radar (90-degree rotated)
cfg.radarV.position = [0.05; 0; 0];
cfg.radarV.rotation = 90;

%% 7. Drone Initial Parameters

cfg.drone.position = [0; 8; 2];

cfg.drone.velocity = [0.5; 0; 0.1];

cfg.drone.RCS = 0.05;

cfg.drone.classID = 1;

%% 8. Radar Processing Parameters

cfg.rangeWindow = 'hann';
cfg.dopplerWindow = 'hann';

cfg.SNR_dB = 20;

%% 9. Dataset Configuration

cfg.dataset.numFrames = 1000;

cfg.dataset.numClasses = 2;

cfg.dataset.outputFolder = 'CubeDN_dataset';

if ~exist(cfg.dataset.outputFolder, 'dir')
    mkdir(cfg.dataset.outputFolder);
end

%% 10. Save Configuration

save(fullfile(cfg.dataset.outputFolder, ...
    'setup_config.mat'), 'cfg');

%% 11. Display Configuration



fprintf('Carrier Frequency : %.2f GHz\n', cfg.fc/1e9);

fprintf('Bandwidth         : %.2f GHz\n', cfg.B/1e9);

fprintf('Range Resolution  : %.3f m\n', cfg.rangeRes);

fprintf('Wavelength        : %.4f m\n', cfg.lambda);

fprintf('ADC Samples       : %d\n', cfg.numSamples);

fprintf('Chirps per Frame  : %d\n', cfg.numChirps);

fprintf('Virtual Antennas  : %d\n', cfg.numVirtualAntennas);

fprintf('Radar Separation  : %.2f m\n', ...
    norm(cfg.radarV.position - cfg.radarH.position));

fprintf('Drone Position    : [%.2f %.2f %.2f] m\n', ...
    cfg.drone.position);


disp('Configuration Saved Successfully!');