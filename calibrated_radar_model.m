clc;
close all;

%% 1. Load Configuration

S = load('CubeDN_dataset/setup_config.mat','cfg');
cfg = S.cfg;

%% 2. Physical Constants

c = 299792458;
kB = 1.380649e-23;
T0 = 290;

fc = 77e9;
lambda = c/fc;

assert(abs(cfg.fc-fc)<1e8,'Configuration frequency must be 77 GHz.');


radar.fc = fc;
radar.lambda = lambda;

radar.txPower_dBm = 12;
radar.txGain_dBi = 12;
radar.rxGain_dBi = 12;

radar.noiseFigure_dB = 10;
radar.systemLoss_dB = 6;

radar.temperature_K = T0;

radar.adcSamplingRate = cfg.fs;
radar.chirpDuration = cfg.Tchirp;
radar.chirpSlope = cfg.slope;

radar.numSamples = cfg.numSamples;
radar.numChirps = cfg.numChirps;
radar.numAntennas = cfg.numVirtualAntennas;

%% 4. Convert Units

Pt = 10^((radar.txPower_dBm-30)/10);

Gt = 10^(radar.txGain_dBi/10);
Gr = 10^(radar.rxGain_dBi/10);

NF = 10^(radar.noiseFigure_dB/10);
L = 10^(radar.systemLoss_dB/10);

%% 5. Receiver Noise Model



Bnoise = cfg.fs;

noisePower_W = kB*T0*Bnoise*NF;

noisePower_dBm = 10*log10(noisePower_W/1e-3);

%% 6. Range Resolution

bandwidth = cfg.slope*cfg.Tchirp;

rangeResolution = c/(2*bandwidth);

maxBeatFrequency = cfg.fs/2;

maxRange = c*maxBeatFrequency/(2*cfg.slope);

%% 7. Drone Radar Cross Section

droneRCS = cfg.drone.RCS;

assert(droneRCS>0,'Drone RCS must be positive.');

droneRCS_dBsm = 10*log10(droneRCS);

%% 8. Radar Equation

rangeAxis = linspace(1,maxRange,500);

receivedPower_W = Pt*Gt*Gr*lambda^2*droneRCS ./ ((4*pi)^3*rangeAxis.^4*L);

receivedPower_dBm = 10*log10(receivedPower_W/1e-3);

%% 9. Single-Sample Input SNR

inputSNR_dB = receivedPower_dBm-noisePower_dBm;

%% 10. Calculate Drone Link Budget

dronePosition = cfg.drone.position;

radarHPosition = cfg.radarH.position;
radarVPosition = cfg.radarV.position;

rangeH = norm(dronePosition-radarHPosition);
rangeV = norm(dronePosition-radarVPosition);

PrH_W = Pt*Gt*Gr*lambda^2*droneRCS/((4*pi)^3*rangeH^4*L);
PrV_W = Pt*Gt*Gr*lambda^2*droneRCS/((4*pi)^3*rangeV^4*L);

PrH_dBm = 10*log10(PrH_W/1e-3);
PrV_dBm = 10*log10(PrV_W/1e-3);

SNR_H_dB = PrH_dBm-noisePower_dBm;
SNR_V_dB = PrV_dBm-noisePower_dBm;

%% 11. Receiver Noise Voltage


receiverImpedance = 50;

noiseVoltageRMS = sqrt(noisePower_W*receiverImpedance);

%% 12. Display Calibration Results



fprintf('Frequency: %.2f GHz\n',fc/1e9);
fprintf('Wavelength: %.4f mm\n',lambda*1000);

fprintf('Transmit Power: %.2f dBm\n',radar.txPower_dBm);
fprintf('TX Gain: %.2f dBi\n',radar.txGain_dBi);
fprintf('RX Gain: %.2f dBi\n',radar.rxGain_dBi);

fprintf('Noise Figure: %.2f dB\n',radar.noiseFigure_dB);
fprintf('System Loss: %.2f dB\n',radar.systemLoss_dB);

fprintf('Noise Bandwidth: %.2f MHz\n',Bnoise/1e6);
fprintf('Thermal Receiver Noise: %.2f dBm\n',noisePower_dBm);

fprintf('Range Resolution: %.4f m\n',rangeResolution);
fprintf('Maximum Range: %.4f m\n',maxRange);

fprintf('Drone RCS: %.2f dBsm\n',droneRCS_dBsm);

fprintf('\nHorizontal Radar\n');
fprintf('Range: %.3f m\n',rangeH);
fprintf('Received Power: %.2f dBm\n',PrH_dBm);
fprintf('Input SNR: %.2f dB\n',SNR_H_dB);

fprintf('\nVertical Radar\n');
fprintf('Range: %.3f m\n',rangeV);
fprintf('Received Power: %.2f dBm\n',PrV_dBm);
fprintf('Input SNR: %.2f dB\n',SNR_V_dB);

%% 13. Plot Received Power

figure('Name','CubeDN 77 GHz Radar Link Budget');

plot(rangeAxis,receivedPower_dBm,'LineWidth',2);
hold on;

yline(noisePower_dBm,'r--','Noise Floor');

xlabel('Range (m)');
ylabel('Received Power (dBm)');
title('77 GHz Radar Received Power vs Range');

legend('Drone Echo','Receiver Noise Floor');
grid on;

%% 14. Plot Input SNR

figure('Name','CubeDN Radar Input SNR');

plot(rangeAxis,inputSNR_dB,'LineWidth',2);
hold on;

yline(0,'r--','0 dB');

xlabel('Range (m)');
ylabel('Input SNR (dB)');
title('77 GHz Radar - Reference Input SNR');

grid on;

%% 15. Store Calibration Metadata

calibration.frequency_Hz = fc;
calibration.wavelength_m = lambda;

calibration.txPower_dBm = radar.txPower_dBm;
calibration.txGain_dBi = radar.txGain_dBi;
calibration.rxGain_dBi = radar.rxGain_dBi;

calibration.noiseFigure_dB = radar.noiseFigure_dB;
calibration.systemLoss_dB = radar.systemLoss_dB;

calibration.noiseBandwidth_Hz = Bnoise;
calibration.noisePower_W = noisePower_W;
calibration.noisePower_dBm = noisePower_dBm;

calibration.rangeResolution_m = rangeResolution;
calibration.maxRange_m = maxRange;

calibration.droneRCS_dBsm = droneRCS_dBsm;

calibration.receivedPowerH_dBm = PrH_dBm;
calibration.receivedPowerV_dBm = PrV_dBm;

calibration.inputSNR_H_dB = SNR_H_dB;
calibration.inputSNR_V_dB = SNR_V_dB;

calibration.noiseVoltageRMS = noiseVoltageRMS;

calibration.isMeasured = false;
calibration.isValidated = false;
calibration.modelType = 'Illustrative 77 GHz monostatic radar link budget';

%% 16. Save Calibration

outputFolder = cfg.dataset.outputFolder;

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

outputFile = fullfile(outputFolder,'radar_calibration.mat');

save(outputFile,'calibration','radar','-v7.3');

%% 17. Final Summary



fprintf('Saved: %s\n',outputFile);

disp('Physical reference model generated.');
disp('Hardware measurement calibration pending.');