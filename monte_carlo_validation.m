clear;
clc;
close all;

%% 1. Load Configuration

S = load('CubeDN_dataset/setup_config.mat','cfg');
cfg = S.cfg;

C = load('CubeDN_dataset/radar_calibration.mat','calibration');
cal = C.calibration;

assert(abs(cfg.fc-77e9)<1e8,'Expected 77 GHz configuration.');
assert(cfg.drone.RCS>0,'Drone RCS must be positive.');

%% 2. Experiment Settings

numTrials = 200;
nominalPfa = 1e-4;

trainingRange = 10;
trainingDoppler = 8;

guardRange = 3;
guardDoppler = 2;

matchRangeBins = 2;
matchDopplerBins = 2;

seed = 20261010;
numClutter = 72;

rng(seed,'twister');

%% 3. Radar Parameters

c = 299792458;
lambda = c/cfg.fc;

Ns = cfg.numSamples;
Nc = cfg.numChirps;

fs = cfg.fs;
slope = cfg.slope;
T = cfg.Tchirp;

assert(fs>0 && slope>0 && T>0,'Invalid FMCW parameters.');

Nrange = 2^nextpow2(max(Ns,512));
Ndoppler = 2^nextpow2(Nc);

fastTime = (0:Ns-1)'/fs;
slowTime = (0:Nc-1)*T;

rangeAxis = (0:floor(Nrange/2)-1)'*fs/Nrange*c/(2*slope);
velocityAxis = ((0:Ndoppler-1)-floor(Ndoppler/2))*lambda/(2*Ndoppler*T);

Nr = numel(rangeAxis);

%% 4. FFT Windows

rangeWin = 0.5-0.5*cos(2*pi*(0:Ns-1)'/Ns);
dopWin = 0.5-0.5*cos(2*pi*(0:Nc-1)/Nc);

window2D = single(rangeWin*dopWin);

%% 5. Physical Reference Model

kB = 1.380649e-23;
T0 = 290;

Pt = 10^((cal.txPower_dBm-30)/10);
Gt = 10^(cal.txGain_dBi/10);
Gr = 10^(cal.rxGain_dBi/10);

NF = 10^(cal.noiseFigure_dB/10);
L = 10^(cal.systemLoss_dB/10);

noiseW = kB*T0*cal.noiseBandwidth_Hz*NF;
noiseStd = sqrt(noiseW/2);

assert(noiseW>0 && isfinite(noiseW),'Invalid noise power.');

%% 6. CFAR Training Kernel

outerR = trainingRange+guardRange;
outerD = trainingDoppler+guardDoppler;

kernel = ones(2*outerR+1,2*outerD+1);

kernel(outerR+1-guardRange:outerR+1+guardRange,outerD+1-guardDoppler:outerD+1+guardDoppler) = 0;

Ntrain = sum(kernel(:));

alpha = Ntrain*(nominalPfa^(-1/Ntrain)-1);

validMask = false(Nr,Ndoppler);

validMask(outerR+1:Nr-outerR,outerD+1:Ndoppler-outerD) = true;

Nvalid = nnz(validMask);

assert(Nvalid>0,'CFAR window too large.');

%% 7. Drone Ground Truth

radarPos = [cfg.radarH.position,cfg.radarV.position];
radarPos = reshape(radarPos,3,2);

truthR = zeros(1,2);
truthD = zeros(1,2);

truthRange = zeros(1,2);
truthVelocity = zeros(1,2);

for q = 1:2

    rel = cfg.drone.position-radarPos(:,q);

    truthRange(q) = norm(rel);

    truthVelocity(q) = dot(cfg.drone.velocity,rel)/truthRange(q);

    [~,truthR(q)] = min(abs(rangeAxis-truthRange(q)));

    [~,truthD(q)] = min(abs(velocityAxis-truthVelocity(q)));

end

for q = 1:2

    assert(validMask(truthR(q),truthD(q)),'Drone outside valid CFAR region.');

end

%% 8. Allocate Result Arrays

hitPresent = false(numTrials,2);
hitAbsent = false(numTrials,2);

noiseCount = zeros(numTrials,2);
absentCount = zeros(numTrials,2);
presentCount = zeros(numTrials,2);

inputSNR = zeros(numTrials,2);
inputSCR = zeros(numTrials,2);

%% 9. Monte Carlo Simulation

fprintf('Running %d independent paired trials...\n',numTrials);

for trial = 1:numTrials

    %% Generate Random Clutter Scene

    clutterPos = [-10+20*rand(1,numClutter);2+18*rand(1,numClutter);0.1+6*rand(1,numClutter)];

    clutterVel = zeros(3,numClutter);

    moveMask = rand(1,numClutter)<0.12;

    clutterVel(1,moveMask) = -0.7+1.4*rand(1,nnz(moveMask));

    clutterVel(2,moveMask) = -0.7+1.4*rand(1,nnz(moveMask));

    clutterRCS = 10.^((-32+28*rand(1,numClutter))/10);

    clutterPhase = 2*pi*rand(1,numClutter);

    dronePhase = 2*pi*rand;

    %% Process Horizontal and Vertical Radars

    for q = 1:2

        pos = radarPos(:,q);

        clutterIQ = complex(zeros(Ns,Nc,'single'));

        %% Generate Clutter Echoes

        for j = 1:numClutter

            rel0 = clutterPos(:,j)-pos;

            R0 = norm(rel0);

            if R0<0.2
                continue;
            end

            u = rel0/R0;

            vr = dot(clutterVel(:,j),u);

            beam = max(0.05,u(2))^2;

            Pr = Pt*Gt*Gr*lambda^2*clutterRCS(j)*beam^2/((4*pi)^3*R0^4*L);

            amp = sqrt(max(Pr,0));

            fb = 2*slope*R0/c;

            fd = 2*vr/lambda;

            fastPhase = 2*pi*fb*fastTime;

            slowPhase = 2*pi*fd*slowTime+4*pi*R0/lambda+clutterPhase(j);

            clutterIQ = clutterIQ+single(amp*exp(1i*(fastPhase+slowPhase)));

        end

        %% Generate Drone Echo

        rel = cfg.drone.position-pos;

        R = norm(rel);

        u = rel/R;

        vr = dot(cfg.drone.velocity,u);

        beam = max(0.05,u(2))^2;

        PrDrone = Pt*Gt*Gr*lambda^2*cfg.drone.RCS*beam^2/((4*pi)^3*R^4*L);

        ampDrone = sqrt(max(PrDrone,0));

        fbDrone = 2*slope*R/c;

        fdDrone = 2*vr/lambda;

        droneIQ = single(ampDrone*exp(1i*(2*pi*fbDrone*fastTime+2*pi*fdDrone*slowTime+4*pi*R/lambda+dronePhase)));

        %% Generate Independent Receiver Noise

        noiseIQ = single(noiseStd)*complex(randn(Ns,Nc,'single'),randn(Ns,Nc,'single'));

        %% Paired Scenes

        iqNoise = noiseIQ;

        iqAbsent = clutterIQ+noiseIQ;

        iqPresent = iqAbsent+droneIQ;

        %% Input SNR and SCR

        targetPower = mean(abs(double(droneIQ(:))).^2);

        noisePower = mean(abs(double(noiseIQ(:))).^2);

        clutterPower = mean(abs(double(clutterIQ(:))).^2);

        inputSNR(trial,q) = 10*log10((targetPower+realmin)/(noisePower+realmin));

        inputSCR(trial,q) = 10*log10((targetPower+realmin)/(clutterPower+realmin));

        %% Process All Three Cases

        iqList = {iqNoise,iqAbsent,iqPresent};

        maps = cell(1,3);

        for mode = 1:3

            fftRD = fftshift(fft(fft(iqList{mode}.*window2D,Nrange,1),Ndoppler,2),2);

            pwr = abs(double(fftRD(1:Nr,:))).^2;

            trainingMean = conv2(pwr,kernel,'same')/Ntrain;

            maps{mode} = (pwr>alpha*trainingMean)&validMask;

        end

        %% Count CFAR Detections

        noiseCount(trial,q) = nnz(maps{1});

        absentCount(trial,q) = nnz(maps{2});

        presentCount(trial,q) = nnz(maps{3});

        %% Ground Truth Matching After Detection

        rr = max(1,truthR(q)-matchRangeBins):min(Nr,truthR(q)+matchRangeBins);

        dd = max(1,truthD(q)-matchDopplerBins):min(Ndoppler,truthD(q)+matchDopplerBins);

        hitPresent(trial,q) = any(maps{3}(rr,dd),'all');

        hitAbsent(trial,q) = any(maps{2}(rr,dd),'all');

    end

    %% Progress

    if mod(trial,25)==0 || trial==numTrials

        fprintf('Completed %d/%d trials\n',trial,numTrials);

    end

end

%% 10. Calculate Detection Probability

PdH = mean(hitPresent(:,1));

PdV = mean(hitPresent(:,2));

PdAND = mean(all(hitPresent,2));

PdOR = mean(any(hitPresent,2));

%% 11. Drone-Absent Chance Matches

PnearAbsentH = mean(hitAbsent(:,1));

PnearAbsentV = mean(hitAbsent(:,2));

PnearAbsentAND = mean(all(hitAbsent,2));

PnearAbsentOR = mean(any(hitAbsent,2));

%% 12. Noise-Only Exceedance

noiseCellExceedH = sum(noiseCount(:,1))/(numTrials*Nvalid);

noiseCellExceedV = sum(noiseCount(:,2))/(numTrials*Nvalid);

%% 13. Clutter-Absent Exceedance

clutterCellExceedH = sum(absentCount(:,1))/(numTrials*Nvalid);

clutterCellExceedV = sum(absentCount(:,2))/(numTrials*Nvalid);

%% 14. Create Validation Report

report.numTrials = numTrials;

report.seed = seed;

report.frequency_Hz = cfg.fc;

report.nominalPfa = nominalPfa;

report.cfarAlpha = alpha;

report.trainingCells = Ntrain;

report.validCellsPerMap = Nvalid;

report.truthRange_m = truthRange;

report.truthVelocity_mps = truthVelocity;

report.matchRangeBins = matchRangeBins;

report.matchDopplerBins = matchDopplerBins;

report.PdH = PdH;

report.PdV = PdV;

report.PdAND = PdAND;

report.PdOR = PdOR;

report.absentNearTruthH = PnearAbsentH;

report.absentNearTruthV = PnearAbsentV;

report.absentNearTruthAND = PnearAbsentAND;

report.absentNearTruthOR = PnearAbsentOR;

report.noiseOnlyCellExceedH = noiseCellExceedH;

report.noiseOnlyCellExceedV = noiseCellExceedV;

report.clutterAbsentCellExceedH = clutterCellExceedH;

report.clutterAbsentCellExceedV = clutterCellExceedV;

report.meanAbsentDetectionsH = mean(absentCount(:,1));

report.meanAbsentDetectionsV = mean(absentCount(:,2));

report.meanPresentDetectionsH = mean(presentCount(:,1));

report.meanPresentDetectionsV = mean(presentCount(:,2));

report.meanInputSNR_H_dB = mean(inputSNR(:,1));

report.meanInputSNR_V_dB = mean(inputSNR(:,2));

report.meanInputSCR_H_dB = mean(inputSCR(:,1));

report.meanInputSCR_V_dB = mean(inputSCR(:,2));

report.industryValidated = false;

report.modelDescription = 'Synthetic point-scatterer paired FMCW CA-CFAR benchmark';

report.notes = 'Pd uses ground-truth neighborhood matching after blind CFAR detection. Absent-near-truth measures chance matches. Correlated bins and clutter can invalidate nominal Pfa assumptions.';

%% 15. Save Results

outputFolder = cfg.dataset.outputFolder;

if ~exist(outputFolder,'dir')

    mkdir(outputFolder);

end

outputFile = fullfile(outputFolder,'monte_carlo_validation.mat');

save(outputFile,'report','hitPresent','hitAbsent','noiseCount','absentCount','presentCount','inputSNR','inputSCR','-v7.3');

%% 16. Print Results

fprintf('\n========================================\n');

fprintf(' CubeDN Monte Carlo Validation Results\n');

fprintf('========================================\n');

fprintf('Trials: %d\n',numTrials);

fprintf('CFAR Training Cells: %d\n',Ntrain);

fprintf('CFAR Alpha: %.4f\n',alpha);

fprintf('Pd Horizontal: %.4f\n',PdH);

fprintf('Pd Vertical: %.4f\n',PdV);

fprintf('Pd AND Fusion: %.4f\n',PdAND);

fprintf('Pd OR Fusion: %.4f\n',PdOR);

fprintf('Absent Near-Truth H: %.4f\n',PnearAbsentH);

fprintf('Absent Near-Truth V: %.4f\n',PnearAbsentV);

fprintf('Noise-Only Exceedance H: %.6g\n',noiseCellExceedH);

fprintf('Noise-Only Exceedance V: %.6g\n',noiseCellExceedV);

fprintf('Clutter-Absent Exceedance H: %.6g\n',clutterCellExceedH);

fprintf('Clutter-Absent Exceedance V: %.6g\n',clutterCellExceedV);

fprintf('Mean Absent Detections H: %.2f\n',report.meanAbsentDetectionsH);

fprintf('Mean Absent Detections V: %.2f\n',report.meanAbsentDetectionsV);

fprintf('Mean Input SNR H: %.2f dB\n',report.meanInputSNR_H_dB);

fprintf('Mean Input SNR V: %.2f dB\n',report.meanInputSNR_V_dB);

fprintf('Saved: %s\n',outputFile);

%% 17. Detection Probability Plot

figure('Name','CubeDN Monte Carlo Detection Rates');

bar([PdH,PdV,PdAND,PdOR;PnearAbsentH,PnearAbsentV,PnearAbsentAND,PnearAbsentOR]');

set(gca,'XTickLabel',{'Horizontal','Vertical','AND','OR'});

ylabel('Fraction of Trials');

legend('Drone Present','Drone Absent Near Truth','Location','best');

title('CFAR Detection and Chance Matches');

ylim([0 1]);

grid on;

%% 18. CFAR Exceedance Plot

figure('Name','CubeDN Monte Carlo CFAR Exceedance');

bar([noiseCellExceedH,noiseCellExceedV;clutterCellExceedH,clutterCellExceedV]');

set(gca,'XTickLabel',{'Horizontal','Vertical'});

ylabel('Fraction of Valid Cells');

legend('Noise Only','Clutter + Noise','Location','best');

title('Empirical CFAR Cell Exceedance');

grid on;

%% 19. Completion

disp('Monte Carlo validation completed.');

disp('This benchmark is not hardware calibrated.');