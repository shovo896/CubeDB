clear;
clc;
close all;

%% 1. Load Radar Configuration

configFile = fullfile('CubeDN_dataset','setup_config.mat');

assert(isfile(configFile),'setup_config.mat not found. Run setup first.');

S = load(configFile,'cfg');

cfg = S.cfg;

%% 2. Dataset Settings

numScenes = 1500;

numTrainScenes = 1050;

numValScenes = 300;

numTestScenes = 150;

assert(numTrainScenes+numValScenes+numTestScenes==numScenes,'Split counts mismatch.');

baseSeed = 20261010;

datasetRoot = fullfile('CubeDN_dataset','CubeDN_3000_Int8');

overwriteExisting = false;

numScenesPerProgress = 25;

%% 3. Radar Configuration Compatibility

if isfield(cfg,'fc')

    fc = cfg.fc;

else

    fc = 77e9;

end

c = 299792458;

lambda = c/fc;

if isfield(cfg,'fs')

    fs = cfg.fs;

elseif isfield(cfg,'sampleRate')

    fs = cfg.sampleRate;

else

    error('Sampling frequency missing from cfg.');

end

if isfield(cfg,'Tchirp')

    Tchirp = cfg.Tchirp;

elseif isfield(cfg,'chirpTime')

    Tchirp = cfg.chirpTime;

else

    error('Chirp time missing from cfg.');

end

if isfield(cfg,'slope')

    slope = cfg.slope;

elseif isfield(cfg,'bandwidth')

    slope = cfg.bandwidth/Tchirp;

else

    error('Chirp slope missing from cfg.');

end

if isfield(cfg,'numSamples')

    Ns = cfg.numSamples;

elseif isfield(cfg,'samplesPerChirp')

    Ns = cfg.samplesPerChirp;

else

    error('Number of ADC samples missing.');

end

Nc = cfg.numChirps;

Na = cfg.numVirtualAntennas;

if isfield(cfg,'antennaSpacing')

    d = cfg.antennaSpacing;

elseif isfield(cfg,'arraySpacing')

    d = cfg.arraySpacing;

else

    d = lambda/2;

end

assert(Na==8,'This dataset is configured for 8 virtual antennas per radar.');

assert(Ns>0 && Nc>0,'Invalid radar dimensions.');

assert(fs>0 && Tchirp>0 && slope>0,'Invalid waveform parameters.');

assert(abs(fc-77e9)<1e9,'Expected approximately 77 GHz carrier.');

%% 4. Radar Geometry

radarHPos = cfg.radarH.position(:);

radarVPos = cfg.radarV.position(:);

assert(numel(radarHPos)==3,'Horizontal radar position must contain 3 coordinates.');

assert(numel(radarVPos)==3,'Vertical radar position must contain 3 coordinates.');

radarPositions = [radarHPos,radarVPos];

arrayAxes = [1 0;0 0;0 1];

antennaOffsets = ((0:Na-1)-(Na-1)/2)*d;

%% 5. Reference Radar Hardware Model

% These are illustrative hardware values, not measured calibration.

txPower_dBm = 12;

txGain_dBi = 12;

rxGain_dBi = 12;

noiseFigure_dB = 10;

systemLoss_dB = 6;

noiseBandwidth_Hz = fs;

calibrationFile = fullfile('CubeDN_dataset','radar_calibration.mat');

if isfile(calibrationFile)

    K = load(calibrationFile);

    if isfield(K,'calibration')

        cal = K.calibration;

        if isfield(cal,'txPower_dBm')

            txPower_dBm = cal.txPower_dBm;

        end

        if isfield(cal,'txGain_dBi')

            txGain_dBi = cal.txGain_dBi;

        end

        if isfield(cal,'rxGain_dBi')

            rxGain_dBi = cal.rxGain_dBi;

        end

        if isfield(cal,'noiseFigure_dB')

            noiseFigure_dB = cal.noiseFigure_dB;

        end

        if isfield(cal,'systemLoss_dB')

            systemLoss_dB = cal.systemLoss_dB;

        end

        if isfield(cal,'noiseBandwidth_Hz')

            noiseBandwidth_Hz = cal.noiseBandwidth_Hz;

        end

    end

end

Pt = 10^((txPower_dBm-30)/10);

Gt = 10^(txGain_dBi/10);

Gr = 10^(rxGain_dBi/10);

NF = 10^(noiseFigure_dB/10);

L = 10^(systemLoss_dB/10);

kB = 1.380649e-23;

noisePower_W = kB*290*noiseBandwidth_Hz*NF;

referenceNoiseStd = sqrt(noisePower_W/2);

assert(isfinite(referenceNoiseStd) && referenceNoiseStd>0,'Invalid receiver noise.');

%% 6. Time and Array Coordinates

fastTime = (0:Ns-1)'/fs;

slowTime = (0:Nc-1)*Tchirp;

%% 7. Create Dataset Folders

splitNames = {'train','val','test'};

for s = 1:numel(splitNames)

    folder = fullfile(datasetRoot,splitNames{s});

    if ~exist(folder,'dir')

        mkdir(folder);

    end

end

%% 8. Manifest Initialization

manifest = repmat(struct( ...
    'sceneID',0, ...
    'split','', ...
    'presentFile','', ...
    'absentFile','', ...
    'sceneSeed',0, ...
    'sceneType','', ...
    'droneRange_m',0, ...
    'droneRCS_m2',0, ...
    'clutterCount',0),numScenes,1);

quantizationDiagnostics = nan(numScenes,4);

%% 9. Main Dataset Generation Loop


fprintf('Scenes: %d\n',numScenes);

fprintf('Samples: %d\n',2*numScenes);

fprintf('IQ Shape: [%d %d %d]\n',Ns,Nc,Na);

fprintf('Storage: int8 I/Q\n');

fprintf('Output: %s\n\n',datasetRoot);

for sceneIdx = 1:numScenes

    %% 9A. Assign Scene Split

    if sceneIdx<=numTrainScenes

        split = 'train';

    elseif sceneIdx<=numTrainScenes+numValScenes

        split = 'val';

    else

        split = 'test';

    end

    sceneSeed = baseSeed+sceneIdx*1009;

    presentFile = fullfile(datasetRoot,split,sprintf('scene_%05d_present.mat',sceneIdx));

    absentFile = fullfile(datasetRoot,split,sprintf('scene_%05d_absent.mat',sceneIdx));

    %% 9B. Resume Existing Completed Pairs

    if ~overwriteExisting && isfile(presentFile) && isfile(absentFile)

        manifest(sceneIdx).sceneID = sceneIdx;

        manifest(sceneIdx).split = split;

        manifest(sceneIdx).presentFile = presentFile;

        manifest(sceneIdx).absentFile = absentFile;

        manifest(sceneIdx).sceneSeed = sceneSeed;

        manifest(sceneIdx).sceneType = 'existing';

        continue;

    end

    rng(sceneSeed,'twister');

    %% 9C. Select Diverse Environment

    environmentIndex = mod(sceneIdx-1,5)+1;

    switch environmentIndex

        case 1

            sceneType = 'open_field';

            clutterCount = randi([15,40]);

            movingFraction = 0.05;

            clutterRCSRange_dBsm = [-40,-18];

        case 2

            sceneType = 'vegetation';

            clutterCount = randi([55,110]);

            movingFraction = 0.25;

            clutterRCSRange_dBsm = [-38,-12];

        case 3

            sceneType = 'urban';

            clutterCount = randi([70,140]);

            movingFraction = 0.10;

            clutterRCSRange_dBsm = [-32,0];

        case 4

            sceneType = 'mixed_outdoor';

            clutterCount = randi([40,100]);

            movingFraction = 0.18;

            clutterRCSRange_dBsm = [-38,-8];

        otherwise

            sceneType = 'high_clutter';

            clutterCount = randi([100,180]);

            movingFraction = 0.30;

            clutterRCSRange_dBsm = [-30,2];

    end

    %% 9D. Drone Position and Motion

    droneRange_m = 3+13*rand;

    droneAzimuth_deg = -40+80*rand;

    droneElevation_deg = 4+31*rand;

    az = deg2rad(droneAzimuth_deg);

    el = deg2rad(droneElevation_deg);

    droneDirection = [cos(el)*sin(az);cos(el)*cos(az);sin(el)];

    dronePos = radarHPos+droneRange_m*droneDirection;

    droneSpeed = 0.2+5.8*rand;

    velocityAzimuth = 2*pi*rand;

    verticalVelocity = -0.8+1.6*rand;

    horizontalSpeed = sqrt(max(0,droneSpeed^2-verticalVelocity^2));

    droneVel = [horizontalSpeed*cos(velocityAzimuth);horizontalSpeed*sin(velocityAzimuth);verticalVelocity];

    droneRCS_dBsm = -30+20*rand;

    droneRCS_m2 = 10^(droneRCS_dBsm/10);

    dronePhase = 2*pi*rand;

    %% 9E. Clutter Geometry

    clutterX = -15+30*rand(1,clutterCount);

    clutterY = 1+23*rand(1,clutterCount);

    clutterZ = 0.05+7.95*rand(1,clutterCount);

    clutterPos = [clutterX;clutterY;clutterZ];

    clutterVel = zeros(3,clutterCount);

    movingMask = rand(1,clutterCount)<movingFraction;

    movingCount = nnz(movingMask);

    clutterVel(1,movingMask) = -1.5+3*rand(1,movingCount);

    clutterVel(2,movingMask) = -1.5+3*rand(1,movingCount);

    clutterVel(3,movingMask) = -0.2+0.4*rand(1,movingCount);

    clutterRCS_dBsm = clutterRCSRange_dBsm(1)+(clutterRCSRange_dBsm(2)-clutterRCSRange_dBsm(1))*rand(1,clutterCount);

    clutterRCS_m2 = 10.^(clutterRCS_dBsm/10);

    clutterPhase = 2*pi*rand(1,clutterCount);

    %% 9F. Scene-Level Hardware and Noise Variation

    receiverNoiseVariation_dB = -3+6*rand;

    noiseMultiplier = 10^(receiverNoiseVariation_dB/20);

    noiseStd = referenceNoiseStd*noiseMultiplier;

    multipathProbability = 0.05+0.25*rand;

    %% 9G. Allocate Paired IQ

    presentIQ = cell(1,2);

    absentIQ = cell(1,2);

    trueRange = zeros(1,2);

    trueRadialVelocity = zeros(1,2);

    trueDirectionCosine = zeros(1,2);

    %% 10. Generate Both Radar Signals

    for q = 1:2

        radarPos = radarPositions(:,q);

        arrayAxis = arrayAxes(:,q);

        backgroundIQ = complex(zeros(Ns,Nc,Na,'single'));

        %% 10A. Clutter Returns

        for scatterIdx = 1:clutterCount

            relativePosition = clutterPos(:,scatterIdx)-radarPos;

            R = norm(relativePosition);

            if R<0.3

                continue;

            end

            direction = relativePosition/R;

            radialVelocity = dot(clutterVel(:,scatterIdx),direction);

            antennaBeam = max(0.05,direction(2))^2;

            receivedPower = Pt*Gt*Gr*lambda^2*clutterRCS_m2(scatterIdx)*antennaBeam^2/((4*pi)^3*R^4*L);

            amplitude = sqrt(max(receivedPower,0));

            beatFrequency = 2*slope*R/c;

            dopplerFrequency = 2*radialVelocity/lambda;

            phase = 2*pi*beatFrequency*fastTime+2*pi*dopplerFrequency*slowTime+4*pi*R/lambda+clutterPhase(scatterIdx);

            for ant = 1:Na

                spatialPhase = 2*pi/lambda*antennaOffsets(ant)*dot(direction,arrayAxis);

                backgroundIQ(:,:,ant) = backgroundIQ(:,:,ant)+single(amplitude*exp(1i*(phase+spatialPhase)));

            end

            %% 10B. Simplified Secondary Reflection

            if rand<multipathProbability

                secondaryRange = R+0.4+2.0*rand;

                secondaryAmplitude = amplitude*(0.05+0.20*rand);

                secondaryPhase = 2*pi*rand;

                secondaryBeat = 2*slope*secondaryRange/c;

                phaseSecondary = 2*pi*secondaryBeat*fastTime+2*pi*dopplerFrequency*slowTime+4*pi*secondaryRange/lambda+secondaryPhase;

                for ant = 1:Na

                    spatialPhase = 2*pi/lambda*antennaOffsets(ant)*dot(direction,arrayAxis);

                    backgroundIQ(:,:,ant) = backgroundIQ(:,:,ant)+single(secondaryAmplitude*exp(1i*(phaseSecondary+spatialPhase)));

                end

            end

        end

        %% 10C. Receiver Noise

        receiverNoise = single(noiseStd)*complex(randn(Ns,Nc,Na,'single'),randn(Ns,Nc,Na,'single'));

        absent = backgroundIQ+receiverNoise;

        %% 10D. Drone Return

        relativeDronePosition = dronePos-radarPos;

        droneRange = norm(relativeDronePosition);

        droneDirectionRadar = relativeDronePosition/droneRange;

        droneRadialVelocity = dot(droneVel,droneDirectionRadar);

        antennaBeam = max(0.05,droneDirectionRadar(2))^2;

        dronePower = Pt*Gt*Gr*lambda^2*droneRCS_m2*antennaBeam^2/((4*pi)^3*droneRange^4*L);

        droneAmplitude = sqrt(max(dronePower,0));

        droneBeat = 2*slope*droneRange/c;

        droneDoppler = 2*droneRadialVelocity/lambda;

        dronePhase2D = 2*pi*droneBeat*fastTime+2*pi*droneDoppler*slowTime+4*pi*droneRange/lambda+dronePhase;

        targetIQ = complex(zeros(Ns,Nc,Na,'single'));

        for ant = 1:Na

            spatialPhase = 2*pi/lambda*antennaOffsets(ant)*dot(droneDirectionRadar,arrayAxis);

            targetIQ(:,:,ant) = single(droneAmplitude*exp(1i*(dronePhase2D+spatialPhase)));

        end

        %% 10E. Paired Radar Samples

        absentIQ{q} = absent;

        presentIQ{q} = absent+targetIQ;

        trueRange(q) = droneRange;

        trueRadialVelocity(q) = droneRadialVelocity;

        trueDirectionCosine(q) = dot(droneDirectionRadar,arrayAxis);

    end

    %% 11. Shared Quantization Scale

    % One scale per scene, shared across both radars and both classes.
    % This avoids class leakage through the scale field.

    maxAbsComponent = 0;

    for q = 1:2

        maxAbsComponent = max(maxAbsComponent,max(abs(real(presentIQ{q}(:)))));

        maxAbsComponent = max(maxAbsComponent,max(abs(imag(presentIQ{q}(:)))));

        maxAbsComponent = max(maxAbsComponent,max(abs(real(absentIQ{q}(:)))));

        maxAbsComponent = max(maxAbsComponent,max(abs(imag(absentIQ{q}(:)))));

    end

    iqScale = single(max(maxAbsComponent/127,realmin('single')));

    %% 12. Quantize Present Sample

    rawH_I = int8(max(-127,min(127,round(real(presentIQ{1})/iqScale))));

    rawH_Q = int8(max(-127,min(127,round(imag(presentIQ{1})/iqScale))));

    rawV_I = int8(max(-127,min(127,round(real(presentIQ{2})/iqScale))));

    rawV_Q = int8(max(-127,min(127,round(imag(presentIQ{2})/iqScale))));

    %% 13. Present Quantization Diagnostics

    reconstructedH = iqScale*complex(single(rawH_I),single(rawH_Q));

    reconstructedV = iqScale*complex(single(rawV_I),single(rawV_Q));

    errorH = mean(abs(double(reconstructedH(:))-double(presentIQ{1}(:))).^2);

    errorV = mean(abs(double(reconstructedV(:))-double(presentIQ{2}(:))).^2);

    powerH = mean(abs(double(presentIQ{1}(:))).^2);

    powerV = mean(abs(double(presentIQ{2}(:))).^2);

    presentSQNR_H_dB = 10*log10((powerH+realmin)/(errorH+realmin));

    presentSQNR_V_dB = 10*log10((powerV+realmin)/(errorV+realmin));

    %% 14. Scene Metadata and Ground Truth

    sceneMetadata.sceneID = sceneIdx;

    sceneMetadata.sceneSeed = sceneSeed;

    sceneMetadata.split = split;

    sceneMetadata.environment = sceneType;

    sceneMetadata.frequency_Hz = fc;

    sceneMetadata.sampleRate_Hz = fs;

    sceneMetadata.chirpTime_s = Tchirp;

    sceneMetadata.chirpSlope_HzPerS = slope;

    sceneMetadata.rawShape = [Ns,Nc,Na];

    sceneMetadata.radarPositions_m = radarPositions;

    sceneMetadata.arrayAxes = arrayAxes;

    sceneMetadata.clutterCount = clutterCount;

    sceneMetadata.movingClutterCount = movingCount;

    sceneMetadata.receiverNoiseVariation_dB = receiverNoiseVariation_dB;

    sceneMetadata.multipathProbability = multipathProbability;

    sceneMetadata.quantizationBits = 8;

    sceneMetadata.quantizationScale = iqScale;

    sceneMetadata.calibrationMeasured = false;

    sceneMetadata.simulationModel = 'Idealized FMCW point scatterers with simplified secondary echoes';

    groundTruth.dronePresent = true;

    groundTruth.className = 'drone';

    groundTruth.classID = 1;

    groundTruth.position_m = dronePos;

    groundTruth.velocity_mps = droneVel;

    groundTruth.RCS_m2 = droneRCS_m2;

    groundTruth.range_m = trueRange;

    groundTruth.radialVelocity_mps = trueRadialVelocity;

    groundTruth.directionCosine = trueDirectionCosine;

    groundTruth.azimuth_deg = droneAzimuth_deg;

    groundTruth.elevation_deg = droneElevation_deg;

    groundTruth.positionReference = 'global_xyz_m';

    label = uint8(1);

    %% 15. Save Drone-Present Sample

    save(presentFile,'rawH_I','rawH_Q','rawV_I','rawV_Q','iqScale','label','groundTruth','sceneMetadata','-v7');

    %% 16. Quantize Drone-Absent Sample

    rawH_I = int8(max(-127,min(127,round(real(absentIQ{1})/iqScale))));

    rawH_Q = int8(max(-127,min(127,round(imag(absentIQ{1})/iqScale))));

    rawV_I = int8(max(-127,min(127,round(real(absentIQ{2})/iqScale))));

    rawV_Q = int8(max(-127,min(127,round(imag(absentIQ{2})/iqScale))));

    %% 17. Absent Quantization Diagnostics

    reconstructedH = iqScale*complex(single(rawH_I),single(rawH_Q));

    reconstructedV = iqScale*complex(single(rawV_I),single(rawV_Q));

    errorH = mean(abs(double(reconstructedH(:))-double(absentIQ{1}(:))).^2);

    errorV = mean(abs(double(reconstructedV(:))-double(absentIQ{2}(:))).^2);

    powerH = mean(abs(double(absentIQ{1}(:))).^2);

    powerV = mean(abs(double(absentIQ{2}(:))).^2);

    absentSQNR_H_dB = 10*log10((powerH+realmin)/(errorH+realmin));

    absentSQNR_V_dB = 10*log10((powerV+realmin)/(errorV+realmin));

    %% 18. Save Drone-Absent Sample

    groundTruth = struct();

    groundTruth.dronePresent = false;

    groundTruth.className = 'background';

    groundTruth.classID = 0;

    groundTruth.position_m = [NaN;NaN;NaN];

    groundTruth.velocity_mps = [NaN;NaN;NaN];

    label = uint8(0);

    save(absentFile,'rawH_I','rawH_Q','rawV_I','rawV_Q','iqScale','label','groundTruth','sceneMetadata','-v7');

    %% 19. Store Manifest

    manifest(sceneIdx).sceneID = sceneIdx;

    manifest(sceneIdx).split = split;

    manifest(sceneIdx).presentFile = presentFile;

    manifest(sceneIdx).absentFile = absentFile;

    manifest(sceneIdx).sceneSeed = sceneSeed;

    manifest(sceneIdx).sceneType = sceneType;

    manifest(sceneIdx).droneRange_m = droneRange_m;

    manifest(sceneIdx).droneRCS_m2 = droneRCS_m2;

    manifest(sceneIdx).clutterCount = clutterCount;

    quantizationDiagnostics(sceneIdx,:) = [presentSQNR_H_dB,presentSQNR_V_dB,absentSQNR_H_dB,absentSQNR_V_dB];

    %% 20. Progress and Checkpoint

    if mod(sceneIdx,numScenesPerProgress)==0 || sceneIdx==numScenes

        fprintf('Completed %d/%d scenes | %d/%d samples\n',sceneIdx,numScenes,2*sceneIdx,2*numScenes);

        save(fullfile(datasetRoot,'generation_checkpoint.mat'),'manifest','quantizationDiagnostics','sceneIdx','-v7');

    end

end

%% 21. Final Dataset Metadata

metadata.datasetName = 'CubeDN_3000_Int8';

metadata.datasetVersion = '1.0_synthetic';

metadata.numScenes = numScenes;

metadata.numSamples = 2*numScenes;

metadata.numTrainSamples = 2*numTrainScenes;

metadata.numValidationSamples = 2*numValScenes;

metadata.numTestSamples = 2*numTestScenes;

metadata.radarFrequency_Hz = fc;

metadata.numRadars = 2;

metadata.numAntennasPerRadar = Na;

metadata.rawIQShape = [Ns,Nc,Na];

metadata.quantizationBits = 8;

metadata.quantizationFormat = 'Separate signed int8 I/Q arrays with shared scene scale';

metadata.pairedScenes = true;

metadata.sceneLevelSplit = true;

metadata.measuredHardwareCalibration = false;

metadata.industryValidated = false;

metadata.notes = 'Synthetic reference dataset. Point-scatterer drone, simplified multipath and clutter; requires independent validation.';

%% 22. Save Dataset Manifest

manifestFile = fullfile(datasetRoot,'dataset_manifest.mat');

save(manifestFile,'manifest','metadata','quantizationDiagnostics','-v7');

%% 23. Final Report



fprintf('Train samples: %d\n',metadata.numTrainSamples);

fprintf('Validation samples: %d\n',metadata.numValidationSamples);

fprintf('Test samples: %d\n',metadata.numTestSamples);

fprintf('Total samples: %d\n',metadata.numSamples);

fprintf('Output folder: %s\n',datasetRoot);

validSQNR = quantizationDiagnostics(isfinite(quantizationDiagnostics));

if ~isempty(validSQNR)

    fprintf('Mean quantization SQNR: %.2f dB\n',mean(validSQNR));

    fprintf('Minimum quantization SQNR: %.2f dB\n',min(validSQNR));

end

fprintf('Manifest: %s\n',manifestFile);

disp('CubeDN dataset generation finished.');