clc;
close all;

%% 1. Load Configuration

load('CubeDN_dataset/setup_config.mat', 'cfg');
rng(42);



%% 2. Radar Parameters

c = cfg.c;
fc = cfg.fc;
lambda = cfg.lambda;
fs = cfg.fs;
Tchirp = cfg.Tchirp;
slope = cfg.slope;
Ns = cfg.numSamples;
Nc = cfg.numChirps;
Na = cfg.numVirtualAntennas;
d = cfg.antennaSpacing;
PRI = Tchirp;

tFast = (0:Ns-1)' / fs;
tSlow = (0:Nc-1) * PRI;

%% 3. Radar Geometry

radarHPos = cfg.radarH.position;
radarVPos = cfg.radarV.position;

arrayAxisH = [1; 0; 0];
arrayAxisV = [0; 0; 1];

antennaIndex = (0:Na-1) - (Na-1)/2;
antennaOffsets = antennaIndex * d;

%% 4. Drone Parameters

dronePos = cfg.drone.position;
droneVel = cfg.drone.velocity;
droneRCS = cfg.drone.RCS;

%% 5. Environment Configuration

numGround = 35;
numBuildings = 12;
numTrees = 20;
numMoving = 5;

groundX = -10 + 20*rand(1,numGround);
groundY = 2 + 18*rand(1,numGround);
groundZ = zeros(1,numGround);
groundPos = [groundX; groundY; groundZ];

buildingX = -8 + 16*rand(1,numBuildings);
buildingY = 5 + 15*rand(1,numBuildings);
buildingZ = 1 + 7*rand(1,numBuildings);
buildingPos = [buildingX; buildingY; buildingZ];

treeX = -10 + 20*rand(1,numTrees);
treeY = 3 + 17*rand(1,numTrees);
treeZ = 0.5 + 5*rand(1,numTrees);
treePos = [treeX; treeY; treeZ];

movingX = -6 + 12*rand(1,numMoving);
movingY = 4 + 12*rand(1,numMoving);
movingZ = 0.3 + 1.5*rand(1,numMoving);
movingPos = [movingX; movingY; movingZ];

movingVel = [-0.5 + rand(1,numMoving); -0.5 + rand(1,numMoving); zeros(1,numMoving)];

%% 6. Scatterer RCS Models

groundRCS = 10.^((-35 + 15*rand(1,numGround))/10);
buildingRCS = 10.^((-15 + 20*rand(1,numBuildings))/10);
treeRCS = 10.^((-30 + 15*rand(1,numTrees))/10);
movingRCS = 10.^((-25 + 15*rand(1,numMoving))/10);

%% 7. Combine Scene Objects

targetPos = [dronePos, groundPos, buildingPos, treePos, movingPos];
targetVel = [droneVel, zeros(3,numGround), zeros(3,numBuildings), zeros(3,numTrees), movingVel];
targetRCS = [droneRCS, groundRCS, buildingRCS, treeRCS, movingRCS];
targetClass = [1, 2*ones(1,numGround), 3*ones(1,numBuildings), 4*ones(1,numTrees), 5*ones(1,numMoving)];

numTargets = size(targetPos,2);

fprintf('Total scatterers: %d\n', numTargets);

%% 8. Initialize Raw IQ Data

rawH = complex(zeros(Ns,Nc,Na,'single'));
rawV = complex(zeros(Ns,Nc,Na,'single'));

%% 9. Antenna Pattern

antennaGain = @(u) max(0.05,u(2))^2;

%% 10. Receiver Noise Configuration

referenceAmplitude = sqrt(droneRCS);
noisePower = referenceAmplitude^2 / (10^(cfg.SNR_dB/10));
noiseStd = sqrt(noisePower/2);

%% 11. Generate Radar Returns

disp('Generating radar echoes...');

for targetIdx = 1:numTargets

    initialPos = targetPos(:,targetIdx);
    velocity = targetVel(:,targetIdx);
    baseRCS = targetRCS(targetIdx);
    objectClass = targetClass(targetIdx);
    initialPhase = 2*pi*rand;

    if objectClass == 1
        fluctuationDepth = 0.35;
        fluctuationFrequency = 15;
    elseif objectClass == 4
        fluctuationDepth = 0.15;
        fluctuationFrequency = 3;
    else
        fluctuationDepth = 0;
        fluctuationFrequency = 0;
    end

    for chirpIdx = 1:Nc

        slowTime = tSlow(chirpIdx);
        currentPos = initialPos + velocity*slowTime;
        rcsScale = 1 + fluctuationDepth*sin(2*pi*fluctuationFrequency*slowTime + initialPhase);
        currentRCS = baseRCS * max(0.05,rcsScale)^2;

        %% Horizontal Radar Geometry

        relH = currentPos - radarHPos;
        RH = norm(relH);

        if RH < 0.1
            continue;
        end

        uH = relH/RH;
        gainH = antennaGain(uH);
        beatH = 2*slope*RH/c;
        ampH = sqrt(currentRCS)*gainH/max(RH,1)^2;
        carrierPhaseH = 4*pi*RH/lambda;

        %% Vertical Radar Geometry

        relV = currentPos - radarVPos;
        RV = norm(relV);

        if RV < 0.1
            continue;
        end

        uV = relV/RV;
        gainV = antennaGain(uV);
        beatV = 2*slope*RV/c;
        ampV = sqrt(currentRCS)*gainV/max(RV,1)^2;
        carrierPhaseV = 4*pi*RV/lambda;

        %% Antenna Responses

        for antIdx = 1:Na

            offset = antennaOffsets(antIdx);
            spatialPhaseH = 2*pi/lambda*offset*dot(uH,arrayAxisH);
            spatialPhaseV = 2*pi/lambda*offset*dot(uV,arrayAxisV);

            phaseH = 2*pi*beatH*tFast + carrierPhaseH + spatialPhaseH + initialPhase;
            phaseV = 2*pi*beatV*tFast + carrierPhaseV + spatialPhaseV + initialPhase;

            echoH = ampH*exp(1j*phaseH);
            echoV = ampV*exp(1j*phaseV);

            rawH(:,chirpIdx,antIdx) = rawH(:,chirpIdx,antIdx) + single(echoH);
            rawV(:,chirpIdx,antIdx) = rawV(:,chirpIdx,antIdx) + single(echoV);

        end

    end

end

%% 12. Simplified Ground Multipath

reflectionCoefficient = 0.25*exp(1j*pi/3);

droneImagePos = dronePos;
droneImagePos(3) = -droneImagePos(3);

droneImageVel = droneVel;
droneImageVel(3) = -droneImageVel(3);

radarPositions = {radarHPos, radarVPos};
arrayAxes = {arrayAxisH, arrayAxisV};

for radarIdx = 1:2

    radarPos = radarPositions{radarIdx};
    arrayAxis = arrayAxes{radarIdx};

    if radarIdx == 1
        raw = rawH;
    else
        raw = rawV;
    end

    for chirpIdx = 1:Nc

        slowTime = tSlow(chirpIdx);
        imagePosition = droneImagePos + droneImageVel*slowTime;
        relativePosition = imagePosition - radarPos;
        Rimage = norm(relativePosition);

        if Rimage < 0.1
            continue;
        end

        unitVector = relativePosition/Rimage;
        beatFrequency = 2*slope*Rimage/c;
        echoAmplitude = sqrt(droneRCS)/max(Rimage,1)^2;

        for antIdx = 1:Na

            offset = antennaOffsets(antIdx);
            spatialPhase = 2*pi/lambda*offset*dot(unitVector,arrayAxis);
            phase = 2*pi*beatFrequency*tFast + 4*pi*Rimage/lambda + spatialPhase;
            reflectedEcho = reflectionCoefficient*echoAmplitude*exp(1j*phase);

            raw(:,chirpIdx,antIdx) = raw(:,chirpIdx,antIdx) + single(reflectedEcho);

        end

    end

    if radarIdx == 1
        rawH = raw;
    else
        rawV = raw;
    end

end

%% 13. Add Receiver Noise

disp('Adding receiver noise...');

noiseH = single(noiseStd)*complex(randn(Ns,Nc,Na,'single'),randn(Ns,Nc,Na,'single'));
noiseV = single(noiseStd)*complex(randn(Ns,Nc,Na,'single'),randn(Ns,Nc,Na,'single'));

rawH = rawH + noiseH;
rawV = rawV + noiseV;

%% 14. Receiver DC Offset

dcOffsetH = single(0.001*exp(1j*2*pi*rand));
dcOffsetV = single(0.001*exp(1j*2*pi*rand));

rawH = rawH + dcOffsetH;
rawV = rawV + dcOffsetV;

%% 15. Drone Ground Truth

relativeH = dronePos - radarHPos;
relativeV = dronePos - radarVPos;

rangeH = norm(relativeH);
rangeV = norm(relativeV);

azimuthH = atan2d(relativeH(1),relativeH(2));
elevationV = atan2d(relativeV(3),hypot(relativeV(1),relativeV(2)));

vrH = dot(droneVel,relativeH/rangeH);
vrV = dot(droneVel,relativeV/rangeV);

groundTruth.position = dronePos;
groundTruth.velocity = droneVel;
groundTruth.classID = cfg.drone.classID;
groundTruth.rangeH = rangeH;
groundTruth.rangeV = rangeV;
groundTruth.azimuth = azimuthH;
groundTruth.elevation = elevationV;

%% 16. Visualize Environment

figure('Name','CubeDN Realistic Outdoor Environment');
hold on;

scatter3(groundPos(1,:),groundPos(2,:),groundPos(3,:),15,[0.55 0.35 0.2],'filled');
scatter3(buildingPos(1,:),buildingPos(2,:),buildingPos(3,:),60,[0.4 0.4 0.4],'filled');
scatter3(treePos(1,:),treePos(2,:),treePos(3,:),35,[0 0.6 0],'filled');
scatter3(movingPos(1,:),movingPos(2,:),movingPos(3,:),45,[1 0.5 0],'filled');

plot3(dronePos(1),dronePos(2),dronePos(3),'rp','MarkerSize',15,'LineWidth',2);
plot3(radarHPos(1),radarHPos(2),radarHPos(3),'bo','MarkerSize',10,'LineWidth',2);
plot3(radarVPos(1),radarVPos(2),radarVPos(3),'ms','MarkerSize',10,'LineWidth',2);

xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');

title('CubeDN - Synthetic Outdoor Radar Scene');

legend('Ground','Buildings','Vegetation','Moving Clutter','Drone','Radar H','Radar V','Location','best');

grid on;
axis equal;
view(35,25);

%% 17. Save Scene and Raw IQ Data

outputFile = fullfile(cfg.dataset.outputFolder,'dual_radar_data.mat');

save(outputFile,'rawH','rawV','dronePos','droneVel','radarHPos','radarVPos','arrayAxisH','arrayAxisV','antennaOffsets','azimuthH','elevationV','rangeH','rangeV','vrH','vrV','groundTruth','targetPos','targetVel','targetRCS','targetClass','-v7.3');

%% 18. Final Summary

fprintf('\n==========================================\n');
fprintf(' CubeDN Realistic Dual Radar Complete\n');
fprintf('==========================================\n');

fprintf('Ground Scatterers     : %d\n',numGround);
fprintf('Building Scatterers   : %d\n',numBuildings);
fprintf('Vegetation Scatterers : %d\n',numTrees);
fprintf('Moving Scatterers     : %d\n',numMoving);

fprintf('Horizontal IQ Size    : [%s]\n',num2str(size(rawH)));
fprintf('Vertical IQ Size      : [%s]\n',num2str(size(rawV)));

fprintf('Drone Range H         : %.3f m\n',rangeH);
fprintf('Drone Range V         : %.3f m\n',rangeV);

fprintf('Saved: %s\n',outputFile);

disp('Realistic synthetic radar IQ generated successfully.');