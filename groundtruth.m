clc;
close all;

%% 1. Load Configuration

load('CubeDN_dataset/setup_config.mat','cfg');
data = load('CubeDN_dataset/Cube4D_data.mat');

Cube4D = data.Cube4D;
rangeAxis = data.rangeAxis;
azimuthAxis = data.azimuthAxis;
elevationAxis = data.elevationAxis;
velocityAxis = data.velocityAxis;

%% 2. Load Ground Truth

gt = data.groundTruth;

dronePos = gt.position;
droneVel = gt.velocity;
classID = gt.classID;

fprintf('Drone Position: [%.2f %.2f %.2f] m\n',dronePos);
fprintf('Drone Class: %d\n',classID);

%% 3. Get Cube Dimensions

[D,R,A,E] = size(Cube4D);

fprintf('Cube Dimensions: [%d %d %d %d]\n',D,R,A,E);

%% 4. Calculate Drone Geometry

radarHPos = cfg.radarH.position;
radarVPos = cfg.radarV.position;

relativeH = dronePos - radarHPos;
relativeV = dronePos - radarVPos;

trueRange = norm(relativeH);

trueAzimuth = asind(relativeH(1)/norm(relativeH));
trueElevation = asind(relativeV(3)/norm(relativeV));

%% 5. Find Nearest Radar Bins

[~,rIdx] = min(abs(rangeAxis-trueRange));
[~,aIdx] = min(abs(azimuthAxis-trueAzimuth));
[~,eIdx] = min(abs(elevationAxis-trueElevation));

fprintf('Range Bin: %d\n',rIdx);
fprintf('Azimuth Bin: %d\n',aIdx);
fprintf('Elevation Bin: %d\n',eIdx);

%% 6. Gaussian Parameters

if classID == 1
    sigma = 1;
else
    sigma = 2;
end

peakConfidence = 1.0;
confidenceThreshold = 0.05;

%% 7. Generate 3D Gaussian Confidence Cube

[rGrid,aGrid,eGrid] = ndgrid(1:R,1:A,1:E);

distanceSquared = (rGrid-rIdx).^2 + (aGrid-aIdx).^2 + (eGrid-eIdx).^2;

ConfidenceCube = peakConfidence*exp(-distanceSquared/(2*sigma^2));

ConfidenceCube(ConfidenceCube<confidenceThreshold) = 0;

ConfidenceCube = single(ConfidenceCube);

%% 8. Create Class Confidence Labels

numClasses = cfg.dataset.numClasses;

ClassConfidence = zeros(R,A,E,numClasses,'single');

assert(classID>=1 && classID<=numClasses,'Invalid drone class ID.');

ClassConfidence(:,:,:,classID) = ConfidenceCube;

%% 9. Visualize Range-Azimuth Label

RA_label = squeeze(max(ConfidenceCube,[],3));

figure('Name','CubeDN Ground Truth Range Azimuth');

imagesc(azimuthAxis,rangeAxis,RA_label);
axis xy;
colorbar;
xlabel('Azimuth Spatial Angle (deg)');
ylabel('Range (m)');
title('Ground Truth - Range Azimuth');

%% 10. Visualize Range-Elevation Label

RE_label = squeeze(max(ConfidenceCube,[],2));

figure('Name','CubeDN Ground Truth Range Elevation');

imagesc(elevationAxis,rangeAxis,RE_label);
axis xy;
colorbar;
xlabel('Elevation Spatial Angle (deg)');
ylabel('Range (m)');
title('Ground Truth - Range Elevation');

%% 11. Visualize Azimuth-Elevation Label

AE_label = squeeze(max(ConfidenceCube,[],1));

figure('Name','CubeDN Ground Truth Azimuth Elevation');

imagesc(elevationAxis,azimuthAxis,AE_label);
axis xy;
colorbar;
xlabel('Elevation Spatial Angle (deg)');
ylabel('Azimuth Spatial Angle (deg)');
title('Ground Truth - Azimuth Elevation');

%% 12. Save Ground Truth

outputFile = fullfile(cfg.dataset.outputFolder,'groundtruth_data.mat');

save(outputFile,'ConfidenceCube','ClassConfidence','dronePos','droneVel','classID','trueRange','trueAzimuth','trueElevation','rIdx','aIdx','eIdx','rangeAxis','azimuthAxis','elevationAxis','-v7.3');

%% 13. Final Output



fprintf('Confidence Cube Size: [%s]\n',num2str(size(ConfidenceCube)));
fprintf('Class Confidence Size: [%s]\n',num2str(size(ClassConfidence)));
fprintf('Saved: %s\n',outputFile);