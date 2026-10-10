clc;
close all;

%% 1. Load Configuration

load('CubeDN_dataset/setup_config.mat', 'cfg');

data = load('CubeDN_dataset/RAD_RED_data.mat');

RAD = data.RAD;
RED = data.RED;

rangeAxis = data.rangeAxis;
velocityAxis = data.velocityAxis;
azimuthAxis = data.azimuthAxis;
elevationAxis = data.elevationAxis;



%% 2. Check Input Dimensions

[D1, R1, A] = size(RAD);
[D2, R2, E] = size(RED);

assert(D1 == D2, 'RAD and RED Doppler dimensions must match.');
assert(R1 == R2, 'RAD and RED Range dimensions must match.');

D = D1;
R = R1;

fprintf('Doppler Bins   : %d\n', D);
fprintf('Range Bins     : %d\n', R);
fprintf('Azimuth Bins   : %d\n', A);
fprintf('Elevation Bins : %d\n', E);

%% 3. Prepare RAD and RED for Fusion

% RAD dimensions: [D x R x A]
% RED dimensions: [D x R x E]

RAD4 = reshape(RAD, D, R, A, 1);
RED4 = reshape(RED, D, R, 1, E);

%% 4. Multiplicative Fusion

disp('Generating 4D radar cube...');

% Implicit expansion
% [D x R x A x 1] .* [D x R x 1 x E]

Cube4D_raw = RAD4 .* RED4;

Cube4D_raw = single(Cube4D_raw);

fprintf('Cube4D Size: [%s]\n', num2str(size(Cube4D_raw)));

%% 5. Log Normalization

% Save raw fused magnitude separately.
% Normalized version is for initial visualization
% and experimental neural-network input.

Cube4D_log = log1p(Cube4D_raw);

maxValue = max(Cube4D_log(:));

if maxValue > 0
    Cube4D = Cube4D_log ./ maxValue;
else
    Cube4D = zeros(size(Cube4D_log), 'single');
end

Cube4D = single(Cube4D);

disp('4D radar cube normalized successfully.');

%% 6. Find Strongest 4D Response

[peakValue, peakIndex] = max(Cube4D_raw(:));

[dIdx, rIdx, aIdx, eIdx] = ind2sub(size(Cube4D_raw), peakIndex);

detectedRange = rangeAxis(rIdx);
detectedVelocity = velocityAxis(dIdx);
detectedAzimuth = azimuthAxis(aIdx);
detectedElevation = elevationAxis(eIdx);

fprintf('\n==========================================\n');
fprintf('       Strongest 4D Radar Response        \n');
fprintf('==========================================\n');

fprintf('Peak Magnitude   : %.4f\n', peakValue);
fprintf('Range            : %.3f m\n', detectedRange);
fprintf('Radial Velocity  : %.3f m/s\n', detectedVelocity);
fprintf('Azimuth Axis     : %.3f deg\n', detectedAzimuth);
fprintf('Elevation Axis   : %.3f deg\n', detectedElevation);

%% 7. Range-Doppler Projection

% Maximum over azimuth and elevation

RD_projection = max(max(Cube4D, [], 4), [], 3);

figure('Name', 'CubeDN 4D Range-Doppler Projection');

imagesc(velocityAxis, rangeAxis, RD_projection.');

axis xy;
colorbar;

xlabel('Radial Velocity (m/s)');
ylabel('Range (m)');
title('CubeDN - Fused Range Doppler Projection');

grid on;

%% 8. Range-Azimuth Projection

% Maximum over Doppler and elevation

RA_projection = squeeze(max(max(Cube4D, [], 4), [], 1));

figure('Name', 'CubeDN Range-Azimuth Projection');

imagesc(azimuthAxis, rangeAxis, RA_projection);

axis xy;
colorbar;

xlabel('Horizontal Spatial Angle (deg)');
ylabel('Range (m)');
title('CubeDN - Fused Range Azimuth Projection');

grid on;

%% 9. Range-Elevation Projection

% Maximum over Doppler and azimuth

RE_projection = squeeze(max(max(Cube4D, [], 3), [], 1));

figure('Name', 'CubeDN Range-Elevation Projection');

imagesc(elevationAxis, rangeAxis, RE_projection);

axis xy;
colorbar;

xlabel('Vertical Spatial Angle (deg)');
ylabel('Range (m)');
title('CubeDN - Fused Range Elevation Projection');

grid on;

%% 10. Azimuth-Elevation Projection

% Maximum over Doppler and range

AE_projection = squeeze(max(max(Cube4D, [], 2), [], 1));

figure('Name', 'CubeDN Azimuth-Elevation Projection');

imagesc(elevationAxis, azimuthAxis, AE_projection);

axis xy;
colorbar;

xlabel('Vertical Spatial Angle (deg)');
ylabel('Horizontal Spatial Angle (deg)');
title('CubeDN - Azimuth Elevation Projection');

grid on;

%% 11. Visualize 3D Radar Response

% Collapse Doppler dimension
% Result: [Range x Azimuth x Elevation]

Cube3D = squeeze(max(Cube4D, [], 1));

% Select high-intensity voxels

threshold = 0.70;

[rGrid, aGrid, eGrid] = ndgrid(rangeAxis, azimuthAxis, elevationAxis);

mask = Cube3D >= threshold;

selectedRange = rGrid(mask);
selectedAzimuth = aGrid(mask);
selectedElevation = eGrid(mask);
selectedIntensity = Cube3D(mask);

% Convert direction-cosine coordinates into XYZ.
% Horizontal array measures ux.
% Vertical array measures uz.

ux = sind(selectedAzimuth);
uz = sind(selectedElevation);

uySquared = 1 - ux.^2 - uz.^2;

validPoints = uySquared >= 0;

selectedRange = selectedRange(validPoints);
selectedIntensity = selectedIntensity(validPoints);

ux = ux(validPoints);
uz = uz(validPoints);

uy = sqrt(uySquared(validPoints));

X = selectedRange .* ux;
Y = selectedRange .* uy;
Z = selectedRange .* uz;

figure('Name', 'CubeDN 3D Radar Response');

if ~isempty(X)

    scatter3(X, Y, Z, 20, selectedIntensity, 'filled');

    colorbar;

else

    text(0.1, 0.5, 'No voxels above threshold', 'Units', 'normalized');

end

xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');

title('CubeDN - 3D Fused Radar Response');

grid on;
axis equal;
view(35, 25);


groundTruth.position = cfg.drone.position;
groundTruth.velocity = cfg.drone.velocity;
groundTruth.classID = cfg.drone.classID;

% Range and direction cosines relative to Radar H

relativePosition = cfg.drone.position - cfg.radarH.position;

groundTruth.range = norm(relativePosition);

groundTruth.azimuth = asind(relativePosition(1) / groundTruth.range);

% Elevation spatial angle relative to Radar V

relativePositionV = cfg.drone.position - cfg.radarV.position;

groundTruth.elevation = asind(relativePositionV(3) / norm(relativePositionV));

%% 13. Save CubeDN Input

outputFile = fullfile(cfg.dataset.outputFolder, 'Cube4D_data.mat');

save(outputFile, 'Cube4D', 'Cube4D_raw', 'groundTruth', 'rangeAxis', 'velocityAxis', 'azimuthAxis', 'elevationAxis', 'detectedRange', 'detectedVelocity', 'detectedAzimuth', 'detectedElevation', '-v7.3');

%% 14. Final Summary



fprintf('Cube4D Dimensions : [%s]\n', num2str(size(Cube4D)));
fprintf('Ground Truth XYZ  : [%.2f %.2f %.2f] m\n', groundTruth.position);
fprintf('Output File       : %s\n', outputFile);

disp('CubeDN 4D radar cube saved successfully.');