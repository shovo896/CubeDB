clc;
close all;



load('CubeDN_dataset/setup_config.mat', 'cfg');

disp('======================================');
disp('    CubeDN Single Drone Simulation    ');
disp('======================================');



c = cfg.c;
fc = cfg.fc;
lambda = cfg.lambda;

fs = cfg.fs;
Tchirp = cfg.Tchirp;
slope = cfg.slope;

numSamples = cfg.numSamples;
numChirps = cfg.numChirps;

numRangeBins = cfg.numRangeBins;
numDopplerBins = cfg.numDopplerBins;



radarPos = cfg.radarH.position;

dronePos = cfg.drone.position;
droneVel = cfg.drone.velocity;

droneRCS = cfg.drone.RCS;

% Initial drone range
relativePos = dronePos - radarPos;

trueRange = norm(relativePos);

% Radial velocity: positive means moving away
radialVelocity = dot(droneVel,relativePos)/trueRange;

% Doppler frequency
fd = 2*radialVelocity/lambda;

fprintf('True Drone Range    : %.3f m\n',trueRange);
fprintf('True Radial Velocity: %.3f m/s\n',radialVelocity);
fprintf('Doppler Frequency   : %.2f Hz\n',fd);

%% 4. Generate Raw FMCW IQ Data

% Fast-time samples
tFast = (0:numSamples-1)'/fs;



rawIQ = complex(zeros(numSamples,numChirps));

% Set reproducible noise
rng(42);

% Simplified signal amplitude
signalAmplitude = sqrt(droneRCS);

% Chirp repetition interval
PRI = Tchirp;

for chirpIdx = 1:numChirps

    % Slow-time instant
    tSlow = (chirpIdx-1)*PRI;

    % Updated drone range
    currentRange = trueRange + radialVelocity*tSlow;

    % Round-trip delay
    tau = 2*currentRange/c;

    % FMCW beat frequency
    fBeat = slope*tau;

    % Positive-frequency beat signal convention
    phase = 2*pi*(fBeat*tFast + fd*tSlow);

    % Complex received beat signal
    receivedSignal = signalAmplitude*exp(1j*phase);

    % Add complex Gaussian noise
    signalPower = mean(abs(receivedSignal).^2);

    noisePower = signalPower/10^(cfg.SNR_dB/10);

    noise = sqrt(noisePower/2)*( ...
        randn(numSamples,1) + ...
        1j*randn(numSamples,1));

    rawIQ(:,chirpIdx) = receivedSignal + noise;

end

disp('Raw FMCW IQ data generated successfully.');

%% 5. Range FFT

% Hann window along fast-time
rangeWindow = hann(numSamples);

windowedIQ = rawIQ .* rangeWindow;

rangeFFT = fft(windowedIQ,numRangeBins,1);

% Positive beat frequencies only
numPositiveBins = floor(numRangeBins/2);

rangeFFT = rangeFFT(1:numPositiveBins,:);

% Range axis
rangeAxis = (0:numPositiveBins-1)' * ...
    fs/numRangeBins * c/(2*slope);

%% 6. Doppler FFT

dopplerWindow = hann(numChirps).';

windowedRangeFFT = rangeFFT .* dopplerWindow;

RD = fftshift( ...
    fft(windowedRangeFFT,numDopplerBins,2),2);

% Doppler frequency axis
dopplerAxis = ( ...
    -numDopplerBins/2 : numDopplerBins/2-1) ...
    /(numDopplerBins*PRI);

% Velocity axis
velocityAxis = dopplerAxis*lambda/2;

%% 7. Range-Doppler Magnitude

RDmag = abs(RD);

RDdB = 20*log10(RDmag + eps);

% Normalize for display
RDdB = RDdB - max(RDdB(:));

%% 8. Find Drone Peak

[~,maxIdx] = max(RDmag(:));

[rangeIdx,dopplerIdx] = ind2sub( ...
    size(RDmag),maxIdx);

estimatedRange = rangeAxis(rangeIdx);

estimatedVelocity = velocityAxis(dopplerIdx);



fprintf('True Range       : %.3f m\n',trueRange);
fprintf('Estimated Range  : %.3f m\n',estimatedRange);

fprintf('True Velocity    : %.3f m/s\n',radialVelocity);
fprintf('Estimated Velocity: %.3f m/s\n',estimatedVelocity);


%% 9. Plot Raw IQ Signal

figure('Name','Raw FMCW IQ Signal');

plot(tFast*1e6,real(rawIQ(:,1)),'b');
hold on;

plot(tFast*1e6,imag(rawIQ(:,1)),'r');

xlabel('Fast Time (\mus)');
ylabel('Amplitude');

title('Single Drone - Raw FMCW IQ Signal');

legend('I','Q');
grid on;

%% 10. Plot Range Profile

figure('Name','Range Profile');

rangeProfile = mean(abs(rangeFFT),2);

plot(rangeAxis, ...
    20*log10(rangeProfile + eps),'LineWidth',1.5);

xlabel('Range (m)');
ylabel('Magnitude (dB)');

title('Single Drone - Range FFT');

xlim([0 cfg.maxRange]);
grid on;

%% 11. Plot Range-Doppler Map

figure('Name','Range Doppler Map');

imagesc(velocityAxis,rangeAxis,RDdB);

axis xy;
colorbar;

xlabel('Radial Velocity (m/s)');
ylabel('Range (m)');

title('Single Drone - Range Doppler Map');

ylim([0 cfg.maxRange]);

clim([-60 0]);

hold on;

plot(estimatedVelocity,estimatedRange, ...
    'rx','MarkerSize',12,'LineWidth',2);

legend('Detected Drone');

%% 12. Save Data

outputFile = fullfile( ...
    cfg.dataset.outputFolder, ...
    'single_drone_data.mat');

save(outputFile, ...
    'rawIQ', ...
    'rangeFFT', ...
    'RD', ...
    'rangeAxis', ...
    'velocityAxis', ...
    'dronePos', ...
    'droneVel', ...
    'trueRange', ...
    'radialVelocity', ...
    'estimatedRange', ...
    'estimatedVelocity', ...
    '-v7.3');

disp('Single Drone Simulation Complete!');
disp(['Data saved to: ',outputFile]);