%  Question 1
Nb = 1000;
Seq = randi([0, 1], 1, Nb);
fc = 0.2;
T = 10;
Ts=4*T;

% Question 2:
A= 2/sqrt(10);

a_k = zeros(1, Nb/2);
j = 1;
for i = 1:2:length(Seq)
    if Seq(i) == 0 && Seq(i+1) == 0
        a_k(j) = 3*A;
    elseif Seq(i) == 1 && Seq(i+1) == 0
        a_k(j) = A;
    elseif Seq(i) == 1 && Seq(i+1) == 1
        a_k(j) = -A;
    else
        a_k(j) = -3*A;
    end
    j = j+1;
end

% Question 3
S1 = a_k(1:2:end);
S2 = a_k(2:2:end); %Impaires

%Question 4
Ns = length(S1);
peine_dirac_S1 = zeros(1, Ns * Ts);
peine_dirac_S2 = zeros(1, Ns * Ts);
j = 1;
% Modulation QAM
for k = 1:Ns
    % S1 (k*Ts + 1)
    peine_dirac_S1((k-1)*Ts + 1) = S1(j);
    % S2 (k*Ts + 1)
    peine_dirac_S2((k-1)*Ts + 1) = S2(j);
    j = j+1;
end
figure();
subplot(2, 2, 1);
stem(peine_dirac_S1);
xlabel('Temp');
ylabel('Amplitude');
title('Peine de Dirac QAM généré avec des symboles S1');

subplot(2, 2, 2);
stem(peine_dirac_S2);
xlabel('Temp');
ylabel('Amplitude');
title('Peine de Dirac QAM généré avec des symboles S2');

%Quesiton 5
resp_imp = (1/sqrt(Ts))*ones(1,Ts);
I = filter(resp_imp, 1, peine_dirac_S1);
Q = filter(resp_imp, 1, peine_dirac_S2);

subplot(2, 2, 3);
plot(I);
title('Signal de sortie après filtrage 1');
xlabel('Tiempo');
ylabel('Amplitud');

subplot(2, 2, 4);
plot(Q);
title('Signal de sortie après filtrage 2');
xlabel('Tiempo');
ylabel('Amplitud');

%Question 6
% 16-QAM
figure;
subplot(2, 2, 1);
scatter(I, Q, 'filled');
xlabel('Axe I');
ylabel('Axe Q');
title('Diagrama IQ de Señales 16-QAM');


%Question 7
t = 1:1:length(I);

% Generate signal x(t)
x_t = I .* cos(2*pi*fc*t) - Q .* sin(2*pi*fc*t);

% Plot the generated signal
subplot(2, 2, 2);
plot(t, x_t);
xlabel('Time');
ylabel('Amplitude');
title('Modulated Signal x(t)');

%Question 8
% Define some placeholder variables
fe = 1; % 10 MHz

% Compute DSP_x using the given formula
X = fftshift(fft(x_t));
DSP_x = X.*cos(X)./length(X);
Axe_frec = -fe/2:fe/length(X):(fe/2 - 1/length(X));

subplot(2, 2, 3);
semilogy(Axe_frec, DSP_x);
xlabel('Frequency (Hz)');
ylabel('Power/Frequency');
title('Power Spectral Density of x(t)');

%Question 9
Eb = 1;
SNR_dB = 10;
SNR_lin = 10^(SNR_dB/10);
No = Eb/SNR_lin;

sigma_2 = sqrt(No/2);

bruit = sigma_2 * randn(size(x_t));
%bruit = 0;
r_t = x_t + bruit;

subplot(2, 2, 4);
plot(r_t);
title('Signal de sortie avec bruit gaussien');
xlabel('Temp');
ylabel('Amplitude');

%Quesiton 10
r1_t = 2.*r_t.*cos(2*pi*fc*t);
r2_t = -2.*r_t.*sin(2*pi*fc*t);

figure();
subplot(2, 2, 1);
plot(r1_t);
title('r_1 (t)');
xlabel('Temp');
ylabel('Amplitude');

subplot(2, 2, 2);
plot(r2_t);
title('r_2 (t)');
xlabel('Temp');
ylabel('Amplitude');

%Question 11
R1 = fftshift(fft(r1_t));
DSP_r1 = R1.*cos(R1)./length(R1);
Axe_frec_R1 = -fe/2:fe/length(R1):(fe/2 - 1/length(R1));

R2 = fftshift(fft(r2_t));
DSP_r2 = R2.*cos(R2)./length(R2);
Axe_frec_R2 = -fe/2:fe/length(R2):(fe/2 - 1/length(R2));

subplot(2, 2, 3);
semilogy(Axe_frec_R1, DSP_r1);
xlabel('Frequency (Hz)');
ylabel('Power/Frequency');
title('Power Spectral Density of r_1(t)');

subplot(2, 2, 4);
semilogy(Axe_frec_R2, DSP_r2);
xlabel('Frequency (Hz)');
ylabel('Power/Frequency');
title('Power Spectral Density of r_2(t)');

%Question 12
g_t = resp_imp; 
g_t_invertida = fliplr(g_t);
r1_filtre = filter(g_t_invertida, 1, r1_t);
r2_filtre = filter(g_t_invertida, 1, r2_t);

figure();
subplot(2, 2, 1);
plot(r1_filtre);
title('Signal filtré avec filtre adapté (r_1(t))');
xlabel('Temps');
ylabel('Amplitude');

subplot(2, 2, 2);
plot(r2_filtre);
title('Signal filtré avec filtre adapté (r_2(t))');
xlabel('Temps');
ylabel('Amplitude');


%Question 13
R1_filre = fftshift(fft(r1_filtre));
DSP_r1_filtre = R1_filre.*cos(R1_filre)./length(R1_filre);
Axe_frec_R1_filtre = -fe/2:fe/length(R1_filre):(fe/2 - 1/length(R1_filre));

R2_filtre = fftshift(fft(r2_filtre));
DSP_r2_filtre = R2_filtre.*cos(R2_filtre)./length(R2_filtre);
Axe_frec_R2_filtre = -fe/2:fe/length(R2_filtre):(fe/2 - 1/length(R2_filtre));

subplot(2, 2, 3);
semilogy(Axe_frec_R1_filtre, DSP_r1_filtre);
xlabel('Frequency (Hz)');
ylabel('Power/Frequency');
title('Power Spectral Density of r_1(t) filtre');

subplot(2, 2, 4);
semilogy(Axe_frec_R2_filtre, DSP_r2_filtre);
xlabel('Frequency (Hz)');
ylabel('Power/Frequency');
title('Power Spectral Density of r_2(t) filtre');


%Question 14
NT = Nb/4*Ts;

r1_fil_desp = r1_filtre(Ts:end);
r2_fil_desp = r2_filtre(Ts:end);

vect1 = downsample(r1_fil_desp, Ts);
vect2 = downsample(r2_fil_desp, Ts);

r1_echant =zeros (1, NT+1);
r2_echant =zeros (1, NT+1);

for i = 1:length(vect1)
    index = (i-1)*Ts + 1;
    r1_echant(index) = vect1(i);
    r2_echant(index) = vect2(i); 
end

figure();
subplot(2, 2, 1);
stem(r1_echant);
title('Signal échantillonné à vitesse de symbole Ts pour r_1(t)');
xlabel('Échantillon');
ylabel('Amplitude');

subplot(2, 2, 2);
stem(r2_echant);
title('Signal échantillonné à vitesse de symbole Ts pour r_2(t)');
xlabel('Échantillon');
ylabel('Amplitude');

%Question 15
decode_r1 = zeros(1, Nb/4);
decode_r2 = zeros(1, Nb/4);

for i = 1:length(vect1)
    index = (i-1)*Ts + 1;
    if r1_echant(index) > 2*A
        decode_r1(i) = 3*A;
    elseif r1_echant(index) <= 2*A && r1_echant(index) > 0
        decode_r1(i) = A;
    elseif r1_echant(index) >= -2*A && r1_echant(index) <= 0
        decode_r1(i) = -A;
    elseif r1_echant(index) < -2*A
        decode_r1(i) = -3*A;
    end
end

for i = 1:length(vect2)
    index = (i-1)*Ts + 1;
    if r2_echant(index) > 2*A
        decode_r2(i) = 3*A;
    elseif r2_echant(index) <= 2*A && r2_echant(index) > 0
        decode_r2(i) = A;
    elseif r2_echant(index) >= -2*A && r2_echant(index) <= 0
        decode_r2(i) = -A;
    elseif r2_echant(index) < -2*A
        decode_r2(i) = -3*A;
    end
end


%Question 16
Seq_S1 = zeros(1, Nb/2);
Seq_S2 = zeros(1, Nb/2);
j = 1;

for i = 1:2:Nb/2
    if decode_r1(j) == 3*A
        Seq_S1(i) = 0;
        Seq_S1(i+1) = 0;
    elseif decode_r1(j) == A
        Seq_S1(i) = 1;
        Seq_S1(i+1) = 0;
    elseif decode_r1(j) == -A
        Seq_S1(i) = 1;
        Seq_S1(i+1) = 1;
    else
        Seq_S1(i) = 0;
        Seq_S1(i+1) = 1;
    end
    j = j+1;
end

j = 1;
for i = 1:2:Nb/2
    if decode_r2(j) == 3*A
        Seq_S2(i) = 0;
        Seq_S2(i+1) = 0;
    elseif decode_r2(j) == A
        Seq_S2(i) = 1;
        Seq_S2(i+1) = 0;
    elseif decode_r2(j) == -A
        Seq_S2(i) = 1;
        Seq_S2(i+1) = 1;
    else
        Seq_S2(i) = 0;
        Seq_S2(i+1) = 1;
    end
    j = j+1;
end


Seq_recu = zeros(1, Nb);
j = 1;
for i = 1:4:length(Seq_recu)
    Seq_recu(i) = Seq_S1(j);
    Seq_recu(i+1) = Seq_S1(j+1);
    Seq_recu(i+2) = Seq_S2(j);
    Seq_recu(i+3) = Seq_S2(j+1);
    j = j+2;
end

error_S1_sym = sum(decode_r1 ~= S1);
error_S2_sym = sum(decode_r2 ~= S2);
error_bit = sum(Seq ~= Seq_recu);


disp(['Au total,  ', num2str(Nb), ' données ont été envoyées']);
if (error_S1_sym + error_S2_sym) > 0
    disp(['Ils ont été détectés pour les Symboles S1 ', num2str(error_S1_sym), ' erreurs de transmission.']);
    disp(['Ils ont été détectés pour les Symboles S2 ', num2str(error_S2_sym), ' erreurs de transmission.']);
    disp(['Ils ont été détectés pour les bits ', num2str(error_bit), ' erreurs de transmission.']);
else
    disp('Aucune erreur de transmission n´a été détectée.');
end


%QuestionS 17, 18 et 19
EbN0_dB = 0:1:16;
EbN0_lin = 10.^(EbN0_dB/10);
Eb = 1;
M = 4;

BER_mean_S1 = zeros(size(EbN0_dB));
BER_mean_S2 = zeros(size(EbN0_dB));
BER_mean_t = zeros(size(EbN0_dB));
Pe = zeros(size(EbN0_dB));


for k = 1:length(EbN0_dB)
    No = Eb/EbN0_lin(k);
    sigma = sqrt(No/2);
    bruit = sigma * randn(size(x_t));
    r_t = x_t + bruit;

    r1_t = 2.*r_t.*cos(2*pi*fc*t);
    r2_t = -2.*r_t.*sin(2*pi*fc*t);

    g_t = resp_imp; 
    g_t_invertida = fliplr(g_t);
    r1_filtre = filter(g_t_invertida, 1, r1_t);
    r2_filtre = filter(g_t_invertida, 1, r2_t);
    
    r1_fil_desp = r1_filtre(Ts:end);
    r2_fil_desp = r2_filtre(Ts:end);
    
    vect1 = downsample(r1_fil_desp, Ts);
    vect2 = downsample(r2_fil_desp, Ts);

    
    r1_echant =zeros (1, NT+1);
    r2_echant =zeros (1, NT+1);
    
    for i = 1:length(vect1)
        index = (i-1)*Ts + 1;
        r1_echant(index) = vect1(i);
        r2_echant(index) = vect2(i); 
    end

    decode_r1 = zeros(1, Nb/4);
    decode_r2 = zeros(1, Nb/4);
    
    for i = 1:length(vect1)
        index = (i-1)*Ts + 1;
        if r1_echant(index) > 2*A
            decode_r1(i) = 3*A;
        elseif r1_echant(index) <= 2*A && r1_echant(index) > 0
            decode_r1(i) = A;
        elseif r1_echant(index) >= -2*A && r1_echant(index) <= 0
            decode_r1(i) = -A;
        elseif r1_echant(index) < -2*A
            decode_r1(i) = -3*A;
        end
    end
    
    for i = 1:length(vect2)
        index = (i-1)*Ts + 1;
        if r2_echant(index) > 2*A
            decode_r2(i) = 3*A;
        elseif r2_echant(index) <= 2*A && r2_echant(index) > 0
            decode_r2(i) = A;
        elseif r2_echant(index) >= -2*A && r2_echant(index) <= 0
            decode_r2(i) = -A;
        elseif r2_echant(index) < -2*A
            decode_r2(i) = -3*A;
        end
    end
    
    Seq_S1 = zeros(1, Nb/2);
    Seq_S2 = zeros(1, Nb/2);
    j = 1;
    
    for i = 1:2:Nb/2
        if decode_r1(j) == 3*A
            Seq_S1(i) = 0;
            Seq_S1(i+1) = 0;
        elseif decode_r1(j) == A
            Seq_S1(i) = 1;
            Seq_S1(i+1) = 0;
        elseif decode_r1(j) == -A
            Seq_S1(i) = 1;
            Seq_S1(i+1) = 1;
        else
            Seq_S1(i) = 0;
            Seq_S1(i+1) = 1;
        end
        j = j+1;
    end
    
    j = 1;
    for i = 1:2:Nb/2
        if decode_r2(j) == 3*A
            Seq_S2(i) = 0;
            Seq_S2(i+1) = 0;
        elseif decode_r2(j) == A
            Seq_S2(i) = 1;
            Seq_S2(i+1) = 0;
        elseif decode_r2(j) == -A
            Seq_S2(i) = 1;
            Seq_S2(i+1) = 1;
        else
            Seq_S2(i) = 0;
            Seq_S2(i+1) = 1;
        end
        j = j+1;
    end
    
    Seq_recu = zeros(1, Nb);
    j = 1;
    for i = 1:4:length(Seq_recu)
        Seq_recu(i) = Seq_S1(j);
        Seq_recu(i+1) = Seq_S1(j+1);
        Seq_recu(i+2) = Seq_S2(j);
        Seq_recu(i+3) = Seq_S2(j+1);
        j = j+2;
    end
    
    error_S1_sym = sum(decode_r1 ~= S1);
    error_S2_sym = sum(decode_r2 ~= S2);
    error_bit = sum(Seq ~= Seq_recu);

    BER_mean_S1(k) = error_S1_sym / length(S1);
    BER_mean_S2(k) = error_S2_sym / length(S2);

    Ne = error_bit;
    BER_mean_t(k) = Ne / Nb;

    Pe(k) = ((sqrt(M)-1)/sqrt(M))*erfc(sqrt((3*log2(sqrt(M))*Eb)/(2*(M-1)*No)));

end

figure();
subplot(2,2,1);
semilogy(EbN0_dB, BER_mean_t, 'o-');
xlabel('Eb/N0 (dB)');
ylabel('Error Total (BER)');
title('BER vs Eb/N0');
grid on;
legend('Total', 'S1', 'S2');

subplot(2,2,2);
semilogy(EbN0_dB, BER_mean_S1, 'o-');
hold on;
semilogy(EbN0_dB, BER_mean_S2, 'o-');
xlabel('Eb/N0 (dB)');
ylabel('Error S1 et S2 (BER)');
title('BER vs Eb/N0');
grid on;
legend('S1', 'S2');


subplot(2,2,3);
semilogy(EbN0_dB, Pe, 'o-');
xlabel('Eb/N0 (dB)');
ylabel('Error Théorique (BER)');
title('BER vs Eb/N0');
grid on;

subplot(2,2,4);
semilogy(EbN0_dB, BER_mean_t, 'o-');
hold on;
semilogy(EbN0_dB, BER_mean_S1, 'o-');
semilogy(EbN0_dB, BER_mean_S2, 'o-');
semilogy(EbN0_dB, Pe, 'o-');
xlabel('Eb/N0 (dB)');
ylabel('Error (BER)');
title('BER vs Eb/N0');
grid on;
legend('Total', 'S1', 'S2', 'Théorique');
