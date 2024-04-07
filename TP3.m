%  Question 1
Nb = 1000;
Seq = randi([0, 1], 1, Nb);

% Question 2:
A= 1;

a_k = zeros(1, Nb);

for i = 1:length(Seq)
    if Seq(i) == 0
        a_k(i) = -A;
    else
        a_k(i) = A;
    end
end
figure();
subplot(2, 2, 1);
stem(1:numel(a_k), Seq, 'filled');
xlabel('Time');
ylabel('Amplitude');
title('Symboles a_k');

% Question 3
Q = a_k(1:2:end);
I = a_k(2:2:end);

subplot(2, 2, 2);
scatter(I, Q, 'filled');
hold on;
plot([min(I), max(I)], [0, 0], 'k--'); % Horizontal axis
plot([0, 0], [min(Q), max(Q)], 'k--'); % Vertical axis
xlabel('Axe I');
ylabel('Axe Q');
title('Diagrama IQ de Señales M-QAM');


% Question 4
% Parámetros de la señal QPSK
fc = 0.2;
T = 10;
Eb = 1;
Ts=4*T;
P = 2*Eb/Ts;

phi_k = zeros(1, Nb/2);

for k = 1:length(I)
    if I(k) == 1 && Q(k) == 1
        phi_k(k) = pi/4;
    elseif I(k) == 1 && Q(k) == -1
        phi_k(k) = 7*pi/4;
    elseif I(k) == -1 && Q(k) == 1
        phi_k(k) = 3*pi/4;
    elseif I(k) == -1 && Q(k) == -1
        phi_k(k) = 7*pi/4;
    end
end

t = 1:1:length(I);
%x_t_2 = sqrt(2*P) * (cos(2*pi*fc*t + phi_k) .* I - sin(2*pi*fc*t + phi_k) .* Q);
x_t = I + 1i*Q;

% Plot de la señal modulada
subplot(2, 2, 3);
plot(t, x_t, 'LineWidth', 1);
xlabel('Tiempo');
ylabel('Amplitud');
title('x_t');


% Question 5
% Complex Gaussian distribution (circular Gaussian)
h = (randn(1, Nb/2) + 1i*randn(1, Nb/2)) / sqrt(2); 

subplot(2, 2, 4);
plot(t, h, 'LineWidth', 1);
xlabel('Time');
ylabel('Amplitude');
title('Signal h(t)');

% Question 6
SNR_dB = 15;
SNR_lin = 10^(SNR_dB/10);
N0 = Eb/SNR_lin;

% Generate in-phase and quadrature components of noise
bi = sqrt(N0 / 2) * randn(1, Nb/2);   
bq = sqrt(N0 / 2) * randn(1, Nb/2);  

% Construct the complex envelope of noise
%b = bi.*cos(2*pi*fc*t) - bq.*sin(2* pi*fc*t);
b = bi + 1i*bq;

figure();
subplot(2, 2, 1);
plot(t, b, 'LineWidth', 1);
xlabel('Time');
ylabel('Amplitude');
title('Bruit Signal b(t)');


% Question 7
r_k = abs(h).^2 .* x_t + conj(h) .* b;

subplot(2, 2, 2);
plot(t, r_k, 'LineWidth', 1);
xlabel('Time');
ylabel('Amplitude');
title('Output Signal r(t)');


% Question 8
recu = zeros(1, Nb);
for i = 1:Nb/2
    if imag(r_k(i)) > 0
        recu((i-1)*2+1) = 1;
    else
        recu((i-1)*2+1) = 0;
    end

    if real(r_k(i)) > 0
        recu((i-1)*2+2) = 1;
    else
        recu((i-1)*2+2) = 0;
    end
end

subplot(2, 2, 3);
stem(1:Nb, recu, 'filled')
num_errores = sum(recu ~= Seq)

% Question 9
EbN0_dB = 0:1:20;
EbN0_lin = 10.^(EbN0_dB/10);
Eb = 1;
BER_mean = zeros(size(EbN0_dB));

for k = 1:length(EbN0_dB)
    N0 = Eb/EbN0_lin(k);

    bi = sqrt(N0 / 2) * randn(1, Nb/2);   
    bq = sqrt(N0 / 2) * randn(1, Nb/2);  
    b = bi + 1i*bq;
    
    r_k = abs(h).^2 .* x_t + conj(h) .* b;

    recu = zeros(1, Nb);
    for i = 1:Nb/2
        if imag(r_k(i)) > 0
            recu((i-1)*2+1) = 1;
        else
            recu((i-1)*2+1) = 0;
        end
    
        if real(r_k(i)) > 0
            recu((i-1)*2+2) = 1;
        else
            recu((i-1)*2+2) = 0;
        end
    end
    Ne = sum(recu ~= Seq);
    BER_mean(k) = Ne / Nb;
end

figure();
subplot(2, 2, 1);
semilogy(EbN0_dB, BER_mean, 'o-');
grid on;
xlabel('Eb/N0 (dB)');
ylabel('BER / Pb');
legend('BER simulada', 'Pb teórica');


% Question 10
EbN0_dB = 0:1:20;
EbN0_lin = 10.^(EbN0_dB/10);
Eb = 1;
BER_mean_2 = zeros(size(EbN0_dB));
h = ones(Nb/2);

for k = 1:length(EbN0_dB)
    N0 = Eb/EbN0_lin(k);

    bi = sqrt(N0 / 2) * randn(1, Nb/2);   
    bq = sqrt(N0 / 2) * randn(1, Nb/2);  
    b = bi + 1i*bq;
    
    %r_k = abs(h).^2 .* x_t + conj(h) .* b;
    r_k = x_t + b;

    recu = zeros(1, Nb);
    for i = 1:Nb/2
        if imag(r_k(i)) > 0
            recu((i-1)*2+1) = 1;
        else
            recu((i-1)*2+1) = 0;
        end
    
        if real(r_k(i)) > 0
            recu((i-1)*2+2) = 1;
        else
            recu((i-1)*2+2) = 0;
        end
    end
    Ne = sum(recu ~= Seq);
    BER_mean_2(k) = Ne / Nb;
end

subplot(2, 2, 2);
semilogy(EbN0_dB, BER_mean_2, 'o-', 'Color', 'blue', 'LineWidth', 2);
hold on;
semilogy(EbN0_dB, BER_mean, 'o-', 'Color', 'red', 'LineWidth', 2);
grid on;
xlabel('Eb/N0 (dB)');
ylabel('BER / Pb');
legend('Sans h_k', 'Avec h_k');


figure();
subplot(2, 2, 1);
stem(1:numel(Q), Q, 'filled');
xlabel('Time');
ylabel('Q');
title('Symboles Q');

subplot(2, 2, 2);
stem(1:numel(I), I, 'filled');
xlabel('Time');
ylabel('I');
title('Symboles I');