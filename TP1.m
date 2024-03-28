Fs = 1;
Ts = 1;
Db = 0.1;
Tb = 10;
Fc = 0.2;

%Nb = 10;
Nb = 100;
A= 1;

%Question 1:
figure();
Seq = randi([0, 1], 1, Nb);
a_k = zeros(1, Nb);

for i = 1:length(Seq)
    if Seq(i) == 0
        a_k(i) = -A;
    else
        a_k(i) = A;
    end
end

%Question 2:
subplot(2, 2, 1);
scatter(1:numel(a_k), a_k);
xlabel('Time');
ylabel('Amplitude');
title('Symboles a_k');


% Question 3:
NT = Nb * Tb;
dirac_comb = zeros(1, NT);

for i = 1:Nb
    dirac_comb((i-1)*Tb + 1) = a_k(i); % Colocar el símbolo a_k cada 10 puntos
    %dirac_comb((i)*Tb + 1) = a_k(i);
end

subplot(2, 2, 2);
stem(dirac_comb);
xlabel('Tiempo');
ylabel('Amplitud');
title('Peine de Dirac con símbolos a_k');


% Question 4:
resp_imp = (1/sqrt(Tb))*ones(1,Tb);
output = filter(resp_imp, 1, dirac_comb);

subplot(2, 2, 3);
plot(output);
title('Señal de Salida después del Filtrado');
xlabel('Tiempo');
ylabel('Amplitud');



%Question 5:
unif_rand = rand(1, 1000);

figure();
subplot(2, 2, 1);
hist(unif_rand, 20); % 20 bins for the histogram
title('Histogram of Uniformly Distributed Random Variable');
xlabel('Value');
ylabel('Frequency');

sum_variable = sum(rand(100, 1000));

% Plot the histogram of the sum of 100 uniformly distributed random variables
subplot(2, 2, 2);
hist(sum_variable, 20); 
title('Histogram of the Sum of 100 Uniformly Distributed Random Variables');
xlabel('Value');
ylabel('Frequency');

%Question 6
miu = 50; 
sigma = 10;

% Generar muestras de la distribución gaussiana
var_gaus = sigma * randn(1000, 1) + miu;

% Trazar el histograma
subplot(2, 2, 3);
histogram(var_gaus, 'Normalization', 'probability');
title('Histograma de Distribución Gaussiana');
xlabel('Valor');
ylabel('Probabilidad');


%Question 8
Eb = 1;
SNR_dB = 3;
SNR_lin = 10^(SNR_dB/10);
No = Eb/SNR_lin;

sigma_2 = sqrt(No/2);

bruit = sigma_2 * randn(size(output));
signal_bruit = output + bruit;

figure();
subplot(2, 2, 1);
plot(signal_bruit);
title('Señal de Salida con Ruido Gaussiano');
xlabel('Tiempo');
ylabel('Amplitud');

%Question 9
g_t = resp_imp; 
g_t_invertida = fliplr(g_t);
%output_filtre = filter(g_t, 1, signal_bruit);
output_filtre = filter(g_t_invertida, 1, signal_bruit);

% Trazar señal filtrada
subplot(2, 2, 2);
plot(output_filtre);
title('Señal Filtrada con Filtro Adaptado');
xlabel('Tiempo');
ylabel('Amplitud');

%Question 10
NT = Nb * Tb;
retard_g = length(g_t) - 1; % Retraso introducido por el filtro de conformación
retard_g_inv = length(g_t_invertida) - 1; % Retraso introducido por el filtro adaptado
retard = round((retard_g + retard_g_inv)*0.1);

%output_echant = downsample(output_filtre, Tb, retard);
vect = downsample(output_filtre, Tb);
output_echant = circshift(vect, [0, -1]);
output_echant_2 =zeros (1, NT+1);
output_echant_3 =zeros (1, NT+1);
for i = 1:Nb
    output_echant_2((i-1)*Tb + 1) = output_echant(i); % Colocar el símbolo a_k cada 10 puntos
    if output_echant_2((i-1)*Tb + 1) > 0
        output_echant_3((i-1)*Tb + 1) = 1;
    else
        output_echant_3((i-1)*Tb + 1) = -1;
    end
end

subplot(2, 2, 3);
stem(output_echant_2);
title('Señal Muestreada a Velocidad de Símbolo');
xlabel('Muestra');
ylabel('Amplitud');

subplot(2, 2, 4);
stem(output_echant_3);
title('Señal 2 Velocidad de Símbolo');
xlabel('Muestra');
ylabel('Amplitud');

%Question 11
a_k_recu = zeros(1, Nb);

for i = 1:Nb
    if output_echant_3((i-1)*Tb + 1) == 1
        a_k_recu(i) = 1;
    else
        a_k_recu(i) = 0;
    end
end

%Question 12
num_errores = sum(a_k_recu ~= Seq);

% Determinar si hay errores de transmisión
if num_errores > 0
    disp(['Se detectaron ', num2str(num_errores), ' errores de transmisión.']);
else
    disp('No se detectaron errores de transmisión.');
end

%Question 13

EbN0_dB = 0:1:10;  % Rango de valores de Eb/N0 en dB
EbN0_lin = 10.^(EbN0_dB/10);  % Convertir de dB a escala lineal


BER_mean = zeros(size(EbN0_dB));
BER_variance = zeros(size(EbN0_dB));

% Simular la transmisión y calcular la BER para cada valor de Eb/N0
for j = 1:length(EbN0_dB)
    Eb = 1;
    No = Eb/EbN0_lin(j);
    sigma_2 = sqrt(No/2);
    bruit = sigma_2 * randn(size(output));
    signal_bruit = output + bruit;

    g_t = resp_imp; 
    g_t_invertida = fliplr(g_t);
    output_filtre = filter(g_t, 1, signal_bruit);
    
    vect = downsample(output_filtre, Tb);
    output_echant = circshift(vect, [0, -1]);
    output_echant_2 =zeros (1, NT+1);
    output_echant_3 =zeros (1, NT+1);
    for i = 1:Nb
        output_echant_2((i-1)*Tb + 1) = output_echant(i); % Colocar el símbolo a_k cada 10 puntos
        if output_echant_2((i-1)*Tb + 1) > 0
            output_echant_3((i-1)*Tb + 1) = 1;
        else
            output_echant_3((i-1)*Tb + 1) = -1;
        end
    end
    
    a_k_recu = zeros(1, Nb);
    
    for i = 1:Nb
        if output_echant_3((i-1)*Tb + 1) == 1
            a_k_recu(i) = 1;
        else
            a_k_recu(i) = 0;
        end
    end
    
    Ne = sum(a_k_recu ~= Seq);

    % Calcular la media de BER para este valor de Eb/N0
    BER_mean(j) = Ne / Nb;
    % Calcular la varianza de BER para este valor de Eb/N0
    BER_variance(j) = BER_mean(j) * (1 - BER_mean(j)) / Nb;
end


%Question 14
Pb_theoretical = 0.5 * erfc(sqrt(EbN0_lin));  % Utilizando la función erfc

% Graficar la tasa de error binaria (BER) simulada y teórica
figure;
subplot(2, 2, 1);
errorbar(EbN0_dB, BER_mean, sqrt(BER_variance), 'o-', 'LineWidth', 2);
hold on;
plot(EbN0_dB, Pb_theoretical, 'r--', 'LineWidth', 2);
grid on;
xlabel('Eb/N0 (dB)');
ylabel('BER / Pb');
title('Comparación de la tasa de error binaria (BER) simulada con la teórica Pb');
legend('BER simulada', 'Pb teórica');

%Question 15

% Definir parámetros
EbN0_dB = 0:1:10;  % Rango de valores de Eb/N0 en dB
EbN0_lin = 10.^(EbN0_dB/10);  % Convertir de dB a escala lineal


BER_mean = zeros(size(EbN0_dB));
BER_variance = zeros(size(EbN0_dB));


% Simular la transmisión y calcular la BER para cada valor de Eb/N0
for k = 1:3
    for j = 1:length(EbN0_dB)
        Eb = 1;
        No = Eb/EbN0_lin(j);
        sigma_2 = sqrt(No/2);
        bruit = sigma_2 * randn(size(output));
        signal_bruit = output + bruit;
        
        if k == 1 
            g_t = randn(1, Tb);
            h = fliplr(g_t);
        elseif k == 2
            g_t = ones(1, Tb);
            h = [0:Tb/2 Tb/2-1:-1:1];
        else
            g_t = randn(1, Tb);
            h = g_t;
        end

   
        output_filtre = filter(h, 1, signal_bruit);
        
        vect = downsample(output_filtre, Tb);
        output_echant = circshift(vect, [0, -1]);
        output_echant_2 =zeros (1, NT+1);
        output_echant_3 =zeros (1, NT+1);
        for i = 1:Nb
            output_echant_2((i-1)*Tb + 1) = output_echant(i); % Colocar el símbolo a_k cada 10 puntos
            if output_echant_2((i-1)*Tb + 1) > 0
                output_echant_3((i-1)*Tb + 1) = 1;
            else
                output_echant_3((i-1)*Tb + 1) = -1;
            end
        end
        
        a_k_recu = zeros(1, Nb);
        
        for i = 1:Nb
            if output_echant_3((i-1)*Tb + 1) == 1
                a_k_recu(i) = 1;
            else
                a_k_recu(i) = 0;
            end
        end
        
        Ne = sum(a_k_recu ~= Seq);
    
        % Calcular la media de BER para este valor de Eb/N0
        BER_mean(j) = Ne / Nb;
        % Calcular la varianza de BER para este valor de Eb/N0
        BER_variance(j) = BER_mean(j) * (1 - BER_mean(j)) / Nb;
    end
    subplot(2, 2, k+1);
    errorbar(EbN0_dB, BER_mean, sqrt(BER_variance), 'o-', 'LineWidth', 2);
    hold on;
    grid on;
    xlabel('Eb/N0 (dB)');
    ylabel('BER / Pb');
    titulo = sprintf('Tasa de error binaria (BER) %d', k);
    title(titulo);
end    


%Question 16
t = -2*Tb:2*Tb;
alpha = 0.22;
sigma =4*Tb;


f2 = exp(-t.^2/(2*sigma)); % Gaussien

f1 = zeros(size(t)); %Nyquist

for i = 1:length(t)
    if t(i) == 0
        f1(i) = 1;
    else
        f1(i) = (sin(pi*t(i)/Tb).*cos(pi*alpha*t(i)/Tb))./(pi*t(i)/Tb.*(1-4*(alpha*t(i)/Tb).^2));
    end
end

%Question 17
% Respuesta impulsiva del filtro f1 (Nyquist)
figure;
subplot(2, 2, 1);
stem(t, f1);
xlabel('Tiempo');
ylabel('Amplitud');
title('Respuesta Impulsiva del Filtro Nyquist (f1)');

% Respuesta impulsiva del filtro f2 (Gaussiano)
subplot(2, 2, 2);
stem(t, f2);
xlabel('Tiempo');
ylabel('Amplitud');
title('Respuesta Impulsiva del Filtro Gaussiano (f2)');

%para cumplir con el criterio de Nyquist, la respuesta impulsiva del filtro debe tener un ancho 
%de banda limitado y sin interferencia entre símbolos adyacentes, lo que significa que las 
%muestras de la respuesta impulsiva deberían ser cero fuera de un intervalo de símbolo.
%Parece acercarse más a esto la f2


%Question 18

% Filtrar el peine de Dirac con los filtros f1 y f2
output1 = filter(f1, 1, dirac_comb);
output2 = filter(f2, 1, dirac_comb);


subplot(2, 2, 3);
plot(output1);
xlabel('Tiempo');
ylabel('Amplitud');
title('Salida Filtrada por f1');

subplot(2, 2, 4);
plot(output2);
xlabel('Tiempo');
ylabel('Amplitud');
title('Salida Filtrada por f2');

%Question 19


%Question 20


%Question 21


%Question 22
Nb = 10;

P_0 = 0.75;
P_1 = 0.25;


a_k = zeros(1, Nb);
for i = 1:Nb
    r = rand; % Generar número aleatorio entre 0 y 1
    if r <= P_0
        a_k(i) = 0; % Asignar 0 con probabilidad P_0
    else
        a_k(i) = 1; % Asignar 1 con probabilidad P_1
    end
end

figure()
subplot(2, 2, 1);
scatter(1:numel(a_k), a_k);
xlabel('Time');
ylabel('Amplitude');
title('Symboles a_k');


NT = Nb * Tb;
dirac_comb = zeros(1, NT);

for i = 1:Nb
    dirac_comb((i-1)*Tb + 1) = a_k(i);
end


subplot(2, 2, 2);
stem(dirac_comb);
xlabel('Tiempo');
ylabel('Amplitud');
title('Peine de Dirac con símbolos a_k');


resp_imp = (1/sqrt(Tb))*ones(1,Tb);
output = filter(resp_imp, 1, dirac_comb);

subplot(2, 2, 3);
plot(output);
title('Señal de Salida después del Filtrado');
xlabel('Tiempo');
ylabel('Amplitud');


Eb = 1;
SNR_dB = 3;
SNR_lin = 10^(SNR_dB/10);
No = Eb/SNR_lin;

sigma_2 = sqrt(No/2);

bruit = sigma_2 * randn(size(output));
signal_bruit = output + bruit;

figure();
subplot(2, 2, 1);
plot(signal_bruit);
title('Señal de Salida con Ruido Gaussiano');
xlabel('Tiempo');
ylabel('Amplitud');

g_t = resp_imp; 
g_t_invertida = fliplr(g_t);
output_filtre = filter(g_t_invertida, 1, signal_bruit);

subplot(2, 2, 2);
plot(output_filtre);
title('Señal Filtrada con Filtro Adaptado');
xlabel('Tiempo');
ylabel('Amplitud');


vect = downsample(output_filtre, Tb);
output_echant = circshift(vect, [0, -1]);
output_echant_2 =zeros (1, NT+1);
for i = 1:Nb
    output_echant_2((i-1)*Tb + 1) = output_echant(i);
end

subplot(2, 2, 3);
stem(output_echant_2);
title('Señal Muestreada a Velocidad de Símbolo');
xlabel('Muestra');
ylabel('Amplitud');


%Question 24

% Démodulation avec seuil fixe (S = 0)
seuil_fixe = 0;
symboles_recus_fixe = sign(output_echant - seuil_fixe); %Cette fonction MATLAB renvoie la valeur signée de chaque élément 
% de l'entrée. Si l'élément est positif, il renvoie 1 ; si l'élément est négatif, il renvoie -1 ; si l'élément est zéro, 
% il renvoie zéro

% Démodulation avec seuil adaptatif (S = (sigma)^2/2 * ln(P0/P1))
seuil_adaptatif = (sigma_2^2 / 2) * log(P_0 / P_1);
symboles_recus_adaptatif = (output_echant <= seuil_adaptatif);

% Affichage des résultats
figure;
subplot(2, 1, 1);
stem(symboles_recus_fixe);
title('Démodulation avec seuil fixe (S = 0)');
xlabel('Muestra');
ylabel('Symbole recu');

subplot(2, 1, 2);
stem(symboles_recus_adaptatif);
title('Démodulation avec seuil adaptatif');
xlabel('Muestra');
ylabel('Symbole recu');


%Question 25
% Calcul des erreurs binaires pour chaque seuil de décision
erreur_fixe = sum(symboles_recus_fixe ~= a_k) / Nb;
erreur_adaptatif = sum(symboles_recus_adaptatif ~= a_k) / Nb;

% Affichage des résultats
figure;
stem(1, erreur_fixe, 'r', 'DisplayName', 'Seuil fixe (S = 0)');
hold on;
stem(2, erreur_adaptatif, 'b', 'DisplayName', 'Seuil adaptatif');
title('Taux d''erreur binaire pour chaque seuil de décision');
xlabel('Seuil de décision');
ylabel('Taux d''erreur binaire');
legend('Location', 'best');




