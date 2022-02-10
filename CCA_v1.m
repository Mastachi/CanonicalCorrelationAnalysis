%CCA v2
%9/ 2/2022
%Ricardo Mastachi Torres
%Trabajo de tésis maestría en ciencias y tecnologías biomédcias

EEG = load("Dataset2/EEG/GCRF0001.mat");
EEG.sampleRate = 512;
fNIRS.data = readmatrix("Dataset2\fNIRS\2020-11-26\2020-11-26_001\NIRS-2020-11-26_001.txt")';
fNIRS.sampleRate = 6.25;

%Resampling
%EEG.timeInstants = [1,2,3];
%EEG.data = resample(EEG.y', EEG.timeInstants)'
%Colocamos los canales como las filas, y el tiempo en las columnas
sliceSize = int16(EEG.sampleRate/fNIRS.sampleRate); %Tamaño de rebanadas
X = zeros(size(EEG.y,1),size(fNIRS.data,2)); %Alojamiento de arreglo
for i = 1:size(X,1)%Filas
    for j = 1:size(X,2) %Columnas
        X(i,j) = EEG.y(i,j*sliceSize);
    end
end
Y = fNIRS.data;

%Y = fNIRS.Data';

%X = 100*rand(33,4777);

%Matriz de tamaño igual al paper de Malacarne
%X = 100*rand(2,4);
%Y = 100*rand(2,4);

%Standarized to have unit variance, for channel
Xstd = X - mean(X,2);
Ystd = Y - mean(Y,2);
%N = normalize(A)
%Xstd = Xstd/std(Xstd(:));
%Ystd = Ystd/std(Ystd(:));
Xstd = normalize(Xstd,2);
Ystd = normalize(Ystd,2);

%Concatenate transpose of X and Y
Z = [Xstd', Ystd'];
%lambda = 100*eps(); %Regularización Tikhonov
%Z = Z+lambda*eye();

%Calculate covaniance of Z
M1 = cov(Z);

%Extract submatrices
Sxx = M1(1:size(X,1)            , 1:size(X,1));
Syy = M1((size(X,1)+1):size(Z,2), (size(X,1)+1):size(Z,2));
Sxy = M1(1:size(X,1)            , (size(X,1)+1):size(Z,2));
Syx = Sxy'; %Syx2 = M1((size(X,1)+1):size(Z,2),  1:size(X,1));

%Defining K
K = Sxx^(-1/2) * Sxy * Syy^(-1/2);
%K = sqrt(Sxx^-1) * Sxy * sqrt(Syy^-1); %Esto ocasiona problemas

%%%%%%%%%%%%Check with matlab
%EEG.data = randn(200,6*1);
%fNIRS.data = randn(200,4*2);
[A,B, r, U, V] = mycanoncorr(Xstd',Ystd');
%[A,B, r, U, V] = mycanoncorr(Z(:,1:size(X,1)),Z(:,size(X,1)+1:end));

%Singular Value Decomposition
[gamma, lambda, delta] = svd(K, 'econ');

%Check...
%N1 = inv(Sxx) * Sxy * inv(Syy) * Syx;
%N2 = inv(Syy) * Syx * inv(Sxx) * Sxy;
%EigenvaluesN1 = (eig(N1)).^(1/2);
%EigenvaluesN2 = (eig(N2)).^(1/2);
%N3 = K * K';
%EigenvaluesN3 = (eig(N3)).^(-1/2);
%N4 = K' * K;
%EigenvaluesN4 = (eig(N4)).^(-1/2);

myA = Sxx^(-1/2) * gamma;
myB = Syy^(-1/2) * delta;

%Canonical variates
myU = Xstd' * myA;
myV = Ystd' * myB;
