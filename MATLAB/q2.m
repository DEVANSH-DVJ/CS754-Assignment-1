clc;
clear;
close all;

addpath("MMread");
video = mmread('../data/cars.avi');

x_min = 169;
x_max = 288;
y_min = 113;
y_max = 352;
H = x_max - x_min + 1;
W = y_max - y_min + 1;
T=3;

F = zeros(H,W,T,'uint8');
for i=1:T
    F(:,:,i) = rgb2gray(video.frames(i).cdata(x_min:x_max, y_min:y_max, :));
    figure;
    imshow(F(:,:,i));
end

C = randi([0, 1], H, W, T, 'uint8');

E = sum(C.*F, 3)/T + 2*randn(H,W);
figure;
imshow(cast(E,'uint8'));

D1 = dctmtx(8);
D2 = kron(D1,D1);
psi = kron(D2, dctmtx(T));

R = zeros(H,W,T,'uint8');

for i=1:H/8
    for j=1:W/8
        y = reshape(E(8*(i-1)+1:8*i,8*(j-1)+1:8*j), [8*8 1]);

        phi = zeros(8*8*T, 8*8, 'double');
        for k=1:T
            phi(8*8*(k-1)+1:8*8*k,:) = diag(reshape(C(8*(i-1)+1:8*i,8*(j-1)+1:8*j,k), [8*8 1]));
        end
        
        x = omp(phi.'*psi,y,40);
        R(8*(i-1)+1:8*i,8*(j-1)+1:8*j,:) = reshape(psi*x, [8 8 T]);
    
%        finish;
    end
end

for i=1:T
    figure;
    imshow(R(:,:,i));
end
