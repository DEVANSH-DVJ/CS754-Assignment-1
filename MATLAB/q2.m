clc;
clear;
close all;

addpath("MMread");
rng(100);

video = mmread('../data/cars.avi');
x_min = 169;
x_max = 288;
y_min = 113;
y_max = 352;
H = x_max - x_min + 1;
W = y_max - y_min + 1;
noise_std = 2;
T = 3;

F = zeros(H,W,T,'double');
for i=1:T
    F(:,:,i) = rgb2gray(video.frames(i).cdata(x_min:x_max, y_min:y_max, :));
%     figure;
%     imshow(cast(F(:,:,i), 'uint8'));
end

C = randi([0, 1], H, W, T, 'double');

E = sum(C.*F, 3) + noise_std*randn(H,W);
E = E/T; C = C/T;
figure;
imshow(cast(E,'uint8'));
imwrite(cast(E, 'uint8'), sprintf('plots/%i_coded_snapshot.jpg',T));

D1 = dctmtx(8);
D2 = kron(D1,D1);
psi = kron(D2, dctmtx(T));

R = zeros(H,W,T,'double');
avg_mat = zeros(H,W,T,'double');

% for i=1:H/8
%     for j=1:W/8
%         y = reshape(E(8*(i-1)+1:8*i,8*(j-1)+1:8*j), [8*8 1]);
% 
%         phi = zeros(8*8*T, 8*8, 'double');
%         for k=1:T
%             phi(8*8*(k-1)+1:8*8*k,:) = diag(reshape(C(8*(i-1)+1:8*i,8*(j-1)+1:8*j,k), [8*8 1]));
%         end
%         
%         x = omp(phi.'*psi, y, 40);
% %         x = omp_e(phi.'*psi, y, noise_std^2);
%         R(8*(i-1)+1:8*i,8*(j-1)+1:8*j,:) = reshape(psi*x, [8 8 T]);
%         
% %         finish;
%     end
% end

tic;
for i=1:H-7
    for j=1:W-7
        y = reshape(E(i:i+7,j:j+7), [8*8 1]);

        phi = zeros(8*8*T, 8*8, 'double');
        for k=1:T
            phi(8*8*(k-1)+1:8*8*k,:) = diag(reshape(C(i:i+7,j:j+7,k), [8*8 1]));
        end
        
%         x = omp(phi.'*psi, y, 10);
        x = omp_e(phi.'*psi, y, 9*8*8*noise_std^2);
        R(i:i+7,j:j+7,:) = R(i:i+7,j:j+7,:) + reshape(psi*x, [8 8 T]);
        avg_mat(i:i+7,j:j+7,:) = avg_mat(i:i+7,j:j+7,:) + ones(8,8,T);
        i, j
    end
end

for i=1:T
    figure;
%     imshow(cast(R(:,:,i)./avg_mat(:,:,i), 'uint8'));
    imshow(cast([R(:,:,i)./avg_mat(:,:,i), F(:,:,i)], 'uint8'));
    imwrite(cast([R(:,:,i)./avg_mat(:,:,i), F(:,:,i)], 'uint8'), sprintf('plots/%i_%i.png',T,i));
end

% l = R(:,:,1)./avg_mat(:,:,1);
% norm(l-F(:,:,1), 'fro')/norm(F(:,:,1), 'fro');
toc;

function theta = omp_e(A, y, e)
    [N, K] = size(A); % N:dim of signal, K:#atoms in dictionary

    theta = zeros(K,1);      % coefficient (output)
    r = y;                   % residual of y
    T = [];                  % support set
    i = 0;                   % iteration
    A_omega = [];

    while(i < 64 && norm(r)^2 > e)
        i = i + 1;
        x_tmp = zeros(K,1);
        indices = setdiff(1:K, T); % iterate all columns except for the chosen ones
        for ind=indices
            x_tmp(ind) = A(:,ind)' * r / norm(A(:,ind)); % sol of min ||a'x-b||
        end
        [~,j] = max(abs(x_tmp));
        T = [T j];
        A_omega = [A_omega A(:,j)];
        theta_s = pinv(A_omega) * y;
        r = y - A_omega * theta_s;
    end
    i+0.1

    for j=1:i
        theta(T(j)) = theta_s(j);
    end
end
