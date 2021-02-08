addpath("MMread");
video = mmread('../data/cars.avi');

x_min = 168;
x_max = 288;
y_min = 112;
y_max = 352;
H = x_max - x_min + 1;
W = y_max - y_min + 1;
T=3;

F = zeros(H,W,T,'uint8');
for i=1:T
   F(:,:,i) = rgb2gray(video.frames(i).cdata(x_min:x_max, y_min:y_max, :));
%    figure;
%    imshow(F(:,:,i));
end

C = randi([0, 1], H, W, T, 'uint8');

E = sum(C.*F, 3) + 2*randn(H,W);
figure;
imshow(E);

