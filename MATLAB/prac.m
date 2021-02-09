clc;
clear;
close all;

addpath("MMread");
rng(20);

video = mmread('../data/cars.avi');

I = reshape(cast(rgb2gray(video.frames(1).cdata(171:200, 171:200, :)), 'double'), [900 1]);

% D1 = dctmtx(288);
D2 = kron(dctmtx(30), dctmtx(30));
% D2 = dctmtx(900);

plot(sort(abs(D2*I)));
