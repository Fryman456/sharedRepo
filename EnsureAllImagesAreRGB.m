%% use imresize to transform the images 
% should only need to do this once
%trainingData must best a datastore type
%which is broken into 3 columns
%ImageData  BoundingBoxes Bounding Box Labels
%We begin by getting our data from the db
ImportFile("Training Data/cars_train_annos");
%The structure of this data is
%bbx1 bby1 bbx2 bby2 class fname
%here bb1 is the top left corner and bb2 is the bottom right
%% to begin our table we first need the seperate the required data into a table
%first we gather the image datastore
imageds = imageDatastore("Training Data\cars_train\cars_train\");
% we also need to augment the images, to ensure they are all the right size
%this also needs to happen to the bounding box
%so we are only going to change the bounding box
%% this was the old way, DOES NOT WORK

imageFldPaths = "Training Data\cars_train\cars_train\spare 128x228\";
imds = imageDatastore(imageFldPaths);
read(imds);
numImages = numel(annotations);
for i=1:numImages
    imagePath = imageFldPaths + annotations(i).fname + ".jpg";
    im=imread(imagePath);
    imsz=size(im);
    imszsz=size(imsz);
    if imszsz(2)==2
    im1=cat(3,im,im);
    im=cat(3,im1,im);
    end
    fname="Training Data\cars_train\cars_train\Perfect 128x228\" + annotations(i).fname;
    imwrite(im, fname);
end
