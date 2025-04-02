numImagesTested=60;
%%
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

imageFldPaths = "Training Data\cars_train\cars_train\Scaled 128x222\";
imds = imageDatastore(imageFldPaths);
read(imds);


%% use imresize to transform the images 
% should only need to do this once
%{
numImages = numel(annotations);
for i=1:numImages
    imagePath = "Training Data\cars_train\cars_train\" + annotations(i).fname;
    im=imread(imagePath);
    resizedIm=imresize(im,[128 228]);
    fname="Training Data\cars_train\cars_train\Scaled 128x222\" + annotations(i).fname + ".jpg";
    imwrite(resizedIm, fname);
end
%}
%% then need to gather all the bounding boxes
%%editted for 100 images
%preallocate the 8144x2 table table object
bbT = table(cell(numImagesTested,1), strings(numImagesTested,1), 'VariableNames', {'car', 'Label'});
numImages = numel(annotations);
% define input size
inputSz = [128 228 3];
for i=1:numImagesTested
    %width and height found through difference
    w=annotations(i).bbox_x2-annotations(i).bbox_x1;
    h=annotations(i).bbox_y2-annotations(i).bbox_y1;
    %we then need to find the scale factor
    imfn=imageds.Files(i);
    im=imread(string(imfn));
    imsz=size(im);
    a=size(imsz);
    if a(2) == 3
        targetSF= inputSz./imsz;
        %before we convert to cell
        bb =[annotations(i).bbox_x1, ...
                          annotations(i).bbox_y1, ...
                          w, ...
                          h];
        mbb=round(bboxresize(bb,targetSF));
    else
        targetSF= inputSz(1,1:2)./imsz;
        %before we convert to cell
        bb =[annotations(i).bbox_x1, ...
                          annotations(i).bbox_y1, ...
                          w, ...
                          h];
        mbb=round(bboxresize(bb,targetSF));
    end
    bbT.car{i}=mbb;
  %{
    this is just to prove that everything was scaled correctly
    a=imds.readByIndex(1);
    a=table2array(a);
    a=cell2mat(a);
    figure
    I = insertObjectAnnotation(a,'Rectangle',mbb,'car');
    imshow(I);
  %}
    %in column 2 we will place the label, which is always "car"
    bbT.Label{i}='car';
    %redundant
end

%%
%example to explain HOW TF boxLabelDatastores are meant to be created
%{
data = load("vehicleTrainingData.mat");
trainingData = data.vehicleTrainingData;
blds = boxLabelDatastore(trainingData(:,2:end));
classes = trainingData.Properties.VariableNames(2:end);
%}
%% this table can then be converted to a ds
bbds = boxLabelDatastore(bbT(:,1));
%% combine the ds
ds1 = combine(imds, bbds);
%% partition training data
numImages = numpartitions(ds1);
shuffledIndices = randperm(numImages);
dsTrain = subset(ds1,shuffledIndices(1:round(0.8*numImages)));
dsVal1 = subset(ds1,shuffledIndices(numpartitions(dsTrain)+1:end));
%% specify the class names
classes = {'car'};
%% initilise detector
detectorIn = yoloxObjectDetector("tiny-coco", classes, InputSize = inputSz);
%% estimate the anchor boxes
%{
Apparently not needed anymore

numAnchors = 5; %arbitary
anchorBoxes = estimateAnchorBoxes(bbds, numAnchors);
%}
disp(size(bbT, 1));   % Check bounding box table size
disp(numel(imds.Files)); % Check number of images in the datastore


%% options
%currently taken directly from the example
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.001, ...
    MiniBatchSize=4,...
    MaxEpochs=4, ...
    ResetInputNormalization=false, ...
    Metrics=mAPObjectDetectionMetric(Name="mAP50"), ...
    ObjectiveMetricName="mAP50", ...
    ValidationData=dsVal1, ...
    ValidationFrequency=20, ...
    VerboseFrequency=2);


%% ultimate objective is the run the function
dnet = trainYOLOXObjectDetector(dsTrain, detectorIn,options);
%where trainingData is a datastore
%detectorIn a pretrained yoloxObjectDetector
%options is a TrainingOptionsSGDM object
%% test this shiezner

c=imread("C:\Users\adria\Documents\GitHub\CNNCarValidator\Training Data\cars_train\cars_train\spare 128x228\00053.jpg.jpg");
[boxes,scores,labels] = detect(dnet,c,Threshold=0.5);
detectedImage = insertObjectAnnotation(c,"rectangle",boxes,labels);
figure
imshow(detectedImage)