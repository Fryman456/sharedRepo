
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
% then need to gather all the bounding boxes
numImages = numel(annotations);
%preallocate the 8144x2 table table object
bbT = table(cell(8144,1), strings(8144,1), 'VariableNames', {'car', 'Label'});

for i=1:numImages
    %width and height found through difference
    w=annotations(i).bbox_x2-annotations(i).bbox_x1;
    h=annotations(i).bbox_y2-annotations(i).bbox_y1;
    %therfore
    bbT.car{i}=[annotations(i).bbox_x1, ...
                      annotations(i).bbox_y1, ...
                      w, ...
                      h];
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
ds = combine(imageds, bbds);
%% define input size
inputSz = [128 228 3];
%% partition training data
numImages = numpartitions(ds);
shuffledIndices = randperm(numImages);
dsTrain = subset(ds,shuffledIndices(1:round(0.8*numImages)));
dsVal = subset(ds,shuffledIndices(numpartitions(dsTrain)+1:end));
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
%% options
%currently taken directly from the example
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.001, ...
    MiniBatchSize=16,...
    MaxEpochs=4, ...
    ResetInputNormalization=false, ...
    Metrics=mAPObjectDetectionMetric(Name="mAP50"), ...
    ObjectiveMetricName="mAP50", ...
    ValidationData=dsVal, ...
    ValidationFrequency=20, ...
    VerboseFrequency=2);
%% ultimate objective is the run the function
dnet = trainYOLOXObjectDetector(dsTrain, detectorIn,options);
%where trainingData is a datastore
%detectorIn a pretrained yoloxObjectDetector
%options is a TrainingOptionsSGDM object