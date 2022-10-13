# frcnn_medium_sample
Sample code and data for Medium post on https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5

Pytorch version: 1.2.0
Pycocotools: 2.0.0

# training
## sample
$ python train.py

## train on your own data
- prepare train image data 
- prepare coco formatted annotation file
- edit config.py "train_data_dir, train_coco" parameters
- $ python train.py

# testing
## sample
$ python inference.py

## infer on your own data
- prepare test data
- edit config.py "test_data_dir, test_img_format"
- $ python inference.py