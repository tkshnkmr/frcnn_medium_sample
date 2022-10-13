import torch
import config
import utils


# load data
my_dataset = utils.myOwnDataset(
    root=config.train_data_dir, annotation=config.train_coco, transforms=get_transform()
)
data_loader = torch.utils.data.DataLoader(
    my_dataset,
    batch_size=config.train_batch_size,
    shuffle=config.train_shuffle_dl,
    num_workers=config.num_workers_dl,
    collate_fn=collate_fn,
)

# load model
model = utils.get_model_object_detector(config.num_classes)
model.eval()

# inference
for imgs, annotations in data_loader:
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    with torch.no_grad():
        predictions = model(imgs)
