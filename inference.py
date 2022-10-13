import torch
import config
import utils


# load model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
saved_name='result/last_model.pth'
checkpoint = torch.load(saved_name, map_location=device)
model = utils.get_model_object_detector(config.num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

# load data
test_dir = 'data/test'
img_format="jpg"
test_imgs = glob.glob(f"{test_dir}/*.{img_format}")

# prepare drawing
class_colors = np.random.uniform(0, 255, size=(len(config.num_classes), 3))

# inference
for imgs, annotations in data_loader:
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    with torch.no_grad():
        predictions = model(imgs)
