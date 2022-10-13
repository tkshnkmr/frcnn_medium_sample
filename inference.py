import torch
import config
import utils
import glob
import numpy as np
import cv2

def draw_bboxes(img, preds, thre, class_colors):
    preds = [{k: v.to('cpu') for k,v in t.items()} for t in preds]

    if len(preds[0]['boxes']) != 0:
        boxes = preds[0]['boxes'].data.numpy()
        scores = preds[0]['scores'].data.numpy()
        
        boxes = boxes[scores >= thre].astype(np.int32)
        pred_classes = [i for i in preds[0]['labels'].cpu().numpy() ]

        for j, box in enumerate(boxes):
            color = class_colors[pred_classes[j]]
            cv2.rectangle(img,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        color, 2)
        cv2.imshow('prediction', img)
        cv2.waitKey(0)

        
def inference_1img(model, img_name, device, thre, class_colors):
    in_img = cv2.imread(img_name)

    # display
    cv2.imshow("input image", in_img)
    cv2.waitKey(1)

    # convert to tensor
    img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = np.transpose(img, (2,0,1)) # HWC -> CHW
    img = torch.tensor(img, dtype=torch.float).to(device)
    img = torch.unsqueeze(img,0) # add batch dim

    # run inference
    with torch.no_grad():
        preds = model(img)
    print(f"inference on {img_name} done.")

    draw_bboxes(in_img, preds, thre, class_colors)


def main():
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

    # prepare for drawing
    class_colors = np.random.uniform(0, 255, size=(config.num_classes, 3))
    detection_thre = 0.8

    # inference
    for i in range(len(test_imgs)):
        img_name = test_imgs[i]
        inference_1img(model, img_name, device, detection_thre, class_colors)

        
if __name__ == "__main__":
    main()
