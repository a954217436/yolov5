import time
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device

from tqdm import tqdm



def detect(model, img_path, img_size = 640, save_txt=True, save_img=True, save_prefix="res/", conf_thres=0.25, iou_thres=0.45):    
    global colors, names, device, half
    
    # Run inference
    t0 = time.time()
    
    img0 = cv2.imread(img_path)  # BGR
    file_name = img_path.split("/")[-1]
    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)\
    
    pred = model(img, augment=True)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    save_path = save_prefix + file_name
    txt_path = save_prefix + file_name[:-4] + '.txt'
    for i, det in enumerate(pred):  # detections per image 
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
            
            with open(txt_path, 'a') as f:
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #line = (cls, *xywh, conf)
                        line = (cls, *xywh)    #不写confidence置信度
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
    if save_img:
        cv2.imwrite(save_path, img0)
    
    
def loadModel(model_path):
    global names, colors, device, half
    
    device = select_device("0")
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(model_path, map_location=device)  # load FP32 model

    if half:
        model.half()  # to FP16

    # Get names and colors    
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
    return model


model = loadModel("runs/train_new_folder3_640_reLabel/exp/weights/best.pt")


#dataroot = "data/yw_test_new/"
#fs = os.listdir(dataroot)
#os.makedirs("res/" + dataroot, exist_ok=True)

val_txt = "/zhanghao/dataset/YW_new/val_fold_3.txt"
lines = open(val_txt, "r").readlines()

os.makedirs("_res/", exist_ok=True)

for f in tqdm(lines):
    detect(model, f.strip(), 640, save_prefix="_res/", conf_thres=0.25, iou_thres=0.4)
