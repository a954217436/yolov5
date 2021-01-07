'''评估测试集结果：
作用：对生成的txt与标注xml对比，寻找漏检、错检图片
    python test.py --weights runs/train_new_640/exp/weights/best.pt --data data/voc.yaml --task val --verbose --save-txt --name val_0.3 --conf-thres 0.3
    对val.txt推理并生成txt在 "runs/test/val_0.3/labels" 下
注意：加上--save-txt后mAP会变得非常高，这是yolov5的一个bug，本意是生成txt为自动标注用的'''

import os
import cv2
import shutil

def iou(box1, box2):
    '''
    box1, box2: [classID, x1, y1, x2, y2]
    '''
    _, left1, top1, right1, down1 = box1
    _, left2, top2, right2, down2 = box2
    
    area1 = (right1 - left1) * (down1 - top1)
    area2 = (right2 - left2) * (down2 - top2)
    area_sum = area1 + area2
    
    #print(area1, area2, area_sum)
    
    left   = max(left1, left2)
    right  = min(right1, right2)
    top    = max(top1, top2)
    bottom = min(down1, down2)
        
    if left > right or top > bottom:
        return 0
    else:
        inter = (right-left) * (bottom-top)
        return inter / (area_sum-inter)
    
    
def box_compare(gt_boxes, dt_boxes):
    '''
    gt_boxes: [[0, 10,20,100,100], [1, 50,60,200,200]] - [classID, x1, y1, x2, y2]
    dt_boxes: [[1, 51,61,198,200], [1, 100,300,900,1000], [0, 200,300,900,1000]] - [classID, x1, y1, x2, y2]
    '''
    gt_nums = len(gt_boxes)
    dt_nums = len(dt_boxes)
    
    true_nums  = 0        #检测到的是对的
    miss_nums  = gt_nums  #没有检测到的
    wrong_nums = 0        #检测到的是错的
    
    if dt_nums == 0:
        return true_nums, miss_nums, wrong_nums
    if gt_nums == 0:
        wrong_nums = dt_nums
        return true_nums, miss_nums, wrong_nums
    
    for gt_box in gt_boxes:
        hit = 0
        for dt_box in dt_boxes:
            #print(iou(gt_box, dt_box))
            if gt_box[0] == dt_box[0] and iou(gt_box, dt_box) > 0.5:
                hit += 1
        
        if hit > 0:
            miss_nums -= 1
            true_nums += 1
    
    wrong_nums = dt_nums - true_nums
    #print("true_nums = {:2} , wrong_nums = {:2} , miss_nums = {:2} ".format(true_nums, wrong_nums, miss_nums))
    return true_nums, wrong_nums, miss_nums


def get_box_from_txt(txt_path):
    '''
    xywh
    '''
    gt_boxes = []
    if not os.path.exists(txt_path):
        return gt_boxes
    with open(txt_path, "r") as t:
        lines = t.readlines()
        #print(lines)
    
    for line in lines:
        line_strs = line.strip().split(" ")
        box = []
        for line_str in line_strs:
            box.append(float(line_str))
        gt_boxes.append(box)
    return gt_boxes    # xywh
    
def box_to_abs(boxes, img_path):
    # boxes: xywh
    img = cv2.imread(img_path)
    h, w, c = img.shape
    for box in boxes:
        box[0] = int(box[0])
        box[1] *= w
        box[2] *= h
        box[3] *= w
        box[4] *= h
        box[3] += box[1]
        box[4] += box[2]
    return boxes  # xyxy
    
def compare_2folder_txts(folder_gt, folder_dt, file_list_txt):
    files = open(file_list_txt, "r").readlines()
    
    all_gt_nums = 0
    all_dt_nums = 0
    all_true_nums = 0
    all_wrong_nums = 0
    all_miss_nums = 0
    all_file_nums = len(files)
    all_problem_files_nums = 0
    
    for f in files:
        jpg_path = f.strip()
        file_name = jpg_path.split("/")[-1][:-4]
        
        gt_boxes = get_box_from_txt(folder_gt + file_name + ".txt")
        gt_boxes = box_to_abs(gt_boxes, jpg_path)
        
        dt_boxes = get_box_from_txt(folder_dt + file_name + ".txt")
        dt_boxes = box_to_abs(dt_boxes, jpg_path)
        
        true_nums, wrong_nums, miss_nums = box_compare(gt_boxes, dt_boxes)
        
        all_gt_nums += len(gt_boxes)
        all_dt_nums += len(dt_boxes)
        all_true_nums += true_nums
        all_wrong_nums += wrong_nums
        all_miss_nums += miss_nums
        
        if miss_nums > 0 or wrong_nums > 0:
            print("{:30} missing {:2} objects ! wrong {:2} boxes ! ".format(file_name, miss_nums, wrong_nums))
            all_problem_files_nums += 1
        
        if miss_nums > 0:
            shutil.copy(folder_dt + file_name + ".jpg", "./_miss/" + file_name + ".jpg")
        if wrong_nums > 0:
            shutil.copy(folder_dt + file_name + ".jpg", "./_wrong/" + file_name + ".jpg")
    
    print("Total result: {:4} gts, {:4} dts, miss {:4} objs, wrong {:4} objs, true {:4} objs, {:4} files not all right !".format(all_gt_nums, 
                                                                                                                                 all_dt_nums, 
                                                                                                                                 all_miss_nums, 
                                                                                                                                 all_wrong_nums, 
                                                                                                                                 all_true_nums, 
                                                                                                                                 all_problem_files_nums))
                                                                                                                                 
compare_2folder_txts("/zhanghao/dataset/YW_new/VOCdevkit/VOC2007/labels/", "./_res/", "/zhanghao/dataset/YW_new/val_fold_3.txt")
