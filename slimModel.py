import torch
import argparse

def convert(weights, output, is_half):
    print("input weights: ", weights, " output: ", output, " is_half: ", is_half)
    try:
        full_model = torch.load(weights)
        
        del full_model['optimizer']
        print("delete optimizer success ...")
        
        if is_half:
            full_model['model'].half()
            print("convert to half success ...")
            
        torch.save(full_model, output)
        print("convert success ! ")
    except:
        print("delete or half failed !!! ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/all_640_5l_reLabel/exp/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--output', type=str, default='slim.pth', help='output path')
    parser.add_argument('--half', action='store_true', help='whether to save half')
    
    opt = parser.parse_args()
    convert(opt.weights, opt.output, opt.half)
