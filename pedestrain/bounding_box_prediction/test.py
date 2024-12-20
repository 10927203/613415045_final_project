
import time
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score

from pedestrian.bounding_box_prediction.datasets.jaad import JAAD as jaad
from pedestrian.bounding_box_prediction.network import *
from pedestrian.bounding_box_prediction.utils import speed2pos, data_loader

# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train PV-LSTM network')
    
    parser.add_argument('--data_dir', type=str, default = './pedestrian/JAAD/processed_annotations', help='Path to dataset')
    parser.add_argument('--dataset', type=str, default = 'jaad', help='Datasets supported: jaad, jta, nuscenes')
    parser.add_argument('--out_dir', type=str, default = './data', help='Path to save output' )  
    parser.add_argument('--task', type=str, default = '2D_bounding_box-intention', help='Task the network is performing, choose between 2D_bounding_box-intention, 3D_bounding_box, 3D_bounding_box-attribute' )
    parser.add_argument('--model_path', type=str, default = './pedestrian/bounding_box_prediction/output/jaad_16_16_16/')
    # data configuration
    parser.add_argument('--input', type=int, default = 2, help='Input sequence length in frames' )
    parser.add_argument('--output', type=int, default = 2, help='Output sequence length in frames' )
    parser.add_argument('--stride', type=int, default = 1, help='Input and output sequence stride in frames' )  
    parser.add_argument('--skip', type=int, default=1)  
    parser.add_argument('--is_3D', type=bool, default=False) 

    # data loading / saving           
    parser.add_argument('--dtype', type=str, default='train')
    parser.add_argument("--from_file", type=bool, default = False)       
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--log_name', type=str, default='testing')
    parser.add_argument('--loader_workers', type=int, default=1)
    parser.add_argument('--loader_shuffle', type=bool, default=False)
    parser.add_argument('--pin_memory', type=bool, default=False)

    # training
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=int, default=1e-5)
    parser.add_argument('--lr_scheduler', type=bool, default=False)

    # network
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--hardtanh_limit', type=int, default=100)
    args = parser.parse_args()
    return args


# For 2D datasets
def test_2d(net, test_loader):
    
    #print('='*100)
    """
    print('Testing ...')
    print('Task: ' + str(args.task))
    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('Hidden layer size: ' + str(args.hidden_size) + '\n')
    """
    '''
    file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
    modelname = 'model_' + file + '.pkl'

    net.load_state_dict(torch.load(os.path.join(args.model_path + modelname)))
    net.eval()
    '''
    
    """
    
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    val_s_scores   = []
    val_c_scores   = []

    ade  = 0
    fde  = 0
    aiou = 0
    fiou = 0
    avg_acc = 0
    avg_rec = 0
    avg_pre = 0
    mAP = 0

    avg_epoch_val_s_loss   = 0
    avg_epoch_val_c_loss   = 0

    counter=0
    state_preds = []
    state_targets = []
    intent_preds = []
    intent_targets = []

    
    """
    future_bounding_box_info = []
    bounding_box_info = []
    
    for idx, (obs_s, target_s, obs_p, target_p) in enumerate(test_loader):
        obs_s = obs_s.to(device='cuda')
        obs_p = obs_p.to(device='cuda')
        
        with torch.no_grad():
            speed_preds, temp = net(speed=obs_s, pos=obs_p)
            preds_p = speed2pos(speed_preds, obs_p)
            bounding_box_info.append( obs_p[0][1].cpu().numpy().tolist() )
            future_bounding_box_info.append( preds_p[0][1].cpu().numpy().tolist() )
    
    return bounding_box_info, future_bounding_box_info 
    
    
    '''
    future_bounding_box_info = []
    bounding_box_info = []
    
    for idx, (obs_s, target_s, obs_p, target_p, target_c, label_c) in enumerate(test_loader):
        #print("!!!!!!!!!!!!!!", idx)
        
        #counter+=1
        obs_s    = obs_s.to(device='cuda')
        #target_s = target_s.to(device='cuda')
        obs_p    = obs_p.to(device='cuda')
        #target_p = target_p.to(device='cuda')
        #target_c = target_c.to(device='cuda')
        
        
        with torch.no_grad():
            #speed_preds, crossing_preds, intentions = net(speed=obs_s, pos=obs_p, average=False)
            speed_preds, temp = net(speed=obs_s, pos=obs_p, average=False)
            #speed_loss    = mse(speed_preds, target_s)/100
            """
            crossing_loss = 0
            for i in range(target_c.shape[1]):
                crossing_loss += bce(crossing_preds[:,i], target_c[:,i])
            crossing_loss /= target_c.shape[1]
            """
            
            preds_p = speed2pos(speed_preds, obs_p)
            
            #print('idx: ', idx )
            #print('obs_p: ', obs_p )
            #print('preds_p: ', preds_p )
            bounding_box_info.append( obs_p[0][1].cpu().numpy().tolist() )
            future_bounding_box_info.append( preds_p[0][1].cpu().numpy().tolist() )
            """
            target_c = target_c[:,:,1].view(-1).cpu().numpy()
            crossing_preds = np.argmax(crossing_preds.view(-1,2).detach().cpu().numpy(), axis=1)

            label_c = label_c.view(-1).cpu().numpy()
            intentions = intentions.view(-1).detach().cpu().numpy()

            state_preds.extend(crossing_preds)
            state_targets.extend(target_c)
            intent_preds.extend(intentions)
            intent_targets.extend(label_c)
            """
            
    etest = time.time()
    print('test: ', etest - stest )    
    return bounding_box_info, future_bounding_box_info     
    '''
    
def pedestrian_main() :
    args = parse_args()

    '''
    # create output dir
    if not args.log_name:
        args.log_name = '{}'.format(args.dataset) 
    if not os.path.isdir(os.path.join(args.out_dir)):
        os.mkdir(os.path.join(args.out_dir))

    # select dataset
    if args.dataset == 'jaad': args.is_3D = False
    '''
    

    # load data
    test_set = eval(args.dataset)(
                data_dir=args.data_dir,
                out_dir=args.out_dir,
                dtype='test',
                input=args.input,
                output=args.output,
                stride=args.stride
                )

    test_loader = data_loader(args, test_set)
    # initiate network
    net = PV_LSTM(args).to(args.device)
    
    #file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
    modelname = 'model_' + '{}_{}'.format(str(args.lr), str(args.hidden_size)) + '.pkl'
    net.load_state_dict(torch.load(os.path.join(args.model_path + modelname)))

    net.eval()
    # training
    return test_2d(net, test_loader)

if __name__ == '__main__':
    pedestrian_main()

###########################################下面code是原本的!!!
'''


import time
import os
import argparse

import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import torchvision
# import torchvision.transforms as transforms
    
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score




# import DataLoader
import datasets
import network
import utils
from utils import data_loader

# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train PV-LSTM network')
    parser.add_argument('--data_dir', type=str, default = 'D:/pedestrian/JAAD/processed_annotations', help='Path to dataset')
    parser.add_argument('--dataset', type=str, default = 'jaad', help='Datasets supported: jaad, jta, nuscenes')
    parser.add_argument('--out_dir', type=str, default = 'D:/pedestrian/bounding_box_prediction/output', help='Path to save output' )  
    parser.add_argument('--task', type=str, default = '2D_bounding_box-intention',
                        help='Task the network is performing, choose between 2D_bounding_box-intention, \
                            3D_bounding_box, 3D_bounding_box-attribute')
    
    # data configuration
    parser.add_argument('--input', type=int, default = 1,
                        help='Input sequence length in frames')
    parser.add_argument('--output', type=int, default = 1,
                        help='Output sequence length in frames')
    parser.add_argument('--stride', type=int, default = 1, 
                        help='Input and output sequence stride in frames')  
    parser.add_argument('--skip', type=int, default=1)  
    parser.add_argument('--is_3D', type=bool, default=False) 

    # data loading / saving           
    parser.add_argument('--dtype', type=str, default='train')
    parser.add_argument("--from_file", type=bool, default=False)       
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--loader_workers', type=int, default=10)
    parser.add_argument('--loader_shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)

    # training
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=int, default=1e-5)
    parser.add_argument('--lr_scheduler', type=bool, default=False)

    # network
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--hardtanh_limit', type=int, default=100)

    args = parser.parse_args()

    return args


# For 2D datasets
def test_2d(args, net, test_loader):
    print('='*100)
    print('Testing ...')
    print('Task: ' + str(args.task))
    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('Hidden layer size: ' + str(args.hidden_size) + '\n')

    file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
    modelname = 'model_' + file + '.pkl'

    net.load_state_dict(torch.load(os.path.join(args.out_dir, args.log_name, modelname)))
    net.eval()


if __name__ == '__main__':
    args = parse_args()

    # create output dir
    if not args.log_name:
        args.log_name = '{}_{}_{}_{}'.format(args.dataset, str(args.input),\
                                str(args.output), str(args.stride)) 
    if not os.path.isdir(os.path.join(args.out_dir, args.log_name)):
        os.mkdir(os.path.join(args.out_dir, args.log_name))

    # select dataset
    if args.dataset == 'jaad':
        args.is_3D = False
    elif args.dataset == 'jta':
        args.is_3D = True
    elif args.dataset == 'nuscenes':
        args.is_3D = True
    else:
        print('Unknown dataset entered! Please select from available datasets: jaad, jta, nuscenes...')

    # load data
    test_set = eval('datasets.' + args.dataset)(
                data_dir=args.data_dir,
                out_dir=os.path.join(args.out_dir, args.log_name),
                dtype='test',
                input=args.input,
                output=args.output,
                stride=args.stride,
                skip=args.skip,
                task=args.task,
                from_file=args.from_file,
                save=args.save
                )

    test_loader = data_loader(args, test_set)

    # initiate network
    net = network.PV_LSTM(args).to(args.device)

    # training
    test_2d(args, net, test_loader)
    
'''