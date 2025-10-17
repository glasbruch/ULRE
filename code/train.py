import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse

from utils.wandb_upload import *
from utils.img_utils import *
from utils.logger import *
from utils.pyt_utils import eval_ood_measure
from utils.helper import *
from utils.transforms import *

from model.mynn import initialize_weights

from model.hook_network import Network
from model.classifier import Classifier, SNGPBlock

from config.config import config

from dataset import blending_data_loader

from dataset.validation.fishyscapes import Fishyscapes
from dataset.validation.segment_me_if_you_can import SegmentMeIfYouCan

from loss import EDLLoss, BCELoss, FocalLoss, DiceLoss
import torch.nn.functional as F
       
def parse_args():
    parser = argparse.ArgumentParser(description="Training script with command line arguments")
    
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--loss", type=str, default="log", choices=["log", "mse", "bce", "focal"], help="Loss function to use")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD Momentum")
    parser.add_argument("--lr-scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD"], help="Loss function to use")
    parser.add_argument("--img-size", type=int, default=700, help="Image size for training.")
    parser.add_argument("--mode", type=str, default="probs", choices=["probs", "llr"], help="Loss function to use")
    parser.add_argument("--bias", action="store_true", help="Use bias in hidden layers.")
    
    parser.add_argument("--seed", type=int, default=0, help="Seed")

    parser.add_argument("--blending", action="store_true", help="Use blending dataloader.")
    #parser.add_argument("--mse", action="store_true", help="Use MSE loss.")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Dimension of per-pixel embeddings")
    parser.add_argument("--annealing-param", type=int, default=10, help="Annealing coefficient for KL regularization.")
    parser.add_argument("--precision-reg", type=float, default=0, help="Regularization strength of Dirichlet parameters.")
    parser.add_argument("--beta", type=float, default=1, help="Scaling of the KL divergence regularization.")

    parser.add_argument("--per-pixel", action="store_true", help="Use per-pixel classifier")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--leaky-relu", action="store_true", help="Use Leaky ReLU instead of ReLU")
    parser.add_argument("--hook-layers", nargs="+", default=["last_hidden_state"], help="Layers to use for input features")
    parser.add_argument("--embedding-dim", type=int, default=304, help="Dimension of per-pixel embeddings")
    parser.add_argument("--activation", type=str, default="exp", help="Activation function for evidence")

    parser.add_argument("--drop", type=float, default=0, help="Dropout rate")

    parser.add_argument("--path", type=str, default="/home/iq58tapy/Documents/BEDL/ckpts", help="Path for saving checkpoints")
    parser.add_argument("--experiment-name", type=str, default="pebal_test", help="Experiment name used in wandb")
    parser.add_argument("--noise-std", type=float, default=None, help="Std of gaussian noise augmentation.")
    
    return parser.parse_args()

def train(model, backbone, epoch, train_loader, criterion, optimizer, args):
    model.train()

    loader_len = len(train_loader)
    tbar = tqdm(range(loader_len), ncols=137, leave=True)
    train_loader = iter(train_loader)

    for batch_idx in tbar:
        city_mix_imgs, city_mix_targets = next(train_loader)

        city_mix_imgs = city_mix_imgs.cuda(non_blocking=True)
        city_mix_targets = city_mix_targets.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        curr_idx = epoch * loader_len + batch_idx

        with torch.no_grad():
            feature_map = backbone.get_intermediate_layers(city_mix_imgs, 
                                                           n=LAYER_IDX,
                                                           reshape=True, 
                                                           norm=False)
            feature_map = torch.cat(feature_map, dim=1)

        if args.noise_std != None:
            feature_map += torch.randn_like(feature_map) * args.noise_std
        
        bin_logits = model(feature_map)
        
        bin_logits = nn.functional.interpolate(bin_logits, size=city_mix_targets.shape[1:], mode='bilinear',
                                     align_corners=True)

        if args.loss == "bce" or args.loss == "focal":
            loss = criterion(bin_logits, city_mix_targets)
        else:
            loss = criterion(bin_logits, city_mix_targets, epoch)

        loss.backward()

        curr_info = {}

        optimizer.step()
        # Free cache for fixed params.
        torch.cuda.empty_cache()
                
        curr_info[args.loss] = loss
        visual_tool.upload_wandb_info(current_step=curr_idx, info_dict=curr_info)
        tbar.set_description("epoch ({}) | "
                            "loss: {:.3f}  ".format(epoch, curr_info[args.loss]))
        
def valid_anomaly(model, backbone, test_set, data_name=None, epoch=None, my_wandb=None, logger=None,
                  upload_img_num=4, args=None):
    curr_info = {}
    model.eval()

    logger.info("validating {} dataset ...".format(data_name))
    tbar = tqdm(range(len(test_set)), ncols=137, leave=True)

    anomaly_score_list = []
    ood_gts_list = []
    focus_area = []

    with torch.no_grad():
        for idx in tbar:
            img, label = test_set[idx]
       
            feature_map = backbone.get_intermediate_layers(img.unsqueeze(0).to("cuda"),  
                                                           n=LAYER_IDX, 
                                                           reshape=True, 
                                                           norm=False)
            feature_map = torch.cat(feature_map, dim=1)
 
            bin_logits = model(feature_map)

            bin_logits = nn.functional.interpolate(bin_logits, size=img.shape[1:], mode='bilinear',
                                     align_corners=True)           

            if args.loss == "bce" or args.loss == "focal":
                anomaly_score = torchvision.transforms.GaussianBlur(7, sigma=1)(bin_logits).squeeze()
                anomaly_score = anomaly_score.cpu().numpy()
            else:
                anomaly_score = compute_anomaly_score(bin_logits, mode=args.mode, activation=args.activation)        
            
            ood_gts_list.append(np.expand_dims(label.detach().cpu().numpy(), 0))
            anomaly_score_list.append(np.expand_dims(anomaly_score, 0))
            if len(focus_area) < upload_img_num:
                anomaly_score[(label != test_set.train_id_out) & (label != test_set.train_id_in)] = 0
                focus_area.append(anomaly_score)

    # evaluation
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    roc_auc, prc_auc, fpr = eval_ood_measure(anomaly_scores, ood_gts, test_set.train_id_in, test_set.train_id_out)

    curr_info['{}_auroc'.format(data_name)] = roc_auc
    curr_info['{}_fpr95'.format(data_name)] = fpr
    curr_info['{}_auprc'.format(data_name)] = prc_auc
    logger.critical(f'AUROC score for {data_name}: {roc_auc}')
    logger.critical(f'AUPRC score for {data_name}: {prc_auc}')
    logger.critical(f'FPR@TPR95 for {data_name}: {fpr}')

    if my_wandb is not None:
        my_wandb.upload_wandb_info(current_step=epoch, info_dict=curr_info)

        my_wandb.upload_ood_image(current_step=epoch, energy_map=focus_area,
                                  img_number=upload_img_num, data_name=data_name)


    del curr_info
    return roc_auc, prc_auc, fpr      
        
def main(args):

    # Initilaize the pretrained model with hooks
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    backbone.eval()

    model = Classifier(per_pixel=args.per_pixel,
                       embedding_dim=args.embedding_dim,
                       leaky_relu=args.leaky_relu,
                       out_channels=1 if args.loss == "bce" or args.loss == "focal" else 2,
                       bias=args.bias,
                       hidden_dim=args.hidden_dim
                       ).to(device)
    
    initialize_weights(model)
    
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # TODO uncomment to undo dice loss
    if args.loss == "mse":
        criterion = EDLLoss(log_loss=False, 
                            annealing_param=args.annealing_param, 
                            precision_reg = args.precision_reg, 
                            activation=args.activation,
                            beta=args.beta)
    elif args.loss == "log":
        criterion = EDLLoss(log_loss=True, 
                            annealing_param=args.annealing_param, 
                            precision_reg = args.precision_reg,  
                            activation=args.activation,
                            beta=args.beta)
    elif args.loss == "bce":
        criterion = BCELoss()
    
    elif args.loss == "focal":
        criterion = FocalLoss()
    
    if args.blending:
        train_loader = blending_data_loader.get_mix_loader(cityscape_root='/home/iq58tapy/Documents/DATA_NO_BACKUP/data/city_scape',
                                         coco_root='/home/iq58tapy/Documents/DATA_NO_BACKUP/data/coco',
                                         img_size=args.img_size,
                                         batch_size=args.batch_size,
                                         adjust_brightness=True, color_transfer=True)
    else:
        train_loader = blending_data_loader.get_mix_loader(cityscape_root='/home/iq58tapy/Documents/DATA_NO_BACKUP/data/city_scape',
                                         coco_root='/home/iq58tapy/Documents/DATA_NO_BACKUP/data/coco',
                                         img_size=args.img_size,
                                         batch_size=args.batch_size,
                                         adjust_brightness=False, color_transfer=False)
    
    testing_transform = Compose([
             ToTensorSN(),
             ResizeLongestSideDivisible(1792, 14, eval_mode=True, randomcrop=False),
             NormalizeSN(mean=config.image_mean, std=config.image_std)
        ])
    
    segment_me_anomaly = SegmentMeIfYouCan(split='road_anomaly', root=config.segment_me_root_path, transform=testing_transform)

    logger.info('Training begin...')

    for curr_epoch in range(args.num_epochs):

        train(model=model, 
              backbone=backbone,
              epoch=curr_epoch, 
              train_loader=train_loader,
              criterion=criterion,
              optimizer=optimizer,
              args=args
              )

        valid_anomaly(model=model, 
                          backbone=backbone,
                          epoch=curr_epoch, 
                          test_set=segment_me_anomaly,
                          data_name='segment_me_anomaly', 
                          my_wandb=visual_tool, 
                          logger=logger,
                          args=args
                          )

        model_path = f"{args.path}/{args.experiment_name}"

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path =f"{model_path}/{curr_epoch}"
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse the command line arguments
    args = parse_args()
    set_all_seeds(args.seed)

    config.experiment_name = args.experiment_name
    LAYER_IDX = [int(layer) for layer in args.hook_layers]

    # Setup logging
    visual_tool = Tensorboard(config=config)
    logger = logging.getLogger("pebal")
    logger.propagate = False

    main(args)