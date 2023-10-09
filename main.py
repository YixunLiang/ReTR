import argparse
from re import I
from stat import UF_OPAQUE
from tqdm import tqdm
import math
import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint


PI = math.pi
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------- main function
if __name__ == "__main__":

    seed_everything(0, workers=True)    

    # -------------------------------- args for training and models ---------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', dest='root_dir', type=str,
        help='directory of training dataset')
    parser.add_argument('--load_ckpt', dest='load_ckpt', type=str, default=False,
        help='load pretrained lightning ckpt')
    parser.add_argument('--train_ray_num', dest='train_ray_num', type=int, default=1024,
        help='ray number in one image')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001,
        help='learning rate')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=2,
        help='batch size')
    parser.add_argument('--max_epochs', dest='max_epochs', type=int, default=16,
        help='max num of epochs')
    parser.add_argument('--val_only', dest='val_only', action="store_true",
        help='only validate')
    parser.add_argument('--volume_reso', dest='volume_reso', type=int, default=96, 
        help="3D feature volume resolution") # set as 0 to disable
    parser.add_argument('--coarse_sample', dest='coarse_sample', type=int, default=64,
        help='number of coarse samples during training')
    parser.add_argument('--fine_sample', dest='fine_sample', type=int, default=64,
        help='number of fine samples during training')
    parser.add_argument('--devices', dest='devices', type=str, default="0,1,2,3",
        help='the devices choose for training')
    # loss weights
    # loss and optimizer hyperparams
    parser.add_argument("--coarse_weight_decay", type=float, default=0.1)
    parser.add_argument("--lr_init", type=float, default=5e-3)
    parser.add_argument("--lr_final", type=float, default=5e-5)
    parser.add_argument("--lr_delay_steps", type=int, default=3)
    parser.add_argument("--scan", type=int, default=None)
    parser.add_argument("--lr_delay_mult", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=0.001) 
    parser.add_argument('--weight_rgb', dest='weight_rgb', type=float, default=1.0)
    parser.add_argument('--weight_depth', dest='weight_depth', type=float, default=0.)
    parser.add_argument('--weight_sparse', dest='weight_sparse', type=float, default=0.)
    parser.add_argument('--exp_name', default='test', help='the exp_dir to save checkpoints/logs')
    # -------------------------------- args for testing --------------------------------
    parser.add_argument('--test_dir', dest='test_dir', type=str,
        help='directory of test dataset')
    parser.add_argument('--out_dir', dest='out_dir', type=str,
        help='directory of to save test result')
    parser.add_argument('--extract_geometry', dest='extract_geometry', action='store_true', 
        help='if you only want to extract geometry')
    parser.add_argument('--dense_recon', action="store_true",help='reconstruction use full 49 images')
    parser.add_argument('--full_virtual_render', dest='full_virtual_render', action="store_true",
        help='render 49 virtual views')
    parser.add_argument('--use_ref_view', dest='use_ref_view', action="store_true",
        help='include ref view for rendering')
    parser.add_argument('--occ_trans', dest='occ_trans', action='store_true', 
        help='occ transformer')
    parser.add_argument('--out_weights', dest='out_weights', action='store_true', 
        help='output_weights')
    parser.add_argument('--test_ray_num', dest='test_ray_num', type=int, default=1200)
    parser.add_argument('--test_sample_coarse', dest='test_sample_coarse', type=int, default=64)
    parser.add_argument('--test_sample_fine', dest='test_sample_fine', type=int, default=64)
    parser.add_argument('--test_coarse_only', dest='test_coarse_only', action="store_true",
        help='only use coarse samples during testing')
    parser.add_argument('--test_n_view', dest='test_n_view', type=int, default=3)
    parser.add_argument('--set', dest='set', type=int, default=0,
        help='two sets are provided by SparseNeuS')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_49_views", action="store_true", default=True)
    parser.add_argument('--use_res_color', action="store_true",
        help='use color from src view to predict')
    parser.add_argument('--ori_size', action="store_true",
        help='use ori_size')
    args = parser.parse_args()
    
    if args.occ_trans:
        from code.model import ReTR
        print('Using OCC transformer... ')

    batch_size = args.batch_size
    num_workers = 12
    if args.use_49_views:
        selected_pair_filepath = 'code/dataset/dtu/pair.txt' 
    if args.debug:
        devices = [0]
    else:
        devices = args.devices
    if args.ori_size:
        img_wh = (1600, 1200)
        print(f'using original image size, {img_wh}')
    else:
        img_wh = [800, 600]
    args.logdir = os.path.join("./ckpts", args.exp_name)
    os.makedirs(args.logdir, exist_ok=True)
    # -------------------------------- dataset ----------------------------------------
    if not args.extract_geometry:
        # training
        from code.dataset.dtu_train import MVSDataset
        dtu_dataset_train = MVSDataset(            
                root_dir=args.root_dir,
                split="train",
                split_filepath="code/dataset/dtu/lists/train.txt",
                pair_filepath=selected_pair_filepath,
                n_views=5,
                )   
        dtu_dataset_val = MVSDataset(            
                root_dir=args.root_dir,
                split="test",
                split_filepath="code/dataset/dtu/lists/test.txt",
                pair_filepath=selected_pair_filepath,
                n_views=5,
                test_ref_views = [23],  # only use view 23
                )            

        print("dtu_dataset_train:", len(dtu_dataset_train))
        print("dtu_dataset_val:", len(dtu_dataset_val))

        dataloader_train = DataLoader(dtu_dataset_train,
                                        batch_size=batch_size, 
                                        num_workers=num_workers, 
                                        shuffle=True)  
        dataloader_val = DataLoader(dtu_dataset_val,
                                        batch_size=batch_size, 
                                        num_workers=num_workers, 
                                        shuffle=False)  
    else:
        # testing on dtu
        dataloader_test = []
        
        from code.dataset.dtu_test_sparse import DtuFitSparse
        print('Sparse view recon... run Table 1 results... ')
        if args.scan is not None:
            scan_list = [args.scan]
        else:
            scan_list =  [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
        for scan in scan_list:
            dataset_tmp = DtuFitSparse(root_dir=args.test_dir, 
                                split="test", 
                                scan_id='scan%d'%scan, 
                                pair_filepath=selected_pair_filepath,
                                full_virtual_render=args.full_virtual_render,
                                img_wh=img_wh, use_ref_view=args.use_ref_view,
                                n_views=args.test_n_view)
            dataloader_tmp = DataLoader(dataset_tmp,
                                            batch_size=1, 
                                            num_workers=2, 
                                            shuffle=False)  
            dataloader_test.append(dataloader_tmp)

    # -------------------------------- lightning module -------------------------------
    if args.load_ckpt:
        retr = ReTR.load_from_checkpoint(checkpoint_path=args.load_ckpt, args=args)
        print("Model loaded:", args.load_ckpt)
    else:
        retr = ReTR(args)
    

    logger = WandbLogger(
        project="retr",
        name =  args.exp_name,
        save_dir = args.logdir,
        offline=args.debug,
    )


    # -------------------------------- trainer ---------------------------------------

    trainer = pl.Trainer(
        accelerator="gpu" if device=="cuda" else "cpu", 
        devices=devices,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1, 
        logger=logger,
        num_sanity_val_steps=1,
        gradient_clip_val=args.grad_clip_norm,
        )

    print(f'coarse and fine sample points {args.coarse_sample}, {args.fine_sample}, {args.test_sample_coarse}, {args.test_sample_fine}')
    ModelSummary(retr, max_depth=1)

    # -------------------------------- train or/and testing --------------------------------
    if not args.extract_geometry:
        if args.val_only:
            print("[only validation]")
            trainer.validate(retr, dataloader_train)
        else:
            print("[start training]")
            trainer.fit(retr, dataloader_train, dataloader_val)
    else:
        for dataloader_test1 in tqdm(dataloader_test):
            trainer.validate(retr, dataloader_test1)

    print("end")