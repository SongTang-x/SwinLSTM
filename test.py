import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from SwinLSTM_D import SwinLSTM
from configs import get_args
from dataset import Moving_MNIST_Test
from functions import test
from utils import set_seed, make_dir, init_logger

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    model = SwinLSTM(img_size=args.input_img_size, patch_size=args.patch_size,
                     in_chans=args.input_channels, embed_dim=args.embed_dim,
                     depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                     num_heads=args.heads_number, window_size=args.window_size).to(args.device)

    criterion = nn.MSELoss()
    
    test_dataset = Moving_MNIST_Test(args)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
                             num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=True)

    model.load_state_dict(torch.load('./Pretrained/trained_model_state_dict'))

    start_time = time.time()

    _, mse, ssim = test(args, logger, 0, model, test_loader, criterion, cache_dir)

    print(f'[Metrics]  MSE:{mse:.4f} SSIM:{ssim:.4f}')
    print(f'Time usage per epoch: {time.time() - start_time:.0f}s')

