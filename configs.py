import argparse


def get_args():
    parser = argparse.ArgumentParser('SwinLSTM training and evaluation script', add_help=False)

    # Setup parameters
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--seed', default=1234, type=int)

    # Moving_MNIST dataset parameters
    parser.add_argument('--num_frames_input', default=10, type=int, help='Input sequence length')
    parser.add_argument('--num_frames_output', default=10, type=int, help='Output sequence length')
    parser.add_argument('--image_size', default=(28, 28), type=int, help='Original resolution')
    parser.add_argument('--input_size', default=(64, 64), help='Input resolution')
    parser.add_argument('--step_length', default=0.1, type=float)
    parser.add_argument('--num_objects', default=[2], type=int)
    parser.add_argument('--train_samples', default=[0, 10000], type=int, help='Number of samples in training set')
    parser.add_argument('--valid_samples', default=[10000, 13000], type=int, help='Number of samples in validation set')
    parser.add_argument('--train_data_dir', default='./data/train-images-idx3-ubyte.gz')
    parser.add_argument('--test_data_dir', default='./data/mnist_test_seq.npy')

    # model parameters
    parser.add_argument('--model', default='SwinLSTM-D', type=str, choices=['SwinLSTM-B', 'SwinLSTM-D'],
                        help='Model type')
    parser.add_argument('--input_channels', default=1, type=int, help='Number of input image channels')
    parser.add_argument('--input_img_size', default=64, type=int, help='Input image size')
    parser.add_argument('--patch_size', default=2, type=int, help='Patch size of input images')
    parser.add_argument('--embed_dim', default=128, type=int, help='Patch embedding dimension')
    parser.add_argument('--depths', default=[12], type=int, help='Depth of Swin Transformer layer for SwinLSTM-B')
    parser.add_argument('--depths_down', default=[2, 6], type=int, help='Downsample of SwinLSTM-D')
    parser.add_argument('--depths_up', default=[6, 2], type=int, help='Upsample of SwinLSTM-D')
    parser.add_argument('--heads_number', default=[4, 8], type=int,
                        help='Number of attention heads in different layers')
    parser.add_argument('--window_size', default=4, type=int, help='Window size of Swin Transformer layer')
    parser.add_argument('--drop_rate', default=0., type=float, help='Dropout rate')
    parser.add_argument('--attn_drop_rate', default=0., type=float, help='Attention dropout rate')
    parser.add_argument('--drop_path_rate', default=0.1, type=float, help='Stochastic depth rate')

    # Training parameters
    parser.add_argument('--train_batch_size', default=16, type=int, help='Batch size for training')
    parser.add_argument('--valid_batch_size', default=16, type=int, help='Batch size for validation')
    parser.add_argument('--test_batch_size', default=16, type=int, help='Batch size for testing')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--epoch_valid', default=10, type=int)
    parser.add_argument('--log_train', default=100, type=int)
    parser.add_argument('--log_valid', default=60, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')

    args = parser.parse_args()

    return args
