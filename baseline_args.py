import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='IMDB',
                        help='Options: IMDB, YELP_13, YELP_14, Amazon')
    parser.add_argument('--area', type=str, default='app',
                        help='Options: app, cds, kindle, movie, elec, home, video')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--output-channel', type=int, default=300)
    parser.add_argument('--out_features', type=int, default=200)
    parser.add_argument('--out_dim', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--feature_dim', type=int, default=300)
    parser.add_argument('--is_att', action='store_true')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--id', type=str, default='test')
    parser.add_argument('--num_dir', type=int, default=0, choices=[0,1], ## 0: uni-direction, 1: bi-direction
                        help='choices of uni- or bi-direction')
    parser.add_argument('--baseline', type=str,
                        choices=['lstm', 'stack','cnnlstm','caps'],
                        help='choices of baseline in training process')
    return parser