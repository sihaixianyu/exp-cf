import argparse

parser = argparse.ArgumentParser()
# Model relevant args
parser.add_argument('-m', '--model_name', default='semf', type=str,
                    help='chosea a target models')
parser.add_argument('-ld', '--latent_dim', default=256, type=int,
                    help='the dimision of latent space for user and item')
parser.add_argument('-ln', '--layer_num', default=3, type=int,
                    help='the layer number for gcn based models')
# Regularization relevant args
parser.add_argument('-wd', '--weight_decay', default=1e-3, type=float,
                    help='the weight decay of l2 regular')
parser.add_argument('-tp', '--theta_plus', default=1e-1, type=float,
                    help='the threshold of item similar regularization term')
parser.add_argument('-tm', '--theta_minus', default=1e-4, type=float,
                    help='the threshold of item similar regularization term')
parser.add_argument('-a', '--alpha', default=1e-4, type=float,
                    help='the coefficient of item similar regularization term')
parser.add_argument('-b', '--beta', default=1e-4, type=float,
                    help='the coefficient for user and positive item regularization term')
parser.add_argument('-g', '--gamma', default=1e-4, type=float,
                    help='the coefficient for user and negative item regularization term')
# Dataset relevant args
parser.add_argument('-d', '--data_name', default='ml-2k', type=str,
                    help='chose a target dataset')
parser.add_argument('-s', '--similarity', default='cosine', type=str,
                    help='the similarity calculation method for training data')
parser.add_argument('-n', '--neighbor_num', default=25, type=int,
                    help='the neighbor number for building explainable matrix')
# Train relevant args
parser.add_argument('-bs', '--batch_size', default=512, type=int,
                    help='the number of training data in each batch')
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                    help='the learning rate of trining process')
parser.add_argument('-e', '--epoch_num', default=500, type=int,
                    help='the total number of training epoch')
parser.add_argument('-r', '--retrain', action='store_true', default=True,
                    help='retrain the models from scratch')
# Test relevant args
parser.add_argument('-t', '--topk', default=10, type=int,
                    help='the number of items accepted by user in recommendation')
parser.add_argument('-i', '--interval', default=10, type=int,
                    help='the test interval of training epoch')
