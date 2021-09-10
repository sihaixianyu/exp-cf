import os
import warnings
from os import path

import numpy as np
import toml
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import util
from dataset import Dataset
from model import LGCN, MF, EMF, EGCN
from model.segcn import SEGCN
from tester import Tester
from trainer import Trainer
from parser import parser

# root_dir = 'drive/MyDrive/egcn/'
root_dir = '/Users/sihaixianyu/Projects/PythonProjects/egcn/'

if __name__ == '__main__':
    np.random.seed(2021)
    warnings.filterwarnings('ignore')

    config = vars(parser.parse_args())
    util.sep_print(config, desc='Config')

    data_dir = path.join(root_dir, 'data/', config['data_name'])
    dataset = Dataset(data_dir, config)

    train_loader = DataLoader(dataset.get_train_data(), batch_size=config['batch_size'], shuffle=True)
    # Warning: we must set tester's batch_size=100 due to our leave-on-out evaluation strategy
    test_loader = DataLoader(dataset.get_test_data(), batch_size=100, shuffle=False)

    model_name = config['model_name']
    if model_name == 'mf':
        model = MF(dataset, config)
        optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif model_name == 'emf':
        model = EMF(dataset, config)
        optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif model_name == 'lgcn':
        model = LGCN(dataset, config)
        optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    elif model_name == 'egcn':
        model = EGCN(dataset, config)
        optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    elif model_name == 'segcn':
        model = SEGCN(dataset, config)
        optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    else:
        raise ValueError('No such model: %s' % model_name)

    trainer = Trainer(train_loader, model, optimizer)
    tester = Tester(test_loader, model, config['topk'])

    model_dir = path.join(root_dir, 'ckpt', config['data_name'])
    if not path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = model.get_model_suffix(model_dir)
    if not path.exists(model_path) or config['retrain']:
        print('Training model {} from scratch...'.format(model_name))
        torch.save(model.state_dict(), model_path)
    else:
        try:
            state_dict = torch.load(model_path, map_location=model.device)
            model.load_state_dict(state_dict)
            print('Loading trained model {}...'.format(model_name))
        except RuntimeError:
            warnings.warn('Loading model {} failed and train from scratch...'.format(model_name))
            torch.save(model.state_dict(), model_path)

    util.color_print('[TEST]')
    hr, ndcg, test_time = tester.test()
    best_epoch = {'epoch': 0, 'hr': hr, 'ndcg': ndcg}
    print('Result: hr=%.4f, ndcg=%.4f, test_time=%.4f' % (hr, ndcg, test_time))

    # best_epoch = {'epoch': 0, 'hr': 0, 'ndcg': 0}
    for epoch in range(1, config['epoch_num'] + 1):
        loss, train_time = trainer.train()
        print('Epoch[%d/%d], loss=%.4f, train_time=%.4f' % (epoch, config['epoch_num'], loss, train_time))

        if epoch % config['interval'] == 0:
            util.color_print('[TEST]')
            hr, ndcg, test_time = tester.test()
            print('Result: hr=%.4f, ndcg=%.4f, test_time=%.4f' % (hr, ndcg, test_time))

            if best_epoch['hr'] <= hr:
                best_epoch['epoch'] = epoch
                best_epoch['hr'] = hr
                best_epoch['ndcg'] = ndcg
                torch.save(model.state_dict(), model_path)

    util.sep_print('Best: epoch=%.4f, hr=%.4f, ndcg=%.4f' % (best_epoch['epoch'], best_epoch['hr'], best_epoch['ndcg']))
