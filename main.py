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
from tester import Tester
from trainer import Trainer

# root_dir = 'drive/MyDrive/egcn/'
root_dir = '/Users/sihaixianyu/Projects/PythonProjects/egcn/'

if __name__ == '__main__':
    np.random.seed(2021)
    warnings.filterwarnings('ignore')

    config = toml.load(path.join(root_dir, 'config.toml'))
    util.sep_print(config['data_name'], desc='Dataset Name', end=False)

    data_dir = path.join(root_dir, 'data/', config['data_name'])
    dataset = Dataset(data_dir, config['neighbor_num'])

    dataset.set_sample_method(config['sample_method'])
    util.sep_print(config['sample_method'], desc='Sample Method', end=False)

    model_name = config['model_name']
    model_config = config[model_name]
    util.sep_print(model_config, desc='Model {}'.format(model_name.upper()))

    if model_name == 'mf':
        model = MF(dataset, model_config)
        optimizer = Adam(model.parameters(), lr=model_config['lr'], weight_decay=model_config['weight_decay'])
    elif model_name == 'emf':
        model = EMF(dataset, model_config)
        optimizer = Adam(model.parameters(), lr=model_config['lr'], weight_decay=model_config['weight_decay'])
    elif model_name == 'lgcn':
        model = LGCN(dataset, model_config)
        optimizer = Adam(model.parameters(), lr=model_config['lr'])
    elif model_name == 'egcn':
        model = EGCN(dataset, model_config)
        optimizer = Adam(model.parameters(), lr=model_config['lr'])
    else:
        raise ValueError('No such model: %s' % model_config['model_name'])

    # Warning: we must set tester's batch_size=100 due to our leave-on-out evaluation strategy
    train_loader = DataLoader(dataset.get_train_data(), batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset.get_test_data(), batch_size=100, shuffle=False)

    trainer = Trainer(train_loader, model, optimizer)
    tester = Tester(test_loader, model, config['topk'])

    model_dir = path.join(root_dir, 'ckpt/')
    if not path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = path.join(model_dir, '{}_{}_{}.pth'.format(config['data_name'],
                                                            config['model_name'].lower(),
                                                            config['sample_method']))
    if not path.exists(model_path):
        torch.save(model.state_dict(), model_path)
    else:
        try:
            state_dict = torch.load(model_path, map_location=model.device)
            model.load_state_dict(state_dict)
        except RuntimeError:
            torch.save(model.state_dict(), model_path)

    util.color_print('[TEST]')
    hr, ndcg, test_time = tester.test()
    best_epoch = {'epoch': 0, 'hr': hr, 'ndcg': ndcg}
    print('Result: hr=%.4f, ndcg=%.4f, test_time=%.4f' % (hr, ndcg, test_time))

    # best_epoch = {'epoch': 0, 'hr': 0, 'ndcg': 0}
    for epoch in range(1, config['epoch_num'] + 1):
        loss, train_time = trainer.train()
        print('Epoch[%d/%d], loss=%.4f, train_time=%.4f' % (epoch, config['epoch_num'], loss, train_time))

        if epoch % config['test_interval'] == 0:
            util.color_print('[TEST]')
            hr, ndcg, test_time = tester.test()
            print('Result: hr=%.4f, ndcg=%.4f, test_time=%.4f' % (hr, ndcg, test_time))

            if best_epoch['hr'] <= hr:
                best_epoch['epoch'] = epoch
                best_epoch['hr'] = hr
                best_epoch['ndcg'] = ndcg
                torch.save(model.state_dict(), model_path)

    util.sep_print('Best: epoch=%.4f, hr=%.4f, ndcg=%.4f' % (best_epoch['epoch'], best_epoch['hr'], best_epoch['ndcg']))
