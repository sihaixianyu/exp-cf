import os
import warnings
from os import path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import models
import util
from dataset import Dataset
from parser import parser
from tester import Tester
from trainer import Trainer

# root_dir = 'drive/MyDrive/egcn/'
root_dir = '/Users/sihaixianyu/Projects/PythonProjects/egcn/'

if __name__ == '__main__':
    np.random.seed(2021)
    warnings.filterwarnings('ignore')

    config = vars(parser.parse_args())
    util.sep_print(config, desc='Config')

    data_dir = path.join(root_dir, 'data/', config['data_name'])
    dataset = Dataset(data_dir, config)

    # Warning: we must set test loader batch_size=100 due to our leave-on-out evaluation strategy
    train_loader = DataLoader(dataset.get_train_data(), batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset.get_test_data(), batch_size=100, shuffle=False)

    model_name = config['model_name']
    model_class = getattr(models, model_name.upper())
    model = model_class(dataset, config)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    trainer = Trainer(train_loader, model, optimizer)
    tester = Tester(test_loader, model, config['topk'])

    model_dir = path.join(root_dir, 'ckpt', config['data_name'])
    if not path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = model.get_model_suffix(model_dir)
    if not path.exists(model_path) or config['retrain']:
        print('Training models {} from scratch...'.format(model_name))
        torch.save(model.state_dict(), model_path)
    else:
        try:
            state_dict = torch.load(model_path, map_location=model.device)
            model.load_state_dict(state_dict)
            print('Loading trained models {}...'.format(model_name))
        except RuntimeError:
            warnings.warn('Loading models {} failed and train from scratch...'.format(model_name))
            torch.save(model.state_dict(), model_path)

    util.color_print('[TEST]')
    hr, ndcg, test_time = tester.test()
    best_epoch = {'epoch': 0, 'hr': hr, 'ndcg': ndcg}
    print('Result: hr=%.4f, ndcg=%.4f, test_time=%.4f' % (hr, ndcg, test_time))

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

    util.sep_print('%s Best: epoch=%.4f, hr=%.4f, ndcg=%.4f' % (model_name.upper(),
                                                                best_epoch['epoch'],
                                                                best_epoch['hr'],
                                                                best_epoch['ndcg']))
