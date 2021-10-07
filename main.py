import os
import warnings
from os import path
from parser import parser

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import models
import util
from dataset import Dataset
from recorder import Recorder
from tester import Tester
from trainer import Trainer

root_dir = '.'
# root_dir = 'drive/MyDrive/exp-cf/'

if __name__ == '__main__':
    np.random.seed(2021)
    warnings.filterwarnings('ignore')

    config = vars(parser.parse_args())
    util.sep_print(config, desc='Config')

    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_num']

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
    tester = Tester(test_loader, model, dataset.full_exp_mat, config['topk'])

    # Record results of each epoch
    recorder = Recorder(config['interval'], cmp_key='hr')

    model_dir = path.join(root_dir, 'ckpts', config['data_name'])
    util.check_dir(model_dir)

    model_path = model.get_model_path(model_dir)
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
    hr, ndcg, mep, wmep, test_time = tester.test()

    recorder.record(0, hr, ndcg, mep, wmep)
    print('Result: hr=%.4f, ndcg=%.4f, mep=%.4f, wmep=%.4f, test_time=%.4f' % (hr, ndcg, mep, wmep, test_time))

    for epoch in range(1, config['epoch_num'] + 1):
        loss, train_time = trainer.train()
        print('Epoch[%d/%d], loss=%.4f, train_time=%.4f' % (epoch, config['epoch_num'], loss, train_time))

        if epoch % config['interval'] == 0:
            util.color_print('[TEST]')
            hr, ndcg, mep, wmep, test_time = tester.test()

            is_update, not_improve = recorder.record(epoch, hr, ndcg, mep, wmep)
            print('Result: hr=%.4f, ndcg=%.4f, mep=%.4f, wmep=%.4f, test_time=%.4f' % (hr, ndcg, mep, wmep, test_time))

            if is_update:
                torch.save(model.state_dict(), model_path)
            if not_improve >= config['early_stop']:
                break

    recorder.print_best(model_name.upper(), keys=['hr', 'ndcg', 'mep', 'wmep'])

