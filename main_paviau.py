import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from util import seed_torch, adjust_learning_rate, getDatasetInfo
from NetworkPre import FeatureNet
from dataset import sanity_check, load_data, get_target_dataset, Traindata
from torchmetrics import Accuracy
from sklearn.metrics import precision_score, classification_report,cohen_kappa_score
from torch.optim import Adam


def parse_args():
    parser = argparse.ArgumentParser(description="FSOSR")
    parser.add_argument('--dataset', type=str, default='paviaU', choices=['paviaU', 'trento','pavia'])
    parser.add_argument("-m", "--test_class_num", type=int, default=8, choices=[8, 6])
    parser.add_argument("-z", "--test_lsample_num_per_class", type=int, default=5, help='5 4 3 2 1')
    parser.add_argument('--patch', type=int, default=13, choices=[7, 9, 11, 13, 15, 17])
    parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=19, metavar='N', help='Number of query in test')
    parser.add_argument('--episodes', type=int, default=40, help='Number of training episodes', choices=[40, 50])
    parser.add_argument('--n_train_para', type=int, default=1, metavar='train_batch_size',
                        help='Size of training batch)')
    parser.add_argument('--feature_dim', type=int, default=64, help='128 or 640 or 3')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,200',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--gamma', type=float, default=1.0, help='loss cofficient for mse loss')
    parser.add_argument('--funit', type=float, default=1.0)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--topk', type=int, default=24)
    parser.add_argument('--spectral_size', type=int, default=100)
    args = parser.parse_args()

    if args.dataset == 'paviaU':
        args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8]
        args.unknown_classes = [9, ]
        args.n_ways = 8
        args.n_open_ways = args.n_ways
    elif args.dataset == 'trento':
        args.known_classes = [1, 2, 3, 4, 5, 6]
        args.unknown_classes = [7, ]
        args.n_ways = 6
        args.n_open_ways = args.n_ways
    if args.dataset == 'pavia':
        args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8]
        args.unknown_classes = [9, ]
        args.n_ways = 8
        args.n_open_ways = args.n_ways
    return args

def F_measure(labels, preds, unknown=-1): # F1
    true_pos = 0.
    false_pos = 0.
    false_neg = 0.
    for i in range(len(labels)):
        true_pos += 1 if preds[i] == labels[i] and labels[i] != unknown else 0
        false_pos += 1 if preds[i] != labels[i] and preds[i] != unknown else 0
        false_neg += 1 if preds[i] != labels[i] and labels[i] != unknown else 0

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    return 2 * ((precision * recall) / (precision + recall))

def dataload(args):
    # load source domain data set
    patch=str(args.patch)
    name='TRIAN_META_DATA_imdb_ocbs_'+patch+'.pickle'
    with open(os.path.join('/mnt/HDD/data/zwj/model_2/my_model/data', name), 'rb') as handle:
        source_imdb = pickle.load(handle)
    source_imdb['data'] = np.array(source_imdb['data'])
    source_imdb['Labels'] = np.array(source_imdb['Labels'], dtype='int')
    source_imdb['set'] = np.array(source_imdb['set'], dtype='int')

    # process source domain data set
    keys_all_train = sorted(list(set(source_imdb['Labels'])))  # class [0,...,18]
    label_encoder_train = {}
    for i in range(len(keys_all_train)):
        label_encoder_train[keys_all_train[i]] = i
    del keys_all_train

    train_set = {}
    for class_, path in zip(source_imdb['Labels'], source_imdb['data']):
        if label_encoder_train[class_] not in train_set:
            train_set[label_encoder_train[class_]] = []
        train_set[label_encoder_train[class_]].append(path)
    del label_encoder_train
    data = sanity_check(train_set)  # 200 labels samples per class
    del source_imdb
    del train_set

    for class_ in data:
        for i in range(len(data[class_])):
            image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
            data[class_][i] = image_transpose
    del image_transpose
    # load target domain data set
    test_data = '/mnt/HDD/data/zwj/model_2/my_model/data/' + args.dataset + '/' + args.dataset + '.mat'
    test_label = '/mnt/HDD/data/zwj/model_2/my_model/data/' + args.dataset + '/' + args.dataset + '_gt.mat'

    Data_Band_Scaler, GroundTruth = load_data(test_data, test_label)
    del test_data
    del test_label
    test_known_loader, test_loader, target_da_metatrain_data, _ = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=args.test_class_num,
        shot_num_per_class=args.test_lsample_num_per_class, args=args)
    train_data = dict(data)
    del data
    train_data.update(target_da_metatrain_data)
    train_loader = DataLoader(Traindata(train_data, args), batch_size=args.n_train_para, shuffle=False)
    del train_data
    return train_loader, test_known_loader, test_loader


def train(args, model, train_loader):
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    train_acc_meter = Accuracy(task="multiclass", num_classes=args.n_ways).to(device=0)
    train_open_acc_meter = Accuracy(task="multiclass", num_classes=args.n_ways + 1).to(device=0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        adjust_learning_rate(epoch, args, optimizer, 0.0001)
        train_acc_meter.reset()
        train_open_acc_meter.reset()
        with tqdm(train_loader, total=len(train_loader), leave=False) as pbar:
            for idx, data in enumerate(pbar):
                support_data, support_label, query_data, query_label, suppopen_data, suppopen_label, openset_data, openset_label = data
                support_data, support_label = support_data.float().cuda(), support_label.cuda().long()
                query_data, query_label = query_data.float().cuda(), query_label.cuda().long()
                suppopen_data, suppopen_label = suppopen_data.float().cuda(), suppopen_label.cuda().long()
                openset_data, openset_label = openset_data.float().cuda(), openset_label.cuda().long()
                openset_label = args.n_ways * torch.ones_like(openset_label)
                the_img = (support_data, query_data, suppopen_data, openset_data)
                the_label = (support_label, query_label, suppopen_label, openset_label)
                probs, loss = model(the_img, the_label)
                (loss_cls, loss_open_hinge, loss_funit) = loss
                loss_open = args.gamma * loss_open_hinge + args.funit * loss_funit
                loss = loss_open + loss_cls

                # ===================backward=====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Closed Set Accuracy
                close_pred = torch.argmax(probs[0][:, :, :args.n_ways].view(-1, args.n_ways), -1)
                close_label = query_label.view(-1)
                open_pred = torch.argmax(
                    torch.cat((probs[0].view(-1, args.n_ways + 1), probs[1].view(-1, args.n_ways + 1)), dim=0),
                    -1)
                open_label = torch.cat((query_label.view(-1), openset_label.view(-1)))
                train_acc_meter.update(close_pred, close_label)
                train_open_acc_meter.update(open_pred, open_label)
                pbar.set_description(f'Epoch [{epoch}/{args.epochs}]')
                pbar.set_postfix({"loss": '{0:.2f}'.format(loss),
                                  "Acc": '{0:.2f}'.format(train_acc_meter.compute().item()),
                                  "Open_Acc": '{0:.2f}'.format(train_open_acc_meter.compute().item())})


def test(args, model, test_known_loader, test_loader):
    model.eval()
    torch.cuda.empty_cache()
    prediction = []
    label = []
    support_data, support_label = next(iter(test_known_loader))
    support_data, support_label = torch.unsqueeze(support_data, dim=0).float().cuda(), torch.unsqueeze(
        support_label, dim=0).cuda().long()
    with tqdm(test_loader, total=len(test_loader), leave=False) as pbar:
        for idx, data in enumerate(pbar):
            query_data, query_label = data
            query_data, query_label = torch.unsqueeze(query_data, dim=0).float().cuda(), torch.unsqueeze(
                query_label, dim=0).cuda().long()
            the_img = (support_data, query_data, support_data, query_data)
            the_label = (support_label, query_label, support_label, query_label)
            probs = model(the_img, the_label, test=True)
            prediction.append(torch.argmax(probs.view(-1, args.n_ways + 1), -1).cpu().detach().numpy())
            label.append(query_label.view(-1).cpu().detach().numpy())
        label = np.concatenate(label)
        prediction = np.concatenate(prediction)
        kappa=cohen_kappa_score(label,prediction)
        f1 = F_measure(label, prediction, unknown=args.n_ways)
        oa = precision_score(label, prediction, average="micro")
        aa = precision_score(label, prediction, average="macro")
        all_result = classification_report(label, prediction, digits=4)
        print('kappa',kappa)
        print('test_oa:', oa)
        print('test_aa:', aa)
        print('f1_micro:', f1)
        print(all_result)
        return kappa, f1, oa, aa, all_result


def main(seed):
    args = parse_args()
    data_info = getDatasetInfo()
    data_info['depth']=args.depth
    data_info['topk']=args.topk
    data_info['spectral_size']=args.spectral_size
    data_info['patch']=args.patch
    train_loader, test_known_loader, test_loader = dataload(args)
    model = FeatureNet(args.n_ways, args.feature_dim, data_info)
    model.cuda()
    train(args, model, train_loader)
    kappa, f1, oa, aa, all_result = test(args, model, test_known_loader, test_loader)
    with open('/mnt/HDD/data/zwj/model_2/my_model/result_paviau.txt', 'a') as f:
        f.write('number ')
        f.write(str(args.test_lsample_num_per_class))
        f.write('\n')
        f.write('seed ')
        f.write(str(seed))
        f.write('\n')
        f.write('kappa: ')
        f.write(str(kappa))
        f.write('\n')
        f.write('f1: ')
        f.write(str(f1))
        f.write('\n')
        f.write('oa: ')
        f.write(str(oa))
        f.write('\n')
        f.write('aa: ')
        f.write(str(aa))
        f.write('\n')
        f.write(all_result)
        f.write('\n')


if __name__ == "__main__":
    seedx = [100,200,300,400,500]
    for i in range(5):
        torch.cuda.empty_cache()
        seed = seedx[i]
        seed_torch(seed)
        main(seed)
