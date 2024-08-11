import torch
import tqdm, gc, time, os
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
from models.emb_MLPs import *

from dataset import AvazuDataset, Movielens1MDataset, CriteoDataset


def get_dataset(name, dataset_path, root_path):
    if name == 'movielens1M':
        return Movielens1MDataset(dataset_path)
    elif name == 'avazu':
        # return AvazuDataset(dataset_path, cache_path=os.path.join(root_path, 'avazu_root_cache'))
        return AvazuDataset(dataset_path, cache_path=os.path.join(root_path, '.avazu'))
    elif name == 'criteo':
        return CriteoDataset(dataset_path, cache_path=os.path.join(root_path, 'criteo_root_cache'))
        # return CriteoDataset(dataset_path, cache_path=os.path.join(root_path, '.criteo'))


def get_model(name, args):
    if name == 'NoSlct':
        return MLP(args)
    # elif name == 'AdaFS_soft':
    #     return AdaFS_soft(args)
    elif name == 'AdaFS_hard':
        return AdaFS_hard(args)
    elif name == 'AdaFS_hard_attention':
        return AdaFS_hard(args)
    elif name == 'AEFS_emb_align_addLoss':
        return AEFS_emb_align_addLoss(args)
    elif name == 'AEFS':
        return AEFS(args)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save({'state_dict': model.state_dict()}, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train_fs_synchronize(model, train_data_loader, valid_data_loader,
                          device, log_interval):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(train_data_loader, smoothing=0, mininterval=1.0)

    for i, (fields, target) in enumerate(tk0):
        # if model.stage == 1: val_fields.append(fields); val_target.append(target)
        fields, target = fields.to(device), target.to(device)
        loss = model(fields, target.float(), step=i)

        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
        # if i > 100: break


def test(model, data_loader, device):
    model.eval()
    targets, predicts, infer_time = list(), list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            start = time.time()
            y = model.predict(fields)
            infer_cost = time.time() - start
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            infer_time.append(infer_cost)
    return roc_auc_score(targets, predicts), log_loss(targets, predicts), sum(infer_time)


def main(dataset_name,
         dataset_path,
         model_name,
         args,
         epoch,
         learning_rate,
         learning_rate_darts,
         batch_size,
         darts_frequency,
         weight_decay,
         device,
         pretrain,
         save_dir,
         param_dir):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path, args.root_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length), generator=torch.Generator().manual_seed(42))

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers)

    model = get_model(model_name, args)
    if pretrain == 0:
        print("trained_mlp_params:", param_dir)
        model.load_state_dict(torch.load(param_dir), strict=False)
    if args.load_model_dir != "None" and args.do_train == 0:
        print("load_model_params:", args.load_model_dir)
        model.load_state_dict(torch.load(args.load_model_dir)['state_dict'], strict=True)

    model = model.to(device)


    if pretrain == 1 and args.do_train:
        print(
            '\n********************************************* Pretrain *********************************************\n')
        model.stage = 0
        early_stopper = EarlyStopper(num_trials=3, save_path=f'{save_dir}/{model_name}:{dataset_name}_pretrain.pt')
        for epoch_i in range(epoch[0]):
            print('Pretrain epoch:', epoch_i)
            train_fs_synchronize(model, train_data_loader,
                                 valid_data_loader,  device, 100)

            auc, logloss, infer_time = test(model, valid_data_loader, device)
            if not early_stopper.is_continuable(model, auc):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break
            print('Pretrain epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss)

        auc, logloss, infer_time = test(model, test_data_loader, device)
        print(f'Pretrain test auc: {auc} logloss: {logloss}, infer time:{infer_time}\n')

    start_time = time.time()
    if args.do_train:
        print(
            '\n********************************************* Main_train *********************************************\n')
        model.stage = 1

        if args.controller:
            early_stopper = EarlyStopper(
                num_trials=3, save_path=f'{save_dir}/{model_name}:{dataset_name}_controller.pt')
        else:
            early_stopper = EarlyStopper(num_trials=3,
                                         save_path=f'{save_dir}/{model_name}:{dataset_name}_noController.pt')
        for epoch_i in range(epoch[1]):
            print('epoch:', epoch_i)
            train_fs_synchronize(model, train_data_loader,
                                 valid_data_loader, device, 100)

            auc, logloss, _ = test(model, valid_data_loader, device)
            if not early_stopper.is_continuable(model, auc):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break

            print('epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss)

    if args.do_eval:
        model.stage = 1

        auc, logloss, infer_time = test(model, test_data_loader, device)
        print(f'test auc: {auc} logloss: {logloss}\n')
        if args.do_train:
            with open('Record/%s_%s.txt' % (model_name, dataset_name), 'a') as the_file:
                the_file.write(
                    '\nModel:%s,Controller:%s,pretrain_type:%s,pretrain_eopch:%s\nDataset:%s,useBN:%s\ntrain Time:%.2f,train Epoches: %d\n test auc:%.8f,logloss:%.8f, darts_frequency:%s\n'
                    % (
                    model_name, str(args.controller), str(args.pretrain), str(epoch[0]), dataset_name, str(args.useBN),
                    (time.time() - start_time) / 60, epoch_i + 1, auc, logloss, str(darts_frequency)))
                if args.model_name == 'AdaFS_hard':
                    result_temp = ""
                    if args.AdaFS_hard_threshold > 0:
                        result_temp = result_temp + 'AdaFS_hard_threshold:%.2f\t' % (float(args.AdaFS_hard_threshold))
                    else:
                        result_temp = result_temp + 'k:%s\t' % (str(args.k))
                    result_temp = result_temp + '\t useWeight:%s, reWeight:%s\t' % (
                    str(args.useWeight), str(args.reWeight))
                    the_file.write(result_temp + "\n")
                if pretrain == 0:
                    the_file.write('trained_mlp_params:%s\n' % (str(param_dir)))

        with open('Record/eval_%s_%s_controller_%s.txt' % (model_name, dataset_name, args.controller_type), 'a') as the_file:
            result_temp = "test auc:%.8f\t logloss:%.8f\t" % (auc, logloss)
            result_temp = result_temp + str(time.asctime(time.localtime(time.time()))) + '\t'
            result_temp = result_temp + save_dir + '\t'
            if args.model_name == 'AdaFS_hard':
                if args.AdaFS_hard_threshold > 0:
                    result_temp = result_temp + 'AdaFS_hard_threshold:%.2f\t' % (float(args.AdaFS_hard_threshold))
                else:
                    result_temp = result_temp + 'k:%s\t' % (str(args.k))
                result_temp = result_temp + '\t useWeight:%s, reWeight:%s\t' % (str(args.useWeight), str(args.reWeight))
            the_file.write(result_temp + "\n")


def get_setting(args):
    temp = "2023"
    if args.AdaFS_hard_threshold:
        temp = temp + 'AdaFS_hard_threshold_%.2f_' % (args.AdaFS_hard_threshold)

    if args.model_name == 'AdaFS_hard_attention':
        temp = temp + 'AdaFS_hard_attention_'
        if args.AttentionWithAve:
            temp = temp + 'AttentionWithAve_'
    elif args.model_name == 'AdaFS_twoModel_emb_align':
        temp = temp + 'AdaFS_twoModel_emb_align_'
        temp = temp + 'embDim_%d_small_%d_' % (args.embed_dim, args.embed_dim_small)
        temp = temp + "optimizerType_" + args.two_model_optimizer_type + "_"
        temp = temp + "controllerPosition_" + args.two_model_controller_position + "_"
    elif args.model_name in ['AEFS_emb_align_addLoss', 'AEFS']:
        temp = temp + '%s_' % args.model_name
        temp = temp + 'embDim_%d_small_%d_' % (args.embed_dim, args.embed_dim_small)
        temp = temp + 'scoreAlign_%s_fieldAlign_%s_Weight_%s_' % (args.score_align_loss,
                                                        args.field_align_loss, args.score_and_field_align_loss_weight)
        if args.small_loss_weight != 1:
            temp = temp + "small_loss_weight_%.1f_" % (args.small_loss_weight)
        temp = temp + "controllerPosition_" + args.two_model_controller_position + "_"
        if args.model_name == 'AEFS':
            temp = temp + "dense_%s_D_Small_%s_" % (args.dense_type, args.dense_type_small)
    elif args.model_name == 'AdaFS_twoModel':
        temp = temp + 'AdaFS_twoModel_'
        temp = temp + 'embDim_%d_small_%d_' % (args.embed_dim, args.embed_dim_small)
    elif args.model_name == 'NoSlct':
        temp = temp + 'NoSlct_'

    if args.controller_type != "controller_mlp":
        temp = temp + "controller_%s_"%(args.controller_type)

    if args.k_ratio != 0.5:
        temp = temp + "_k_ratio_%.2f_"%(args.k_ratio)

    if args.TrainFSSynchronize:
        temp = temp + 'TrainFSSynchronize_'


    if args.softmaxAddNorm != 'None':  # only for AdaFS_soft
        assert args.model_nam == 'AdaFS_soft'
        temp = temp + '%s_' % args.softmaxAddNorm

    if args.pretrain != 1:
        temp = temp + 'pretrain_%d_' % args.pretrain
    if args.reWeight != True:
        temp = temp + 'noReWeight_'
    return temp


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        # os.environ['CUDA_VISIBLE_DEVICES'] = "1"

        sys.argv = "main.py --root_path /home/hf/code/AdaFS-main/  --model_name AEFS  --num_workers 1 --dataset_name avazu " \
                   "--save_dir Ada_twoModel/chkpt --pretrain 2 --embed_dim 32 " \
                   "--embed_dim_small 4 --two_model_controller_position small_emb --field_align_loss MSE --score_align_loss MSE " \
                   "--dense_type DeepFM --dense_type_small MultiLayerPerceptron " \
                   "--controller_type MvFS --k_ratio 0.5 --do_train 1 --small_loss_weight 0".split()


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1M', help='criteo, avazu, movielens1M')
    parser.add_argument('--model_name', default='AdaFS_soft',
                        help='AdaFS_soft, AdaFS_hard, AdaFS_hard_attention, AdaFS_twoModel')
    parser.add_argument('--k', type=int, default=0)  # for AdaFS_hard
    parser.add_argument('--useWeight', type=bool, default=True)
    parser.add_argument('--reWeight', type=int, default=1)
    parser.add_argument('--useBN', type=bool, default=True)
    parser.add_argument('--mlp_dims', type=int, default=[16, 8], help='original=16')
    parser.add_argument('--epoch', type=int, default=[2, 50], nargs='+', help='pretrain/main_train epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_darts', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--darts_frequency', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--dropout', type=int, default=0.2)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='cuda:0')
    parser.add_argument('--save_dir', default='chkpt', help='the subpath of root_path')
    parser.add_argument('--controller', default=True, help='True:Use controller in model; False:Do not use controller')
    parser.add_argument('--pretrain', type=int, default=2, help='0:pretrain to converge, 1:pretrain, 2:no pretrain')
    parser.add_argument('--repeat_experiments', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=8)


    # add
    parser.add_argument('--root_path', default='./',
                        help='the root path "./"')
    parser.add_argument('--do_train', type=int, default=1)  #
    parser.add_argument('--do_eval', type=int, default=1)  #
    parser.add_argument('--AdaFS_hard_threshold', type=float, default=0)  #
    parser.add_argument('--load_model_dir', default='None', help='load_model_dir')
    ## For attention
    parser.add_argument('--AttentionWithAve', type=int, default=1)
    ## For train FS synchronize
    parser.add_argument('--TrainFSSynchronize', type=int, default=1)
    parser.add_argument('--softmaxAddNorm', default='None', help='Max-min, Max-min_0.1_1, Max-min_0.5_1, ...')


    ## for AEFS model
    parser.add_argument('--embed_dim', type=int, default=32, help='original=32')
    parser.add_argument('--embed_dim_small', type=int, default=4, help='original=4')
    parser.add_argument('--dense_type', default='MultiLayerPerceptron',
                        help='the dense layer type: MultiLayerPerceptron, FM, DeepFM, DeepCrossNet, InnerProductNet')
    parser.add_argument('--dense_type_small', default='MultiLayerPerceptron',
                        help='the dense layer type of small model')
    parser.add_argument('--two_model_optimizer_type', default='simultaneous', help='simultaneous, split, for `AdaFS_twoModel_emb_align` model')
    parser.add_argument('--two_model_controller_position', default='small_emb', help='normal_emb, small_emb, for `AdaFS_twoModel_emb_align` model')

    parser.add_argument('--weight_align_loss', default='None', help='None, pearson, huber, for `AEFS` model')
    parser.add_argument('--score_align_loss', default='MSE', help='MSE, kl_loss, huber, for `AEFS` model')
    parser.add_argument('--field_align_loss', default='MSE', help='MSE, kl_loss, huber, for `AEFS` model')
    parser.add_argument('--score_and_field_align_loss_weight', default='0.3_0.3', help='for `AEFS` model')
    parser.add_argument('--small_loss_weight', type=float, default=1, help='for `AEFS` model')
    parser.add_argument('--controller_type', default="controller_mlp", help='controller_mlp, MvFS, attention')
    parser.add_argument('--k_ratio', type=float, default=0.5, help='the ratio of selected feature') #


    args = parser.parse_args()

    args.reWeight = bool(args.reWeight)

    args.save_dir = os.path.join(args.root_path, args.save_dir, get_setting(args))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    param_dir = args.save_dir
    if args.dataset_name == 'criteo':
        dataset_path = os.path.join(args.root_path, 'dataset/criteo/train.txt')
        param_dir += '/mlp:criteo_noController.pt'
    if args.dataset_name == 'avazu':
        dataset_path = os.path.join(args.root_path, 'dataset/avazu/train')
        param_dir += '/mlp:avazu_noController.pt'
    if args.dataset_name == 'movielens1M':
        dataset_path = os.path.join(args.root_path, 'dataset/ml-1m/train.txt')
        param_dir += '/mlp:movielens1M_noController.pt'

    if args.dataset_name == 'movielens1M':
        args.field_dims = [3706,301,81,6040,21,7,2,3402]
    elif args.dataset_name == 'avazu':
        args.field_dims = [241, 8, 8, 3697, 4614, 25, 5481, 329,
            31, 381763, 1611748, 6793, 6, 5, 2509, 9, 10, 432, 5, 68, 169, 61]
    elif args.dataset_name == 'criteo':
        args.field_dims = [    49,    101,    126,     45,    223,    118,     84,     76,
           95,      9,     30,     40,     75,   1458,    555, 193949,
       138801,    306,     19,  11970,    634,      4,  42646,   5178,
       192773,   3175,     27,  11422, 181075,     11,   4654,   2032,
            5, 189657,     18,     16,  59697,     86,  45571]

    if args.model_name == 'NoSlct':
        args.controller = False

    if args.controller:
        if args.k == 0:
            args.k = int(len(args.field_dims) / 2)
        print(f'\nk = {args.k},\t',
              f'useWeight = {args.useWeight},\t',
              f'reWeight = {args.reWeight}', )
    print(f'\nrepeat_experiments = {args.repeat_experiments}')
    print(f'\ndataset_name = {args.dataset_name},\t',
          f'dataset_path = {dataset_path},\t',
          f'model_name = {args.model_name},\t',
          f'Controller = {args.controller},\t',
          f'useBN = {args.useBN},\t',
          f'mlp_dim = {args.mlp_dims},\t',
          f'epoch = {args.epoch},\t',
          f'learning_rate = {args.learning_rate},\t',
          f'learning_rate_darts = {args.learning_rate_darts},\t',
          f'batch_size = {args.batch_size},\t',
          f'darts_frequency = {args.darts_frequency},\t',
          f'weight_decay = {args.weight_decay},\t',
          f'device = {args.device},\t',
          f'pretrain_type = {args.pretrain},\t',
          f'save_dir = {args.save_dir}\n')
    for i in range(args.repeat_experiments):
        save_dir = args.save_dir + "_%d" % i
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        time_start = time.time()
        main(args.dataset_name,
             dataset_path,
             args.model_name,
             args,
             args.epoch,
             args.learning_rate,
             args.learning_rate_darts,
             args.batch_size,
             args.darts_frequency,
             args.weight_decay,
             args.device,
             args.pretrain,
             save_dir,
             param_dir)

        print(f'\ndataset_name = {args.dataset_name},\t',
              f'dataset_path = {dataset_path},\t',
              f'model_name = {args.model_name},\t',
              f'Controller = {args.controller},\t',
              f'mlp_dim = {args.mlp_dims},\t',
              f'epoch = {args.epoch},\t',
              f'learning_rate = {args.learning_rate},\t',
              f'learning_rate_darts = {args.learning_rate_darts},\t',
              f'batch_size = {args.batch_size},\t',
              f'darts_frequency = {args.darts_frequency},\t',
              f'weight_decay = {args.weight_decay},\t',
              f'device = {args.device},\t',
              f'pretrain_type = {args.pretrain},\t',
              f'save_dir = {save_dir},\t',
              f'training time = {(time.time() - time_start) / 3600}\n')


