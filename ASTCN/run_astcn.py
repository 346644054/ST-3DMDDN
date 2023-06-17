import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
import time
import sys
import datetime
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import STData
from astcn_model import ASTCN, Regularization
from logger import Logger
import torch

torch.cuda.current_device()
torch.cuda._initialized = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

params = argparse.ArgumentParser()
params.add_argument('--windows_size', type = int, default = 7 * 48, help = ' ')
params.add_argument('--flow_channel', type = int, default = 2, help = ' ')
params.add_argument('--lr', type = int, default = 6e-4, help = ' ')
params.add_argument('--batch_size', type = int, default = 32, help = ' ')
params.add_argument('--l2_weight_decay', type = int, default = 0, help = ' ')
params.add_argument('--epochs', type = int, default = 50, help = ' ')
params.add_argument('--max_flow', type = int, default = 1292, help = ' ')
params.add_argument('--min_flow', type = int, default = 0, help = ' ')
params = params.parse_args()

def main(run_fn, begin_time):
    min_validate_RMSE = torch.randn(1)
    min_validate_RMSE[0] = 9999
    min_validate_RMSE.share_memory_()
    run_fn(begin_time, min_validate_RMSE)

def setup():
    torch.manual_seed(2469)

def build_network(device):
    astcn = ASTCN(2, 2, [8, 8, 16, 16, 32, 32, 64, 64], [16, 16, 32, 32, 64, 64, 128, 128],
                  [3, 3, 3], 0.0, params.windows_size, use_ext_in_att=True)
    astcn.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(astcn.parameters(), lr=params.lr)
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return astcn, criterion, optimizer, stepLR

def inverse_mmn(img):
    img = (img + 1.) / 2.
    img = 1. * img * (params.max_flow - params.min_flow) + 0.0
    return img

def cal_test_rmse(model, criterion, device):
    test_set = STData(dataset_type="test", windows_size=params.windows_size, dir_path='./data/processed/TaxiBJ/')
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=params.batch_size, shuffle=False, num_workers=4)

    total_loss = []
    count_all_pred_val = 0
    model.eval()

    att_va_arr_save = []

    batch_pred_real_save = []
    batch_output_real_save = []

    with torch.no_grad():
        for i, batchdata in enumerate(test_loader):
            batch_input = batchdata[0].to(device)
            batch_exter = batchdata[1].to(device)
            batch_output = batchdata[2].to(device)
            batch_pred, att_va_arr = model([batch_input, batch_exter])

            criterion_loss = criterion(batch_pred, batch_output)

            batch_pred_real = inverse_mmn(batch_pred).float()
            batch_output_real = inverse_mmn(batch_output).float()

            if (i == 0):
                att_va_arr_save = att_va_arr
            batch_pred_real_save.append(batch_pred_real)
            batch_output_real_save.append(batch_output_real)

            loss = (batch_pred_real - batch_output_real) ** 2
            total_loss.append(loss.sum().item())
            count_all_pred_val += batch_pred_real.shape.numel()
    RMSE = np.sqrt(np.array(total_loss).sum() / count_all_pred_val)

    # save attention value
    for idx_layer in range(len(att_va_arr_save)):
        for idx_type in range(len(att_va_arr_save[0])):
            att_va_arr_save[idx_layer][idx_type] = np.array(att_va_arr_save[idx_layer][idx_type].cpu().numpy())
    att_va_arr = np.array(att_va_arr_save)
    np.save("att_va_arr", att_va_arr)

    # save result
    batch_pred_real_save = torch.cat(batch_pred_real_save)
    batch_output_real_save = torch.cat(batch_output_real_save)
    pred_result = np.concatenate([batch_pred_real_save.cpu().numpy(), batch_output_real_save.cpu().numpy()], 0)
    np.save("figure_pred_targ", pred_result)
    return RMSE, criterion_loss.item()


def cal_val_rmse(model, criterion, device):
    val_set = STData(dataset_type="validate", windows_size=params.windows_size, dir_path='./data/processed/TaxiBJ/')
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=params.batch_size, shuffle=False, num_workers=4)

    total_loss = []
    count_all_pred_val = 0
    model.eval()
    with torch.no_grad():
        for i, batchdata in enumerate(val_loader):
            batch_input = batchdata[0].to(device)
            batch_exter = batchdata[1].to(device)
            batch_output = batchdata[2].to(device)
            batch_pred, att_va_arr = model([batch_input, batch_exter])

            criterion_loss = criterion(batch_pred, batch_output)

            batch_pred_real = inverse_mmn(batch_pred).float()
            batch_output_real = inverse_mmn(batch_output).float()

            loss = (batch_pred_real - batch_output_real) ** 2
            total_loss.append(loss.sum().item())
            count_all_pred_val += batch_pred_real.shape.numel()
    RMSE = np.sqrt(np.array(total_loss).sum() / count_all_pred_val)
    return RMSE, criterion_loss.item()


def run(begin_time, min_validate_RMSE):
    logger_path = os.path.join('./tensorboard/', begin_time)
    setup()
    logger = Logger(logger_path)
    device = 0

    stcnn, criterion, optimizer, stepLR = build_network(device)
    if params.l2_weight_decay != 0:
        reg_loss = Regularization(stcnn, params.l2_weight_decay, p=2).to(device)

    par = list(stcnn.parameters())
    s = sum([np.prod(list(d.size())) for d in par])
    print("Parameter of stcnn:", s)

    train_set = STData(dataset_type="train_all", windows_size=params.windows_size,
                           dir_path='./data/processed/TaxiBJ/')
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=params.batch_size,
                                                   shuffle=True,
                                                   num_workers=4, pin_memory=True, drop_last=True)

    for i in range(params.epochs):
        stepLR.step()
        for j, batchdata in enumerate(train_loader, 1):
            stcnn.train()
            last_time = time.time()
            # shape of batch_input: [D, B, H, W, C{in}]
            batch_input = batchdata[0].to(device, non_blocking=True)
            batch_exter = batchdata[1].to(device, non_blocking=True)
            batch_output = batchdata[2].to(device, non_blocking=True)

            batch_input.requires_grad = True
            batch_exter.requires_grad = True

            batch_pred, att_va_arr = stcnn([batch_input, batch_exter])
            loss = criterion(batch_pred, batch_output)
            if params.l2_weight_decay != 0:
                loss = loss + reg_loss(stcnn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sys.stdout.write(
                "\rBatch: {} - iter {} - sacled_loss {} - loss {} - iter/s: {}".format(i, j, loss.item(),
                                                                                       loss.item(),
                                                                                       1. / (time.time() - last_time)))
            sys.stdout.flush()
        print("\n")
        with torch.no_grad():
            stcnn.eval()
            RMSE, loss = cal_val_rmse(stcnn, criterion, device)
            logger.scalar_summary('RMSE of val set', RMSE, i + 1)
            logger.scalar_summary('loss of val set', loss, i + 1)
            print('\tVALIDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t loss:{}'.format(i, RMSE, loss))
            if RMSE < min_validate_RMSE[0]:
                save_model_path = os.path.join("./restore/", begin_time)
                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)
                torch.save(stcnn.state_dict(), os.path.join(save_model_path, 'stcnn_params.pkl'))
                min_validate_RMSE[0] = RMSE
                print('\tSave model!'.ljust(12), '\tEpoch:{}\t\tVal_RMSE: {}'.format(i, RMSE))
                
            RMSE, loss = cal_test_rmse(stcnn, criterion, device)
            logger.scalar_summary('RMSE of test set', RMSE, i + 1)
            logger.scalar_summary('loss of test set', loss, i + 1)
            print('\tTEST'.ljust(12), '\tEpoch:{}\t\tRMSE:     {}'.format(i, RMSE))



def load_checkpoint(begin_time, min_validate_RMSE):
    setup()
    import torch
    torch.cuda.current_device()
    torch.cuda._initialized = True
    device = 0
    torch.cuda.set_device(device)
    print(torch.cuda.device_count())
    stcnn, criterion, optimizer, stepLR = build_network(device)
    par = list(stcnn.parameters())
    s = sum([np.prod(list(d.size())) for d in par])
    print("Parameter of stcnn:", s)

    dir_name = u'./restore/'
    list_dir = os.listdir(dir_name)
    list_dir = sorted(list_dir, key=lambda x: os.path.getmtime(os.path.join(dir_name, x)))

    latest_dir = dir_name + list_dir[-1] + '/'
    # latest_dir = './restore/2021-06-09 15 26 36/'
    file_name = os.listdir(latest_dir)
    file_name = sorted(file_name, key=lambda x: os.path.getmtime(os.path.join(latest_dir, x)))

    CHECKPOINT_PATH = latest_dir + file_name[-1]

    stcnn.load_state_dict(torch.load(CHECKPOINT_PATH))
    optimizer.zero_grad()
    RMSE, loss = cal_test_rmse(stcnn, criterion, device)
    print("evaluating using the final model using {} ".format(CHECKPOINT_PATH))
    print("RMSE: ", RMSE)


if __name__ == '__main__':
    print("Build logger path")
    begin_time = datetime.datetime.now().strftime("%Y-%m-%d %H %M %S")
    main(run, begin_time)

    main(load_checkpoint, begin_time)