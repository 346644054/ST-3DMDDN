import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm

class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")


class SelfAttention(nn.Module):
    '''
    Q: Packed queries. 3d tensor. [N, C_in, L_in].
    K: Packed keys. 3d tensor. [N, C_in, L_in].
    V: Packed values. 3d tensor. [N, C_in, L_in].
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    '''

    def __init__(self, dropout):
        super(SelfAttention, self).__init__()
        self.softmax1 = nn.Softmax(-1)
        # self.dropout1 = nn.Dropout(dropout)
        self.count = 1

    def future_mask(self, inputs):
        padding_num = float('-inf')
        diag_vals = torch.ones_like(inputs[0, :, :])  # (L_in, C_in)
        tril = torch.tril(diag_vals)  # (L_in, C_in)
        future_masks = tril.unsqueeze(0).repeat(inputs.size(0), 1, 1)

        paddings = (torch.ones_like(future_masks) * padding_num)
        outputs = torch.where(torch.eq(future_masks, 0.0), paddings, inputs)
        return outputs

    def forward(self, Q, K, V, causality=False):
        d_k = K.size(-1)
        # dot product
        outputs = torch.matmul(Q, torch.transpose(K, -1, -2))
        # scale
        outputs = (outputs / (d_k))
        if causality:
            outputs = self.future_mask(outputs)
        # softmax
        outputs = self.softmax1(outputs)
        return outputs


class MultiHeadAttion(nn.Module):
    def __init__(self, num_heads=4, C_in=2, windows_size=7 * 48, dropout_rate=0.2, causality=True,
                 use_ext_in_att=False, layer_Num = 0):
        super(MultiHeadAttion, self).__init__()
        self.num_heads = num_heads
        self.C_in = C_in
        self.use_ext_in_att = use_ext_in_att

        self.c_in_ext = C_in if not self.use_ext_in_att else C_in * 2

        self.reduce_inputs_shape_as = 64
        self.d_models_middle = self.num_heads * 64
        self.causality = causality
        self.layer_Num = layer_Num

        self.gap = nn.AdaptiveAvgPool3d([windows_size, 1, 1])

        self.ext_seq = nn.Sequential(nn.Conv3d(28, 128, 1, bias=False),
                                     nn.BatchNorm3d(128),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv3d(128, 128, 1, bias=False),
                                     nn.BatchNorm3d(128),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv3d(128, C_in, 1, bias=False),
                                     nn.BatchNorm3d(C_in),
                                     nn.LeakyReLU(inplace=True),
                                     nn.AdaptiveAvgPool3d([windows_size, 1, 1]))

        self.merge_channel = nn.Sequential(nn.Conv3d(self.c_in_ext, self.reduce_inputs_shape_as, 1, bias=False),
                                           nn.BatchNorm3d(self.reduce_inputs_shape_as),
                                           nn.LeakyReLU(inplace=True),
                                           nn.Conv3d(self.reduce_inputs_shape_as, self.reduce_inputs_shape_as, 1, bias=False),
                                           nn.LayerNorm([self.reduce_inputs_shape_as, windows_size, 1, 1], eps=1e-6))

        self.linear1 = nn.Linear(self.reduce_inputs_shape_as, self.d_models_middle, bias=False)
        self.linear2 = nn.Linear(self.reduce_inputs_shape_as, self.d_models_middle, bias=False)
        self.linear3 = nn.Linear(self.reduce_inputs_shape_as, self.d_models_middle, bias=False)
        self.attention1 = SelfAttention(dropout_rate)
        self.linear_out = nn.Linear(self.d_models_middle, 1, bias=False)


    def forward(self, multiVarInputs):
        if not self.use_ext_in_att:
            inputs = multiVarInputs
        else:
            inputs = multiVarInputs[0]
            ext = multiVarInputs[1]

        res_conn_inputs = inputs

        D = inputs.shape[2]
        self.L = D

        inputs_tran = self.gap(inputs)
        if self.use_ext_in_att:
            # ext = ext.reshape(ext.shape[0], ext.shape[1] * ext.shape[-1], ext.shape[2], 1, 1)
            ext = self.ext_seq(ext)

            inputs_tran = torch.cat([inputs_tran, ext], dim=1)

        inputs_tran = self.merge_channel(inputs_tran)

        inputs_tran = inputs_tran.transpose(1, 2)
        inputs_tran = inputs_tran.reshape((inputs_tran.shape[0], inputs_tran.shape[1], -1))

        inputs = inputs.transpose(1, 2)
        original_shape = inputs.shape
        inputs_ = inputs.reshape((original_shape[0], original_shape[1], -1))

        Q = self.linear1(inputs_tran)  # (N, L_in, C_in*num_heads)
        K = self.linear2(inputs_tran)  # (N, L_in, C_in*num_heads)
        V = self.linear3(inputs_tran)  # (N, L_in, C_in*num_heads)

        Q_ = torch.cat(torch.chunk(Q, self.num_heads, 2), 0)  # (N*num_heads, L_in, C_in)
        K_ = torch.cat(torch.chunk(K, self.num_heads, 2), 0)  # (N*num_heads, L_in, C_in)
        V_ = torch.cat(torch.chunk(V, self.num_heads, 2), 0)  # (N*num_heads, L_in, C_in)

        outputs = self.attention1(Q_, K_, V_, self.causality)
        att_softmax = outputs

        outputs = torch.matmul(outputs, V_)
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, 0), 2)
        att_values = outputs
        outputs = self.linear_out(outputs)
        att_values_merge = outputs
        outputs = inputs_ * outputs

        outputs = outputs.reshape(original_shape)
        outputs = outputs.transpose(1, 2)
        return outputs, [att_softmax, att_values, att_values_merge]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, bottlenect_channel, kernel_size, stride, dilation,
                 padding, dropout=0.2, input_len=0, use_ext_in_att=False, layer_Num = 0):
        super(TemporalBlock, self).__init__()
        self.k_s = kernel_size
        self.use_ext_in_att = use_ext_in_att



        self.att = MultiHeadAttion(num_heads=4, C_in=n_inputs, windows_size=input_len,
                                   dropout_rate=dropout, causality=True, use_ext_in_att=self.use_ext_in_att,
                                   layer_Num=layer_Num)
        self.conv1 = nn.Conv3d(n_inputs, n_inputs, kernel_size=kernel_size,
                               padding=[0, padding[1], padding[2]], stride=[2, 1, 1], bias=False)
        self.bn1 = nn.BatchNorm3d(n_inputs)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(n_inputs, n_inputs, kernel_size=[1, kernel_size[1], kernel_size[2]],
                               padding=[0, padding[1], padding[2]], bias=False)
        self.bn2 = nn.BatchNorm3d(n_inputs)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv3d(n_inputs, n_outputs, kernel_size=[1, kernel_size[1], kernel_size[2]],
                               padding=[0, padding[1], padding[2]], bias=False)
        self.bn3 = nn.BatchNorm3d(n_outputs)

        self.relu3 = nn.LeakyReLU(inplace=True)

        self.net1 = nn.Sequential(self.att)
        self.net2 = nn.Sequential(self.conv1, self.bn1, self.relu1,
                                  self.conv2, self.bn2, self.relu2,
                                  self.conv3, self.bn3)

        self.downsample = nn.Sequential(nn.Conv3d(n_inputs, n_outputs, 1, bias=False),
                                        nn.BatchNorm3d(n_outputs)) if n_inputs != n_outputs else None

    def forward(self, inputs):
        if not self.use_ext_in_att:
            x = inputs
        else:
            x = inputs[0]
            ext = inputs[1]
        windows_size = x.shape[2]
        pad_count = np.ceil((windows_size - self.k_s[0]) / 2)
        pad_front = int(pad_count * 2 + self.k_s[0] - windows_size)
        pad_front = self.k_s[0] - windows_size if (pad_count < 0) else pad_front
        x = nn.ConstantPad3d((0, 0, 0, 0, pad_front, 0), 0)(x)

        if self.use_ext_in_att:
            ext = nn.ConstantPad3d((0, 0, 0, 0, pad_front, 0), 0)(ext)

        out = x
        out, att_values = self.net1(out) if not self.use_ext_in_att else self.net1([out, ext])
        # att_values = []
        out = self.net2(out)

        res = x if self.downsample is None else self.downsample(x)

        if __name__ == '__main__':
            print("TemporalBlock: in_D:{}\t;\tout_D:{};\t pad_fron:{}".format(res.shape[2], out.shape[2], pad_front))
        downsample_shape_D = [i * 2 + self.k_s[0] - 1 for i in range(out.shape[2])]

        if not self.use_ext_in_att:
            return self.relu3(out + res[:, :, downsample_shape_D, :, :]), att_values
        else:
            return self.relu3(out + res[:, :, downsample_shape_D, :, :]),\
                   torch.cat(torch.chunk(ext[:, :, -(len(downsample_shape_D)*2):, :, :], 2, 2), -1), att_values


class TemporalConvNet(nn.Module):
    def __init__(self, C_i, num_channels, bottlenect_channels, windows_size, kernel_size, dropout=0.2,
                 use_ext_in_att=False):
        super(TemporalConvNet, self).__init__()
        self.temporal_block = nn.ModuleList()

        self.num_levels = len(num_channels)
        self.skip_channel = num_channels[-1]
        self.windows_size = windows_size
        self.k_s = kernel_size
        self.use_ext_in_att = use_ext_in_att

        self.block_input_len, self.block_output_len = self.cal_block_output_len

        for i in range(self.num_levels):
            dilation_d = 2 ** i
            dilation_h_w = 1
            dilation_size = (dilation_d, dilation_h_w, dilation_h_w)

            padding_d = (kernel_size[0] - 1) * dilation_d
            padding_h_w = int((kernel_size[1] - 1) / 2)
            padding = (padding_d, padding_h_w, padding_h_w)

            in_channels = C_i if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            self.temporal_block.append(
                TemporalBlock(in_channels, out_channels, bottlenect_channels[i], kernel_size, stride=1,
                              dilation=dilation_size, padding=padding, dropout=dropout,
                              input_len=self.block_input_len[i], use_ext_in_att=self.use_ext_in_att, layer_Num=i))

        self.bn_out = nn.BatchNorm3d(self.skip_channel)
        self.relu_out = nn.LeakyReLU(inplace=True)

    @property
    def cal_block_output_len(self):
        every_block_input_len = []
        every_block_output_len = []
        windows_size = self.windows_size

        while (windows_size != 1):
            pad_count = np.ceil((windows_size - self.k_s[0]) / 2)
            pad_count = 0 if (pad_count < 0) else pad_count
            windows_size = int(pad_count * 2 + self.k_s[0])
            every_block_input_len.append(windows_size)
            windows_size = int((windows_size - self.k_s[0]) / 2)
            windows_size += 0 if (self.k_s[0] % 2 == 0) else 1
            every_block_output_len.append(windows_size)
        if __name__ == '__main__':
            print('every_block_input_len', every_block_input_len)
            print('every_block_output_len', every_block_output_len)
        return every_block_input_len, every_block_output_len

    def block(self, input):
        att_val_arr = []
        if not self.use_ext_in_att:
            x = input
            for i in range(self.num_levels):
                x, att_values = self.temporal_block[i](x)
                att_val_arr.append(att_values)
            return x, att_val_arr
        else:
            x = input[0]
            ext = input[1]
            for i in range(self.num_levels):
                x, ext, att_values = self.temporal_block[i]([x, ext])
                att_val_arr.append(att_values)
            return x, att_val_arr

    def forward(self, input):
        return self.block(input)


class ASTCN(nn.Module):
    def __init__(self, C_i, C_o, bottlenect_channels, num_channels, kernel_size, dropout, windows_size, use_ext_in_att):
        super(ASTCN, self).__init__()
        self.num_channels = num_channels
        self.skip_all_channel = num_channels[-1] * len(num_channels)
        self.C_o = C_o
        self.windows_size = windows_size
        self.use_ext_in_att = use_ext_in_att

        self.conv_input = nn.Sequential(nn.Conv3d(C_i, num_channels[0], [1, 7, 7], [1, 2, 2], [0, 3, 3], bias=False),
                                        nn.BatchNorm3d(num_channels[0]),
                                        nn.LeakyReLU(inplace=True))

        self.tcn = TemporalConvNet(num_channels[0], num_channels, bottlenect_channels, windows_size=windows_size,
                                   kernel_size=kernel_size, dropout=dropout, use_ext_in_att=self.use_ext_in_att)

        self.unconv = nn.ConvTranspose3d(num_channels[-1], num_channels[-1], [1, 2, 2], stride=[1, 2, 2])
        self.bn_unconv = nn.BatchNorm3d(num_channels[-1])
        self.relu_unconv = nn.LeakyReLU(inplace=True)

        self.convOut = nn.Sequential(nn.Conv3d(num_channels[-1], num_channels[-1], kernel_size=[1, 3, 3], padding=[0, 1, 1]),
                                     nn.BatchNorm3d(num_channels[-1]),
                                     nn.LeakyReLU(inplace=True),

                                     nn.Conv3d(num_channels[-1], num_channels[-1], kernel_size=[1, 3, 3], padding=[0, 1, 1]),
                                     nn.BatchNorm3d(num_channels[-1]),
                                     nn.LeakyReLU(inplace=True),

                                     nn.Conv3d(num_channels[-1], num_channels[-1], kernel_size=[1, 3, 3], padding=[0, 1, 1]),
                                     )
        
        self.convOut1 = nn.Sequential(nn.Tanh(), 
                                      nn.Conv3d(num_channels[-1], num_channels[-1], kernel_size=[1, 3, 3], padding=[0, 1, 1]),
                                    # nn.BatchNorm3d(num_channels[-1]),
                                      nn.Tanh(), 

                                     nn.Conv3d(num_channels[-1], num_channels[-1], kernel_size=[1, 3, 3], padding=[0, 1, 1]),
                                    #  nn.BatchNorm3d(12num_channels[-1]),
                                     nn.Tanh(), 

                                     nn.Conv3d(num_channels[-1], C_o, kernel_size=1),
                                    #  nn.Tanh()
                                     )
        
#         self.tanh_o = nn.Tanh()


    def forward(self, inputs):
        """Inputs have to have dimension (N, C_{in}, D, H, W)"""
        if self.use_ext_in_att == True:
            ext = inputs[1].transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
            
        # 减小H, W 的维度
        o = self.conv_input(inputs[0])

        o, att_val_arr = self.tcn(o) if not self.use_ext_in_att else self.tcn([o, ext])

        o = self.unconv(o)
        o = self.bn_unconv(o)
        o = self.relu_unconv(o)

        o = self.convOut(o)

        in_tensor = inputs[0][:, 0:1, -1:, :, :].repeat(1, int(self.num_channels[-1] / 2), 1, 1, 1)
        out_tensor = inputs[0][:, 1:2, -1:, :, :].repeat(1, int(self.num_channels[-1] / 2), 1, 1, 1)

        o = o + torch.cat([in_tensor, out_tensor], 1)
        
        o = self.convOut1(o)

        return o, att_val_arr


if __name__ == '__main__':
    import time
    from utils import OneHot, Binary, binary_tensor, STData
    begin_time = time.time()
    windows_size = 7 * 48
    batch_size = 2
    train_set = STData(dataset_type="train", windows_size=windows_size, dir_path='../../../data/processed/TaxiBJ/')
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
    itertrain = iter(train_loader)

    test_data, ext_data, test_y = itertrain.next()
    test_data.requires_grad = True
    ext_data.requires_grad = True

    tcn = ASTCN(2, 2, [8, 8, 16, 16, 32, 32, 64, 64], [16, 16, 32, 32, 64, 64, 128, 128],
                [3, 3, 3], 0.0, windows_size, use_ext_in_att=True)

    stcnn = tcn
    par = list(stcnn.parameters())
    import numpy as np

    s = sum([np.prod(list(d.size())) for d in par])
    print("Parameter of stcnn:", s)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(tcn.parameters(), lr=0.01)
    optimizer.zero_grad()
    test_output, att_val_arr = tcn([test_data, ext_data])
    att1 = att_val_arr[0]
    # att1_np = att1.detach().numpy()
    loss = criterion(test_output, test_y)
    loss.backward()

    print(test_output.shape)
    print('test, running time: {}'.format(time.time() - begin_time))