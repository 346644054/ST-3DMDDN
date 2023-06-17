from ST3DNet import *
import pickle
from utils import *
import os
import math
from keras.utils import plot_model
from keras.optimizers import Adam
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import  time
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定只是用第三块GPU
CUDA_VISIBLE_DEVICES=0



import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
TensorBoardcallback = keras.callbacks.TensorBoard(
    log_dir='./LOGS'
)
# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # loss
        plt.plot(iters, self.losses[loss_type], 'blue', label='train loss')
        if loss_type == 'epoch':
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'red', label='val loss')
        plt.grid(False)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig('E://pic_xinst3dnet.svg')

    def final_loss(self, loss_type):
        return self.losses[loss_type], self.val_loss[loss_type]



start_time = time.time()


nb_epoch = 1500  # number of epoch at training stage
nb_epoch_cont = 200# number of epoch at training (cont) stage
batch_size = 16  # batch size
T = 48  # number of time intervals in one day
lr = 0.0002  # learning rate
len_closeness = 6  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence
nb_residual_unit = 7   # number of residual units
nb_flow = 2  # there are two types of flows: new-flow and end-flow
days_test = 7  # divide data into two subsets: Train & Test, of which the test set is the last 10 days  28
len_test = T * days_test
map_height, map_width = 20, 25  # grid size
# nb_area = 81
# m_factor = math.sqrt(1. * map_height * map_width / nb_area)
m_factor = 1


###60
# evaluating using the final model
# Train score: 504.704526  rmse (real): 22.345625 mae (real): 12.283250 mape (real): 0.282891
# Test score: 631.061401  rmse (real): 25.120935 mae (real): 13.480573 mape (real): 0.307114
# the whole process use 284.2135499238968 minutes


filename = os.path.join("data","BikeNYC", 'NYC_c%d_p%d_t%d_ext'%(len_closeness, len_period, len_trend))
f = open(filename, 'rb')
X_train = pickle.load(f)
Y_train = pickle.load(f)
X_test = pickle.load(f)
Y_test = pickle.load(f)
mmn = pickle.load(f)
external_dim = pickle.load(f)
timestamp_train = pickle.load(f)
timestamp_test = pickle.load(f)

for i in X_train:
    print(i.shape)

Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
Y_test = mmn.inverse_transform(Y_test)

c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None

t_conf = (len_trend, nb_flow, map_height,
          map_width) if len_trend > 0 else None
model = ST3DNet(c_conf=c_conf,p_conf=p_conf, t_conf=t_conf, external_dim=external_dim )

adam = Adam(lr=lr)
model.compile(loss='mse', optimizer=adam, metrics=[rmse,mae,mape])
model.summary()
plot_model(model, to_file='model.png',show_shapes=True)

from keras.callbacks import EarlyStopping, ModelCheckpoint
hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(
        len_closeness, len_period, len_trend, nb_residual_unit, lr)
fname_param = '{}.best.h5'.format(hyperparams_name)

early_stopping = EarlyStopping(monitor='val_rmse', patience=50, mode='min')
model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
historyplot = LossHistory()
print('=' * 10)
print("training model...")
history = model.fit(X_train, Y_train,
                    nb_epoch=300,
                    batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[early_stopping,historyplot,model_checkpoint,TensorBoardcallback],
                    verbose=2)
# historyplot.loss_plot('epoch')

model.save_weights('{}.h5'.format(hyperparams_name), overwrite=True)



print('=' * 10)
print('evaluating using the model that has the best loss on the valid set')
model.load_weights(fname_param)
score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
print('Train score: %.6f  rmse (real): %.6f' %(score[0], score[1] * m_factor))


score = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
print('Test score: %.6f  rmse (real): %.6f' %(score[0], score[1] * m_factor))

print('=' * 10)
print("training model (cont)...")
fname_param = os.path.join('MODEL', '{}.cont.best.h5'.format(hyperparams_name))
model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=2, save_best_only=True, mode='min')
history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=2, batch_size=batch_size, callbacks=[model_checkpoint,TensorBoardcallback], validation_data=(X_test, Y_test))
model.save_weights('{}_cont.h5'.format(hyperparams_name), overwrite=True)

print('=' * 10)
print('evaluating using the final model')
model.load_weights(fname_param)
score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
print('Train score: %.6f  rmse (real): %.6f mae (real): %.6f mape (real): %.6f' %
       (score[0], score[1] * m_factor ,score[2],score[3]))

score = model.evaluate(
    X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
print('Test score: %.6f  rmse (real): %.6f mae (real): %.6f mape (real): %.6f' %
       (score[0], score[1],score[2],score[3]))
end_time = time.time()
del_time = end_time - start_time
print('the whole process use {} minutes'.format(del_time / 60))

for iday in range(7):
#iday=0   #第一天
    pred=model.predict_on_batch(X_test)[iday*48:(iday+1)*48]
    groundtruth=Y_test[iday*48:(iday+1)*48]
    print(pred.shape);
    print("::::");
    print(groundtruth.shape);
    np.save('pred_'+str(iday)+'.npy',pred)
    np.save('groundtruth_'+str(iday)+'.npy',groundtruth)
