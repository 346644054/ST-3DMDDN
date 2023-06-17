from utils import *
from copy import copy

def load_holiday(timeslots, fname):
    f = open(os.path.join(fname, "BJ_Holiday.txt"), 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    print(H.sum())
    # print(timeslots[H==1])
    return H[:, None]


def load_meteorol(timeslots, fname):
    f = h5py.File(os.path.join(fname, 'BJ_Meteorology.h5'), 'r')
    Timeslot = f['date'].value
    WindSpeed = f['WindSpeed'].value
    Weather = f['Weather'].value
    Temperature = f['Temperature'].value
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    print("shape: ", WS.shape, WR.shape, TE.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    # print('meger shape:', merge_data.shape)
    return merge_data


def create_mask(city, city_dict):
    if city == 'NY':
        shape = (32, 32)
    else:
        shape = (32, 32)
    sum_inflow = np.zeros(shape = shape)
    sum_outflow = np.zeros(shape = shape)
    for i in city_dict.keys():
        if 'Inflow' in i:
            sum_inflow += city_dict[i]
        elif 'Outflow' in i:
            sum_outflow += city_dict[i]
    sum_outflow = np.array([0 if x == 0 else 1 for x in sum_outflow.flatten()]).reshape(shape)
    sum_inflow = np.array([0 if x == 0 else 1 for x in sum_inflow.flatten()]).reshape(shape)

    return np.array([sum_outflow, sum_inflow])


def create_dict(data, timestamps):

    # Function that creates a dictionary with inflow (_End) or outflow (_Start) matrix for each timestamp.

    ny_dict = {}
    for index in range(len(data)):
        ny_dict[str(timestamps[index]) + '_Inflow'] = data[index][0].tolist()
        ny_dict[str(timestamps[index]) + '_Outflow'] = data[index][1].tolist()
    return ny_dict




def load_data_BJ(T=48, nb_flow=2, len_closeness=None, len_period=None, len_trend=None, len_test=None, meta_data=True, holiday_data=True, meteorol_data=True):
    assert (len_closeness + len_period + len_trend > 0)
    dir = os.getcwd()
    # load data
    # 13 - 16
    data_all = []
    timestamps_all = list()
    # queshizhi
    # fill_missed_month = [["201311", "201312", "201401", "201402"],
    #                      ["201407", "201408", "201409", "201410", "201411", "201412", "201501", "201502"],
    #                      ["201507", "201508", "201509", "201510"],
    #                      ]
    #
    # flag = 0
    # for year in range(13, 17):
    #     fname = os.path.join(dir, 'data', 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
    #     print("file name: ", fname)
    #     data, timestamps = load_stdata(fname)
    #     # print(timestamps)
    #     # remove a certain day which does not have 48 timestamps
    #     data, timestamps = remove_incomplete_days(data, timestamps, T)
    #     data = data[:, :nb_flow]
    #     data[data < 0] = 0.
    #     data_all.append(data)
    #     timestamps_all.append(timestamps)
    #     print("\n")
    #     if flag != 0:
    #         month_count = 0
    #         for month in fill_missed_month[flag - 1]:
    #             _filled_timestamps = []
    #             for j in range(1, 32):
    #                 if "09" in month or "11" in month or "04" in month:
    #                     if j == 31:
    #                         continue
    #                 if "02" in month:
    #                     if j == 29:
    #                         continue
    #
    #                 for i in range(1, 49):
    #                     _filled_timestamps.append(str(str(month) + str("%02d" % j) + str("%02d" % i)).encode("utf-8"))
    #
    #             first_data_month = timestamps_all[flag - 1][-1][0:-4]
    #             first_data_index_i, first_data_index_j = -1, -1
    #             flag_i = False
    #             for index in range(len(timestamps_all[flag - 1])):
    #                 if first_data_month.decode("utf-8") in timestamps_all[flag - 1][index].decode("utf-8"):
    #                     if flag_i == False:
    #                         first_data_index_i = index
    #                         flag_i = True
    #                         # print("first_data_index_i:", first_data_index_i)
    #                 else:
    #                     if flag_i == True:
    #                         first_data_index_j = index
    #                         # print("first_data_index_j:", first_data_index_j)
    #                         break
    #             if first_data_index_j == -1:
    #                 first_data_index_j = len(timestamps_all[flag - 1])
    #             first_data_index_len = first_data_index_j - first_data_index_i
    #
    #             second_data_month = timestamps_all[flag][month_count][0:-4]
    #             second_data_index_i, second_data_index_j = -1, -1
    #             flag_j = False
    #             for index in range(len(timestamps_all[flag])):
    #                 if second_data_month.decode("utf-8") in timestamps_all[flag][index].decode("utf-8"):
    #                     if flag_j == False:
    #                         second_data_index_i = index
    #                         flag_j = True
    #                 else:
    #                     if flag_j == True:
    #                         second_data_index_j = index
    #                         break
    #             if second_data_index_j == -1:
    #                 second_data_index_j = len(timestamps_all[flag])
    #             second_data_index_len = second_data_index_j - second_data_index_i
    #
    #             month_count += 1
    #
    #             data_index_len = first_data_index_len if second_data_index_len > first_data_index_len else second_data_index_len
    #             data_index_len = len(_filled_timestamps) if data_index_len > len(_filled_timestamps) else data_index_len
    #             _filled_timestamps = _filled_timestamps[0:data_index_len]
    #
    #             _filled_data = (data_all[flag - 1][first_data_index_i: first_data_index_i + data_index_len]
    #                             + data_all[flag][second_data_index_i: second_data_index_i + data_index_len]) // 2
    #
    #             data_all[flag - 1] = np.concatenate([data_all[flag - 1], _filled_data])
    #             timestamps_all[flag - 1] += _filled_timestamps
    #
    #     flag += 1

    for year in range(13, 17):
        fname = os.path.join(dir, 'data','TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("file name: ", fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        # create mask
        ny_dict = create_dict(data, timestamps)
        mask = create_mask('NY', ny_dict)

        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    # minmax_scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is
        # a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        # _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(
        #     len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        # print("create dataset gsn")
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset_3D(len_closeness=len_closeness, len_period=len_period,
                                                                 len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
    if holiday_data:
        # load holiday
        holiday_feature = load_holiday(timestamps_Y, os.path.join(dir, 'data','TaxiBJ'))
        meta_feature.append(holiday_feature)
    if meteorol_data:
        # load meteorol data
        meteorol_feature = load_meteorol(timestamps_Y, os.path.join(dir, 'data','TaxiBJ'))
        meta_feature.append(meteorol_feature)

    meta_feature = np.hstack(meta_feature) if len(
        meta_feature) > 0 else np.asarray(meta_feature)
    metadata_dim = meta_feature.shape[1] if len(
        meta_feature.shape) > 1 else None
    # if metadata_dim < 1:
    #     metadata_dim = None
    if meta_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'mete feature: ', meta_feature.shape)

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[
                                            :-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[
                                        -len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[
                                      :-len_test], timestamps_Y[-len_test:]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)

    if metadata_dim is not None:
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)

    print('X train shape:')
    for _X in X_train:
        print(_X.shape, )
    print()

    print('X test shape:')
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test,mask


T = 48  # number of time intervals in one day
len_closeness = 4  # length of closeness dependent sequence
len_period = 2  # length of peroid dependent sequence
len_trend = 0 # length of trend dependent sequence
nb_flow = 2  # there are two types of flows: new-flow and end-flow
days_test = 7  # last 4 weeks
len_test = T * days_test
map_height, map_width = 32, 32  # grid size
consider_external_info = True


if consider_external_info:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, mask = \
        load_data_BJ(T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend,
                     len_test=len_test, meta_data=consider_external_info, holiday_data=consider_external_info, meteorol_data=consider_external_info)

    dir = os.getcwd()
    filename = os.path.join(dir, 'data', 'TaxiBJ', 'TaxiBJ_c%d_p%d_t%d_ext' % (len_closeness, len_period, len_trend))
    print('filename:', filename)
    f = open(filename, 'wb')
    pickle.dump(X_train, f)
    pickle.dump(Y_train, f)
    pickle.dump(X_test, f)
    pickle.dump(Y_test, f)
    pickle.dump(mmn, f)
    pickle.dump(external_dim, f)
    pickle.dump(timestamp_train, f)
    pickle.dump(timestamp_test, f)
    pickle.dump(mask, f)
    f.close()

else:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test , mask= \
            load_data_BJ(T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test, meta_data=consider_external_info, holiday_data=consider_external_info, meteorol_data=consider_external_info)

    dir = os.getcwd()
    filename = os.path.join(dir, 'data', 'TaxiBJ','TaxiBJ_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))
    print('filename:', filename)
    f = open(filename, 'wb')
    pickle.dump(X_train, f)
    pickle.dump(Y_train, f)
    pickle.dump(X_test, f)
    pickle.dump(Y_test, f)
    pickle.dump(mmn, f)
    pickle.dump(external_dim, f)
    pickle.dump(timestamp_train, f)
    pickle.dump(timestamp_test, f)
    f.close()