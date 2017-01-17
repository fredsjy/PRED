import pickle


def unpickle(file):
    fo = open(file, 'rb')
    # u = pickle._Unpickler(fo)
    # u.encoding = 'latin1'
    dict = pickle.load(fo, encoding = 'latin1')
    fo.close()
    return dict

file_1 = unpickle("data_batch_3")

print(file_1)#['data'].shape)#['data'][2].shape)


