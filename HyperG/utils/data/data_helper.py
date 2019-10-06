from random import shuffle


def split_id(id_list, ratio):
    train_len = int(len(id_list) * ratio)
    shuffle(id_list)

    id_train = id_list[:train_len]
    id_val = id_list[train_len:]
    return id_train, id_val