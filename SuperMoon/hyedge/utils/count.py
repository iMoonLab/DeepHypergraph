def count_node(H, node_num=None):
    return H[0].max().item() + 1 if node_num is None else node_num


def count_hyedge(H, hyedge_num=None):
    return H[1].max().item() + 1 if hyedge_num is None else hyedge_num
