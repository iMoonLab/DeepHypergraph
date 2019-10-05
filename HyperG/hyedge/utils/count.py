def count_node(H):
    return H[0].max().item() + 1


def count_hyedge(H):
    return H[1].max().item() + 1
