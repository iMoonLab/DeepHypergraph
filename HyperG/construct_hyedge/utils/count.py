def node_count(H):
    return H[0].max().item() + 1


def hyedge_count(H):
    return H[1].max().item() + 1
