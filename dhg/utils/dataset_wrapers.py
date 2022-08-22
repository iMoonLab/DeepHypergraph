import random
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset

from .structure import edge_list_to_adj_dict


class UserItemDataset(Dataset):
    r"""The dataset class of user-item bipartite graph for recommendation task.
    
    Args:
        ``num_users`` (``int``): The number of users.
        ``num_items`` (``int``): The number of items.
        ``user_item_list`` (``List[Tuple[int, int]]``): The list of user-item pairs.
        ``train_user_item_list`` (``List[Tuple[int, int]]``, optional): The list of user-item pairs for training. This is only needed for testing to mask those seen items in training. Defaults to ``None``.
        ``strict_link`` (``bool``): Whether to iterate through all interactions in the dataset. If set to ``False``, in training phase the dataset will keep randomly sampling interactions until meeting the same number of original interactions. Defaults to ``True``.
        ``phase`` (``str``): The phase of the dataset can be either ``"train"`` or ``"test"``. Defaults to ``"train"``.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_item_list: List[Tuple[int, int]],
        train_user_item_list: Optional[List[Tuple[int, int]]] = None,
        strict_link: bool = True,
        phase: str = "train",
    ):

        assert phase in ["train", "test"]
        self.phase = phase
        self.num_users, self.num_items = num_users, num_items
        self.user_item_list = user_item_list
        self.adj_dict = edge_list_to_adj_dict(user_item_list)
        self.strict_link = strict_link
        if phase != "train":
            assert (
                train_user_item_list is not None
            ), "train_user_item_list is needed for testing."
            self.train_adj_dict = edge_list_to_adj_dict(train_user_item_list)

    def sample_triplet(self):
        r"""Sample a triple of user, positive item, and negtive item from all interactions.
        """
        user = random.randrange(self.num_users)
        assert len(self.adj_dict[user]) > 0
        pos_item = random.choice(self.adj_dict[user])
        neg_item = self.sample_neg_item(user)
        return user, pos_item, neg_item

    def sample_neg_item(self, user: int):
        r"""Sample a negative item for the sepcified user.

        Args:
            ``user`` (``int``): The index of the specified user.
        """
        neg_item = random.randrange(self.num_items)
        while neg_item in self.adj_dict[user]:
            neg_item = random.randrange(self.num_items)
        return neg_item

    def __getitem__(self, index):
        r"""Return the item at the index. If the phase is ``"train"``, return the (``User``-``PositiveItem``-``NegativeItem``) triplet. If the phase is ``"test"``, return all true positive items for each user.
        
        Args:
            ``index`` (``int``): The index of the item.
        """
        if self.phase == "train":
            if self.strict_link:
                user, pos_item = self.user_item_list[index]
                neg_item = self.sample_neg_item(user)
            else:
                user, pos_item, neg_item = self.sample_triplet()
            return user, pos_item, neg_item
        else:
            train_mask, true_rating = (
                torch.zeros(self.num_items),
                torch.zeros(self.num_items),
            )
            train_items, true_items = self.train_adj_dict[index], self.adj_dict[index]
            train_mask[train_items] = float("-inf")
            true_rating[true_items] = 1.0
            return index, train_mask, true_rating

    def __len__(self):
        r"""Return the length of the dataset. If the phase is ``"train"``, return the number of interactions. If the phase is ``"test"``, return the number of users.
        """
        if self.phase == "train":
            return len(self.user_item_list)
        else:
            return self.num_users
