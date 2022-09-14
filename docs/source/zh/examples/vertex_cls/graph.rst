图
==========================================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

在如下的例子中，我们使用四种典型图/超图神经网络在图关联结构中执行节点分类任务。

模型
---------------------------

- GCN (:py:class:`dhg.models.GCN`), `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ 论文 (ICLR 2017).
- GAT (:py:class:`dhg.models.GAT`), `Graph Attention Networks <https://arxiv.org/pdf/1710.10903>`_ 论文 (ICLR 2018).
- HGNN (:py:class:`dhg.models.HGNN`), `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ 论文 (AAAI 2019).
- HGNN+ (:py:class:`dhg.models.HGNNP`), `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ 论文 (IEEE T-PAMI 2022).

数据集
---------------------------

Cora数据集 (:py:class:`dhg.data.Cora`) 为节点分类任务中使用的引用网络数据集。
更多细节可以参考此 `网页 <https://relational.fit.cvut.cz/dataset/CORA>`_.

结果汇总
----------------

========    ======================  ======================  ======================
模型         验证集的Accuracy         测试集的Accuracy          测试集的F1 score
========    ======================  ======================  ======================
GCN         0.800                   0.823                   0.814
GAT         0.804                   0.824                   0.817
HGNN        0.804                   0.820                   0.811
HGNN+       0.802                   0.827                   0.820
========    ======================  ======================  ======================


Cora上使用GCN
----------------

导入依赖包
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from copy import deepcopy

    import torch
    import torch.optim as optim
    import torch.nn.functional as F

    from dhg import Graph
    from dhg.data import Cora
    from dhg.models import GCN
    from dhg.random import set_seed
    from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator

定义函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def train(net, X, A, lbls, train_idx, optimizer, epoch):
        net.train()

        st = time.time()
        optimizer.zero_grad()
        outs = net(X, A)
        outs, lbls = outs[train_idx], lbls[train_idx]
        loss = F.cross_entropy(outs, lbls)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
        return loss.item()


    @torch.no_grad()
    def infer(net, X, A, lbls, idx, test=False):
        net.eval()
        outs = net(X, A)
        outs, lbls = outs[idx], lbls[idx]
        if not test:
            res = evaluator.validate(lbls, outs)
        else:
            res = evaluator.test(lbls, outs)
        return res

主函数
^^^^^^^

.. note:: 

    更多关于评测器 ``Evaluator`` 的细节可以参照 :doc:`构建指标评测器 </zh/tutorial/metric>` 章节。

.. code-block:: python

    if __name__ == "__main__":
        set_seed(2022)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
        data = Cora()
        X, lbl = data["features"], data["labels"]
        G = Graph(data["num_vertices"], data["edge_list"])
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]

        net = GCN(data["dim_features"], 16, data["num_classes"])
        optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

        X, lbl = X.to(device), lbl.to(device)
        G = G.to(device)
        net = net.to(device)

        best_state = None
        best_epoch, best_val = 0, 0
        for epoch in range(300):
            # train
            train(net, X, G, lbl, train_mask, optimizer, epoch)
            # validation
            if epoch % 1 == 0:
                with torch.no_grad():
                    val_res = infer(net, X, G, lbl, val_mask)
                if val_res > best_val:
                    print(f"update best: {val_res:.5f}")
                    best_epoch = epoch
                    best_val = val_res
                    best_state = deepcopy(net.state_dict())
        print("\ntrain finished!")
        print(f"best val: {best_val:.5f}")
        # test
        print("test...")
        net.load_state_dict(best_state)
        res = infer(net, X, G, lbl, test_mask, test=True)
        print(f"final result: epoch: {best_epoch}")
        print(res)

.. only:: not latex

    输出
    ^^^^^^^^^^^^
    .. code-block:: 

        Epoch: 0, Time: 0.51907s, Loss: 1.95010
        update best: 0.31600
        Epoch: 1, Time: 0.00182s, Loss: 1.94601
        Epoch: 2, Time: 0.00164s, Loss: 1.94383
        Epoch: 3, Time: 0.00167s, Loss: 1.93907
        Epoch: 4, Time: 0.00164s, Loss: 1.93350
        update best: 0.38000
        Epoch: 5, Time: 0.00166s, Loss: 1.92899
        Epoch: 6, Time: 0.00323s, Loss: 1.92461
        Epoch: 7, Time: 0.00164s, Loss: 1.91764
        Epoch: 8, Time: 0.00341s, Loss: 1.91163
        Epoch: 9, Time: 0.00167s, Loss: 1.90064
        Epoch: 10, Time: 0.00154s, Loss: 1.89617
        Epoch: 11, Time: 0.00159s, Loss: 1.88721
        Epoch: 12, Time: 0.00223s, Loss: 1.87626
        Epoch: 13, Time: 0.00178s, Loss: 1.86941
        Epoch: 14, Time: 0.00167s, Loss: 1.86202
        Epoch: 15, Time: 0.00316s, Loss: 1.85207
        Epoch: 16, Time: 0.00215s, Loss: 1.84285
        Epoch: 17, Time: 0.00289s, Loss: 1.83392
        Epoch: 18, Time: 0.00203s, Loss: 1.82120
        Epoch: 19, Time: 0.00202s, Loss: 1.80663
        Epoch: 20, Time: 0.00246s, Loss: 1.79340
        Epoch: 21, Time: 0.00201s, Loss: 1.77829
        Epoch: 22, Time: 0.00203s, Loss: 1.76851
        update best: 0.38800
        Epoch: 23, Time: 0.00162s, Loss: 1.75592
        update best: 0.40200
        Epoch: 24, Time: 0.00159s, Loss: 1.74545
        update best: 0.43000
        Epoch: 25, Time: 0.00175s, Loss: 1.72373
        update best: 0.45000
        Epoch: 26, Time: 0.00157s, Loss: 1.71025
        update best: 0.46000
        Epoch: 27, Time: 0.00164s, Loss: 1.68904
        update best: 0.46400
        Epoch: 28, Time: 0.00211s, Loss: 1.67401
        update best: 0.46600
        Epoch: 29, Time: 0.00168s, Loss: 1.67025
        update best: 0.48400
        Epoch: 30, Time: 0.00176s, Loss: 1.65349
        update best: 0.49200
        Epoch: 31, Time: 0.00250s, Loss: 1.61911
        update best: 0.49800
        Epoch: 32, Time: 0.00177s, Loss: 1.61325
        update best: 0.51400
        Epoch: 33, Time: 0.00192s, Loss: 1.56832
        update best: 0.52600
        Epoch: 34, Time: 0.00173s, Loss: 1.55827
        update best: 0.55000
        Epoch: 35, Time: 0.00172s, Loss: 1.55186
        update best: 0.56200
        Epoch: 36, Time: 0.00183s, Loss: 1.53794
        update best: 0.57400
        Epoch: 37, Time: 0.00222s, Loss: 1.50345
        update best: 0.58600
        Epoch: 38, Time: 0.00169s, Loss: 1.49760
        update best: 0.59600
        Epoch: 39, Time: 0.00164s, Loss: 1.47143
        update best: 0.60200
        Epoch: 40, Time: 0.00171s, Loss: 1.43501
        update best: 0.62800
        Epoch: 41, Time: 0.00170s, Loss: 1.42085
        update best: 0.64800
        Epoch: 42, Time: 0.00360s, Loss: 1.38769
        update best: 0.65400
        Epoch: 43, Time: 0.00156s, Loss: 1.36689
        update best: 0.66200
        Epoch: 44, Time: 0.00152s, Loss: 1.36428
        update best: 0.66800
        Epoch: 45, Time: 0.00167s, Loss: 1.32395
        Epoch: 46, Time: 0.00153s, Loss: 1.29274
        update best: 0.67600
        Epoch: 47, Time: 0.00164s, Loss: 1.30380
        Epoch: 48, Time: 0.00439s, Loss: 1.26099
        update best: 0.68800
        Epoch: 49, Time: 0.00186s, Loss: 1.25379
        Epoch: 50, Time: 0.00175s, Loss: 1.23854
        update best: 0.69800
        Epoch: 51, Time: 0.00171s, Loss: 1.20378
        update best: 0.72200
        Epoch: 52, Time: 0.00170s, Loss: 1.16979
        update best: 0.73200
        Epoch: 53, Time: 0.00326s, Loss: 1.15275
        update best: 0.74800
        Epoch: 54, Time: 0.00183s, Loss: 1.11128
        update best: 0.75200
        Epoch: 55, Time: 0.00183s, Loss: 1.12654
        update best: 0.75600
        Epoch: 56, Time: 0.00172s, Loss: 1.12641
        update best: 0.76400
        Epoch: 57, Time: 0.00171s, Loss: 1.08093
        update best: 0.76600
        Epoch: 58, Time: 0.00228s, Loss: 1.06145
        Epoch: 59, Time: 0.00163s, Loss: 1.03330
        Epoch: 60, Time: 0.00240s, Loss: 1.02479
        Epoch: 61, Time: 0.00179s, Loss: 1.01496
        Epoch: 62, Time: 0.00187s, Loss: 0.93007
        Epoch: 63, Time: 0.00176s, Loss: 0.97366
        Epoch: 64, Time: 0.00296s, Loss: 0.92534
        Epoch: 65, Time: 0.00230s, Loss: 0.91500
        update best: 0.77400
        Epoch: 66, Time: 0.00169s, Loss: 0.93400
        update best: 0.77800
        Epoch: 67, Time: 0.00161s, Loss: 0.86869
        update best: 0.78000
        Epoch: 68, Time: 0.00162s, Loss: 0.89109
        Epoch: 69, Time: 0.00177s, Loss: 0.89371
        Epoch: 70, Time: 0.00259s, Loss: 0.87362
        update best: 0.78200
        Epoch: 71, Time: 0.00159s, Loss: 0.80287
        Epoch: 72, Time: 0.00155s, Loss: 0.88049
        Epoch: 73, Time: 0.00160s, Loss: 0.78692
        Epoch: 74, Time: 0.00163s, Loss: 0.79204
        Epoch: 75, Time: 0.00152s, Loss: 0.81149
        update best: 0.78400
        Epoch: 76, Time: 0.00288s, Loss: 0.79278
        Epoch: 77, Time: 0.00183s, Loss: 0.75974
        update best: 0.78600
        Epoch: 78, Time: 0.00155s, Loss: 0.74237
        Epoch: 79, Time: 0.00162s, Loss: 0.72129
        update best: 0.78800
        Epoch: 80, Time: 0.00154s, Loss: 0.72252
        update best: 0.79000
        Epoch: 81, Time: 0.00170s, Loss: 0.69306
        update best: 0.79200
        Epoch: 82, Time: 0.00274s, Loss: 0.64976
        Epoch: 83, Time: 0.00157s, Loss: 0.66782
        Epoch: 84, Time: 0.00155s, Loss: 0.68008
        Epoch: 85, Time: 0.00160s, Loss: 0.70714
        Epoch: 86, Time: 0.00164s, Loss: 0.64139
        Epoch: 87, Time: 0.00159s, Loss: 0.66335
        Epoch: 88, Time: 0.00223s, Loss: 0.65881
        Epoch: 89, Time: 0.00248s, Loss: 0.65215
        Epoch: 90, Time: 0.00151s, Loss: 0.57064
        Epoch: 91, Time: 0.00155s, Loss: 0.64725
        Epoch: 92, Time: 0.00157s, Loss: 0.58507
        Epoch: 93, Time: 0.00174s, Loss: 0.62494
        Epoch: 94, Time: 0.00158s, Loss: 0.58289
        Epoch: 95, Time: 0.00157s, Loss: 0.56591
        Epoch: 96, Time: 0.00289s, Loss: 0.59959
        Epoch: 97, Time: 0.00157s, Loss: 0.62588
        Epoch: 98, Time: 0.00154s, Loss: 0.58035
        Epoch: 99, Time: 0.00156s, Loss: 0.58727
        Epoch: 100, Time: 0.00158s, Loss: 0.56111
        Epoch: 101, Time: 0.00152s, Loss: 0.54035
        Epoch: 102, Time: 0.00151s, Loss: 0.56815
        Epoch: 103, Time: 0.00233s, Loss: 0.50579
        Epoch: 104, Time: 0.00150s, Loss: 0.53285
        Epoch: 105, Time: 0.00147s, Loss: 0.56204
        Epoch: 106, Time: 0.00153s, Loss: 0.51602
        Epoch: 107, Time: 0.00160s, Loss: 0.52320
        Epoch: 108, Time: 0.00150s, Loss: 0.53845
        Epoch: 109, Time: 0.00151s, Loss: 0.55428
        Epoch: 110, Time: 0.00307s, Loss: 0.52966
        Epoch: 111, Time: 0.00150s, Loss: 0.56845
        Epoch: 112, Time: 0.00148s, Loss: 0.52385
        update best: 0.79400
        Epoch: 113, Time: 0.00155s, Loss: 0.52051
        Epoch: 114, Time: 0.00178s, Loss: 0.51860
        Epoch: 115, Time: 0.00159s, Loss: 0.48878
        Epoch: 116, Time: 0.00375s, Loss: 0.50367
        Epoch: 117, Time: 0.00160s, Loss: 0.49782
        Epoch: 118, Time: 0.00153s, Loss: 0.51155
        Epoch: 119, Time: 0.00153s, Loss: 0.47739
        Epoch: 120, Time: 0.00178s, Loss: 0.50645
        Epoch: 121, Time: 0.00157s, Loss: 0.49175
        Epoch: 122, Time: 0.00157s, Loss: 0.47638
        Epoch: 123, Time: 0.00345s, Loss: 0.46064
        Epoch: 124, Time: 0.00159s, Loss: 0.44845
        Epoch: 125, Time: 0.00153s, Loss: 0.44286
        Epoch: 126, Time: 0.00151s, Loss: 0.46044
        Epoch: 127, Time: 0.00156s, Loss: 0.45707
        Epoch: 128, Time: 0.00177s, Loss: 0.50700
        Epoch: 129, Time: 0.00153s, Loss: 0.46442
        Epoch: 130, Time: 0.00345s, Loss: 0.44911
        Epoch: 131, Time: 0.00153s, Loss: 0.46168
        Epoch: 132, Time: 0.00153s, Loss: 0.47634
        Epoch: 133, Time: 0.00152s, Loss: 0.41177
        Epoch: 134, Time: 0.00162s, Loss: 0.42612
        Epoch: 135, Time: 0.00160s, Loss: 0.46436
        Epoch: 136, Time: 0.00153s, Loss: 0.42374
        Epoch: 137, Time: 0.00380s, Loss: 0.42290
        Epoch: 138, Time: 0.00181s, Loss: 0.43096
        Epoch: 139, Time: 0.00166s, Loss: 0.43386
        Epoch: 140, Time: 0.00170s, Loss: 0.47472
        Epoch: 141, Time: 0.00175s, Loss: 0.40687
        Epoch: 142, Time: 0.00170s, Loss: 0.43927
        Epoch: 143, Time: 0.00347s, Loss: 0.39323
        Epoch: 144, Time: 0.00174s, Loss: 0.42356
        Epoch: 145, Time: 0.00168s, Loss: 0.44625
        Epoch: 146, Time: 0.00165s, Loss: 0.38619
        Epoch: 147, Time: 0.00171s, Loss: 0.40754
        Epoch: 148, Time: 0.00169s, Loss: 0.38543
        Epoch: 149, Time: 0.00166s, Loss: 0.39466
        Epoch: 150, Time: 0.00280s, Loss: 0.43009
        Epoch: 151, Time: 0.00165s, Loss: 0.38695
        Epoch: 152, Time: 0.00166s, Loss: 0.41950
        Epoch: 153, Time: 0.00166s, Loss: 0.41095
        Epoch: 154, Time: 0.00174s, Loss: 0.40313
        Epoch: 155, Time: 0.00167s, Loss: 0.43876
        Epoch: 156, Time: 0.00384s, Loss: 0.40152
        Epoch: 157, Time: 0.00170s, Loss: 0.39797
        update best: 0.80000
        Epoch: 158, Time: 0.00165s, Loss: 0.35990
        Epoch: 159, Time: 0.00168s, Loss: 0.40668
        Epoch: 160, Time: 0.00161s, Loss: 0.39737
        Epoch: 161, Time: 0.00153s, Loss: 0.42709
        Epoch: 162, Time: 0.00174s, Loss: 0.40306
        Epoch: 163, Time: 0.00262s, Loss: 0.44195
        Epoch: 164, Time: 0.00150s, Loss: 0.35434
        Epoch: 165, Time: 0.00154s, Loss: 0.39269
        Epoch: 166, Time: 0.00159s, Loss: 0.32633
        Epoch: 167, Time: 0.00154s, Loss: 0.38579
        Epoch: 168, Time: 0.00155s, Loss: 0.38941
        Epoch: 169, Time: 0.00150s, Loss: 0.38425
        Epoch: 170, Time: 0.00250s, Loss: 0.39287
        Epoch: 171, Time: 0.00153s, Loss: 0.36239
        Epoch: 172, Time: 0.00153s, Loss: 0.37962
        Epoch: 173, Time: 0.00154s, Loss: 0.35394
        Epoch: 174, Time: 0.00159s, Loss: 0.34589
        Epoch: 175, Time: 0.00161s, Loss: 0.38056
        Epoch: 176, Time: 0.00156s, Loss: 0.37199
        Epoch: 177, Time: 0.00309s, Loss: 0.36108
        Epoch: 178, Time: 0.00181s, Loss: 0.37211
        Epoch: 179, Time: 0.00153s, Loss: 0.35234
        Epoch: 180, Time: 0.00155s, Loss: 0.33577
        Epoch: 181, Time: 0.00153s, Loss: 0.37541
        Epoch: 182, Time: 0.00156s, Loss: 0.30629
        Epoch: 183, Time: 0.00149s, Loss: 0.36643
        Epoch: 184, Time: 0.00346s, Loss: 0.34131
        Epoch: 185, Time: 0.00153s, Loss: 0.35421
        Epoch: 186, Time: 0.00146s, Loss: 0.33999
        Epoch: 187, Time: 0.00149s, Loss: 0.36365
        Epoch: 188, Time: 0.00152s, Loss: 0.36926
        Epoch: 189, Time: 0.00152s, Loss: 0.31029
        Epoch: 190, Time: 0.00155s, Loss: 0.32959
        Epoch: 191, Time: 0.00247s, Loss: 0.35637
        Epoch: 192, Time: 0.00208s, Loss: 0.30936
        Epoch: 193, Time: 0.00154s, Loss: 0.32842
        Epoch: 194, Time: 0.00154s, Loss: 0.31046
        Epoch: 195, Time: 0.00156s, Loss: 0.34217
        Epoch: 196, Time: 0.00169s, Loss: 0.35384
        Epoch: 197, Time: 0.00157s, Loss: 0.31096
        Epoch: 198, Time: 0.00307s, Loss: 0.31790
        Epoch: 199, Time: 0.00160s, Loss: 0.29574

        train finished!
        best val: 0.80000
        test...
        final result: epoch: 157
        {'accuracy': 0.8230000138282776, 'f1_score': 0.8135442845966843, 'f1_score -> average@micro': 0.823}

Cora上使用GAT
----------------

导入依赖包
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from copy import deepcopy

    import torch
    import torch.optim as optim
    import torch.nn.functional as F

    from dhg import Graph
    from dhg.data import Cora
    from dhg.models import GAT
    from dhg.random import set_seed
    from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator


定义函数
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def train(net, X, A, lbls, train_idx, optimizer, epoch):
        net.train()

        st = time.time()
        optimizer.zero_grad()
        outs = net(X, A)
        outs, lbls = outs[train_idx], lbls[train_idx]
        loss = F.cross_entropy(outs, lbls)
        # loss = F.nll_loss(outs, lbls)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
        return loss.item()


    @torch.no_grad()
    def infer(net, X, A, lbls, idx, test=False):
        net.eval()
        outs = net(X, A)
        outs, lbls = outs[idx], lbls[idx]
        if not test:
            res = evaluator.validate(lbls, outs)
        else:
            res = evaluator.test(lbls, outs)
        return res

主函数
^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: 

    更多关于评测器 ``Evaluator`` 的细节可以参照 :doc:`构建指标评测器 </zh/tutorial/metric>` 章节。

.. code-block:: python

    if __name__ == "__main__":
        set_seed(2022)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
        data = Cora()
        X, lbl = data["features"], data["labels"]
        G = Graph(data["num_vertices"], data["edge_list"])
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]

        net = GAT(data["dim_features"], 8, data["num_classes"], num_heads=8, drop_rate=0.6)
        optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=5e-4)

        X, lbl = X.cuda(), lbl.cuda()
        G = G.to(device)
        net = net.cuda()

        best_state = None
        best_epoch, best_val = 0, 0
        for epoch in range(300):
            # train
            train(net, X, G, lbl, train_mask, optimizer, epoch)
            # validation
            if epoch % 1 == 0:
                with torch.no_grad():
                    val_res = infer(net, X, G, lbl, val_mask)
                if val_res > best_val:
                    print(f"update best: {val_res:.5f}")
                    best_epoch = epoch
                    best_val = val_res
                    best_state = deepcopy(net.state_dict())
        print("\ntrain finished!")
        print(f"best val: {best_val:.5f}")
        # test
        print("test...")
        net.load_state_dict(best_state)
        res = infer(net, X, G, lbl, test_mask, test=True)
        print(f"final result: epoch: {best_epoch}")
        print(res)

.. only:: not latex

    输出
    ^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code-block:: 

        Epoch: 0, Time: 0.56263s, Loss: 1.94867
        update best: 0.12200
        Epoch: 1, Time: 0.03209s, Loss: 1.94319
        Epoch: 2, Time: 0.03641s, Loss: 1.94076
        Epoch: 3, Time: 0.03197s, Loss: 1.93497
        Epoch: 4, Time: 0.03536s, Loss: 1.92976
        Epoch: 5, Time: 0.03239s, Loss: 1.92725
        update best: 0.18200
        Epoch: 6, Time: 0.03332s, Loss: 1.91903
        update best: 0.38200
        Epoch: 7, Time: 0.03125s, Loss: 1.91507
        update best: 0.49200
        Epoch: 8, Time: 0.02778s, Loss: 1.91092
        update best: 0.50400
        Epoch: 9, Time: 0.03188s, Loss: 1.90407
        update best: 0.51800
        Epoch: 10, Time: 0.02726s, Loss: 1.89345
        update best: 0.54000
        Epoch: 11, Time: 0.03213s, Loss: 1.88819
        update best: 0.56800
        Epoch: 12, Time: 0.03265s, Loss: 1.88074
        update best: 0.58800
        Epoch: 13, Time: 0.03181s, Loss: 1.87387
        update best: 0.61200
        Epoch: 14, Time: 0.02740s, Loss: 1.86807
        update best: 0.63600
        Epoch: 15, Time: 0.02897s, Loss: 1.85900
        update best: 0.68200
        Epoch: 16, Time: 0.02701s, Loss: 1.84736
        update best: 0.69800
        Epoch: 17, Time: 0.02716s, Loss: 1.83908
        update best: 0.72200
        Epoch: 18, Time: 0.02782s, Loss: 1.83323
        update best: 0.74800
        Epoch: 19, Time: 0.02795s, Loss: 1.81363
        update best: 0.77400
        Epoch: 20, Time: 0.02841s, Loss: 1.80020
        update best: 0.78200
        Epoch: 21, Time: 0.02796s, Loss: 1.79673
        update best: 0.79400
        Epoch: 22, Time: 0.02665s, Loss: 1.77684
        update best: 0.79600
        Epoch: 23, Time: 0.02657s, Loss: 1.75575
        Epoch: 24, Time: 0.02713s, Loss: 1.74837
        Epoch: 25, Time: 0.02716s, Loss: 1.74580
        Epoch: 26, Time: 0.02909s, Loss: 1.71996
        Epoch: 27, Time: 0.02656s, Loss: 1.70845
        Epoch: 28, Time: 0.02675s, Loss: 1.69779
        Epoch: 29, Time: 0.02614s, Loss: 1.66469
        Epoch: 30, Time: 0.02627s, Loss: 1.66196
        Epoch: 31, Time: 0.02743s, Loss: 1.65097
        Epoch: 32, Time: 0.02776s, Loss: 1.62630
        Epoch: 33, Time: 0.02752s, Loss: 1.60284
        Epoch: 34, Time: 0.02749s, Loss: 1.58056
        Epoch: 35, Time: 0.02549s, Loss: 1.57601
        Epoch: 36, Time: 0.02724s, Loss: 1.55081
        Epoch: 37, Time: 0.02836s, Loss: 1.53101
        Epoch: 38, Time: 0.02641s, Loss: 1.53054
        Epoch: 39, Time: 0.02638s, Loss: 1.51172
        Epoch: 40, Time: 0.02669s, Loss: 1.45463
        Epoch: 41, Time: 0.02674s, Loss: 1.43432
        Epoch: 42, Time: 0.02729s, Loss: 1.39888
        Epoch: 43, Time: 0.02715s, Loss: 1.40660
        Epoch: 44, Time: 0.02975s, Loss: 1.41301
        Epoch: 45, Time: 0.02658s, Loss: 1.32990
        Epoch: 46, Time: 0.02753s, Loss: 1.31327
        Epoch: 47, Time: 0.02823s, Loss: 1.30501
        Epoch: 48, Time: 0.02904s, Loss: 1.28125
        Epoch: 49, Time: 0.02605s, Loss: 1.23469
        Epoch: 50, Time: 0.02684s, Loss: 1.25209
        Epoch: 51, Time: 0.02576s, Loss: 1.24679
        Epoch: 52, Time: 0.02693s, Loss: 1.20283
        Epoch: 53, Time: 0.02735s, Loss: 1.16539
        Epoch: 54, Time: 0.02733s, Loss: 1.16182
        Epoch: 55, Time: 0.02691s, Loss: 1.12086
        Epoch: 56, Time: 0.02706s, Loss: 1.09962
        Epoch: 57, Time: 0.02628s, Loss: 1.09911
        Epoch: 58, Time: 0.02716s, Loss: 1.05156
        Epoch: 59, Time: 0.02729s, Loss: 1.03817
        Epoch: 60, Time: 0.03020s, Loss: 0.99580
        Epoch: 61, Time: 0.02628s, Loss: 0.98298
        Epoch: 62, Time: 0.02804s, Loss: 0.95318
        Epoch: 63, Time: 0.02650s, Loss: 0.94846
        Epoch: 64, Time: 0.02753s, Loss: 0.94741
        Epoch: 65, Time: 0.02678s, Loss: 0.92977
        Epoch: 66, Time: 0.02639s, Loss: 0.85785
        Epoch: 67, Time: 0.02938s, Loss: 0.87859
        Epoch: 68, Time: 0.02816s, Loss: 0.81501
        Epoch: 69, Time: 0.02799s, Loss: 0.82868
        Epoch: 70, Time: 0.02577s, Loss: 0.83454
        Epoch: 71, Time: 0.03040s, Loss: 0.81279
        Epoch: 72, Time: 0.02764s, Loss: 0.80267
        Epoch: 73, Time: 0.02707s, Loss: 0.77012
        Epoch: 74, Time: 0.02769s, Loss: 0.75785
        Epoch: 75, Time: 0.02844s, Loss: 0.70275
        Epoch: 76, Time: 0.02718s, Loss: 0.73779
        Epoch: 77, Time: 0.02707s, Loss: 0.75283
        Epoch: 78, Time: 0.02642s, Loss: 0.71528
        Epoch: 79, Time: 0.02563s, Loss: 0.65665
        Epoch: 80, Time: 0.02572s, Loss: 0.72648
        Epoch: 81, Time: 0.02690s, Loss: 0.64160
        Epoch: 82, Time: 0.02741s, Loss: 0.67890
        Epoch: 83, Time: 0.03295s, Loss: 0.66671
        Epoch: 84, Time: 0.02697s, Loss: 0.68267
        Epoch: 85, Time: 0.02802s, Loss: 0.62096
        Epoch: 86, Time: 0.02694s, Loss: 0.59566
        Epoch: 87, Time: 0.02695s, Loss: 0.61715
        Epoch: 88, Time: 0.02584s, Loss: 0.56823
        Epoch: 89, Time: 0.02680s, Loss: 0.58922
        Epoch: 90, Time: 0.02628s, Loss: 0.62176
        Epoch: 91, Time: 0.02630s, Loss: 0.56168
        Epoch: 92, Time: 0.02729s, Loss: 0.59730
        Epoch: 93, Time: 0.03309s, Loss: 0.54350
        Epoch: 94, Time: 0.02711s, Loss: 0.52554
        Epoch: 95, Time: 0.03073s, Loss: 0.55863
        Epoch: 96, Time: 0.03009s, Loss: 0.54187
        Epoch: 97, Time: 0.02847s, Loss: 0.51606
        Epoch: 98, Time: 0.02721s, Loss: 0.58703
        Epoch: 99, Time: 0.02683s, Loss: 0.45709
        Epoch: 100, Time: 0.02546s, Loss: 0.48065
        Epoch: 101, Time: 0.02661s, Loss: 0.47521
        Epoch: 102, Time: 0.02708s, Loss: 0.49044
        Epoch: 103, Time: 0.02877s, Loss: 0.54857
        Epoch: 104, Time: 0.02891s, Loss: 0.49147
        Epoch: 105, Time: 0.02831s, Loss: 0.51098
        Epoch: 106, Time: 0.02855s, Loss: 0.47384
        Epoch: 107, Time: 0.02663s, Loss: 0.44903
        Epoch: 108, Time: 0.02739s, Loss: 0.48902
        Epoch: 109, Time: 0.02786s, Loss: 0.47107
        Epoch: 110, Time: 0.02680s, Loss: 0.44998
        Epoch: 111, Time: 0.02667s, Loss: 0.45758
        Epoch: 112, Time: 0.02677s, Loss: 0.48968
        Epoch: 113, Time: 0.03363s, Loss: 0.47052
        Epoch: 114, Time: 0.02720s, Loss: 0.42302
        Epoch: 115, Time: 0.02691s, Loss: 0.46022
        Epoch: 116, Time: 0.02800s, Loss: 0.44152
        Epoch: 117, Time: 0.02809s, Loss: 0.41619
        Epoch: 118, Time: 0.02747s, Loss: 0.42209
        Epoch: 119, Time: 0.02731s, Loss: 0.39555
        Epoch: 120, Time: 0.02757s, Loss: 0.41737
        Epoch: 121, Time: 0.02572s, Loss: 0.43961
        Epoch: 122, Time: 0.02781s, Loss: 0.45638
        Epoch: 123, Time: 0.03219s, Loss: 0.40218
        Epoch: 124, Time: 0.02912s, Loss: 0.39478
        Epoch: 125, Time: 0.02836s, Loss: 0.42770
        Epoch: 126, Time: 0.02821s, Loss: 0.44723
        Epoch: 127, Time: 0.02668s, Loss: 0.44981
        Epoch: 128, Time: 0.02659s, Loss: 0.36467
        Epoch: 129, Time: 0.02790s, Loss: 0.41371
        Epoch: 130, Time: 0.02687s, Loss: 0.43008
        Epoch: 131, Time: 0.02749s, Loss: 0.39013
        Epoch: 132, Time: 0.02737s, Loss: 0.38068
        Epoch: 133, Time: 0.02744s, Loss: 0.41307
        Epoch: 134, Time: 0.02709s, Loss: 0.37499
        Epoch: 135, Time: 0.03620s, Loss: 0.38330
        Epoch: 136, Time: 0.03489s, Loss: 0.36262
        Epoch: 137, Time: 0.03187s, Loss: 0.37654
        Epoch: 138, Time: 0.03120s, Loss: 0.39200
        Epoch: 139, Time: 0.03104s, Loss: 0.38622
        Epoch: 140, Time: 0.03423s, Loss: 0.40245
        Epoch: 141, Time: 0.02714s, Loss: 0.42246
        Epoch: 142, Time: 0.02613s, Loss: 0.38597
        Epoch: 143, Time: 0.02614s, Loss: 0.33846
        Epoch: 144, Time: 0.02727s, Loss: 0.35218
        Epoch: 145, Time: 0.02886s, Loss: 0.34761
        Epoch: 146, Time: 0.02711s, Loss: 0.36396
        Epoch: 147, Time: 0.02971s, Loss: 0.36457
        Epoch: 148, Time: 0.02699s, Loss: 0.34745
        Epoch: 149, Time: 0.02773s, Loss: 0.35060
        Epoch: 150, Time: 0.02763s, Loss: 0.33626
        Epoch: 151, Time: 0.02665s, Loss: 0.31920
        Epoch: 152, Time: 0.02700s, Loss: 0.35494
        Epoch: 153, Time: 0.02631s, Loss: 0.32023
        Epoch: 154, Time: 0.02521s, Loss: 0.33341
        Epoch: 155, Time: 0.02761s, Loss: 0.33163
        Epoch: 156, Time: 0.03211s, Loss: 0.37067
        Epoch: 157, Time: 0.02632s, Loss: 0.31185
        Epoch: 158, Time: 0.02799s, Loss: 0.32024
        Epoch: 159, Time: 0.02868s, Loss: 0.33890
        Epoch: 160, Time: 0.02777s, Loss: 0.34390
        Epoch: 161, Time: 0.02628s, Loss: 0.34751
        Epoch: 162, Time: 0.02660s, Loss: 0.34165
        Epoch: 163, Time: 0.02635s, Loss: 0.32915
        Epoch: 164, Time: 0.02783s, Loss: 0.34125
        Epoch: 165, Time: 0.02822s, Loss: 0.35261
        Epoch: 166, Time: 0.02855s, Loss: 0.31803
        Epoch: 167, Time: 0.02532s, Loss: 0.34157
        Epoch: 168, Time: 0.02748s, Loss: 0.36173
        Epoch: 169, Time: 0.02843s, Loss: 0.29295
        Epoch: 170, Time: 0.02735s, Loss: 0.32935
        Epoch: 171, Time: 0.02742s, Loss: 0.32463
        Epoch: 172, Time: 0.02704s, Loss: 0.34419
        Epoch: 173, Time: 0.02737s, Loss: 0.32393
        Epoch: 174, Time: 0.02667s, Loss: 0.32464
        Epoch: 175, Time: 0.02750s, Loss: 0.32668
        Epoch: 176, Time: 0.02771s, Loss: 0.33835
        Epoch: 177, Time: 0.02783s, Loss: 0.32610
        Epoch: 178, Time: 0.03027s, Loss: 0.31611
        Epoch: 179, Time: 0.02945s, Loss: 0.31614
        Epoch: 180, Time: 0.02750s, Loss: 0.33912
        Epoch: 181, Time: 0.02655s, Loss: 0.29072
        Epoch: 182, Time: 0.02566s, Loss: 0.33455
        Epoch: 183, Time: 0.02669s, Loss: 0.29251
        Epoch: 184, Time: 0.02900s, Loss: 0.32722
        Epoch: 185, Time: 0.02738s, Loss: 0.29612
        Epoch: 186, Time: 0.02708s, Loss: 0.30084
        Epoch: 187, Time: 0.02681s, Loss: 0.28315
        Epoch: 188, Time: 0.02847s, Loss: 0.31396
        Epoch: 189, Time: 0.02638s, Loss: 0.31683
        Epoch: 190, Time: 0.02819s, Loss: 0.33803
        Epoch: 191, Time: 0.02756s, Loss: 0.31791
        Epoch: 192, Time: 0.02695s, Loss: 0.35256
        Epoch: 193, Time: 0.02624s, Loss: 0.30407
        Epoch: 194, Time: 0.02629s, Loss: 0.30797
        Epoch: 195, Time: 0.02591s, Loss: 0.29365
        Epoch: 196, Time: 0.02655s, Loss: 0.28897
        Epoch: 197, Time: 0.02585s, Loss: 0.31783
        Epoch: 198, Time: 0.02900s, Loss: 0.28889
        Epoch: 199, Time: 0.02735s, Loss: 0.31066
        Epoch: 200, Time: 0.02652s, Loss: 0.31168
        Epoch: 201, Time: 0.02635s, Loss: 0.26849
        Epoch: 202, Time: 0.02685s, Loss: 0.29419
        Epoch: 203, Time: 0.02794s, Loss: 0.31236
        update best: 0.79800
        Epoch: 204, Time: 0.02748s, Loss: 0.29655
        Epoch: 205, Time: 0.02772s, Loss: 0.32185
        update best: 0.80000
        Epoch: 206, Time: 0.03271s, Loss: 0.28461
        Epoch: 207, Time: 0.02841s, Loss: 0.28718
        Epoch: 208, Time: 0.02810s, Loss: 0.28859
        Epoch: 209, Time: 0.02825s, Loss: 0.33484
        Epoch: 210, Time: 0.02748s, Loss: 0.25476
        Epoch: 211, Time: 0.02689s, Loss: 0.31217
        Epoch: 212, Time: 0.02616s, Loss: 0.30048
        Epoch: 213, Time: 0.02599s, Loss: 0.25396
        Epoch: 214, Time: 0.02509s, Loss: 0.25659
        Epoch: 215, Time: 0.02558s, Loss: 0.27736
        Epoch: 216, Time: 0.02744s, Loss: 0.29813
        Epoch: 217, Time: 0.02797s, Loss: 0.26633
        Epoch: 218, Time: 0.02972s, Loss: 0.26556
        Epoch: 219, Time: 0.02468s, Loss: 0.26812
        Epoch: 220, Time: 0.02691s, Loss: 0.27502
        Epoch: 221, Time: 0.02941s, Loss: 0.27201
        Epoch: 222, Time: 0.03062s, Loss: 0.24750
        Epoch: 223, Time: 0.02580s, Loss: 0.25536
        Epoch: 224, Time: 0.02601s, Loss: 0.24400
        Epoch: 225, Time: 0.02609s, Loss: 0.26673
        Epoch: 226, Time: 0.02816s, Loss: 0.28496
        Epoch: 227, Time: 0.02798s, Loss: 0.27348
        Epoch: 228, Time: 0.02800s, Loss: 0.30068
        Epoch: 229, Time: 0.02711s, Loss: 0.25621
        Epoch: 230, Time: 0.02845s, Loss: 0.28133
        Epoch: 231, Time: 0.02709s, Loss: 0.26263
        Epoch: 232, Time: 0.02776s, Loss: 0.28019
        Epoch: 233, Time: 0.02760s, Loss: 0.24621
        Epoch: 234, Time: 0.02652s, Loss: 0.25726
        Epoch: 235, Time: 0.02607s, Loss: 0.27996
        Epoch: 236, Time: 0.02545s, Loss: 0.26172
        Epoch: 237, Time: 0.02611s, Loss: 0.28643
        update best: 0.80200
        Epoch: 238, Time: 0.02843s, Loss: 0.27893
        Epoch: 239, Time: 0.02436s, Loss: 0.23068
        Epoch: 240, Time: 0.02698s, Loss: 0.26539
        Epoch: 241, Time: 0.02526s, Loss: 0.26346
        Epoch: 242, Time: 0.02636s, Loss: 0.25852
        Epoch: 243, Time: 0.02681s, Loss: 0.24250
        Epoch: 244, Time: 0.02879s, Loss: 0.26560
        Epoch: 245, Time: 0.02841s, Loss: 0.24397
        Epoch: 246, Time: 0.02649s, Loss: 0.22487
        Epoch: 247, Time: 0.02529s, Loss: 0.28920
        Epoch: 248, Time: 0.02598s, Loss: 0.25361
        Epoch: 249, Time: 0.02651s, Loss: 0.23220
        Epoch: 250, Time: 0.02981s, Loss: 0.24851
        Epoch: 251, Time: 0.02647s, Loss: 0.26154
        Epoch: 252, Time: 0.02915s, Loss: 0.28003
        Epoch: 253, Time: 0.02627s, Loss: 0.27142
        Epoch: 254, Time: 0.02771s, Loss: 0.24000
        Epoch: 255, Time: 0.02807s, Loss: 0.22970
        Epoch: 256, Time: 0.02778s, Loss: 0.25055
        Epoch: 257, Time: 0.02756s, Loss: 0.25298
        Epoch: 258, Time: 0.02604s, Loss: 0.25399
        Epoch: 259, Time: 0.02515s, Loss: 0.23506
        Epoch: 260, Time: 0.02584s, Loss: 0.27011
        Epoch: 261, Time: 0.02733s, Loss: 0.27896
        Epoch: 262, Time: 0.03368s, Loss: 0.27697
        Epoch: 263, Time: 0.02622s, Loss: 0.25122
        Epoch: 264, Time: 0.02557s, Loss: 0.22288
        Epoch: 265, Time: 0.02677s, Loss: 0.24788
        Epoch: 266, Time: 0.02789s, Loss: 0.25024
        Epoch: 267, Time: 0.02766s, Loss: 0.24291
        Epoch: 268, Time: 0.02734s, Loss: 0.23501
        Epoch: 269, Time: 0.02628s, Loss: 0.22473
        update best: 0.80400
        Epoch: 270, Time: 0.02710s, Loss: 0.23869
        Epoch: 271, Time: 0.02704s, Loss: 0.23497
        Epoch: 272, Time: 0.02797s, Loss: 0.27661
        Epoch: 273, Time: 0.02528s, Loss: 0.22743
        Epoch: 274, Time: 0.02586s, Loss: 0.27344
        Epoch: 275, Time: 0.02527s, Loss: 0.24526
        Epoch: 276, Time: 0.02694s, Loss: 0.23004
        Epoch: 277, Time: 0.02799s, Loss: 0.26727
        Epoch: 278, Time: 0.02743s, Loss: 0.24816
        Epoch: 279, Time: 0.02808s, Loss: 0.24808
        Epoch: 280, Time: 0.02596s, Loss: 0.21776
        Epoch: 281, Time: 0.02563s, Loss: 0.21926
        Epoch: 282, Time: 0.02653s, Loss: 0.22270
        Epoch: 283, Time: 0.02805s, Loss: 0.24317
        Epoch: 284, Time: 0.02826s, Loss: 0.26508
        Epoch: 285, Time: 0.02821s, Loss: 0.27642
        Epoch: 286, Time: 0.02656s, Loss: 0.28210
        Epoch: 287, Time: 0.02595s, Loss: 0.21376
        Epoch: 288, Time: 0.02581s, Loss: 0.22294
        Epoch: 289, Time: 0.02792s, Loss: 0.22761
        Epoch: 290, Time: 0.02788s, Loss: 0.21223
        Epoch: 291, Time: 0.02840s, Loss: 0.25497
        Epoch: 292, Time: 0.02945s, Loss: 0.25667
        Epoch: 293, Time: 0.02686s, Loss: 0.28930
        Epoch: 294, Time: 0.02824s, Loss: 0.27815
        Epoch: 295, Time: 0.02799s, Loss: 0.29124
        Epoch: 296, Time: 0.02615s, Loss: 0.23398
        Epoch: 297, Time: 0.02607s, Loss: 0.21476
        Epoch: 298, Time: 0.02598s, Loss: 0.22739
        Epoch: 299, Time: 0.02830s, Loss: 0.26215

        train finished!
        best val: 0.80400
        test...
        final result: epoch: 269
        {'accuracy': 0.8240000009536743, 'f1_score': 0.8174891298012773, 'f1_score -> average@micro': 0.824}


Cora上使用HGNN
----------------

导入依赖包
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from copy import deepcopy

    import torch
    import torch.optim as optim
    import torch.nn.functional as F

    from dhg import Graph, Hypergraph
    from dhg.data import Cora
    from dhg.models import HGNN
    from dhg.random import set_seed
    from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator


定义函数
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def train(net, X, G, lbls, train_idx, optimizer, epoch):
        net.train()

        st = time.time()
        optimizer.zero_grad()
        outs = net(X, G)
        outs, lbls = outs[train_idx], lbls[train_idx]
        loss = F.cross_entropy(outs, lbls)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
        return loss.item()


    @torch.no_grad()
    def infer(net, X, G, lbls, idx, test=False):
        net.eval()
        outs = net(X, G)
        outs, lbls = outs[idx], lbls[idx]
        if not test:
            res = evaluator.validate(lbls, outs)
        else:
            res = evaluator.test(lbls, outs)
        return res

主函数
^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: 

    更多关于评测器 ``Evaluator`` 的细节可以参照 :doc:`构建指标评测器 </zh/tutorial/metric>` 章节。

.. code-block:: python

    if __name__ == "__main__":
        set_seed(2022)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
        data = Cora()
        X, lbl = data["features"], data["labels"]
        G = Graph(data["num_vertices"], data["edge_list"])
        HG = Hypergraph.from_graph_kHop(G, k=1)
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]

        net = HGNN(data["dim_features"], 16, data["num_classes"])
        optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

        X, lbl = X.to(device), lbl.to(device)
        HG = HG.to(device)
        net = net.to(device)

        best_state = None
        best_epoch, best_val = 0, 0
        for epoch in range(200):
            # train
            train(net, X, HG, lbl, train_mask, optimizer, epoch)
            # validation
            if epoch % 1 == 0:
                with torch.no_grad():
                    val_res = infer(net, X, HG, lbl, val_mask)
                if val_res > best_val:
                    print(f"update best: {val_res:.5f}")
                    best_epoch = epoch
                    best_val = val_res
                    best_state = deepcopy(net.state_dict())
        print("\ntrain finished!")
        print(f"best val: {best_val:.5f}")
        # test
        print("test...")
        net.load_state_dict(best_state)
        res = infer(net, X, HG, lbl, test_mask, test=True)
        print(f"final result: epoch: {best_epoch}")
        print(res)


.. only:: not latex

    输出
    ^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code-block:: 

        Epoch: 0, Time: 0.50315s, Loss: 1.94993
        update best: 0.31600
        Epoch: 1, Time: 0.00196s, Loss: 1.94627
        Epoch: 2, Time: 0.00175s, Loss: 1.94413
        Epoch: 3, Time: 0.00200s, Loss: 1.93941
        Epoch: 4, Time: 0.00180s, Loss: 1.93488
        Epoch: 5, Time: 0.00174s, Loss: 1.92980
        update best: 0.32000
        Epoch: 6, Time: 0.00184s, Loss: 1.92559
        update best: 0.36400
        Epoch: 7, Time: 0.00256s, Loss: 1.91934
        update best: 0.46400
        Epoch: 8, Time: 0.00198s, Loss: 1.91385
        Epoch: 9, Time: 0.00177s, Loss: 1.90319
        Epoch: 10, Time: 0.00248s, Loss: 1.89834
        Epoch: 11, Time: 0.00248s, Loss: 1.89127
        Epoch: 12, Time: 0.00173s, Loss: 1.87880
        Epoch: 13, Time: 0.00247s, Loss: 1.87467
        Epoch: 14, Time: 0.00194s, Loss: 1.86688
        Epoch: 15, Time: 0.00181s, Loss: 1.85891
        Epoch: 16, Time: 0.00266s, Loss: 1.85094
        Epoch: 17, Time: 0.00289s, Loss: 1.84161
        Epoch: 18, Time: 0.00179s, Loss: 1.82744
        Epoch: 19, Time: 0.00239s, Loss: 1.81666
        Epoch: 20, Time: 0.00198s, Loss: 1.80902
        Epoch: 21, Time: 0.00177s, Loss: 1.78956
        Epoch: 22, Time: 0.00252s, Loss: 1.78221
        update best: 0.49000
        Epoch: 23, Time: 0.00191s, Loss: 1.76655
        update best: 0.50200
        Epoch: 24, Time: 0.00174s, Loss: 1.76185
        update best: 0.51600
        Epoch: 25, Time: 0.00253s, Loss: 1.74321
        update best: 0.51800
        Epoch: 26, Time: 0.00187s, Loss: 1.72027
        update best: 0.52200
        Epoch: 27, Time: 0.00369s, Loss: 1.70986
        update best: 0.52600
        Epoch: 28, Time: 0.00241s, Loss: 1.69354
        update best: 0.53000
        Epoch: 29, Time: 0.00309s, Loss: 1.69100
        update best: 0.53800
        Epoch: 30, Time: 0.00232s, Loss: 1.66968
        update best: 0.54400
        Epoch: 31, Time: 0.00313s, Loss: 1.65087
        update best: 0.54600
        Epoch: 32, Time: 0.00224s, Loss: 1.64182
        update best: 0.56000
        Epoch: 33, Time: 0.00277s, Loss: 1.60257
        update best: 0.57800
        Epoch: 34, Time: 0.00208s, Loss: 1.58798
        update best: 0.59200
        Epoch: 35, Time: 0.00176s, Loss: 1.58344
        update best: 0.60000
        Epoch: 36, Time: 0.00200s, Loss: 1.56942
        update best: 0.63200
        Epoch: 37, Time: 0.00206s, Loss: 1.53224
        update best: 0.64800
        Epoch: 38, Time: 0.00215s, Loss: 1.53036
        update best: 0.67000
        Epoch: 39, Time: 0.00200s, Loss: 1.50875
        update best: 0.68000
        Epoch: 40, Time: 0.00209s, Loss: 1.46828
        update best: 0.69200
        Epoch: 41, Time: 0.00243s, Loss: 1.45782
        update best: 0.69400
        Epoch: 42, Time: 0.00208s, Loss: 1.42179
        Epoch: 43, Time: 0.00267s, Loss: 1.40893
        Epoch: 44, Time: 0.00176s, Loss: 1.40358
        update best: 0.69800
        Epoch: 45, Time: 0.00175s, Loss: 1.37788
        Epoch: 46, Time: 0.00274s, Loss: 1.34310
        Epoch: 47, Time: 0.00173s, Loss: 1.32779
        update best: 0.70200
        Epoch: 48, Time: 0.00175s, Loss: 1.30572
        update best: 0.71200
        Epoch: 49, Time: 0.00221s, Loss: 1.28909
        update best: 0.71800
        Epoch: 50, Time: 0.00184s, Loss: 1.28903
        update best: 0.72400
        Epoch: 51, Time: 0.00345s, Loss: 1.25486
        update best: 0.73200
        Epoch: 52, Time: 0.00176s, Loss: 1.22994
        update best: 0.74200
        Epoch: 53, Time: 0.00173s, Loss: 1.20690
        update best: 0.75000
        Epoch: 54, Time: 0.00241s, Loss: 1.17115
        Epoch: 55, Time: 0.00198s, Loss: 1.18836
        update best: 0.75600
        Epoch: 56, Time: 0.00279s, Loss: 1.17722
        update best: 0.75800
        Epoch: 57, Time: 0.00204s, Loss: 1.13414
        Epoch: 58, Time: 0.00173s, Loss: 1.12058
        update best: 0.76200
        Epoch: 59, Time: 0.00228s, Loss: 1.09260
        update best: 0.77400
        Epoch: 60, Time: 0.00188s, Loss: 1.07260
        Epoch: 61, Time: 0.00256s, Loss: 1.09610
        Epoch: 62, Time: 0.00280s, Loss: 1.02422
        Epoch: 63, Time: 0.00221s, Loss: 1.03871
        update best: 0.77800
        Epoch: 64, Time: 0.00311s, Loss: 1.00255
        Epoch: 65, Time: 0.00226s, Loss: 0.99640
        update best: 0.78000
        Epoch: 66, Time: 0.00296s, Loss: 0.99191
        update best: 0.78200
        Epoch: 67, Time: 0.00235s, Loss: 0.95631
        update best: 0.78600
        Epoch: 68, Time: 0.00255s, Loss: 0.94336
        Epoch: 69, Time: 0.00183s, Loss: 0.92673
        update best: 0.79000
        Epoch: 70, Time: 0.00165s, Loss: 0.92654
        update best: 0.79600
        Epoch: 71, Time: 0.00188s, Loss: 0.86986
        update best: 0.80000
        Epoch: 72, Time: 0.00170s, Loss: 0.90749
        Epoch: 73, Time: 0.00164s, Loss: 0.86787
        Epoch: 74, Time: 0.00218s, Loss: 0.86549
        Epoch: 75, Time: 0.00182s, Loss: 0.86944
        Epoch: 76, Time: 0.00189s, Loss: 0.83897
        Epoch: 77, Time: 0.00167s, Loss: 0.82139
        Epoch: 78, Time: 0.00168s, Loss: 0.81658
        Epoch: 79, Time: 0.00198s, Loss: 0.78883
        Epoch: 80, Time: 0.00207s, Loss: 0.78880
        Epoch: 81, Time: 0.00209s, Loss: 0.77039
        Epoch: 82, Time: 0.00170s, Loss: 0.74785
        Epoch: 83, Time: 0.00185s, Loss: 0.74238
        Epoch: 84, Time: 0.00293s, Loss: 0.73360
        Epoch: 85, Time: 0.00164s, Loss: 0.76029
        Epoch: 86, Time: 0.00163s, Loss: 0.71382
        Epoch: 87, Time: 0.00162s, Loss: 0.72503
        Epoch: 88, Time: 0.00202s, Loss: 0.70878
        Epoch: 89, Time: 0.00172s, Loss: 0.71945
        Epoch: 90, Time: 0.00180s, Loss: 0.65032
        Epoch: 91, Time: 0.00302s, Loss: 0.71030
        Epoch: 92, Time: 0.00157s, Loss: 0.67237
        Epoch: 93, Time: 0.00161s, Loss: 0.68624
        Epoch: 94, Time: 0.00161s, Loss: 0.65738
        Epoch: 95, Time: 0.00203s, Loss: 0.65683
        Epoch: 96, Time: 0.00171s, Loss: 0.63819
        Epoch: 97, Time: 0.00177s, Loss: 0.66612
        Epoch: 98, Time: 0.00231s, Loss: 0.64060
        Epoch: 99, Time: 0.00161s, Loss: 0.63596
        Epoch: 100, Time: 0.00161s, Loss: 0.62215
        Epoch: 101, Time: 0.00195s, Loss: 0.59992
        Epoch: 102, Time: 0.00184s, Loss: 0.63610
        Epoch: 103, Time: 0.00168s, Loss: 0.60803
        Epoch: 104, Time: 0.00174s, Loss: 0.60519
        Epoch: 105, Time: 0.00203s, Loss: 0.61317
        update best: 0.80200
        Epoch: 106, Time: 0.00163s, Loss: 0.56701
        Epoch: 107, Time: 0.00160s, Loss: 0.58649
        Epoch: 108, Time: 0.00202s, Loss: 0.60864
        Epoch: 109, Time: 0.00171s, Loss: 0.59734
        Epoch: 110, Time: 0.00174s, Loss: 0.58395
        Epoch: 111, Time: 0.00262s, Loss: 0.59959
        Epoch: 112, Time: 0.00166s, Loss: 0.57178
        Epoch: 113, Time: 0.00162s, Loss: 0.57493
        Epoch: 114, Time: 0.00166s, Loss: 0.56720
        Epoch: 115, Time: 0.00207s, Loss: 0.57864
        Epoch: 116, Time: 0.00174s, Loss: 0.55171
        Epoch: 117, Time: 0.00201s, Loss: 0.56022
        Epoch: 118, Time: 0.00295s, Loss: 0.54393
        Epoch: 119, Time: 0.00162s, Loss: 0.54266
        Epoch: 120, Time: 0.00162s, Loss: 0.54640
        Epoch: 121, Time: 0.00165s, Loss: 0.51695
        Epoch: 122, Time: 0.00193s, Loss: 0.53059
        Epoch: 123, Time: 0.00175s, Loss: 0.49817
        Epoch: 124, Time: 0.00168s, Loss: 0.49963
        Epoch: 125, Time: 0.00280s, Loss: 0.50499
        Epoch: 126, Time: 0.00165s, Loss: 0.51792
        Epoch: 127, Time: 0.00162s, Loss: 0.48759
        Epoch: 128, Time: 0.00188s, Loss: 0.52524
        Epoch: 129, Time: 0.00192s, Loss: 0.49752
        Epoch: 130, Time: 0.00182s, Loss: 0.48539
        Epoch: 131, Time: 0.00178s, Loss: 0.51904
        Epoch: 132, Time: 0.00210s, Loss: 0.51619
        Epoch: 133, Time: 0.00164s, Loss: 0.46799
        Epoch: 134, Time: 0.00168s, Loss: 0.47253
        Epoch: 135, Time: 0.00220s, Loss: 0.50235
        Epoch: 136, Time: 0.00179s, Loss: 0.48068
        Epoch: 137, Time: 0.00181s, Loss: 0.48230
        Epoch: 138, Time: 0.00311s, Loss: 0.47752
        Epoch: 139, Time: 0.00165s, Loss: 0.46344
        Epoch: 140, Time: 0.00168s, Loss: 0.50513
        Epoch: 141, Time: 0.00175s, Loss: 0.45315
        Epoch: 142, Time: 0.00234s, Loss: 0.45984
        Epoch: 143, Time: 0.00184s, Loss: 0.45598
        Epoch: 144, Time: 0.00181s, Loss: 0.48745
        Epoch: 145, Time: 0.00208s, Loss: 0.47391
        Epoch: 146, Time: 0.00167s, Loss: 0.42658
        Epoch: 147, Time: 0.00164s, Loss: 0.44139
        Epoch: 148, Time: 0.00211s, Loss: 0.44337
        Epoch: 149, Time: 0.00174s, Loss: 0.43854
        Epoch: 150, Time: 0.00194s, Loss: 0.45141
        Epoch: 151, Time: 0.00337s, Loss: 0.43659
        Epoch: 152, Time: 0.00223s, Loss: 0.45104
        Epoch: 153, Time: 0.00217s, Loss: 0.45788
        Epoch: 154, Time: 0.00256s, Loss: 0.44208
        Epoch: 155, Time: 0.00216s, Loss: 0.47642
        Epoch: 156, Time: 0.00289s, Loss: 0.41826
        Epoch: 157, Time: 0.00219s, Loss: 0.44075
        Epoch: 158, Time: 0.00212s, Loss: 0.39873
        Epoch: 159, Time: 0.00235s, Loss: 0.43970
        Epoch: 160, Time: 0.00170s, Loss: 0.41875
        Epoch: 161, Time: 0.00185s, Loss: 0.42697
        Epoch: 162, Time: 0.00185s, Loss: 0.44240
        Epoch: 163, Time: 0.00165s, Loss: 0.45397
        Epoch: 164, Time: 0.00217s, Loss: 0.38061
        Epoch: 165, Time: 0.00187s, Loss: 0.40102
        Epoch: 166, Time: 0.00194s, Loss: 0.39496
        Epoch: 167, Time: 0.00208s, Loss: 0.41661
        Epoch: 168, Time: 0.00187s, Loss: 0.41864
        Epoch: 169, Time: 0.00262s, Loss: 0.41757
        Epoch: 170, Time: 0.00188s, Loss: 0.41356
        Epoch: 171, Time: 0.00180s, Loss: 0.38835
        Epoch: 172, Time: 0.00213s, Loss: 0.42775
        Epoch: 173, Time: 0.00187s, Loss: 0.39169
        Epoch: 174, Time: 0.00164s, Loss: 0.41415
        Epoch: 175, Time: 0.00290s, Loss: 0.39668
        update best: 0.80400
        Epoch: 176, Time: 0.00161s, Loss: 0.42034
        Epoch: 177, Time: 0.00164s, Loss: 0.40507
        Epoch: 178, Time: 0.00206s, Loss: 0.39741
        Epoch: 179, Time: 0.00181s, Loss: 0.40042
        Epoch: 180, Time: 0.00163s, Loss: 0.37404
        Epoch: 181, Time: 0.00167s, Loss: 0.40175
        Epoch: 182, Time: 0.00217s, Loss: 0.35673
        Epoch: 183, Time: 0.00162s, Loss: 0.39076
        Epoch: 184, Time: 0.00157s, Loss: 0.39327
        Epoch: 185, Time: 0.00208s, Loss: 0.38354
        Epoch: 186, Time: 0.00172s, Loss: 0.36611
        Epoch: 187, Time: 0.00174s, Loss: 0.38952
        Epoch: 188, Time: 0.00276s, Loss: 0.39074
        Epoch: 189, Time: 0.00160s, Loss: 0.36561
        Epoch: 190, Time: 0.00164s, Loss: 0.37361
        Epoch: 191, Time: 0.00162s, Loss: 0.37590
        Epoch: 192, Time: 0.00188s, Loss: 0.36160
        Epoch: 193, Time: 0.00173s, Loss: 0.37451
        Epoch: 194, Time: 0.00170s, Loss: 0.36310
        Epoch: 195, Time: 0.00285s, Loss: 0.39782
        Epoch: 196, Time: 0.00160s, Loss: 0.36185
        Epoch: 197, Time: 0.00161s, Loss: 0.35991
        Epoch: 198, Time: 0.00191s, Loss: 0.37487
        Epoch: 199, Time: 0.00219s, Loss: 0.36310

        train finished!
        best val: 0.80400
        test...
        final result: epoch: 175
        {'accuracy': 0.8209999799728394, 'f1_score': 0.8113491851888245, 'f1_score -> average@micro': 0.821}    

Cora上使用HGNN+
----------------

导入依赖包
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from copy import deepcopy

    import torch
    import torch.optim as optim
    import torch.nn.functional as F

    from dhg import Graph, Hypergraph
    from dhg.data import Cora
    from dhg.models import HGNNP
    from dhg.random import set_seed
    from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator


定义函数
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def train(net, X, G, lbls, train_idx, optimizer, epoch):
        net.train()

        st = time.time()
        optimizer.zero_grad()
        outs = net(X, G)
        outs, lbls = outs[train_idx], lbls[train_idx]
        loss = F.cross_entropy(outs, lbls)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
        return loss.item()


    @torch.no_grad()
    def infer(net, X, G, lbls, idx, test=False):
        net.eval()
        outs = net(X, G)
        outs, lbls = outs[idx], lbls[idx]
        if not test:
            res = evaluator.validate(lbls, outs)
        else:
            res = evaluator.test(lbls, outs)
        return res

主函数
^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: 

    更多关于评测器 ``Evaluator`` 的细节可以参照 :doc:`构建指标评测器 </zh/tutorial/metric>` 章节。

.. code-block:: python

    if __name__ == "__main__":
        set_seed(2022)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
        data = Cora()
        X, lbl = data["features"], data["labels"]
        G = Graph(data["num_vertices"], data["edge_list"])
        HG = Hypergraph.from_graph(G)
        HG.add_hyperedges_from_graph_kHop(G, k=1)
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]

        net = HGNNP(data["dim_features"], 16, data["num_classes"])
        optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

        X, lbl = X.to(device), lbl.to(device)
        HG = HG.to(device)
        net = net.to(device)

        best_state = None
        best_epoch, best_val = 0, 0
        for epoch in range(200):
            # train
            train(net, X, HG, lbl, train_mask, optimizer, epoch)
            # validation
            if epoch % 1 == 0:
                with torch.no_grad():
                    val_res = infer(net, X, HG, lbl, val_mask)
                if val_res > best_val:
                    print(f"update best: {val_res:.5f}")
                    best_epoch = epoch
                    best_val = val_res
                    best_state = deepcopy(net.state_dict())
        print("\ntrain finished!")
        print(f"best val: {best_val:.5f}")
        # test
        print("test...")
        net.load_state_dict(best_state)
        res = infer(net, X, HG, lbl, test_mask, test=True)
        print(f"final result: epoch: {best_epoch}")
        print(res)


.. only:: not latex

    输出
    ^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code-block:: 

        Epoch: 0, Time: 0.50397s, Loss: 1.95489
        update best: 0.31600
        Epoch: 1, Time: 0.00688s, Loss: 1.95044
        Epoch: 2, Time: 0.00594s, Loss: 1.94790
        Epoch: 3, Time: 0.00777s, Loss: 1.94277
        Epoch: 4, Time: 0.00543s, Loss: 1.93662
        Epoch: 5, Time: 0.00805s, Loss: 1.93121
        Epoch: 6, Time: 0.00549s, Loss: 1.92640
        update best: 0.31800
        Epoch: 7, Time: 0.00687s, Loss: 1.91871
        update best: 0.37600
        Epoch: 8, Time: 0.00722s, Loss: 1.91161
        update best: 0.41000
        Epoch: 9, Time: 0.00553s, Loss: 1.90009
        update best: 0.50400
        Epoch: 10, Time: 0.00770s, Loss: 1.89464
        update best: 0.57000
        Epoch: 11, Time: 0.00566s, Loss: 1.88557
        Epoch: 12, Time: 0.00769s, Loss: 1.87337
        Epoch: 13, Time: 0.00549s, Loss: 1.86598
        Epoch: 14, Time: 0.00767s, Loss: 1.85734
        Epoch: 15, Time: 0.00546s, Loss: 1.84511
        Epoch: 16, Time: 0.00752s, Loss: 1.83575
        Epoch: 17, Time: 0.00545s, Loss: 1.82488
        Epoch: 18, Time: 0.00840s, Loss: 1.80935
        Epoch: 19, Time: 0.00536s, Loss: 1.79647
        Epoch: 20, Time: 0.00756s, Loss: 1.78831
        Epoch: 21, Time: 0.00538s, Loss: 1.76364
        Epoch: 22, Time: 0.00797s, Loss: 1.75609
        Epoch: 23, Time: 0.00601s, Loss: 1.74039
        Epoch: 24, Time: 0.00737s, Loss: 1.73402
        update best: 0.57200
        Epoch: 25, Time: 0.00510s, Loss: 1.70649
        Epoch: 26, Time: 0.00626s, Loss: 1.68333
        update best: 0.57600
        Epoch: 27, Time: 0.00489s, Loss: 1.67384
        Epoch: 28, Time: 0.00637s, Loss: 1.64703
        Epoch: 29, Time: 0.00569s, Loss: 1.65015
        Epoch: 30, Time: 0.00616s, Loss: 1.61904
        Epoch: 31, Time: 0.00482s, Loss: 1.60483
        Epoch: 32, Time: 0.00657s, Loss: 1.58717
        update best: 0.57800
        Epoch: 33, Time: 0.00671s, Loss: 1.54870
        update best: 0.58400
        Epoch: 34, Time: 0.00547s, Loss: 1.53594
        update best: 0.59800
        Epoch: 35, Time: 0.00591s, Loss: 1.52464
        update best: 0.61000
        Epoch: 36, Time: 0.00569s, Loss: 1.50577
        update best: 0.62800
        Epoch: 37, Time: 0.00447s, Loss: 1.47224
        update best: 0.64400
        Epoch: 38, Time: 0.00566s, Loss: 1.46083
        update best: 0.65800
        Epoch: 39, Time: 0.00448s, Loss: 1.44008
        update best: 0.67400
        Epoch: 40, Time: 0.00560s, Loss: 1.39763
        update best: 0.68800
        Epoch: 41, Time: 0.00452s, Loss: 1.38902
        update best: 0.69600
        Epoch: 42, Time: 0.00592s, Loss: 1.34805
        update best: 0.70600
        Epoch: 43, Time: 0.00460s, Loss: 1.32505
        update best: 0.71200
        Epoch: 44, Time: 0.00575s, Loss: 1.32579
        update best: 0.71600
        Epoch: 45, Time: 0.00456s, Loss: 1.29263
        update best: 0.72200
        Epoch: 46, Time: 0.00590s, Loss: 1.25758
        update best: 0.72800
        Epoch: 47, Time: 0.00457s, Loss: 1.25460
        update best: 0.73000
        Epoch: 48, Time: 0.00577s, Loss: 1.21283
        update best: 0.73200
        Epoch: 49, Time: 0.00555s, Loss: 1.22506
        update best: 0.73800
        Epoch: 50, Time: 0.00590s, Loss: 1.20866
        update best: 0.74200
        Epoch: 51, Time: 0.00607s, Loss: 1.17283
        update best: 0.75800
        Epoch: 52, Time: 0.00558s, Loss: 1.14841
        update best: 0.78000
        Epoch: 53, Time: 0.00534s, Loss: 1.12203
        update best: 0.78800
        Epoch: 54, Time: 0.00525s, Loss: 1.07957
        update best: 0.79000
        Epoch: 55, Time: 0.00598s, Loss: 1.09576
        update best: 0.79200
        Epoch: 56, Time: 0.00518s, Loss: 1.08737
        update best: 0.79400
        Epoch: 57, Time: 0.00666s, Loss: 1.03506
        Epoch: 58, Time: 0.00471s, Loss: 1.02326
        Epoch: 59, Time: 0.00623s, Loss: 1.01210
        Epoch: 60, Time: 0.00557s, Loss: 0.99087
        Epoch: 61, Time: 0.00454s, Loss: 0.99048
        Epoch: 62, Time: 0.00614s, Loss: 0.92911
        Epoch: 63, Time: 0.00461s, Loss: 0.96758
        Epoch: 64, Time: 0.00739s, Loss: 0.90397
        Epoch: 65, Time: 0.00469s, Loss: 0.89135
        Epoch: 66, Time: 0.00745s, Loss: 0.90936
        Epoch: 67, Time: 0.00459s, Loss: 0.85870
        Epoch: 68, Time: 0.00657s, Loss: 0.86560
        Epoch: 69, Time: 0.00534s, Loss: 0.84675
        Epoch: 70, Time: 0.00564s, Loss: 0.85727
        Epoch: 71, Time: 0.00590s, Loss: 0.79680
        Epoch: 72, Time: 0.00453s, Loss: 0.82477
        Epoch: 73, Time: 0.00614s, Loss: 0.79762
        Epoch: 74, Time: 0.00452s, Loss: 0.78480
        Epoch: 75, Time: 0.00735s, Loss: 0.81077
        Epoch: 76, Time: 0.00463s, Loss: 0.77174
        Epoch: 77, Time: 0.00706s, Loss: 0.74386
        Epoch: 78, Time: 0.00569s, Loss: 0.73486
        Epoch: 79, Time: 0.00738s, Loss: 0.70369
        update best: 0.79600
        Epoch: 80, Time: 0.00563s, Loss: 0.70949
        Epoch: 81, Time: 0.00649s, Loss: 0.68134
        Epoch: 82, Time: 0.00542s, Loss: 0.65184
        update best: 0.79800
        Epoch: 83, Time: 0.00635s, Loss: 0.66273
        Epoch: 84, Time: 0.00545s, Loss: 0.65232
        Epoch: 85, Time: 0.00696s, Loss: 0.69817
        Epoch: 86, Time: 0.00574s, Loss: 0.64078
        Epoch: 87, Time: 0.00686s, Loss: 0.65521
        Epoch: 88, Time: 0.00470s, Loss: 0.63180
        Epoch: 89, Time: 0.00449s, Loss: 0.65444
        Epoch: 90, Time: 0.00605s, Loss: 0.56861
        Epoch: 91, Time: 0.00456s, Loss: 0.64074
        Epoch: 92, Time: 0.00659s, Loss: 0.59132
        update best: 0.80200
        Epoch: 93, Time: 0.00465s, Loss: 0.62925
        Epoch: 94, Time: 0.00662s, Loss: 0.60163
        Epoch: 95, Time: 0.00453s, Loss: 0.58727
        Epoch: 96, Time: 0.00693s, Loss: 0.57620
        Epoch: 97, Time: 0.00481s, Loss: 0.60987
        Epoch: 98, Time: 0.00702s, Loss: 0.57996
        Epoch: 99, Time: 0.00462s, Loss: 0.56781
        Epoch: 100, Time: 0.00570s, Loss: 0.54706
        Epoch: 101, Time: 0.00507s, Loss: 0.54080
        Epoch: 102, Time: 0.00444s, Loss: 0.57735
        Epoch: 103, Time: 0.00613s, Loss: 0.52275
        Epoch: 104, Time: 0.00452s, Loss: 0.53871
        Epoch: 105, Time: 0.00667s, Loss: 0.54541
        Epoch: 106, Time: 0.00565s, Loss: 0.51127
        Epoch: 107, Time: 0.00738s, Loss: 0.52514
        Epoch: 108, Time: 0.00540s, Loss: 0.54392
        Epoch: 109, Time: 0.00604s, Loss: 0.54753
        Epoch: 110, Time: 0.00465s, Loss: 0.53154
        Epoch: 111, Time: 0.00629s, Loss: 0.53460
        Epoch: 112, Time: 0.00568s, Loss: 0.52337
        Epoch: 113, Time: 0.00587s, Loss: 0.52842
        Epoch: 114, Time: 0.00562s, Loss: 0.50907
        Epoch: 115, Time: 0.00454s, Loss: 0.51616
        Epoch: 116, Time: 0.00561s, Loss: 0.50364
        Epoch: 117, Time: 0.00459s, Loss: 0.49458
        Epoch: 118, Time: 0.00545s, Loss: 0.49913
        Epoch: 119, Time: 0.00529s, Loss: 0.48824
        Epoch: 120, Time: 0.00519s, Loss: 0.52106
        Epoch: 121, Time: 0.00555s, Loss: 0.46541
        Epoch: 122, Time: 0.00459s, Loss: 0.47356
        Epoch: 123, Time: 0.00539s, Loss: 0.44043
        Epoch: 124, Time: 0.00468s, Loss: 0.44389
        Epoch: 125, Time: 0.00569s, Loss: 0.45298
        Epoch: 126, Time: 0.00500s, Loss: 0.46986
        Epoch: 127, Time: 0.00551s, Loss: 0.45141
        Epoch: 128, Time: 0.00533s, Loss: 0.48571
        Epoch: 129, Time: 0.00460s, Loss: 0.43895
        Epoch: 130, Time: 0.00600s, Loss: 0.44426
        Epoch: 131, Time: 0.00457s, Loss: 0.47401
        Epoch: 132, Time: 0.00579s, Loss: 0.46865
        Epoch: 133, Time: 0.00464s, Loss: 0.41215
        Epoch: 134, Time: 0.00528s, Loss: 0.42941
        Epoch: 135, Time: 0.00642s, Loss: 0.46532
        Epoch: 136, Time: 0.00538s, Loss: 0.42108
        Epoch: 137, Time: 0.00690s, Loss: 0.41919
        Epoch: 138, Time: 0.00617s, Loss: 0.44285
        Epoch: 139, Time: 0.00577s, Loss: 0.42653
        Epoch: 140, Time: 0.00548s, Loss: 0.45898
        Epoch: 141, Time: 0.00539s, Loss: 0.41800
        Epoch: 142, Time: 0.00467s, Loss: 0.40399
        Epoch: 143, Time: 0.00487s, Loss: 0.38347
        Epoch: 144, Time: 0.00509s, Loss: 0.42234
        Epoch: 145, Time: 0.00721s, Loss: 0.42908
        Epoch: 146, Time: 0.00489s, Loss: 0.37335
        Epoch: 147, Time: 0.00664s, Loss: 0.40119
        Epoch: 148, Time: 0.00465s, Loss: 0.38477
        Epoch: 149, Time: 0.00451s, Loss: 0.40037
        Epoch: 150, Time: 0.00553s, Loss: 0.40168
        Epoch: 151, Time: 0.00454s, Loss: 0.38555
        Epoch: 152, Time: 0.00729s, Loss: 0.40183
        Epoch: 153, Time: 0.00465s, Loss: 0.40610
        Epoch: 154, Time: 0.00669s, Loss: 0.39806
        Epoch: 155, Time: 0.00463s, Loss: 0.43478
        Epoch: 156, Time: 0.00641s, Loss: 0.37409
        Epoch: 157, Time: 0.00509s, Loss: 0.39802
        Epoch: 158, Time: 0.00453s, Loss: 0.34516
        Epoch: 159, Time: 0.00563s, Loss: 0.39663
        Epoch: 160, Time: 0.00456s, Loss: 0.37089
        Epoch: 161, Time: 0.00711s, Loss: 0.39547
        Epoch: 162, Time: 0.00455s, Loss: 0.41472
        Epoch: 163, Time: 0.00645s, Loss: 0.40523
        Epoch: 164, Time: 0.00465s, Loss: 0.33511
        Epoch: 165, Time: 0.00565s, Loss: 0.35864
        Epoch: 166, Time: 0.00575s, Loss: 0.33017
        Epoch: 167, Time: 0.00785s, Loss: 0.36668
        Epoch: 168, Time: 0.00604s, Loss: 0.36207
        Epoch: 169, Time: 0.00650s, Loss: 0.37902
        Epoch: 170, Time: 0.00473s, Loss: 0.38248
        Epoch: 171, Time: 0.00664s, Loss: 0.34953
        Epoch: 172, Time: 0.00556s, Loss: 0.38132
        Epoch: 173, Time: 0.00686s, Loss: 0.34698
        Epoch: 174, Time: 0.00619s, Loss: 0.36063
        Epoch: 175, Time: 0.00468s, Loss: 0.34594
        Epoch: 176, Time: 0.00545s, Loss: 0.37555
        Epoch: 177, Time: 0.00457s, Loss: 0.35946
        Epoch: 178, Time: 0.00718s, Loss: 0.35694
        Epoch: 179, Time: 0.00458s, Loss: 0.34922
        Epoch: 180, Time: 0.00693s, Loss: 0.30437
        Epoch: 181, Time: 0.00461s, Loss: 0.34730
        Epoch: 182, Time: 0.00632s, Loss: 0.31228
        Epoch: 183, Time: 0.00509s, Loss: 0.36002
        Epoch: 184, Time: 0.00454s, Loss: 0.36114
        Epoch: 185, Time: 0.00546s, Loss: 0.34812
        Epoch: 186, Time: 0.00456s, Loss: 0.33244
        Epoch: 187, Time: 0.00696s, Loss: 0.34411
        Epoch: 188, Time: 0.00459s, Loss: 0.35262
        Epoch: 189, Time: 0.00628s, Loss: 0.32643
        Epoch: 190, Time: 0.00472s, Loss: 0.32591
        Epoch: 191, Time: 0.00451s, Loss: 0.33036
        Epoch: 192, Time: 0.00594s, Loss: 0.31552
        Epoch: 193, Time: 0.00559s, Loss: 0.32376
        Epoch: 194, Time: 0.00627s, Loss: 0.31232
        Epoch: 195, Time: 0.00550s, Loss: 0.33725
        Epoch: 196, Time: 0.00570s, Loss: 0.34083
        Epoch: 197, Time: 0.00508s, Loss: 0.30638
        Epoch: 198, Time: 0.00559s, Loss: 0.33905
        Epoch: 199, Time: 0.00603s, Loss: 0.30302

        train finished!
        best val: 0.80200
        test...
        final result: epoch: 92
        {'accuracy': 0.8270000219345093, 'f1_score': 0.8198394539104813, 'f1_score -> average@micro': 0.827}
