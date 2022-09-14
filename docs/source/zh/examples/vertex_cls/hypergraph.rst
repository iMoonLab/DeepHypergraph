超图
==========================================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

在如下的例子中，我们使用三种典型图/超图神经网络在超图关联结构中执行节点分类任务。

模型
---------------------------

- GCN (:py:class:`dhg.models.GCN`), `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ 论文 (ICLR 2017).
- HGNN (:py:class:`dhg.models.HGNN`), `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ 论文 (AAAI 2019).
- HGNN+ (:py:class:`dhg.models.HGNNP`), `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ 论文 (IEEE T-PAMI 2022).

数据集
---------------------------

Cooking 200 数据集 (:py:class:`dhg.data.Cooking200`) 从 `Yummly.com <https://www.yummly.com/>`_ 收集并用于节点分类任务。
其为超图数据集，顶点代表菜式，超边代表配料。
每种菜式与一种分类信息关联，为该菜式的菜系（中餐、日本菜、法国菜、俄罗斯菜）。

.. note:: 

    数据集为超图数据集，不能直接用于GCN模型。因此，使用 ``clique expansion`` 将超图转为图。

.. note:: 

    数据集不包含顶点特征，因此我们生成一个单位矩阵代表顶点特征矩阵。

.. warning:: 

    生成单位矩阵作为顶点特征会导致训练阶段参数不稳定。因此，batch_norm会用于以下示例中的GCN、HGNN和HGNN+模型中。


结果汇总
----------------

========    ======================  ======================  ======================
模型         验证集的Accuracy         测试集的Accuracy          测试集F1 score
========    ======================  ======================  ======================
GCN         0.500                   0.434                   0.356
HGNN        0.485                   0.495                   0.376
HGNN+       0.475                   0.520                   0.391
========    ======================  ======================  ======================


Cooking200上使用GCN
---------------------------

导入依赖包
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from copy import deepcopy

    import torch
    import torch.optim as optim
    import torch.nn.functional as F

    from dhg import Graph, Hypergraph
    from dhg.data import Cooking200
    from dhg.models import GCN
    from dhg.random import set_seed
    from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator


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
^^^^^^^^^

.. note:: 

    更多关于评测器 ``Evaluator`` 的细节可以参照 :doc:`构建指标评测器 </zh/tutorial/metric>` 章节。

.. code-block:: python


    if __name__ == "__main__":
        set_seed(2021)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
        data = Cooking200()

        X, lbl = torch.eye(data["num_vertices"]), data["labels"]
        ft_dim = X.shape[1]
        HG = Hypergraph(data["num_vertices"], data["edge_list"])
        G = Graph.from_hypergraph_clique(HG, weighted=True)
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]

        net = GCN(ft_dim, 32, data["num_classes"], use_bn=True)
        optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

        X, lbl = X.to(device), lbl.to(device)
        G = G.to(device)
        net = net.to(device)

        best_state = None
        best_epoch, best_val = 0, 0
        for epoch in range(200):
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

        Epoch: 0, Time: 7.29884s, Loss: 3.02374
        update best: 0.05000
        Epoch: 1, Time: 0.02545s, Loss: 2.47223
        Epoch: 2, Time: 0.02411s, Loss: 2.41279
        update best: 0.05500
        Epoch: 3, Time: 0.02656s, Loss: 2.36803
        update best: 0.07500
        Epoch: 4, Time: 0.02486s, Loss: 2.33794
        Epoch: 5, Time: 0.02224s, Loss: 2.30590
        Epoch: 6, Time: 0.02089s, Loss: 2.28631
        Epoch: 7, Time: 0.02136s, Loss: 2.25775
        Epoch: 8, Time: 0.02186s, Loss: 2.24081
        update best: 0.08000
        Epoch: 9, Time: 0.02203s, Loss: 2.22660
        update best: 0.09500
        Epoch: 10, Time: 0.02155s, Loss: 2.20722
        update best: 0.14500
        Epoch: 11, Time: 0.02141s, Loss: 2.19497
        Epoch: 12, Time: 0.02263s, Loss: 2.17880
        Epoch: 13, Time: 0.02199s, Loss: 2.16433
        Epoch: 14, Time: 0.02258s, Loss: 2.15038
        Epoch: 15, Time: 0.02230s, Loss: 2.13811
        Epoch: 16, Time: 0.02135s, Loss: 2.12440
        Epoch: 17, Time: 0.02217s, Loss: 2.11146
        Epoch: 18, Time: 0.02183s, Loss: 2.10333
        Epoch: 19, Time: 0.03591s, Loss: 2.09031
        Epoch: 20, Time: 0.02081s, Loss: 2.07710
        Epoch: 21, Time: 0.02111s, Loss: 2.06423
        Epoch: 22, Time: 0.02114s, Loss: 2.05410
        Epoch: 23, Time: 0.02137s, Loss: 2.04545
        update best: 0.15500
        Epoch: 24, Time: 0.02159s, Loss: 2.03412
        update best: 0.16000
        Epoch: 25, Time: 0.02189s, Loss: 2.01589
        update best: 0.17500
        Epoch: 26, Time: 0.02204s, Loss: 2.01508
        Epoch: 27, Time: 0.02206s, Loss: 1.99630
        Epoch: 28, Time: 0.02180s, Loss: 1.98635
        update best: 0.18500
        Epoch: 29, Time: 0.02168s, Loss: 1.97526
        update best: 0.20000
        Epoch: 30, Time: 0.02155s, Loss: 1.96057
        update best: 0.21000
        Epoch: 31, Time: 0.02147s, Loss: 1.95878
        update best: 0.21500
        Epoch: 32, Time: 0.02174s, Loss: 1.94054
        Epoch: 33, Time: 0.02147s, Loss: 1.93238
        Epoch: 34, Time: 0.02176s, Loss: 1.92268
        update best: 0.23000
        Epoch: 35, Time: 0.02169s, Loss: 1.91224
        update best: 0.24000
        Epoch: 36, Time: 0.02141s, Loss: 1.89593
        update best: 0.25000
        Epoch: 37, Time: 0.02133s, Loss: 1.89175
        update best: 0.25500
        Epoch: 38, Time: 0.02230s, Loss: 1.88137
        Epoch: 39, Time: 0.02201s, Loss: 1.87121
        Epoch: 40, Time: 0.02050s, Loss: 1.85513
        Epoch: 41, Time: 0.02120s, Loss: 1.85149
        Epoch: 42, Time: 0.02102s, Loss: 1.83702
        update best: 0.27000
        Epoch: 43, Time: 0.02095s, Loss: 1.82509
        update best: 0.27500
        Epoch: 44, Time: 0.02139s, Loss: 1.81752
        update best: 0.29000
        Epoch: 45, Time: 0.02115s, Loss: 1.80817
        Epoch: 46, Time: 0.02119s, Loss: 1.79938
        update best: 0.29500
        Epoch: 47, Time: 0.02088s, Loss: 1.78561
        update best: 0.33000
        Epoch: 48, Time: 0.02106s, Loss: 1.78137
        update best: 0.34000
        Epoch: 49, Time: 0.02088s, Loss: 1.76117
        update best: 0.34500
        Epoch: 50, Time: 0.02143s, Loss: 1.75598
        update best: 0.36000
        Epoch: 51, Time: 0.02129s, Loss: 1.74965
        Epoch: 52, Time: 0.02177s, Loss: 1.73695
        Epoch: 53, Time: 0.02160s, Loss: 1.72132
        update best: 0.36500
        Epoch: 54, Time: 0.02177s, Loss: 1.71943
        update best: 0.37000
        Epoch: 55, Time: 0.02115s, Loss: 1.71475
        update best: 0.37500
        Epoch: 56, Time: 0.02157s, Loss: 1.69237
        update best: 0.38500
        Epoch: 57, Time: 0.02164s, Loss: 1.68571
        update best: 0.39500
        Epoch: 58, Time: 0.02150s, Loss: 1.67695
        update best: 0.40000
        Epoch: 59, Time: 0.02156s, Loss: 1.66385
        Epoch: 60, Time: 0.02155s, Loss: 1.65498
        Epoch: 61, Time: 0.02102s, Loss: 1.65138
        update best: 0.41000
        Epoch: 62, Time: 0.02167s, Loss: 1.63215
        update best: 0.42000
        Epoch: 63, Time: 0.02174s, Loss: 1.62920
        update best: 0.43500
        Epoch: 64, Time: 0.02154s, Loss: 1.61913
        update best: 0.44000
        Epoch: 65, Time: 0.02159s, Loss: 1.61141
        Epoch: 66, Time: 0.02195s, Loss: 1.60337
        Epoch: 67, Time: 0.02069s, Loss: 1.58908
        update best: 0.45500
        Epoch: 68, Time: 0.02115s, Loss: 1.57248
        Epoch: 69, Time: 0.02138s, Loss: 1.57386
        update best: 0.46500
        Epoch: 70, Time: 0.02106s, Loss: 1.56231
        Epoch: 71, Time: 0.02118s, Loss: 1.55329
        Epoch: 72, Time: 0.02242s, Loss: 1.54713
        Epoch: 73, Time: 0.02136s, Loss: 1.53178
        Epoch: 74, Time: 0.02172s, Loss: 1.52513
        Epoch: 75, Time: 0.02200s, Loss: 1.51584
        Epoch: 76, Time: 0.02123s, Loss: 1.50966
        update best: 0.47000
        Epoch: 77, Time: 0.02147s, Loss: 1.50546
        update best: 0.47500
        Epoch: 78, Time: 0.02270s, Loss: 1.49482
        Epoch: 79, Time: 0.02264s, Loss: 1.47653
        Epoch: 80, Time: 0.02349s, Loss: 1.46740
        Epoch: 81, Time: 0.02231s, Loss: 1.46205
        Epoch: 82, Time: 0.02251s, Loss: 1.44632
        Epoch: 83, Time: 0.02184s, Loss: 1.44394
        Epoch: 84, Time: 0.02175s, Loss: 1.43398
        Epoch: 85, Time: 0.02109s, Loss: 1.43450
        Epoch: 86, Time: 0.02110s, Loss: 1.41855
        Epoch: 87, Time: 0.02112s, Loss: 1.41488
        Epoch: 88, Time: 0.02119s, Loss: 1.40113
        Epoch: 89, Time: 0.02133s, Loss: 1.38627
        Epoch: 90, Time: 0.02178s, Loss: 1.38061
        Epoch: 91, Time: 0.02106s, Loss: 1.38012
        Epoch: 92, Time: 0.02245s, Loss: 1.36612
        Epoch: 93, Time: 0.02165s, Loss: 1.36384
        Epoch: 94, Time: 0.02169s, Loss: 1.35315
        Epoch: 95, Time: 0.02287s, Loss: 1.33591
        Epoch: 96, Time: 0.02321s, Loss: 1.33441
        Epoch: 97, Time: 0.02267s, Loss: 1.32461
        Epoch: 98, Time: 0.02246s, Loss: 1.31650
        Epoch: 99, Time: 0.02192s, Loss: 1.30920
        Epoch: 100, Time: 0.02145s, Loss: 1.29616
        Epoch: 101, Time: 0.02106s, Loss: 1.28773
        Epoch: 102, Time: 0.02128s, Loss: 1.28913
        Epoch: 103, Time: 0.02125s, Loss: 1.27793
        Epoch: 104, Time: 0.02174s, Loss: 1.27127
        Epoch: 105, Time: 0.02135s, Loss: 1.26090
        Epoch: 106, Time: 0.02187s, Loss: 1.25673
        Epoch: 107, Time: 0.02137s, Loss: 1.23971
        Epoch: 108, Time: 0.02163s, Loss: 1.23427
        Epoch: 109, Time: 0.02173s, Loss: 1.23829
        Epoch: 110, Time: 0.02228s, Loss: 1.21614
        Epoch: 111, Time: 0.02190s, Loss: 1.22033
        Epoch: 112, Time: 0.02146s, Loss: 1.21155
        update best: 0.48000
        Epoch: 113, Time: 0.02183s, Loss: 1.19760
        Epoch: 114, Time: 0.02472s, Loss: 1.20577
        Epoch: 115, Time: 0.02249s, Loss: 1.18268
        Epoch: 116, Time: 0.02274s, Loss: 1.17723
        Epoch: 117, Time: 0.02290s, Loss: 1.16582
        Epoch: 118, Time: 0.02262s, Loss: 1.16943
        Epoch: 119, Time: 0.02180s, Loss: 1.16023
        Epoch: 120, Time: 0.02193s, Loss: 1.14612
        update best: 0.48500
        Epoch: 121, Time: 0.02191s, Loss: 1.14254
        Epoch: 122, Time: 0.02162s, Loss: 1.13199
        Epoch: 123, Time: 0.02136s, Loss: 1.12077
        Epoch: 124, Time: 0.02165s, Loss: 1.11500
        Epoch: 125, Time: 0.02177s, Loss: 1.11730
        Epoch: 126, Time: 0.02150s, Loss: 1.10626
        Epoch: 127, Time: 0.02119s, Loss: 1.09788
        Epoch: 128, Time: 0.02119s, Loss: 1.09148
        Epoch: 129, Time: 0.02130s, Loss: 1.08841
        Epoch: 130, Time: 0.02211s, Loss: 1.08878
        Epoch: 131, Time: 0.02171s, Loss: 1.08039
        Epoch: 132, Time: 0.02172s, Loss: 1.06337
        Epoch: 133, Time: 0.02185s, Loss: 1.05798
        Epoch: 134, Time: 0.02197s, Loss: 1.05995
        Epoch: 135, Time: 0.02310s, Loss: 1.04716
        Epoch: 136, Time: 0.02271s, Loss: 1.03834
        update best: 0.49000
        Epoch: 137, Time: 0.02218s, Loss: 1.03407
        Epoch: 138, Time: 0.02329s, Loss: 1.02641
        Epoch: 139, Time: 0.02310s, Loss: 1.02540
        Epoch: 140, Time: 0.02245s, Loss: 1.02152
        Epoch: 141, Time: 0.02171s, Loss: 1.01990
        Epoch: 142, Time: 0.02151s, Loss: 1.00520
        Epoch: 143, Time: 0.02128s, Loss: 1.01225
        Epoch: 144, Time: 0.02179s, Loss: 1.00302
        Epoch: 145, Time: 0.02164s, Loss: 0.98153
        Epoch: 146, Time: 0.02117s, Loss: 0.97740
        Epoch: 147, Time: 0.02110s, Loss: 0.97149
        Epoch: 148, Time: 0.02131s, Loss: 0.97149
        Epoch: 149, Time: 0.02128s, Loss: 0.97657
        Epoch: 150, Time: 0.02155s, Loss: 0.95241
        Epoch: 151, Time: 0.02171s, Loss: 0.96010
        Epoch: 152, Time: 0.02174s, Loss: 0.94509
        Epoch: 153, Time: 0.02167s, Loss: 0.94987
        Epoch: 154, Time: 0.02262s, Loss: 0.94258
        Epoch: 155, Time: 0.02226s, Loss: 0.93526
        Epoch: 156, Time: 0.02236s, Loss: 0.93201
        Epoch: 157, Time: 0.02148s, Loss: 0.92291
        Epoch: 158, Time: 0.02158s, Loss: 0.93494
        Epoch: 159, Time: 0.02159s, Loss: 0.91413
        Epoch: 160, Time: 0.02150s, Loss: 0.91853
        Epoch: 161, Time: 0.02143s, Loss: 0.90566
        Epoch: 162, Time: 0.02117s, Loss: 0.90713
        Epoch: 163, Time: 0.02124s, Loss: 0.89651
        Epoch: 164, Time: 0.02103s, Loss: 0.89034
        Epoch: 165, Time: 0.02168s, Loss: 0.88661
        Epoch: 166, Time: 0.02163s, Loss: 0.88348
        Epoch: 167, Time: 0.02174s, Loss: 0.87290
        Epoch: 168, Time: 0.02185s, Loss: 0.87435
        Epoch: 169, Time: 0.02155s, Loss: 0.86458
        Epoch: 170, Time: 0.02088s, Loss: 0.87389
        Epoch: 171, Time: 0.02264s, Loss: 0.86114
        Epoch: 172, Time: 0.02286s, Loss: 0.84979
        Epoch: 173, Time: 0.02272s, Loss: 0.85025
        Epoch: 174, Time: 0.02237s, Loss: 0.85343
        Epoch: 175, Time: 0.02243s, Loss: 0.84297
        Epoch: 176, Time: 0.02235s, Loss: 0.84274
        Epoch: 177, Time: 0.02185s, Loss: 0.83616
        Epoch: 178, Time: 0.02188s, Loss: 0.83237
        Epoch: 179, Time: 0.02110s, Loss: 0.83829
        Epoch: 180, Time: 0.02102s, Loss: 0.83292
        Epoch: 181, Time: 0.02157s, Loss: 0.82355
        Epoch: 182, Time: 0.02148s, Loss: 0.82146
        Epoch: 183, Time: 0.02148s, Loss: 0.82488
        Epoch: 184, Time: 0.02128s, Loss: 0.81608
        Epoch: 185, Time: 0.02128s, Loss: 0.81082
        Epoch: 186, Time: 0.02121s, Loss: 0.81338
        Epoch: 187, Time: 0.02183s, Loss: 0.81301
        Epoch: 188, Time: 0.02234s, Loss: 0.79188
        Epoch: 189, Time: 0.02182s, Loss: 0.79709
        update best: 0.50000
        Epoch: 190, Time: 0.02134s, Loss: 0.78706
        Epoch: 191, Time: 0.02183s, Loss: 0.77257
        Epoch: 192, Time: 0.02276s, Loss: 0.77896
        Epoch: 193, Time: 0.02326s, Loss: 0.77773
        Epoch: 194, Time: 0.02287s, Loss: 0.76515
        Epoch: 195, Time: 0.02281s, Loss: 0.76747
        Epoch: 196, Time: 0.02164s, Loss: 0.76833
        Epoch: 197, Time: 0.02182s, Loss: 0.75029
        Epoch: 198, Time: 0.02136s, Loss: 0.76452
        Epoch: 199, Time: 0.02135s, Loss: 0.75916

        train finished!
        best val: 0.50000
        test...
        final result: epoch: 189
        {'accuracy': 0.4340996742248535, 'f1_score': 0.35630662515488015, 'f1_score -> average@micro': 0.43409967156932744}

Cooking200上使用HGNN
---------------------------

导入依赖包
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from copy import deepcopy

    import torch
    import torch.optim as optim
    import torch.nn.functional as F

    from dhg import Hypergraph
    from dhg.data import Cooking200
    from dhg.models import HGNN
    from dhg.random import set_seed
    from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator


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
^^^^^^^^^

.. note:: 

    更多关于评测器 ``Evaluator`` 的细节可以参照 :doc:`构建指标评测器 </zh/tutorial/metric>` 章节。

.. code-block:: python

    if __name__ == "__main__":
        set_seed(2021)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
        data = Cooking200()

        X, lbl = torch.eye(data["num_vertices"]), data["labels"]
        G = Hypergraph(data["num_vertices"], data["edge_list"])
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]

        net = HGNN(X.shape[1], 32, data["num_classes"], use_bn=True)
        optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

        X, lbl = X.to(device), lbl.to(device)
        G = G.to(device)
        net = net.to(device)

        best_state = None
        best_epoch, best_val = 0, 0
        for epoch in range(200):
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

        Epoch: 0, Time: 0.57807s, Loss: 2.99290
        update best: 0.10000
        Epoch: 1, Time: 0.02624s, Loss: 2.28624
        Epoch: 2, Time: 0.02707s, Loss: 2.15988
        Epoch: 3, Time: 0.02373s, Loss: 2.05894
        Epoch: 4, Time: 0.02545s, Loss: 1.99918
        Epoch: 5, Time: 0.02619s, Loss: 1.92948
        Epoch: 6, Time: 0.02215s, Loss: 1.88097
        Epoch: 7, Time: 0.02229s, Loss: 1.83393
        Epoch: 8, Time: 0.02181s, Loss: 1.79070
        Epoch: 9, Time: 0.02256s, Loss: 1.75345
        Epoch: 10, Time: 0.02264s, Loss: 1.70969
        Epoch: 11, Time: 0.02248s, Loss: 1.68242
        Epoch: 12, Time: 0.02248s, Loss: 1.64419
        Epoch: 13, Time: 0.02257s, Loss: 1.60876
        Epoch: 14, Time: 0.02238s, Loss: 1.58108
        Epoch: 15, Time: 0.02194s, Loss: 1.54466
        Epoch: 16, Time: 0.02172s, Loss: 1.52140
        Epoch: 17, Time: 0.02130s, Loss: 1.48225
        Epoch: 18, Time: 0.02156s, Loss: 1.46237
        Epoch: 19, Time: 0.02133s, Loss: 1.43527
        Epoch: 20, Time: 0.02148s, Loss: 1.40451
        Epoch: 21, Time: 0.02133s, Loss: 1.39555
        Epoch: 22, Time: 0.02182s, Loss: 1.36368
        Epoch: 23, Time: 0.02151s, Loss: 1.33732
        Epoch: 24, Time: 0.02178s, Loss: 1.32686
        Epoch: 25, Time: 0.02232s, Loss: 1.30681
        Epoch: 26, Time: 0.02289s, Loss: 1.28287
        Epoch: 27, Time: 0.02245s, Loss: 1.28563
        Epoch: 28, Time: 0.02210s, Loss: 1.24644
        Epoch: 29, Time: 0.02195s, Loss: 1.22813
        Epoch: 30, Time: 0.02205s, Loss: 1.20336
        Epoch: 31, Time: 0.02245s, Loss: 1.20308
        Epoch: 32, Time: 0.02129s, Loss: 1.16802
        Epoch: 33, Time: 0.02144s, Loss: 1.17182
        Epoch: 34, Time: 0.02215s, Loss: 1.14047
        Epoch: 35, Time: 0.02195s, Loss: 1.13377
        Epoch: 36, Time: 0.02233s, Loss: 1.09250
        Epoch: 37, Time: 0.02283s, Loss: 1.09588
        Epoch: 38, Time: 0.02356s, Loss: 1.09042
        Epoch: 39, Time: 0.02211s, Loss: 1.08532
        Epoch: 40, Time: 0.02340s, Loss: 1.04074
        update best: 0.11000
        Epoch: 41, Time: 0.02125s, Loss: 1.05056
        update best: 0.13500
        Epoch: 42, Time: 0.02302s, Loss: 1.02834
        update best: 0.14000
        Epoch: 43, Time: 0.02278s, Loss: 0.99903
        update best: 0.14500
        Epoch: 44, Time: 0.02238s, Loss: 1.01756
        update best: 0.15000
        Epoch: 45, Time: 0.02286s, Loss: 0.99652
        update best: 0.17500
        Epoch: 46, Time: 0.02251s, Loss: 0.97935
        update best: 0.21500
        Epoch: 47, Time: 0.02234s, Loss: 0.97873
        update best: 0.24500
        Epoch: 48, Time: 0.02245s, Loss: 0.95888
        update best: 0.26000
        Epoch: 49, Time: 0.02228s, Loss: 0.95761
        update best: 0.28000
        Epoch: 50, Time: 0.02254s, Loss: 0.94229
        Epoch: 51, Time: 0.02264s, Loss: 0.92833
        update best: 0.29000
        Epoch: 52, Time: 0.02238s, Loss: 0.92601
        update best: 0.30000
        Epoch: 53, Time: 0.02311s, Loss: 0.90252
        update best: 0.31000
        Epoch: 54, Time: 0.02189s, Loss: 0.89501
        update best: 0.32500
        Epoch: 55, Time: 0.02193s, Loss: 0.89724
        Epoch: 56, Time: 0.02246s, Loss: 0.87068
        update best: 0.33500
        Epoch: 57, Time: 0.02181s, Loss: 0.87531
        update best: 0.34000
        Epoch: 58, Time: 0.02287s, Loss: 0.84288
        update best: 0.34500
        Epoch: 59, Time: 0.02227s, Loss: 0.84243
        update best: 0.36500
        Epoch: 60, Time: 0.02149s, Loss: 0.83892
        update best: 0.38500
        Epoch: 61, Time: 0.02253s, Loss: 0.83062
        update best: 0.40000
        Epoch: 62, Time: 0.02271s, Loss: 0.82245
        update best: 0.42000
        Epoch: 63, Time: 0.02195s, Loss: 0.81214
        update best: 0.43000
        Epoch: 64, Time: 0.02162s, Loss: 0.80847
        update best: 0.44000
        Epoch: 65, Time: 0.02136s, Loss: 0.78325
        Epoch: 66, Time: 0.02245s, Loss: 0.79052
        update best: 0.45500
        Epoch: 67, Time: 0.02248s, Loss: 0.78128
        Epoch: 68, Time: 0.02295s, Loss: 0.77049
        Epoch: 69, Time: 0.02315s, Loss: 0.75469
        Epoch: 70, Time: 0.02331s, Loss: 0.74771
        Epoch: 71, Time: 0.02317s, Loss: 0.73701
        Epoch: 72, Time: 0.02307s, Loss: 0.74350
        Epoch: 73, Time: 0.02176s, Loss: 0.73698
        Epoch: 74, Time: 0.02164s, Loss: 0.72565
        Epoch: 75, Time: 0.02148s, Loss: 0.70553
        update best: 0.46500
        Epoch: 76, Time: 0.02136s, Loss: 0.71696
        Epoch: 77, Time: 0.02111s, Loss: 0.72410
        Epoch: 78, Time: 0.02111s, Loss: 0.71131
        update best: 0.47000
        Epoch: 79, Time: 0.02180s, Loss: 0.68748
        Epoch: 80, Time: 0.02095s, Loss: 0.68774
        Epoch: 81, Time: 0.02147s, Loss: 0.70136
        Epoch: 82, Time: 0.02122s, Loss: 0.66882
        Epoch: 83, Time: 0.02164s, Loss: 0.64563
        Epoch: 84, Time: 0.02149s, Loss: 0.66794
        Epoch: 85, Time: 0.02194s, Loss: 0.65860
        Epoch: 86, Time: 0.02157s, Loss: 0.66000
        Epoch: 87, Time: 0.02267s, Loss: 0.65452
        Epoch: 88, Time: 0.02250s, Loss: 0.64512
        Epoch: 89, Time: 0.02169s, Loss: 0.64318
        Epoch: 90, Time: 0.02175s, Loss: 0.63814
        Epoch: 91, Time: 0.02177s, Loss: 0.62040
        Epoch: 92, Time: 0.02108s, Loss: 0.61942
        Epoch: 93, Time: 0.02111s, Loss: 0.61757
        Epoch: 94, Time: 0.02118s, Loss: 0.60520
        Epoch: 95, Time: 0.02112s, Loss: 0.58358
        Epoch: 96, Time: 0.02129s, Loss: 0.58866
        Epoch: 97, Time: 0.02171s, Loss: 0.58599
        Epoch: 98, Time: 0.02220s, Loss: 0.59330
        Epoch: 99, Time: 0.02243s, Loss: 0.56555
        Epoch: 100, Time: 0.02262s, Loss: 0.57273
        Epoch: 101, Time: 0.02240s, Loss: 0.57785
        Epoch: 102, Time: 0.02086s, Loss: 0.56949
        Epoch: 103, Time: 0.02111s, Loss: 0.55187
        Epoch: 104, Time: 0.02136s, Loss: 0.55166
        Epoch: 105, Time: 0.02119s, Loss: 0.54706
        Epoch: 106, Time: 0.02107s, Loss: 0.55239
        Epoch: 107, Time: 0.02136s, Loss: 0.53656
        Epoch: 108, Time: 0.02115s, Loss: 0.53478
        Epoch: 109, Time: 0.02146s, Loss: 0.52564
        Epoch: 110, Time: 0.02189s, Loss: 0.52242
        Epoch: 111, Time: 0.02248s, Loss: 0.52779
        Epoch: 112, Time: 0.02191s, Loss: 0.50813
        Epoch: 113, Time: 0.02182s, Loss: 0.51623
        Epoch: 114, Time: 0.02143s, Loss: 0.51834
        Epoch: 115, Time: 0.02220s, Loss: 0.49232
        Epoch: 116, Time: 0.02117s, Loss: 0.51582
        Epoch: 117, Time: 0.02116s, Loss: 0.49434
        Epoch: 118, Time: 0.02110s, Loss: 0.49518
        Epoch: 119, Time: 0.02147s, Loss: 0.49155
        Epoch: 120, Time: 0.02122s, Loss: 0.48029
        Epoch: 121, Time: 0.02153s, Loss: 0.49079
        Epoch: 122, Time: 0.02151s, Loss: 0.48253
        Epoch: 123, Time: 0.02170s, Loss: 0.46945
        Epoch: 124, Time: 0.02259s, Loss: 0.47764
        Epoch: 125, Time: 0.02228s, Loss: 0.47102
        Epoch: 126, Time: 0.02196s, Loss: 0.45784
        Epoch: 127, Time: 0.02184s, Loss: 0.46020
        Epoch: 128, Time: 0.02245s, Loss: 0.45922
        Epoch: 129, Time: 0.02191s, Loss: 0.46458
        Epoch: 130, Time: 0.02215s, Loss: 0.46924
        Epoch: 131, Time: 0.02222s, Loss: 0.45952
        Epoch: 132, Time: 0.02226s, Loss: 0.44490
        Epoch: 133, Time: 0.02174s, Loss: 0.44763
        Epoch: 134, Time: 0.02143s, Loss: 0.45225
        Epoch: 135, Time: 0.02149s, Loss: 0.42556
        Epoch: 136, Time: 0.02141s, Loss: 0.42714
        Epoch: 137, Time: 0.02150s, Loss: 0.43604
        Epoch: 138, Time: 0.02171s, Loss: 0.42259
        Epoch: 139, Time: 0.02168s, Loss: 0.41784
        Epoch: 140, Time: 0.02149s, Loss: 0.41759
        Epoch: 141, Time: 0.02125s, Loss: 0.41633
        Epoch: 142, Time: 0.02220s, Loss: 0.42547
        Epoch: 143, Time: 0.02271s, Loss: 0.41790
        Epoch: 144, Time: 0.02280s, Loss: 0.39776
        Epoch: 145, Time: 0.02264s, Loss: 0.41429
        Epoch: 146, Time: 0.02128s, Loss: 0.39543
        Epoch: 147, Time: 0.02141s, Loss: 0.39529
        Epoch: 148, Time: 0.02100s, Loss: 0.41145
        Epoch: 149, Time: 0.02103s, Loss: 0.40083
        Epoch: 150, Time: 0.02170s, Loss: 0.39246
        Epoch: 151, Time: 0.02154s, Loss: 0.39613
        Epoch: 152, Time: 0.02188s, Loss: 0.38080
        Epoch: 153, Time: 0.02213s, Loss: 0.39159
        Epoch: 154, Time: 0.02236s, Loss: 0.38570
        Epoch: 155, Time: 0.02209s, Loss: 0.38382
        Epoch: 156, Time: 0.02146s, Loss: 0.37949
        update best: 0.47500
        Epoch: 157, Time: 0.02179s, Loss: 0.37078
        Epoch: 158, Time: 0.02223s, Loss: 0.37063
        Epoch: 159, Time: 0.02219s, Loss: 0.37556
        Epoch: 160, Time: 0.02217s, Loss: 0.37468
        Epoch: 161, Time: 0.02146s, Loss: 0.38581
        update best: 0.48500
        Epoch: 162, Time: 0.02278s, Loss: 0.36664
        Epoch: 163, Time: 0.02172s, Loss: 0.35075
        Epoch: 164, Time: 0.02139s, Loss: 0.35056
        Epoch: 165, Time: 0.02156s, Loss: 0.36339
        Epoch: 166, Time: 0.02149s, Loss: 0.36245
        Epoch: 167, Time: 0.02133s, Loss: 0.34675
        Epoch: 168, Time: 0.02141s, Loss: 0.36043
        Epoch: 169, Time: 0.02148s, Loss: 0.34538
        Epoch: 170, Time: 0.02128s, Loss: 0.34694
        Epoch: 171, Time: 0.02138s, Loss: 0.33723
        Epoch: 172, Time: 0.02260s, Loss: 0.34017
        Epoch: 173, Time: 0.02259s, Loss: 0.33932
        Epoch: 174, Time: 0.02307s, Loss: 0.33170
        Epoch: 175, Time: 0.02290s, Loss: 0.31819
        Epoch: 176, Time: 0.02261s, Loss: 0.33577
        Epoch: 177, Time: 0.02269s, Loss: 0.34146
        Epoch: 178, Time: 0.02284s, Loss: 0.33086
        Epoch: 179, Time: 0.02215s, Loss: 0.34498
        Epoch: 180, Time: 0.02317s, Loss: 0.33026
        Epoch: 181, Time: 0.02228s, Loss: 0.32811
        Epoch: 182, Time: 0.02216s, Loss: 0.33203
        Epoch: 183, Time: 0.02248s, Loss: 0.31955
        Epoch: 184, Time: 0.02239s, Loss: 0.34238
        Epoch: 185, Time: 0.02253s, Loss: 0.30963
        Epoch: 186, Time: 0.02240s, Loss: 0.31527
        Epoch: 187, Time: 0.02199s, Loss: 0.31484
        Epoch: 188, Time: 0.02200s, Loss: 0.32514
        Epoch: 189, Time: 0.02171s, Loss: 0.32029
        Epoch: 190, Time: 0.02169s, Loss: 0.32122
        Epoch: 191, Time: 0.02157s, Loss: 0.30233
        Epoch: 192, Time: 0.02125s, Loss: 0.30417
        Epoch: 193, Time: 0.02159s, Loss: 0.30060
        Epoch: 194, Time: 0.02142s, Loss: 0.29333
        Epoch: 195, Time: 0.02155s, Loss: 0.29596
        Epoch: 196, Time: 0.02158s, Loss: 0.30458
        Epoch: 197, Time: 0.02204s, Loss: 0.29744
        Epoch: 198, Time: 0.02227s, Loss: 0.29473
        Epoch: 199, Time: 0.02259s, Loss: 0.30488

        train finished!
        best val: 0.48500
        test...
        final result: epoch: 161
        {'accuracy': 0.4949307441711426, 'f1_score': 0.37618299381063885, 'f1_score -> average@micro': 0.49493074396687137}


Cooking200上使用HGNN+
---------------------------

导入依赖包
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from copy import deepcopy

    import torch
    import torch.optim as optim
    import torch.nn.functional as F

    from dhg import Hypergraph
    from dhg.data import Cooking200
    from dhg.models import HGNN, HGNNP
    from dhg.random import set_seed
    from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator


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
^^^^^^^^^

.. note:: 

    更多关于评测器 ``Evaluator`` 的细节可以参照 :doc:`构建指标评测器 </zh/tutorial/metric>` 章节。

.. code-block:: python

    if __name__ == "__main__":
        set_seed(2021)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
        data = Cooking200()

        X, lbl = torch.eye(data["num_vertices"]), data["labels"]
        G = Hypergraph(data["num_vertices"], data["edge_list"])
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]

        net = HGNNP(X.shape[1], 32, data["num_classes"], use_bn=True)
        optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

        X, lbl = X.to(device), lbl.to(device)
        G = G.to(device)
        net = net.to(device)

        best_state = None
        best_epoch, best_val = 0, 0
        for epoch in range(200):
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

        Epoch: 0, Time: 0.52802s, Loss: 2.98654
        update best: 0.05000
        Epoch: 1, Time: 0.00738s, Loss: 2.28235
        Epoch: 2, Time: 0.00829s, Loss: 2.15288
        Epoch: 3, Time: 0.00929s, Loss: 2.05343
        Epoch: 4, Time: 0.00716s, Loss: 1.99081
        Epoch: 5, Time: 0.00703s, Loss: 1.92390
        Epoch: 6, Time: 0.01025s, Loss: 1.87569
        Epoch: 7, Time: 0.01015s, Loss: 1.83000
        Epoch: 8, Time: 0.00870s, Loss: 1.78668
        update best: 0.06500
        Epoch: 9, Time: 0.00811s, Loss: 1.75019
        Epoch: 10, Time: 0.00792s, Loss: 1.70593
        Epoch: 11, Time: 0.00855s, Loss: 1.68245
        Epoch: 12, Time: 0.00940s, Loss: 1.64045
        Epoch: 13, Time: 0.00667s, Loss: 1.60735
        Epoch: 14, Time: 0.00808s, Loss: 1.58477
        Epoch: 15, Time: 0.00863s, Loss: 1.54530
        Epoch: 16, Time: 0.00839s, Loss: 1.52168
        Epoch: 17, Time: 0.00863s, Loss: 1.48935
        Epoch: 18, Time: 0.01009s, Loss: 1.46205
        Epoch: 19, Time: 0.00998s, Loss: 1.43605
        Epoch: 20, Time: 0.00808s, Loss: 1.40635
        Epoch: 21, Time: 0.00765s, Loss: 1.39397
        Epoch: 22, Time: 0.00749s, Loss: 1.36317
        Epoch: 23, Time: 0.00791s, Loss: 1.34086
        Epoch: 24, Time: 0.00627s, Loss: 1.32558
        Epoch: 25, Time: 0.00784s, Loss: 1.30849
        Epoch: 26, Time: 0.00752s, Loss: 1.27822
        Epoch: 27, Time: 0.00628s, Loss: 1.28945
        Epoch: 28, Time: 0.00731s, Loss: 1.24414
        Epoch: 29, Time: 0.00741s, Loss: 1.22858
        Epoch: 30, Time: 0.00677s, Loss: 1.20161
        Epoch: 31, Time: 0.00777s, Loss: 1.19882
        Epoch: 32, Time: 0.00707s, Loss: 1.16460
        Epoch: 33, Time: 0.00730s, Loss: 1.16780
        Epoch: 34, Time: 0.00787s, Loss: 1.13391
        update best: 0.07000
        Epoch: 35, Time: 0.00747s, Loss: 1.13935
        update best: 0.08500
        Epoch: 36, Time: 0.00683s, Loss: 1.08887
        update best: 0.12000
        Epoch: 37, Time: 0.00780s, Loss: 1.08907
        Epoch: 38, Time: 0.00782s, Loss: 1.08394
        Epoch: 39, Time: 0.00626s, Loss: 1.07832
        Epoch: 40, Time: 0.00783s, Loss: 1.03877
        update best: 0.12500
        Epoch: 41, Time: 0.00795s, Loss: 1.03990
        update best: 0.13500
        Epoch: 42, Time: 0.00626s, Loss: 1.02008
        update best: 0.14500
        Epoch: 43, Time: 0.00709s, Loss: 0.99529
        update best: 0.16000
        Epoch: 44, Time: 0.00763s, Loss: 1.01162
        update best: 0.17500
        Epoch: 45, Time: 0.00749s, Loss: 0.99196
        update best: 0.20500
        Epoch: 46, Time: 0.00629s, Loss: 0.97237
        update best: 0.21000
        Epoch: 47, Time: 0.00754s, Loss: 0.97511
        update best: 0.22500
        Epoch: 48, Time: 0.00805s, Loss: 0.95078
        update best: 0.23000
        Epoch: 49, Time: 0.00745s, Loss: 0.94715
        update best: 0.24500
        Epoch: 50, Time: 0.00643s, Loss: 0.93461
        update best: 0.25500
        Epoch: 51, Time: 0.00743s, Loss: 0.92102
        update best: 0.27500
        Epoch: 52, Time: 0.00772s, Loss: 0.91536
        update best: 0.29500
        Epoch: 53, Time: 0.00714s, Loss: 0.89386
        update best: 0.30500
        Epoch: 54, Time: 0.00722s, Loss: 0.88108
        Epoch: 55, Time: 0.00777s, Loss: 0.88809
        Epoch: 56, Time: 0.00717s, Loss: 0.85739
        Epoch: 57, Time: 0.00724s, Loss: 0.86278
        update best: 0.31000
        Epoch: 58, Time: 0.00804s, Loss: 0.83276
        update best: 0.32500
        Epoch: 59, Time: 0.00786s, Loss: 0.83001
        update best: 0.35000
        Epoch: 60, Time: 0.00629s, Loss: 0.83385
        update best: 0.37500
        Epoch: 61, Time: 0.00712s, Loss: 0.82473
        update best: 0.39500
        Epoch: 62, Time: 0.00904s, Loss: 0.81101
        update best: 0.41000
        Epoch: 63, Time: 0.00745s, Loss: 0.80212
        Epoch: 64, Time: 0.00715s, Loss: 0.79534
        update best: 0.42000
        Epoch: 65, Time: 0.00705s, Loss: 0.77077
        Epoch: 66, Time: 0.00710s, Loss: 0.77775
        update best: 0.43000
        Epoch: 67, Time: 0.00717s, Loss: 0.77026
        update best: 0.43500
        Epoch: 68, Time: 0.00789s, Loss: 0.75978
        Epoch: 69, Time: 0.00747s, Loss: 0.74209
        Epoch: 70, Time: 0.00639s, Loss: 0.73636
        Epoch: 71, Time: 0.00689s, Loss: 0.72454
        Epoch: 72, Time: 0.00793s, Loss: 0.72910
        Epoch: 73, Time: 0.00729s, Loss: 0.72512
        Epoch: 74, Time: 0.00775s, Loss: 0.71034
        update best: 0.44500
        Epoch: 75, Time: 0.00766s, Loss: 0.69282
        update best: 0.45000
        Epoch: 76, Time: 0.00627s, Loss: 0.70622
        update best: 0.46000
        Epoch: 77, Time: 0.00706s, Loss: 0.70540
        update best: 0.47500
        Epoch: 78, Time: 0.00849s, Loss: 0.69790
        Epoch: 79, Time: 0.00731s, Loss: 0.66718
        Epoch: 80, Time: 0.00748s, Loss: 0.67149
        Epoch: 81, Time: 0.00900s, Loss: 0.68492
        Epoch: 82, Time: 0.00624s, Loss: 0.65467
        Epoch: 83, Time: 0.00713s, Loss: 0.63049
        Epoch: 84, Time: 0.00852s, Loss: 0.65693
        Epoch: 85, Time: 0.00622s, Loss: 0.64821
        Epoch: 86, Time: 0.00717s, Loss: 0.64481
        Epoch: 87, Time: 0.00784s, Loss: 0.64284
        Epoch: 88, Time: 0.00630s, Loss: 0.62653
        Epoch: 89, Time: 0.00726s, Loss: 0.62808
        Epoch: 90, Time: 0.00786s, Loss: 0.62135
        Epoch: 91, Time: 0.00729s, Loss: 0.59833
        Epoch: 92, Time: 0.00731s, Loss: 0.60561
        Epoch: 93, Time: 0.00801s, Loss: 0.60091
        Epoch: 94, Time: 0.00630s, Loss: 0.58819
        Epoch: 95, Time: 0.00763s, Loss: 0.56774
        Epoch: 96, Time: 0.00743s, Loss: 0.57335
        Epoch: 97, Time: 0.00662s, Loss: 0.56947
        Epoch: 98, Time: 0.00899s, Loss: 0.57430
        Epoch: 99, Time: 0.00751s, Loss: 0.56189
        Epoch: 100, Time: 0.00719s, Loss: 0.55171
        Epoch: 101, Time: 0.00791s, Loss: 0.56934
        Epoch: 102, Time: 0.00627s, Loss: 0.54815
        Epoch: 103, Time: 0.00731s, Loss: 0.54027
        Epoch: 104, Time: 0.00817s, Loss: 0.54291
        Epoch: 105, Time: 0.00623s, Loss: 0.52773
        Epoch: 106, Time: 0.00737s, Loss: 0.53735
        Epoch: 107, Time: 0.00790s, Loss: 0.51841
        Epoch: 108, Time: 0.00631s, Loss: 0.51548
        Epoch: 109, Time: 0.00753s, Loss: 0.51153
        Epoch: 110, Time: 0.00822s, Loss: 0.50702
        Epoch: 111, Time: 0.00689s, Loss: 0.50974
        Epoch: 112, Time: 0.00648s, Loss: 0.49094
        Epoch: 113, Time: 0.00768s, Loss: 0.50044
        Epoch: 114, Time: 0.00808s, Loss: 0.50632
        Epoch: 115, Time: 0.00744s, Loss: 0.48155
        Epoch: 116, Time: 0.00774s, Loss: 0.49875
        Epoch: 117, Time: 0.00633s, Loss: 0.48650
        Epoch: 118, Time: 0.00742s, Loss: 0.48026
        Epoch: 119, Time: 0.00928s, Loss: 0.48162
        Epoch: 120, Time: 0.00687s, Loss: 0.46713
        Epoch: 121, Time: 0.00679s, Loss: 0.46894
        Epoch: 122, Time: 0.00891s, Loss: 0.47300
        Epoch: 123, Time: 0.00639s, Loss: 0.45836
        Epoch: 124, Time: 0.00676s, Loss: 0.46030
        Epoch: 125, Time: 0.00940s, Loss: 0.45373
        Epoch: 126, Time: 0.00926s, Loss: 0.44894
        Epoch: 127, Time: 0.00701s, Loss: 0.45110
        Epoch: 128, Time: 0.00710s, Loss: 0.43749
        Epoch: 129, Time: 0.00913s, Loss: 0.45104
        Epoch: 130, Time: 0.00706s, Loss: 0.45284
        Epoch: 131, Time: 0.00693s, Loss: 0.44452
        Epoch: 132, Time: 0.00937s, Loss: 0.43088
        Epoch: 133, Time: 0.00810s, Loss: 0.43557
        Epoch: 134, Time: 0.00713s, Loss: 0.44251
        Epoch: 135, Time: 0.00822s, Loss: 0.41227
        Epoch: 136, Time: 0.00981s, Loss: 0.41414
        Epoch: 137, Time: 0.00706s, Loss: 0.42148
        Epoch: 138, Time: 0.00649s, Loss: 0.40822
        Epoch: 139, Time: 0.00860s, Loss: 0.41343
        Epoch: 140, Time: 0.00616s, Loss: 0.39754
        Epoch: 141, Time: 0.00644s, Loss: 0.39057
        Epoch: 142, Time: 0.00860s, Loss: 0.41271
        Epoch: 143, Time: 0.00631s, Loss: 0.39916
        Epoch: 144, Time: 0.00675s, Loss: 0.37878
        Epoch: 145, Time: 0.00897s, Loss: 0.40234
        Epoch: 146, Time: 0.00621s, Loss: 0.38136
        Epoch: 147, Time: 0.00864s, Loss: 0.38960
        Epoch: 148, Time: 0.00633s, Loss: 0.40494
        Epoch: 149, Time: 0.00629s, Loss: 0.38099
        Epoch: 150, Time: 0.00883s, Loss: 0.37809
        Epoch: 151, Time: 0.00621s, Loss: 0.38888
        Epoch: 152, Time: 0.00633s, Loss: 0.35971
        Epoch: 153, Time: 0.00842s, Loss: 0.37553
        Epoch: 154, Time: 0.00622s, Loss: 0.36924
        Epoch: 155, Time: 0.00739s, Loss: 0.37269
        Epoch: 156, Time: 0.00864s, Loss: 0.36131
        Epoch: 157, Time: 0.00627s, Loss: 0.35630
        Epoch: 158, Time: 0.00854s, Loss: 0.36315
        Epoch: 159, Time: 0.00648s, Loss: 0.37506
        Epoch: 160, Time: 0.00638s, Loss: 0.36177
        Epoch: 161, Time: 0.00867s, Loss: 0.37122
        Epoch: 162, Time: 0.00632s, Loss: 0.35660
        Epoch: 163, Time: 0.00641s, Loss: 0.34108
        Epoch: 164, Time: 0.00873s, Loss: 0.34228
        Epoch: 165, Time: 0.00619s, Loss: 0.34731
        Epoch: 166, Time: 0.00656s, Loss: 0.34604
        Epoch: 167, Time: 0.00881s, Loss: 0.33136
        Epoch: 168, Time: 0.00620s, Loss: 0.35096
        Epoch: 169, Time: 0.00874s, Loss: 0.33567
        Epoch: 170, Time: 0.00766s, Loss: 0.32705
        Epoch: 171, Time: 0.00628s, Loss: 0.32490
        Epoch: 172, Time: 0.00880s, Loss: 0.32892
        Epoch: 173, Time: 0.00619s, Loss: 0.32556
        Epoch: 174, Time: 0.00631s, Loss: 0.32410
        Epoch: 175, Time: 0.00878s, Loss: 0.30940
        Epoch: 176, Time: 0.00629s, Loss: 0.33027
        Epoch: 177, Time: 0.00636s, Loss: 0.32709
        Epoch: 178, Time: 0.00887s, Loss: 0.32104
        Epoch: 179, Time: 0.00625s, Loss: 0.33687
        Epoch: 180, Time: 0.00694s, Loss: 0.31593
        Epoch: 181, Time: 0.00861s, Loss: 0.31409
        Epoch: 182, Time: 0.00627s, Loss: 0.31477
        Epoch: 183, Time: 0.00847s, Loss: 0.30355
        Epoch: 184, Time: 0.00642s, Loss: 0.33237
        Epoch: 185, Time: 0.00630s, Loss: 0.30555
        Epoch: 186, Time: 0.00839s, Loss: 0.29973
        Epoch: 187, Time: 0.00631s, Loss: 0.30695
        Epoch: 188, Time: 0.00645s, Loss: 0.30313
        Epoch: 189, Time: 0.00899s, Loss: 0.30699
        Epoch: 190, Time: 0.00626s, Loss: 0.31283
        Epoch: 191, Time: 0.00654s, Loss: 0.28851
        Epoch: 192, Time: 0.00879s, Loss: 0.28803
        Epoch: 193, Time: 0.00621s, Loss: 0.28213
        Epoch: 194, Time: 0.00846s, Loss: 0.27823
        Epoch: 195, Time: 0.00704s, Loss: 0.29048
        Epoch: 196, Time: 0.00638s, Loss: 0.28898
        Epoch: 197, Time: 0.00894s, Loss: 0.29096
        Epoch: 198, Time: 0.00642s, Loss: 0.27857
        Epoch: 199, Time: 0.00817s, Loss: 0.29117

        train finished!
        best val: 0.47500
        test...
        final result: epoch: 77
        {'accuracy': 0.5203484296798706, 'f1_score': 0.39131907709452823, 'f1_score -> average@micro': 0.5203484221048122}


