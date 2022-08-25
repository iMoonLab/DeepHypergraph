On Simple Graph
==========================================

GCN on Cora
----------------

Import Libraries
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

Define Functions
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

Main
^^^^^^^

.. note:: 

    More details about the metric ``Evaluator`` can be found in the :ref:`Build Evaluator <tutorial_build_evaluator>` section.

.. code-block:: python

    if __name__ == "__main__":
        set_seed(2022)
        evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
        data = Cora()
        X, lbl = data["features"], data["labels"]
        G = Graph(data["num_vertices"], data["edge_list"])
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]

        net = GCN(data["dim_features"], 16, data["num_classes"])
        optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

        x, lbl = x.to(device), lbl.to(device)
        G = G.to(x.device)
        net = net.to(device)

        best_state = None
        best_epoch, best_val = 0, 0
        for epoch in range(300):
            # train
            train(net, x, G, lbl, train_mask, optimizer, epoch)
            # validation
            if epoch % 1 == 0:
                with torch.no_grad():
                    val_res = infer(net, x, G, lbl, val_mask)
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
        res = infer(net, x, G, lbl, test_mask, test=True)
        print(f"final result: epoch: {best_epoch}")
        print(res)

Outputs
^^^^^^^^^^^^
.. code-block:: text

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

GAT on Cora
----------------

Import Libraries
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


Define Functions
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

Main
^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: 

    More details about the metric ``Evaluator`` can be found in the :ref:`Build Evaluator <tutorial_build_evaluator>` section.

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
        G = G.to(X.device)
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

Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

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


HGNN on Cora
----------------

Import Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^


Define Functions
^^^^^^^^^^^^^^^^^^^^^^^^^


Main
^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: 

    More details about the metric ``Evaluator`` can be found in the :ref:`Build Evaluator <tutorial_build_evaluator>` section.



Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^

HGNN+ on Cora
----------------

Import Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^


Define Functions
^^^^^^^^^^^^^^^^^^^^^^^^^


Main
^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: 

    More details about the metric ``Evaluator`` can be found in the :ref:`Build Evaluator <tutorial_build_evaluator>` section.


Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^
