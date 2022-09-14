HGNN+ on Cooking200
=======================

.. hint:: 

    - Author: `Yifan Feng (丰一帆) <https://fengyifan.site/>`_
    - Proof: Xinwei Zhang

In the following example, we present a simple example of Auto-ML for vertex classification on hypergraph.
More details for how to use the :doc:`/api/experiments` to auto tuning your own model can be found in the tutorial <:doc:`/tutorial/auto_ml`>.

Configuration
--------------

- Model: HGNN+ (:py:class:`dhg.models.HGNNP`): `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).
- Dataset: Cooking 200 (:py:class:`dhg.data.Cooking200`): A cooking recipe hypergraph dataset collected from `Yummly.com <https://www.yummly.com/>`_ for vertex classification task.


Import Libraries
---------------------

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from dhg import Hypergraph
    from dhg.data import Cooking200
    from dhg.models import HGNNP
    from dhg.random import set_seed
    from dhg.experiments import HypergraphVertexClassificationTask as Task
    from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator


Define Functions
-------------------

.. code-block:: python

    def structure_builder(trial):
        global hg_base, g
        cur_hg: Hypergraph = hg_base.clone()
        return cur_hg


    def model_builder(trial):
        return HGNNP(dim_features, trial.suggest_int("hidden_dim", 10, 20), num_classes, use_bn=True)


    def train_builder(trial, model):
        optimizer = optim.Adam(
            model.parameters(),
            lr=trial.suggest_loguniform("lr", 1e-4, 1e-2),
            weight_decay=trial.suggest_loguniform("weight_decay", 1e-4, 1e-2),
        )
        criterion = nn.CrossEntropyLoss()
        return {
            "optimizer": optimizer,
            "criterion": criterion,
        }

Main
-----

.. important:: 

    You must change the ``work_root`` to your own work directory.

.. code-block:: python

    if __name__ == "__main__":
        work_root = "/home/fengyifan/OS3D/toolbox/exp_cache/tmp"
        set_seed(2022)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        data = Cooking200()
        dim_features = data["num_vertices"]
        num_classes = data["num_classes"]
        hg_base = Hypergraph(data["num_vertices"], data["edge_list"])
        input_data = {
            "features": torch.eye(data["num_vertices"]),
            "labels": data["labels"],
            "train_mask": data["train_mask"],
            "val_mask": data["val_mask"],
            "test_mask": data["test_mask"],
        }
        evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
        task = Task(
            work_root, input_data, model_builder, train_builder, evaluator, device, structure_builder=structure_builder,
        )
        task.run(200, 50, "maximize")


.. only:: not latex

    Outputs
    -------------

    .. code-block:: 

        [I 2022-08-25 17:46:08,399] Logs will be saved to /home/fengyifan/OS3D/toolbox/exp_cache/tmp/2022-08-25--17-46-08/log.txt
        [I 2022-08-25 17:46:08,399] Files in training will be saved in /home/fengyifan/OS3D/toolbox/exp_cache/tmp/2022-08-25--17-46-08
        [I 2022-08-25 17:46:09,904] Random seed is 2022
        [I 2022-08-25 17:46:09,905] A new study created in memory with name: no-name-9e617917-a809-40dc-a6b9-87aeda5bb6ee
        [I 2022-08-25 17:46:12,361] Trial 0 finished with value: 0.4000000059604645 and parameters: {'hidden_dim': 10, 'lr': 0.0009956704582324435, 'weight_decay': 0.00016856499028548418}. Best is trial 0 with value: 0.4000000059604645.
        [I 2022-08-25 17:46:14,217] Trial 1 finished with value: 0.41999998688697815 and parameters: {'hidden_dim': 10, 'lr': 0.002348633160857829, 'weight_decay': 0.0009418378430920174}. Best is trial 1 with value: 0.41999998688697815.
        [I 2022-08-25 17:46:16,074] Trial 2 finished with value: 0.48500001430511475 and parameters: {'hidden_dim': 19, 'lr': 0.0019719874263090698, 'weight_decay': 0.006221946114841155}. Best is trial 2 with value: 0.48500001430511475.
        [I 2022-08-25 17:46:18,074] Trial 3 finished with value: 0.48500001430511475 and parameters: {'hidden_dim': 17, 'lr': 0.004599459949791714, 'weight_decay': 0.0045199760918655345}. Best is trial 2 with value: 0.48500001430511475.
        [I 2022-08-25 17:46:20,060] Trial 4 finished with value: 0.4950000047683716 and parameters: {'hidden_dim': 19, 'lr': 0.008205190552892963, 'weight_decay': 0.0005446140912512398}. Best is trial 4 with value: 0.4950000047683716.
        [I 2022-08-25 17:46:20,551] Trial 5 pruned. 
        [I 2022-08-25 17:46:21,146] Trial 6 pruned. 
        [I 2022-08-25 17:46:21,751] Trial 7 pruned. 
        [I 2022-08-25 17:46:22,397] Trial 8 pruned. 
        [I 2022-08-25 17:46:22,720] Trial 9 pruned. 
        [I 2022-08-25 17:46:24,731] Trial 10 finished with value: 0.49000000953674316 and parameters: {'hidden_dim': 18, 'lr': 0.009112327540785461, 'weight_decay': 0.0002825142053930118}. Best is trial 4 with value: 0.4950000047683716.
        [I 2022-08-25 17:46:26,677] Trial 11 finished with value: 0.49000000953674316 and parameters: {'hidden_dim': 17, 'lr': 0.009700863338872084, 'weight_decay': 0.00024395653633063402}. Best is trial 4 with value: 0.4950000047683716.
        [I 2022-08-25 17:46:28,785] Trial 12 finished with value: 0.4950000047683716 and parameters: {'hidden_dim': 18, 'lr': 0.009506157011953582, 'weight_decay': 0.00034409703681570236}. Best is trial 4 with value: 0.4950000047683716.
        [I 2022-08-25 17:46:30,801] Trial 13 finished with value: 0.49000000953674316 and parameters: {'hidden_dim': 20, 'lr': 0.004245693592715978, 'weight_decay': 0.00046142123936015995}. Best is trial 4 with value: 0.4950000047683716.
        [I 2022-08-25 17:46:32,813] Trial 14 finished with value: 0.4950000047683716 and parameters: {'hidden_dim': 17, 'lr': 0.00494083746774663, 'weight_decay': 0.0001151901195440639}. Best is trial 4 with value: 0.4950000047683716.
        [I 2022-08-25 17:46:33,293] Trial 15 pruned. 
        [I 2022-08-25 17:46:33,810] Trial 16 pruned. 
        [I 2022-08-25 17:46:34,329] Trial 17 pruned. 
        [I 2022-08-25 17:46:34,840] Trial 18 pruned. 
        [I 2022-08-25 17:46:35,358] Trial 19 pruned. 
        [I 2022-08-25 17:46:35,902] Trial 20 pruned. 
        [I 2022-08-25 17:46:36,895] Trial 21 pruned. 
        [I 2022-08-25 17:46:37,406] Trial 22 pruned. 
        [I 2022-08-25 17:46:39,326] Trial 23 finished with value: 0.49000000953674316 and parameters: {'hidden_dim': 16, 'lr': 0.006943644200360305, 'weight_decay': 0.0006003049507614988}. Best is trial 4 with value: 0.4950000047683716.
        [I 2022-08-25 17:46:41,236] Trial 24 finished with value: 0.48500001430511475 and parameters: {'hidden_dim': 20, 'lr': 0.009971146065887018, 'weight_decay': 0.00035931897741066387}. Best is trial 4 with value: 0.4950000047683716.
        [I 2022-08-25 17:46:41,732] Trial 25 pruned. 
        [I 2022-08-25 17:46:42,160] Trial 26 pruned. 
        [I 2022-08-25 17:46:42,589] Trial 27 pruned. 
        [I 2022-08-25 17:46:43,120] Trial 28 pruned. 
        [I 2022-08-25 17:46:43,631] Trial 29 pruned. 
        [I 2022-08-25 17:46:44,143] Trial 30 pruned. 
        [I 2022-08-25 17:46:44,612] Trial 31 pruned. 
        [I 2022-08-25 17:46:45,187] Trial 32 pruned. 
        [I 2022-08-25 17:46:45,906] Trial 33 pruned. 
        [I 2022-08-25 17:46:46,544] Trial 34 pruned. 
        [I 2022-08-25 17:46:46,965] Trial 35 pruned. 
        [I 2022-08-25 17:46:48,842] Trial 36 finished with value: 0.5049999952316284 and parameters: {'hidden_dim': 17, 'lr': 0.009648904316000167, 'weight_decay': 0.00013498962749734303}. Best is trial 36 with value: 0.5049999952316284.
        [I 2022-08-25 17:46:49,339] Trial 37 pruned. 
        [I 2022-08-25 17:46:51,214] Trial 38 finished with value: 0.48500001430511475 and parameters: {'hidden_dim': 18, 'lr': 0.009528262435822034, 'weight_decay': 0.00013603318896175282}. Best is trial 36 with value: 0.5049999952316284.
        [I 2022-08-25 17:46:51,612] Trial 39 pruned. 
        [I 2022-08-25 17:46:53,637] Trial 40 finished with value: 0.48500001430511475 and parameters: {'hidden_dim': 17, 'lr': 0.005722162043271019, 'weight_decay': 0.0003712595876989976}. Best is trial 36 with value: 0.5049999952316284.
        [I 2022-08-25 17:46:54,125] Trial 41 pruned. 
        [I 2022-08-25 17:46:54,627] Trial 42 pruned. 
        [I 2022-08-25 17:46:55,069] Trial 43 pruned. 
        [I 2022-08-25 17:46:55,541] Trial 44 pruned. 
        [I 2022-08-25 17:46:57,467] Trial 45 finished with value: 0.5 and parameters: {'hidden_dim': 18, 'lr': 0.009996814276559166, 'weight_decay': 0.00030144984469652667}. Best is trial 36 with value: 0.5049999952316284.
        [I 2022-08-25 17:46:58,015] Trial 46 pruned. 
        [I 2022-08-25 17:46:58,499] Trial 47 pruned. 
        [I 2022-08-25 17:46:58,970] Trial 48 pruned. 
        [I 2022-08-25 17:46:59,430] Trial 49 pruned. 
        [I 2022-08-25 17:46:59,483] Best trial:
        [I 2022-08-25 17:46:59,483]     Value: 0.505
        [I 2022-08-25 17:46:59,483]     Params:
        [I 2022-08-25 17:46:59,484]             hidden_dim |-> 17
        [I 2022-08-25 17:46:59,484]             lr |-> 0.009648904316000167
        [I 2022-08-25 17:46:59,484]             weight_decay |-> 0.00013498962749734303
        [I 2022-08-25 17:46:59,496] Final test results:
        [I 2022-08-25 17:46:59,496]     accuracy |-> 0.526
        [I 2022-08-25 17:46:59,497]     f1_score |-> 0.402
        [I 2022-08-25 17:46:59,497]     f1_score -> average@micro |-> 0.526