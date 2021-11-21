# DepthCRNN-PointLSTM

See the `PointDepthScoreFusion` class in [score_fusion_model.py](./score_fusion_model.py) to see how the score-level fusion model is made. It is quite simple.

To see an example of using it, run:

```
python score_fusion_model.py
```

This will:
- Create the fusion model, which I'm tentatively calling `PointDepthScoreFusion`
- Create a dummy dataloader that generates point clouds and depth images, with a very small batch size (2) and very few examples (10)
- Runs an example training process to check that everything works fine (trains for 5 steps)

You will see some logs like:

```
Starting demo run:

(ignore the maxpool warning, if any)

Successfully created score fusion model
Score fusion model has 4078484 parameters.
Shape of output: torch.Size([2, 14])
[+] Successfully trained 1 step. Loss: 2.3820953369140625
Shape of output: torch.Size([2, 14])
[+] Successfully trained 1 step. Loss: 2.7277846336364746
Shape of output: torch.Size([2, 14])
[+] Successfully trained 1 step. Loss: 2.688685894012451
Shape of output: torch.Size([2, 14])
[+] Successfully trained 1 step. Loss: 2.268145799636841
Shape of output: torch.Size([2, 14])
[+] Successfully trained 1 step. Loss: 2.6050686836242676
Completed demo run.
```