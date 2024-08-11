
dataset: Avazu

| Method                 | AUC    | Logloss |
| ---------------------- | ------ | ------- |
| AEFS (mini-AdaFS)      | 0.7785 | 0.3808  |
| - w/o pretrain         | 0.7781 | 0.3809  |
| - w/o l1 normalization | 0.7778 | 0.3819  |

dataset: Criteo

| Method                 | AUC    | Logloss |
| ---------------------- | ------ | ------- |
| AEFS (mini-AdaFS)      | 0.8057 | 0.4465  |
| - w/o pretrain         | 0.8058 | 0.4467  |
| - w/o l1 normalization | 0.8052 | 0.4487  |

From the experimental results, 
it can be observed that the pretraining strategy has little impact on the recommendation performance, while removing L1 normalization has a significant effect on the experimental outcomes.