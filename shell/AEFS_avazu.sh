cd ..
#!/bin/sh
source activate
conda activate AEFS

root_path='~/AEFS/'
dataset_name=avazu

model_name='AEFS'
embed_dim=32
embed_dim_small=4
field_align_loss='MSE'
score_align_loss='MSE'
dense_type='MultiLayerPerceptron'
dense_type_small='MultiLayerPerceptron'

python main.py --model_name ${model_name} --dataset_name ${dataset_name} \
    --embed_dim ${embed_dim} --embed_dim_small ${embed_dim_small} --field_align_loss ${field_align_loss} \
    --score_align_loss ${score_align_loss} --root_path ${root_path}  \
    --dense_type ${dense_type} --dense_type_small ${dense_type_small} --pretrain 2