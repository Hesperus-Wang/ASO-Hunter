# ASO-Hunter
A few-shot meta-learning framework for RNase H-dependent ASO efficacy prediction
# random split Training code

```python standard_model_no_label_noise.py ./cleaned_data_iqr_withsplit_for_meta.csv \
--output_dir ./n_support_test_s-40/seed_42/n_support_10/ \
--lr 5e-4 --max_epochs 50 --hidden_dim 128 --num_layers 3 \
--dropout 0.1 --wandb --wandb_project aso_inhibition --wandb_experiment_name aso_wl \
--num_workers 16 --pin_memory --n_support 10 --n_query 35 --eval_n_query 35 \ 
--update_step_train 5 --update_step_test 5 --meta_batch_size 6 \
--steps_per_epoch 80 --noise 0 --noise_val 0 --freeze_lm_only \
--point_loss_type huber --huber_delta 1.0 --rank_loss_weight 0.2 
--rank_min_delta 0.25 --min_center_run 4 --seed 42```

