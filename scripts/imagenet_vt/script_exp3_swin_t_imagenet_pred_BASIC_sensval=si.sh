#!/bin/bash -l

export CUDA_VISIBLE_DEVICES=""
source activate activation_sense_vt

python ../experiments/3a_masked_prediction.py \
--recompute_predictions 1 \
--batch_size_computing 100 \
--samples_per_class_train 732 \
--samples_per_class_valid 50 \
--augmentation_set_number 3 \
--dataset_part valid \
--Nsamples 50000 \
--Ninner_samples 3 \
--mkl_num_threads 4 \
--data_dirname ../data/imagenet \
--dataset imagenet \
--model_dirname ../torch-models/ \
--desired_image_height 224 \
--desired_image_width 224 \
--use_permutation_variable 1 \
--use_class_variable 1 \
--use_partition_variable 1 \
--alphas "(0.0, 0.5, 1.5)" \
--percentiles "(0.5, 0.6, 0.7, 0.8, 0.9)" \
--top_n_predictions 5 \
--sensitivity_values_dirname ../results/ \
--values_fnm_base imagenet_values \
--output_filename_suffix pred_BASIC \
--device cpu \
--network_name swin_t \
--network_modules '[]' \
--classification_layer_name head \
--subset_random_state_train 6498 \
--subset_random_state_valid 4660 \
--torch_seed 1598 \
--numpy_seed 3689 \
--class_sampler_seed 2196 \
--class_selector_seed 8580 \
--augpar_sampler_seeds 4624 \
--model_filename swin_t-704ceda3.pth \
--results_dirname_path ../results/swin_t/si \
--sensitivity_values_name si
