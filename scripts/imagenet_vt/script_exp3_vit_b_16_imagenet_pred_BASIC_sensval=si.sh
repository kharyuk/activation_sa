#!/bin/bash -l

export CUDA_VISIBLE_DEVICES=""
source activate activation_sense_vt

python ../experiments/3a_masked_prediction.py \
--recompute_predictions 1 \
--batch_size_computing 500 \
--samples_per_class_train 732 \
--samples_per_class_valid 50 \
--augmentation_set_number 3 \
--dataset_part valid \
--Nsamples 50000 \
--Ninner_samples 3 \
--mkl_num_threads 8 \
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
--output_filename_suffix pred_BASIC  \
--device cpu \
--network_name vit_b_16 \
--network_modules '[]' \
--classification_layer_name heads \
--subset_random_state_train 297 \
--subset_random_state_valid 907640 \
--torch_seed 2340098 \
--numpy_seed 80042 \
--class_sampler_seed 102030 \
--class_selector_seed 981924 \
--augpar_sampler_seeds 65298 \
--model_filename vit_b_16-c867db91.pth \
--results_dirname_path ../results/vit_b_16/si \
--sensitivity_values_name si
