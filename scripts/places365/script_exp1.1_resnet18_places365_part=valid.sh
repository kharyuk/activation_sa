#!/bin/bash
export CUDA_VISIBLE_DEVICES=""
source activate activation_sense

python ../experiments/1.1_contrast.py --dataset places365 --batch_size_activations_computing 100 --samples_per_class_train 3068 --samples_per_class_valid 100 --data_dirname ../data/places365 --activations_dirname  /mnt/bulky/pkharyuk/new/activation_sa/results/ --model_dirname ../torch-models/ --model_filename resnet18_places365.pth.tar --mkl_num_threads 9 --device cpu --recompute_activations 0 --remove_activations_hdf5 0 --desired_image_height 224 --desired_image_width 224 --network_name resnet18 --network_modules '["maxpool", "layer1", "layer2", "layer3", "avgpool", "fc"]' --classification_layer_name fc --libstdcpp_path /mnt/bulky/pkharyuk/apd/envs/activation_sense/lib/libstdc++.so.6 --values_buffer_size 1000 --torch_seed 871 --numpy_seed 2323 --class_sampler_seed 987 --class_selector_seed 4948 --augpar_sampler_seeds 9098 --subset_random_state_train 4 --subset_random_state_valid 998 --Nsamples 36500 --Ninner_samples 10 --Njobs_cs_computing 8 --augmentation_set_number 3 --activations_fnm_prefix cs_resnet18_places365_activations --values_fnm_prefix cs_resnet18_places365_values --dataset_part valid

