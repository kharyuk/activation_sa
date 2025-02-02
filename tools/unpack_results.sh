#!/bin/bash

# script for unpacking supplementary files from Zenodo:
# https://zenodo.org/uploads/14788913
# Files must be placed into "zenodo_files" directory

zenodo_directory="../zenodo_files"
results_directory="../results"


series1_file="1_sensitivity_values.7z"
series2_file="2_guided_masking_predictions.7z"
series3_file="3_single_channelled_segments.7z"

# x option preserves directory tree structure
# e option makes it plain (our case)
7z e "$zenodo_directory/$series1_file" -o"$results_directory"
7z e "$zenodo_directory/$series2_file" -o"$results_directory"
7z e "$zenodo_directory/$series3_file" -o"$results_directory/single_unit"



