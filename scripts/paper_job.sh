#!/usr/bin/env bash

CSV_FILE="paper_job.csv"

while IFS=, read -r data_dir basis_config n_iter chain_save; do
	[[ "$data_dir" == "data_dir" ]] && continue

    echo "Running job:"
    echo "  data_dir=$data_dir"
    echo "  basis_config=$basis_config"
    echo "  n_iter=$n_iter"
    echo "  cs=$chain_save"


	python general_sampling.py \
		-dd "$data_dir" \
		-bc "$basis_config" \
		-niter "$n_iter" \
		-cs "$chain_save"
done < "$CSV_FILE"

