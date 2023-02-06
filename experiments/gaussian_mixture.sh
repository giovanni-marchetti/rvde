#!/usr/bin/env bash
set -e

# gaussians
dset="gaussians"
band_low="0.01"
band_high="5"

. train_models.sh
