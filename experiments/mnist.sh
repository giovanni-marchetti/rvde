#!/usr/bin/env bash
set -e

# mnist
dset="mnist"
band_low="0.005"
band_high="0.25"
reduce=resize
dim=100

. train_models.sh
