#!/usr/bin/env bash
set -e

dset='gaussians'
band_low=.001
band_high=10.

# Training
# for est in  "kde" "contvde" "awkde" "cvde" ; do
# for kind in "r" ; do
# #for kind in "r" "e" "g" ; do
#   echo "Current estimator: ${est}_${kind}"
#   python train_estimator.py "${est}" "${dset}" --kind "${kind}" ${reduce:+"--reduce" "${reduce}"} ${dim:+"--dim" "${dim}"} \
#        --band_low ${band_low} --band_high ${band_high}  ${logscale:+"--logscale"}
# done
# done

# exit

# Suggested alpha computation
# for kind in "r" ; do
# #for kind in "r" "e" ; do
#   echo "Estimating alpha: ${est}"
#   python train_estimator.py "contvde" "${dset}" --kind "${kind}" ${reduce:+"--reduce" "${reduce}"} ${dim:+"--dim" "${dim}"} \
#        --band_low ${band_low} --band_high ${band_high} --extra
# done

# Plotting
for kind in "r" ; do
#for kind in "r" "e" "g" ; do
  python graph.py "${dset}" ${reduce:+"--reduce" "${reduce}"} ${dim:+"--dim" "${dim}"} --kind "${kind}"
done
