#!/usr/bin/env bash
# Step D on the idle A30 (PCI bus CA = index 2 in PCI order). Data already built and verified (DATA_OK);
# the zips were deleted, so only rebuild the data if the patch folders are missing.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
D=~/stepd/data
if [ "$(ls $D/patches512_div2k 2>/dev/null | wc -l)" != 3200 ] || [ "$(ls $D/patches512_clic2020 2>/dev/null | wc -l)" != 12872 ]; then
  cd ~/stepd && ./stepd_a30/make_data.sh $D ~/stepd/env/bin/python 2>&1 | tee -a ~/stepd/make_data.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "make_data failed: not starting training"; exec bash; }
fi
cd ~/stepd/stepd_a30 && ./run_stepd.sh $D 2 ~/stepd/env/bin/python 1 2>&1 | tee -a ~/stepd/run_stepd.log
exec bash
