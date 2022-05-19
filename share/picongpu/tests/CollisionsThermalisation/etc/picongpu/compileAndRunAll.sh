#!/bin/bash
BASE_PATH=~/SCRATCH/runs/CollisionThermalization
for i in {0..3}
do
  pic-build -t $i
  tbg -s -t -c etc/picongpu/1.cfg $BASE_PATH/TEST_CASE_N$i
done
