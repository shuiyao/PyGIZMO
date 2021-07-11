#!/bin/bash

model="l25n288-phew-m5-spl"

if [ -e ./tabs/$model.tab ]; then
    sbatch $model"_10-20.slm"
    sbatch $model"_20-30.slm"
    sbatch $model"_30-40.slm"
    # sbatch $model"_40-50.slm"
    # sbatch $model"_50-60.slm"
    # sbatch $model"_60-70.slm"
    # sbatch $model"_70-80.slm"
else
    echo "../"$model".tab Not Found."
fi
