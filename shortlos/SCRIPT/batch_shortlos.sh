#!/bin/bash

modelname=l25n288-phew-m5-spl
#modelname=l25n288-phewoff
mcinit=2.0e38
zcen=0.25
lbox=25.0
ftau=1.0
tabfile="tabs/shortlos_"$modelname # No need for ".tab"
losfile="LoS/loshalo."$modelname".x.lst"

# parallel ../contspecexbin {1} {2} {3} {4} {5} {6} ::: $tabfile :::: $losfile ::: $zcen ::: $lbox ::: $ftau ::: 2     

while read line 
do
    echo "../contspecexbin "$tabfile $line $zcen $lbox $ftau 2 $mcinit
    ../contspecexbin_shortspec $tabfile $line $zcen $lbox $ftau 2 $mcinit
done < $losfile
