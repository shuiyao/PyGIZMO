#!/bin/bash
#parallel contspecexbin_v8 {1} {2} {3} {4} {5} {6} ::: snap_p50n288ezw15 ::: 0.0 ::: 6.0 ::: 50.0 ::: 1.0 :::: angles_10_24

modelname=l25n288-phew-m5-spl
mcinit=2.0e38
zbeg=0.0
zend=0.5
lbox=50.0
ftau=1.0
tabfile="tabs/"$modelname # No need for ".tab"

while read line 
do
    echo "../contspecexbin "$modelname $zbeg $zend $lbox $ftau $line $mcinit
    ../contspecexbin $tabfile $zbeg $zend $lbox $ftau $line $mcinit
done < angles/angles_30_40.dat