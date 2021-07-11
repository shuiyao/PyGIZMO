#!/bin/bash
#parallel contspecexbin_v8 {1} {2} {3} {4} {5} {6} ::: snap_p50n288ezw15 ::: 0.0 ::: 6.0 ::: 50.0 ::: 1.0 :::: angles_10_24

modelname=l25n144-phew-rcloud
mcinit=2.0e38
zbeg=0.0
zend=0.5
lbox=25.0
ftau=1.0
tabfile="tabs/"$modelname # No need for ".tab"

echo "../contspecexbin "$tabfile $zbeg $zend $lbox $ftau $line $mcinit
#gdb --args ../contspecexbin $tabfile $zbeg $zend $lbox $ftau 14.0 $mcinit
../contspecexbin $tabfile $zbeg $zend $lbox $ftau 42.0 $mcinit
