#!/bin/bash
set -e 

bash train_default_test.sh config.sh 

bash train_seg_test.sh config.sh 

bash train_cls_test.sh config.sh 
