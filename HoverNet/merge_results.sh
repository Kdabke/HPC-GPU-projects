#!/bin/bash


awk 'NR>1 {print $0}' $1 > ./temp_results/$1
