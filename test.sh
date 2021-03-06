#!/bin/bash

set -e

scripts=(
    "utils/dataset.py"
    "assignment1.py"
    "assignment2.py"
    "assignment3.py"
    "assignment4.py"
    "assignment4_adagrad.py"
    "assignment5.py"
  )

for i in "${scripts[@]}"
do
  printf "Running $i...\n"
  python $i
done

printf "Finished all scripts without errors.\n"
