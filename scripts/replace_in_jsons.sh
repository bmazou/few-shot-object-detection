#!/bin/bash

for entry in ./*.json
do
    sed -i "s/$1/$2/" "$entry"
done
