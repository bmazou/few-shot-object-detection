#!/bin/bash

cat *.json  | tr "," "\n" | grep "coco_url" | cut -f 2- -d: | cut -f 2 -d\" > links.txt
