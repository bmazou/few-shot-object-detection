#!/bin/bash

for entry in ./*.txt
do
    while read -r line
    do 
        x=$( echo "$line" | cut -d " " -f2 )

        echo "$x"
    done < "$entry"
        # x1=
        # cat "$entry"
        echo "--------"
done

# while read -r line
# do 
#     echo "$line"
#     echo "Další lajna"
# done < "goat1.txt"
