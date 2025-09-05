#!/bin/bash
echo "Arg 1: $1"
rm -r ./.temp
mkdir .temp
scp -r marvin:$1 ./.temp
papers .temp