#!/bin/bash
echo "Arg 1: $1"
scp marvin:$1 ./.temp
papers .temp
