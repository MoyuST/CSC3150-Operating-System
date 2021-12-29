#!/bin/sh

for var in 2 3 4 5 6 7 8 9 10
do
	prefix="normal"
	postfix=".c"
	i=$base$var
	name=$prefix$i$postfix
	eval cp normal1.c $name
	eval sed -i 's/1/$i/g' $name
done
