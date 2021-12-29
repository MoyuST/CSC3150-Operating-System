#!/bin/bash
mknod /dev/mydev c $1 $2
chmod 666 /dev/mydev
ls -l /dev/mydev
