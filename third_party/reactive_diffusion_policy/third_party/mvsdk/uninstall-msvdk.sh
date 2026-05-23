#!/bin/bash

CURDIR=`pwd`
echo "Your current directory is $CURDIR. This is where the MVSDK software will be installed..."
CURR_USER=`whoami`
CURR_ARCH=`arch`

if [ $CURR_USER != 'root' ]; then
   echo "You have to be root to run this script"
   echo "Fail !!!"
   exit 1;
fi

rm /etc/udev/rules.d/88-mvusb.rules 
echo "Remove /etc/udev/rules.d/88-mvusb.rules"

rm /lib/libMVSDK.so
echo "Remove lib/libMVSDK.so"

echo "Successful"
echo "Please  restart system  now!!!"
