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


cp 88-mvusb.rules /etc/udev/rules.d/

if [ $CURR_ARCH == 'x86_64' ]; then
	cp lib/x64/libMVSDK.so /lib
	echo "Copy x64/libMVSDK.so to /lib"
elif [ $CURR_ARCH == 'aarch64' ]; then
	cp lib/arm64/libMVSDK.so /lib
	echo "Copy arm64/libMVSDK.so to /lib"
elif [[ ${CURR_ARCH:2} == '86' ]]; then
	cp lib/x86/libMVSDK.so /lib
	echo "Copy x86/libMVSDK.so to /lib"
else
	cp lib/arm/libMVSDK.so /lib
	echo "Copy arm/libMVSDK.so to /lib"
fi

echo "Successful"
echo "Please  restart system  now!!!"
