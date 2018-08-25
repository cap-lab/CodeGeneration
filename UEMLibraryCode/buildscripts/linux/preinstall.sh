#!/bin/sh

libtoolize --copy -f
if [ "$?" != "0" ]; then
    echo "[Error] libtoolize failed!" 1>&2
    exit 1
fi

aclocal -I m4config
if [ "$?" != "0" ]; then
    echo "[Error] aclocal failed!" 1>&2
    exit 1
fi

autoconf -f
if [ "$?" != "0" ]; then
    echo "[Error] autoconf failed!" 1>&2
    exit 1
fi

autoheader -f
if [ "$?" != "0" ]; then
    echo "[Error] autoheader failed!" 1>&2
    exit 1
fi

automake -c --foreign --add-missing
if [ "$?" != "0" ]; then
    echo "[Error] automake failed!" 1>&2
    exit 1
fi

