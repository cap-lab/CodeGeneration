libtoolize --copy -f
aclocal -I m4config
autoconf -f
autoheader -f
automake -c --foreign --add-missing
