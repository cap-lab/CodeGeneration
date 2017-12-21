aclocal -I m4config
autoconf -f
autoheader -f
automake --foreign --add-missing
