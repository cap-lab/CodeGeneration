#CC = arm-linux-gnueabi-gcc
#LD = arm-linux-gnueabi-gcc
CC = armcc

#LDFLAGS = --info totals --info unused --map --list image.map --ro-base 0x0 --rw-base 0x00040000 --first "vectors.o(Vect)" --entry 0x0 
#LDFLAGS = --info totals --info unused --map --list image.map --ro-base 0x0 --rw-base 0x00040000 --first "vectors.o(Vect)" --entry 0x0 

CFLAGS = -O0

all : esim_lib.o

clean :
	rm -f *.o

esim_lib.o : esim_lib.c
	$(CC) $(CFLAGS) -c esim_lib.c -o esim_lib.o 

