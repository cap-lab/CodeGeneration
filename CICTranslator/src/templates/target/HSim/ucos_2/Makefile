CC = armcc
AS = armasm
AR = armar

#ASFLAGS = -g
ASFLAGS =
#CFLAGS =--arm --cpu ARM926EJ-S -Ospace -c -O0 -g -I./
CFLAGS =--arm --cpu ARM926EJ-S -Ospace -c -O3 -I./

OBJECTS = init.o os_cpu_a.o os_cpu_c.o os_core.o os_time.o os_q.o os_mutex.o os_mem.o os_flag.o os_task.o os_mbox.o os_sem.o

all : HSimUCOS.a vectors.o

HSimUCOS.a: $(OBJECTS)
	$(AR) -r $@ $(OBJECTS)

vectors.o: vectors.s

.s.o :
	$(AS) $(ASFLAGS) $< -o $@

.c.o :
	$(CC) $(CFLAGS) $< -o $@

clean :
	rm -f *.o *.a

