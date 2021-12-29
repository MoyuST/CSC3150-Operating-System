CFILES:= $(shell ls|grep .c)
PROGS:=$(patsubst %.c,%,$(CFILES))

all: $(PROGS)

%:%.c
	$(CC) -o $@ $<

clean:$(PROGS)
	rm $(PROGS)
