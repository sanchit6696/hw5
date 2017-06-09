CC = gcc
FLAGS = -O3 -Wall -g
EXECS = multigrid_2d

all: ${EXECS}

multigrid_2d: multigrid_2d.c
	${CC} ${FLAGS} $^ -o multigrid_2d


clean:
	rm -f ${EXECS}