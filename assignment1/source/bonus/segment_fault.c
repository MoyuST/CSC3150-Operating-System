#include <stdio.h>
#include <signal.h>
#include <unistd.h>

int main(int argc,char* argv[]){
	printf("------------CHILD PROCESS START------------\n");
	printf("This is the SIGSEGV program\n\n");
	raise(SIGSEGV);
	sleep(5);
	printf("------------CHILD PROCESS END------------\n");

	return 0;
}
