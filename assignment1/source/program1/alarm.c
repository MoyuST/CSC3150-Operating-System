#include <stdio.h>
#include <signal.h>
#include <unistd.h>

int main(int argc,char* argv[]){
	printf("------------CHILD PROCESS START------------\n");
	printf("This is the SIGALRM program\n\n");
	alarm(2);
	sleep(5);
	printf("------------CHILD PROCESS END------------\n");

	return 0;
}
