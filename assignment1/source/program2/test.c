#include <unistd.h>
#include <stdio.h>
#include <signal.h>

int main(int argc,char* argv[]){
	int i=0;

	printf("--------USER PROGRAM--------\n");
//	alarm(2);
	raise(SIGBUS);
	sleep(5);
	printf("user process success!!\n");
	printf("--------USER PROGRAM--------\n");
	return 100;
}
