#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <string.h>

#define maxChildProcess 30
#define maxFileLength 50


char* processTerminatedSignal[] = {
	"SIGHUP",      "SIGINT",       "SIGQUIT",      "SIGILL",      "SIGTRAP",
	"SIGABRT",     "SIGBUS",        "SIGFPE",       "SIGKILL",     NULL,
    "SIGSEGV",         NULL,       "SIGPIPE",     "SIGALRM",    "SIGTERM"
};

char* signalInfomation[] = {
	"is hang up by hangup signal",
	"is interrupted by interrupt signal",
	"is quited by quit signal",
	"gets illegal instruction",
	"is terminated by trap signal",
	"is abort by abort signal",
	"gets bus error",
	"gets floating point exception",
	"is killed by kill signal",
	NULL,
	"uses invalid memory reference",
	NULL,
	"writes to pipe with no readers",
	"is terminated by alarm signal",
	"is terminated by termaniation signal",
};

void checking(pid_t parent, pid_t child, int status, int myfork);

static char filenames[maxChildProcess][maxFileLength];


void recursiveProcess(int currentProcess, int processTotal, int * siganlList, pid_t* pidList);

int main(int argc,char *argv[]){
	
	int * siganlListPtr = mmap(NULL, sizeof(int)*maxChildProcess, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
	pid_t * pidListPtr = mmap(NULL, sizeof(pid_t)*maxChildProcess, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);

	for(int i=0;i<argc;i++){
		strcpy(filenames[i], argv[i]);
	}

	pid_t pid;
	int status;
	pid = fork();

	if(pid==-1){
		perror("myfork fork");
		exit(1);
	}
	else{
		if(pid==0){
			recursiveProcess(0, argc, siganlListPtr, pidListPtr);
		}
		else{
			waitpid(pid, &status, WUNTRACED);
			pidListPtr[0] = pid;
			siganlListPtr[0] = status;
			printf("the process tree : %d", pidListPtr[0]);

			for(int i=1;i<argc;i++){
				printf("->%d", pidListPtr[i]);
			}
			printf("\n");

			for(int i=argc-1;i>=1;i--){
				checking(pidListPtr[i], pidListPtr[i-1], siganlListPtr[i], 1);
				printf("\n");
			}

			//checking myfork
			checking(0, pidListPtr[0], 0, 0);
			printf("\n");
			

		}
	}

	return 0;
}


void recursiveProcess(int currentProcess, int processTotal, int *signalList, pid_t* pidList){

	if(currentProcess < processTotal-1){
		pid_t pid;
		int status;
		pid = fork();

		if(pid==-1){
			perror("fork");
			exit(1);
		}
		else{
			//child process
			if(pid==0){
				recursiveProcess(currentProcess+1, processTotal, signalList, pidList);
			}
			//parent process
			else{
				waitpid(pid, &status, WUNTRACED);
				pidList[currentProcess+1] = pid;
				signalList[currentProcess+1] = status;
			}
		}
	}

	//execute the files
	if(strcmp(filenames[currentProcess], "./myfork")){
		int j;
		char *arg[2];
		//copy the execute file
		arg[0] = filenames[currentProcess];
		arg[1] = NULL;
		execve(arg[0],arg,NULL);
		exit(0);
	}
}

void checking(pid_t parent, pid_t child, int status, int myfork){
	if(myfork == 0) printf("myfork(pid=%d) ", child);
	else printf("The child process (pid=%d) of the parent process(pid=%d)", parent, child);
	if(WIFEXITED(status)){
		printf("has normal execution\n");
		printf("the exit status = %d\n",WEXITSTATUS(status));
	}
	//process failure
	else if(WIFSIGNALED(status)){
		int terminationStatus = WTERMSIG(status);
		printf("is termiated by siganl\n");
		printf("Its signal number = %d\n", terminationStatus);

		if(terminationStatus>=1 && terminationStatus <=15 
		&& processTerminatedSignal[terminationStatus-1]!=NULL){
			printf("child process get %s signal\n", processTerminatedSignal[terminationStatus-1]);
			printf("child process %s\n", signalInfomation[terminationStatus-1]);
		}
		else{
			printf("child process get a signal not in samples\n");
		}
		printf("CHILD EXECUTION FAILED!!\n");
	}
	//process stopped
	else if(WIFSTOPPED(status)){
		int stopStatus = WSTOPSIG(status);
		printf("is stopped by siganl\n");
		printf("Its signal number = %d\n", stopStatus);
		if(stopStatus==SIGSTOP){
			printf("child process get SIGSTOP signal\n");
			printf("child process stopped\n");
		}
		else{
			printf("child process get a signal not in the samples\n");
			printf("child process stopped\n");
		}
		printf("CHILD PROCESS STOPPED\n");
	}
	else{
		printf("CHILD PROCESS CONTINUED\n");
	}
}
