#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

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

int main(int argc, char *argv[]){

	pid_t pid;
	int status;

	/* fork a child process */
	printf("Process start to fork\n");
	pid = fork();

	if(pid==-1){
		perror("fork");
		exit(1);
	}
	else{
		//child process
		if(pid==0){
			printf("I'm the Child Process, my pid = %d\n", getpid());
			int i;
			char *arg[argc];
			//copy the execute file
			for(i=0;i<argc-1;i++){
				arg[i]=argv[i+1];
			}
			arg[argc-1]=NULL;

			/* execute test program */ 
			printf("Child process start to execute the program\n");
			execve(arg[0],arg,NULL);
			exit(EXIT_FAILURE);
		}

		//parent process
		else{
			printf("I'm the Parent Process, my pid = %d\n", getpid());

			/* wait for child process terminates */
			waitpid(pid, &status, WUNTRACED);
			printf("Parent process receiving the SIGCHLD signal\n");

			/* check child process'  termination status */
			//normal termination
			if(WIFEXITED(status)){
                printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
            }
			//process failure
            else if(WIFSIGNALED(status)){
				int terminationStatus = WTERMSIG(status);
				if(terminationStatus>=1 && terminationStatus <=15 && processTerminatedSignal[terminationStatus-1]!=NULL){
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
            exit(0);
		}
	}	
}
