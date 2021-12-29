#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>
#include <linux/wait.h>

MODULE_LICENSE("GPL");

struct wait_opts{
		enum pid_type wo_type;
		int wo_flags;
		struct pid *wo_pid;
		struct siginfo __user *wo_info;
		int __user *wo_stat;
		struct rusage __user *wo_rusage;
		wait_queue_t child_wait;
		int notask_error;
};

static struct task_struct *task;
extern long _do_fork( unsigned long clone_flags,
	   				  unsigned long stack_start,
					  unsigned long stack_size,
					  int __user *parent_tidptr,
					  int __user *child_tidptr,
					  unsigned long tls);

extern int do_execve( struct filename *filename,
	 				  const char __user *const __user *__argv,
					  const char __user *const __user *__envp);

extern long do_wait(struct wait_opts *wo);

extern struct filename *getname(const char __user *filename);

char* processTerminatedSignal[] = {
	"SIGHUP",      "SIGINT",       "SIGQUIT",      "SIGILL",      "SIGTRAP",
	"SIGABRT",     "SIGBUS",        "SIGFPE",       "SIGKILL",     NULL,
    "SIGSEGV",         NULL,       "SIGPIPE",     "SIGALRM",    "SIGTERM"
};

int my_WEXITSTATUS(int status){
	return ((status & 0xff00)>>8);
}

int my_WTERMSIG(int status){
	return (status & 0x7f);
}

int my_WSTOPSIG(int status){
	return (my_WEXITSTATUS(status));
}

int my_WIFEXITED(int status){
	return (my_WTERMSIG(status)==0);
}

signed char my_WIFSIGNALED(int status){
	return (((signed char) (((status & 0x7f) + 1) >> 1) ) > 0);
}


int my_WIFSTOPPED(int status){
	return (((status) & 0xff) == 0x7f);
}


//execute the test.c
int my_exec(void){
	int result;
	const char path[] = "/home/seed/work/assignment1/source/program2/test";
	const char *const argv[] = {path, NULL, NULL};
	const char *const envp[] = {"HOME=/", "PATH=/sbin:/user/sbin:/bin:/usr/bin", NULL};

	struct filename * my_filename = getname(path);

	printk("[program2] : child process");
	result = do_execve(my_filename, argv, envp);

	if(!result){
		return 0;
	}

	do_exit(result);
}


int my_wait(pid_t pid){
	int status;
	int a;
	
	// int terminatedStatus;
	struct wait_opts wo;
	struct pid * wo_pid = NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);

	wo.wo_type   = type;
	wo.wo_pid    = wo_pid;
	wo.wo_flags  = WEXITED|WUNTRACED;
	wo.wo_info   = NULL;
	wo.wo_stat   = (int __user*) &status;
	wo.wo_rusage = NULL;

	do_wait(&wo);
	a = *(wo.wo_stat);

	put_pid(wo_pid);

	return a;
}

//implement fork function
int my_fork(void *argc){
	
	
	//set default sigaction for current process
	int i;
	pid_t pid;
	int status;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	
	/* fork a process using do_fork */
	pid = _do_fork(SIGCHLD, (unsigned long)& my_exec, 0, NULL, NULL, 0);


	printk("[program2] : The Child process has pid = %d\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n", (int)current->pid);

	status = my_wait(pid);

	//checking the return status
	if(my_WIFEXITED(status)){
		printk("[program2] : child process gets normal termination\n");
		printk("[program2] : The return signal is %d", status);
	}
	else if(my_WIFSTOPPED(status)){
		int stopStatus = my_WSTOPSIG(status);
		printk("[program2] : CHILD PROCESS STOPPED\n");
		if(stopStatus == 19 ){
			printk("[program2] : child process get SIGSTOP signal\n");
		}
		else{
			printk("[program2] : child process get a siganl not in the samples\n");
		}
		printk("[program2] : The return signal is %d", stopStatus);
	}
	else if(my_WIFSIGNALED(status)){
		int terminationStatus = my_WTERMSIG(status);
		printk("[program2] : CHILD EXECUTION FAILED!!\n");
		if(terminationStatus>=1 && terminationStatus <=15 && processTerminatedSignal[terminationStatus-1]!=NULL){
			printk("[program2] : child process get %s signal\n", processTerminatedSignal[status-1]);
		}
		else{
			printk("[program2] : child process get a signal not in samples\n");
		}
		printk("[program2] : The return signal is %d", terminationStatus);
	}
	else{
		printk("[program2] : CHILD PROCESS CONTINUED\n");
	}
	do_exit(0);

	return 0;
}


static int __init program2_init(void){

	printk("[program2] : module_init\n");
	printk("[program2] : module_init create kthread start\n");
	
	/* create a kernel thread to run my_fork */
	task = kthread_create(&my_fork, NULL, "Mythread");

	//wake up new thread if ok
	if(!IS_ERR(task)){
		printk("[program2] : module_init Kthread starts\n");
		wake_up_process(task);
	}

	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
