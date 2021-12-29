#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"

void *dma_buf;
static int dev_major;
static int dev_minor;
static struct cdev *dev_cdevp;
static unsigned int IRQ_NUM = 1;
static int interrupt_num = 0;

//dev_id
struct mydev_ID
{
	unsigned int id;
	unsigned int IRQ_num;
};

static struct mydev_ID mydev_id;

// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2

// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);
static int drv_open(struct inode*, struct file*);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int drv_release(struct inode*, struct file*);
static long drv_ioctl(struct file *, unsigned int , unsigned long );
static irqreturn_t record_interrupt(int irq, void* dev_id);


// cdev file_operations
static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};

// in and out function
void myoutc(unsigned char data,unsigned short int port);
void myouts(unsigned short data,unsigned short int port);
void myouti(unsigned int data,unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);
int prime(int base, short nth);

// Work routine
static struct work_struct *work_routine;

// For input data structure
struct DataIn {
    char a;
    int b;
    short c;
} *dataIn;


// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct* ws);

static irqreturn_t record_interrupt(int irq, void* dev_id){
	int increment = 0;
	if(irq == IRQ_NUM) increment = 1;
	interrupt_num += increment;

	return IRQ_NONE;
}

// Input and output data from/to DMA
void myoutc(unsigned char data,unsigned short int port) {
    *(volatile unsigned char*)(dma_buf+port) = data;
}
void myouts(unsigned short data,unsigned short int port) {
    *(volatile unsigned short*)(dma_buf+port) = data;
}
void myouti(unsigned int data,unsigned short int port) {
    *(volatile unsigned int*)(dma_buf+port) = data;
}
unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char*)(dma_buf+port);
}
unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short*)(dma_buf+port);
}
unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int*)(dma_buf+port);
}

int prime(int base, short nth)
{
    int fnd=0;
    int i, num, isPrime;

    num = base;
    while(fnd != nth) {
        isPrime=1;
        num++;
        for(i=2;i<=num/2;i++) {
            if(num%i == 0) {
                isPrime=0;
                break;
            }
        }
        
        if(isPrime) {
            fnd++;
        }
    }
    return num;
}


static int drv_open(struct inode* ii, struct file* ff) {
	try_module_get(THIS_MODULE);
    	printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}

static int drv_release(struct inode* ii, struct file* ff) {
	module_put(THIS_MODULE);
    	printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}

static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement read operation for your device */
	//read the answer to the user buffer
	int readable = myini(DMAREADABLEADDR);
	if(readable == 1){
		int answer = myini(DMAANSADDR);
		printk("%s:%s(): ans = %d\n", PREFIX_TITLE, __func__, answer);
		put_user(answer, (int *) buffer);
		//set readable to false and clean the result in DMA
		myouti(0, DMAREADABLEADDR);
		myouti(0, DMAANSADDR);
	}
	else{
		printk("%s:%s(): cannot read the data\n", PREFIX_TITLE, __func__);
	}

	return 0;
}

static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement write operation for your device */
	int IOMode;
	char data_a;
	int data_b;
	short data_c;

	//store the data from user
	int * b_addr = (int *) buffer + 1;
	int * c_addr = (int *) buffer + 2;

 	get_user(data_a, (char *) buffer);
	get_user(data_b, (int *) b_addr);
	get_user(data_c, (short *) c_addr);

	myoutc(data_a, DMAOPCODEADDR);
	myouti(data_b, DMAOPERANDBADDR);
	myouts(data_c, DMAOPERANDCADDR);

	IOMode = myini(DMABLOCKADDR);
	
	INIT_WORK(work_routine, drv_arithmetic_routine);
	printk("%s:%s(): queue work\n", PREFIX_TITLE, __func__);

	// Decide io mode
	if(IOMode == 1) {
		// Blocking IO
		myouti(0, DMAREADABLEADDR);
		printk("%s:%s(): block\n", PREFIX_TITLE, __func__);
		schedule_work(work_routine);
		flush_scheduled_work();
		myouti(1, DMAREADABLEADDR);
    	} 
	else {
		// Non-locking IO
		myouti(0, DMAREADABLEADDR);
		schedule_work(work_routine);
   	 }
	return 0;

}

static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
	/* Implement ioctl setting for your device */
	int * addr = (int *) arg;
	unsigned int value_arg = (unsigned int) *addr;

	if(cmd == HW5_IOCSETSTUID){
		myouti(value_arg, DMASTUIDADDR);
		printk("%s:%s(): My STUID is = %d\n",PREFIX_TITLE, __func__, value_arg);
	}
	else if(cmd == HW5_IOCSETRWOK){
		myouti(value_arg, DMARWOKADDR);
		printk("%s:%s(): RW OK\n",PREFIX_TITLE, __func__);
	}
	else if(cmd == HW5_IOCSETIOCOK){
		myouti(value_arg, DMAIOCOKADDR);
		printk("%s:%s(): IOC OK\n",PREFIX_TITLE, __func__);
	}
	else if(cmd == HW5_IOCSETIRQOK){
		myouti(value_arg, DMAIRQOKADDR);
		printk("%s:%s(): IRQ OK\n",PREFIX_TITLE, __func__);
	}
	else if(cmd == HW5_IOCSETBLOCK){

		myouti(value_arg, DMABLOCKADDR);
		
		if(value_arg == 1){
			printk("%s:%s(): Blocking IO\n", PREFIX_TITLE, __func__);
		}
		else if(value_arg == 0){
			printk("%s:%s(): Non-Blocking IO\n", PREFIX_TITLE, __func__);
		}
		else
		{
			printk("%s:%s(): invalid IO\n", PREFIX_TITLE, __func__);
		}
		
	}
	else if(cmd == HW5_IOCWAITREADABLE){
		int readable_signal;
		readable_signal = myini(DMAREADABLEADDR);
		while(true){
			if(readable_signal == 1){
				break;
			}
			msleep(5000);
			readable_signal = myini(DMAREADABLEADDR);
		}
		put_user(1, addr);
		printk("%s:%s(): wait readable %d\n",PREFIX_TITLE, __func__, readable_signal);
	}
	else{
		printk("%s:%s(): invalid instruction\n",PREFIX_TITLE, __func__);
	}

	return 0;
}

static void drv_arithmetic_routine(struct work_struct* ws) {
	/* Implement arthemetic routine */
    char data_a;
	int data_b;
	short data_c;
	int ans;
	int IOMode;

	myouti(0, DMAREADABLEADDR);
	data_a = (char) myinc(DMAOPCODEADDR);
    data_b = myini(DMAOPERANDBADDR);
    data_c = myins(DMAOPERANDCADDR);


    switch(data_a) {
        case '+':
            ans=data_b+data_c;
            break;
        case '-':
            ans=data_b-data_c;
            break;
        case '*':
            ans=data_b*data_c;
            break;
        case '/':
            ans=data_b/data_c;
            break;
        case 'p':
            ans = prime(data_b, data_c);
            break;
        default:
            ans=0;
    }

	myouti(ans, DMAANSADDR);

	IOMode = myini(DMABLOCKADDR);

	//non-blocking mode
	if(IOMode == 0){
		myouti(1, DMAREADABLEADDR);
	}

	printk("%s:%s(): %d %c %d = %d\n",PREFIX_TITLE, __func__, data_b, data_a, data_c, ans);
}

static int __init init_modules(void) {
	
	dev_t dev;
	int ret = 0;
	int irq_return;

	printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);

	//set dev_id
	mydev_id.id = 0;
	mydev_id.IRQ_num = IRQ_NUM;
	irq_return = request_irq(IRQ_NUM, record_interrupt, IRQF_SHARED, "mydev_interrupt", (void *) & mydev_id);
	if(irq_return != 0){
		printk("%s:%s(): Cannot set request_irq\n", PREFIX_TITLE, __FUNCTION__);
	}
	printk("%s:%s(): request_irq %d return %d\n", PREFIX_TITLE, __FUNCTION__, IRQ_NUM, irq_return);

	/* Register chrdev */ 
	//allocate device number
	ret = alloc_chrdev_region(&dev, 0, 1, "mydev");
	if(ret)
	{
		printk("%s:%s(): Cannot alloc chrdev\n", PREFIX_TITLE, __FUNCTION__);
		return ret;
	}

	dev_major = MAJOR(dev);
	dev_minor = MINOR(dev);

	printk("%s:%s(): register chrdev(%d,%d)\n", PREFIX_TITLE, __FUNCTION__,dev_major,dev_minor);

	/* Init cdev and make it alive */
	dev_cdevp = cdev_alloc();

	cdev_init(dev_cdevp, &fops);
	dev_cdevp->owner = THIS_MODULE;
	ret = cdev_add(dev_cdevp, MKDEV(dev_major, dev_minor), 1);

	if(ret < 0)
	{
		printk("%s:%s(): Add chrdev failed\n", PREFIX_TITLE, __FUNCTION__);
		return ret;
	}

	/* Allocate DMA buffer */
	dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL);
	printk("%s:%s(): allocate dma buffer\n", PREFIX_TITLE, __FUNCTION__);
	
	/* Allocate work routine */
	work_routine = kmalloc(sizeof(typeof(*work_routine)), GFP_KERNEL);

	return 0;
}

static void __exit exit_modules(void) {
	
	dev_t dev;

	printk("%s:%s(): interrupt count=%d\n", PREFIX_TITLE, __func__, interrupt_num);
	free_irq(IRQ_NUM, (void *) &mydev_id);

	/* Free DMA buffer when exit modules */
	kfree(dma_buf);
	printk("%s:%s(): free dma buffer\n", PREFIX_TITLE, __func__);

	/* Delete character device */
	dev = MKDEV(dev_major, dev_minor);
	cdev_del(dev_cdevp);
	unregister_chrdev_region(dev, 1);
	printk("%s:%s(): unregister chrdev\n", PREFIX_TITLE, __func__);

	/* Free work routine */
	kfree(work_routine);

	printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);
