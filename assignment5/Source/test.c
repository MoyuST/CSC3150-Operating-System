#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<fcntl.h>
#include<sys/ioctl.h>
#include <unistd.h>
#include "ioc_hw5.h"

struct dataIn {
    char a;
    int b;
    short c;
};

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

int arithmetic(int fd, char operator, int operand1, short operand2)
{
    struct dataIn data;
    int ans;
    int readable;
    int ret;

    data.a = operator;
    data.b = operand1;
    data.c = operand2;

    switch(data.a) {
        case '+':
            ans=data.b+data.c;
            break;
        case '-':
            ans=data.b-data.c;
            break;
        case '*':
            ans=data.b*data.c;
            break;
        case '/':
            ans=data.b/data.c;
            break;
        case 'p':
            ans = prime(data.b, data.c);
            break;
        default:
            ans=0;
    }

    printf("%d %c %d = %d\n\n", data.b, data.a, data.c, ans);

    /******************Blocking IO******************/
    printf("Blocking IO\n");
    ret = 1;
    if (ioctl(fd, HW5_IOCSETBLOCK, &ret) < 0) {
        printf("set blocking failed\n");
        return -1;
    }

    write(fd, &data, sizeof(data));

    //Do not need to synchronize
    //But need to wait computation completed

    read(fd, &ret, sizeof(int));

    printf("ans=%d ret=%d\n\n", ans, ret);
    /***********************************************/

    /****************Non-Blocking IO****************/
    printf("Non-Blocking IO\n");
    ret = 0;
    if (ioctl(fd, HW5_IOCSETBLOCK, &ret) < 0) {
        printf("set non-blocking failed\n");
        return -1;
    }

    printf("Queueing work\n");
    write(fd, &data, sizeof(data));

    //Can do something here
    //But cannot confirm computation completed

    printf("Waiting\n");
    //synchronize function
    ioctl(fd, HW5_IOCWAITREADABLE, &readable);

    if(readable==1){
        printf("Can read now.\n");
        read(fd, &ret, sizeof(int));
    }

    printf("ans=%d ret=%d\n\n", ans, ret);
    /***********************************************/

    return ans;
}

int main()
{
    printf("...............Start...............\n");

    //open my char device:
    int fd = open("/dev/mydev", O_RDWR);
    if(fd == -1) {
        printf("can't open device!\n");
        return -1;
    }

    int ret;

    ret = 118010224;
    if (ioctl(fd, HW5_IOCSETSTUID, &ret) < 0) {
        printf("set stuid failed\n");
        return -1;
    }

    ret = 1;
    if (ioctl(fd, HW5_IOCSETRWOK, &ret) < 0) {
        printf("set rw failed\n");
        return -1;
    }

    ret = 1;
    if (ioctl(fd, HW5_IOCSETIOCOK, &ret) < 0) {
        printf("set ioc failed\n");
        return -1;
    }

    ret = 1;
    if (ioctl(fd, HW5_IOCSETIRQOK, &ret) < 0) {
        printf("set irq failed\n");
        return -1;
    }

    //arithmetic(fd, '+', 100, 10);
    //arithmetic(fd, '-', 100, 10);
    //arithmetic(fd, '*', 100, 10);
    //arithmetic(fd, '/', 100, 10);
    arithmetic(fd, 'p', 100, 10000);
    //arithmetic(fd, 'p', 100, 20000);


    printf("...............End...............\n");

    return 0;
}
