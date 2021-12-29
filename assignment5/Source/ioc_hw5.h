#ifndef IOC_HW5_H
#define IOC_HW5_H

#define HW5_IOC_MAGIC         'k'
#define HW5_IOCSETSTUID       _IOW(HW5_IOC_MAGIC, 1, int)
#define HW5_IOCSETRWOK        _IOW(HW5_IOC_MAGIC, 2, int)
#define HW5_IOCSETIOCOK       _IOW(HW5_IOC_MAGIC, 3, int)
#define HW5_IOCSETIRQOK       _IOW(HW5_IOC_MAGIC, 4, int)
#define HW5_IOCSETBLOCK       _IOW(HW5_IOC_MAGIC, 5, int)
#define HW5_IOCWAITREADABLE   _IOR(HW5_IOC_MAGIC, 6, int)
#define HW5_IOC_MAXNR         6

#endif




