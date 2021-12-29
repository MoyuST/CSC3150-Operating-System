This is the readme of Assignment_1

CONTENTS:
	1.PROGRAM1
	2.PROGRAM2
	3.BOBUS
	
PROGRAM1:
	Under the 'program1' directory lies all source codes of Task 1 and test cases.
	The 'program1.c' is the main program, the others are for test uses.
	The directory consists of the following files:
	program1.c, Makefile, abort.c, alarm.c, bus.c, floating.c, hangup.c, illegal_instr.c,
	interrupt.c, kill.c, normal.c, pipe.c, quit.c, segment_fault.c, stop.c, terminate.c,
	trap.c.
	
	
	HOW TO COMPILE:
		In the 'program1' directory, type 'make' command and enter.
		
	HOW TO CLEAR:
		In the 'program1' directory, type 'make clean' command and enter.
		
	HOW TO EXECUTE:
		In the 'program1' directory, type './program1 $TEST_CASE $ARG1 $ARG2 ...',
		where $TEST_CASE is the name of test program and $ARG1, $ARG2,... 
		are names of arguments that the test program could have.
		
PROGRAM2:
	Under the 'program2' directory lies all source codes of Task 2 and one test case.
	The 'program2.c' is the main program, and 'test.c' is for test use.
	
	BEFORE PROGRAM COMPILATION AND EXECUTION:
		Revising Linux Kernel is needed, as is shown in the following steps:
		1. Update the Linux source code.
		2. Compile the kernel and boot image, replace the boot image with new one, then reboot.
		
		To compile to test program, simply type 'gcc -o $FILENAME $FILENAME.c', where
		$FILENAME is the file name without the extension.
	
	HOW TO COMPILE:
		In the 'program2' directory, type 'make' command and enter
		
	HOW TO CLEAR:
		In the 'program2' directory, type 'make clean' command and enter.
		
	HOW TO EXECUTE:
		1.Type 'sudo insmod program2.ko' under 'program2' directory and enter
		2.You could see messages appear by typing 'dmesg' command
		  The messages are between the messages 'module init' and 'module exit'.
		3.Type 'sudo rmmod program2' and enter to remove the program2 module.
		
BONUS:
	Under the 'bonus' directory lies all source codes of bonus tasks and several test cases.
	The 'myfork.c' is the main program. Other *.c files are test cases.
	
	HOW TO COMPILE:
		In the 'bonus' directory, type 'make' command and enter. 
		Test programs could be compiled as well.
		
	HOW TO CLEAR:
		In the 'bonus' directory, type 'make clean' command and enter.
		
	HOW TO EXECUTE:
		In the 'bonus' directory, type './myfork $TEST_PRO1 $TEST_PRO2 $TEST_PRO3 ...',
		where $TEST_PRO1, $TEST_PRO2,... are names of programs myfork executes.
