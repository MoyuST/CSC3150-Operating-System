#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void user_program(FileSystem *fs, uchar *input, uchar *output) {
	
	
	/////////////// Test Case 1  ///////////////
	//u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
	//fs_write(fs, input, 64, fp);
	//fp = fs_open(fs, "b.txt\0", G_WRITE);
	//fs_write(fs, input + 32, 32, fp);
	//fp = fs_open(fs, "t.txt\0", G_WRITE);
	//fs_write(fs, input + 32, 32, fp);
	//fp = fs_open(fs, "t.txt\0", G_READ);
	//fs_read(fs, output, 32, fp);
	//fs_gsys(fs,LS_D);
	//fs_gsys(fs, LS_S);
	//fp = fs_open(fs, "b.txt\0", G_WRITE);
	//fs_write(fs, input + 64, 12, fp);
	//fs_gsys(fs, LS_S);
	//fs_gsys(fs, LS_D);
	//fs_gsys(fs, RM, "t.txt\0");
	//fs_gsys(fs, LS_S);


	
	/////////////// Test Case 2  ///////////////
	//u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
	//fs_write(fs,input, 64, fp);
	//fp = fs_open(fs,"b.txt\0", G_WRITE);
	//fs_write(fs,input + 32, 32, fp);
	//fp = fs_open(fs,"t.txt\0", G_WRITE);
	//fs_write(fs,input + 32, 32, fp);
	//fp = fs_open(fs,"t.txt\0", G_READ);
	//fs_read(fs,output, 32, fp);
	//fs_gsys(fs,LS_D);
	//fs_gsys(fs,LS_S);
	//fp = fs_open(fs,"b.txt\0", G_WRITE);
	//fs_write(fs,input + 64, 12, fp);
	//fs_gsys(fs,LS_S);
	//fs_gsys(fs,LS_D);
	//fs_gsys(fs,RM, "t.txt\0");
	//fs_gsys(fs,LS_S);
	//char fname[10][20];
	//for (int i = 0; i < 10; i++)
	//{
	//	fname[i][0] = i + 33;
	//	for (int j = 1; j < 19; j++)
	//		fname[i][j] = 64 + j;
	//	fname[i][19] = '\0';
	//}

	//for (int i = 0; i < 10; i++)
	//{
	//	fp = fs_open(fs,fname[i], G_WRITE);
	//	fs_write(fs,input + i, 24 + i, fp);
	//}

	//fs_gsys(fs,LS_S);

	//for (int i = 0; i < 5; i++)
	//	fs_gsys(fs,RM, fname[i]);

	//fs_gsys(fs,LS_D);
	
	
	/////////////// Test Case 3  ///////////////
	u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input, 64, fp);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_READ);
	fs_read(fs, output, 32, fp);
	fs_gsys(fs, LS_D);
	fs_gsys(fs, LS_S);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 64, 12, fp);
	fs_gsys(fs, LS_S);
	fs_gsys(fs, LS_D);
	fs_gsys(fs, RM, "t.txt\0");
	fs_gsys(fs, LS_S);

	char fname[10][20];
	for (int i = 0; i < 10; i++)
	{
		fname[i][0] = i + 33;
		for (int j = 1; j < 19; j++)
			fname[i][j] = 64 + j;
		fname[i][19] = '\0';
	}

	for (int i = 0; i < 10; i++)
	{
		fp = fs_open(fs, fname[i], G_WRITE);
		fs_write(fs, input + i, 24 + i, fp);
	}

	fs_gsys(fs, LS_S);

	for (int i = 0; i < 5; i++)
		fs_gsys(fs, RM, fname[i]);

	fs_gsys(fs, LS_D);

	char fname2[1018][20];
	int p = 0;

	for (int k = 2; k < 15; k++)
		for (int i = 50; i <= 126; i++, p++)
		{
			fname2[p][0] = i;
			for (int j = 1; j < k; j++)
				fname2[p][j] = 64 + j;
			fname2[p][k] = '\0';
		}

	for (int i = 0; i < 1001; i++)
	{
		fp = fs_open(fs, fname2[i], G_WRITE);
		fs_write(fs, input + i, 24 + i, fp);
	}

	fs_gsys(fs, LS_S);
	fp = fs_open(fs, fname2[1000], G_READ);
	fs_read(fs, output + 1000, 1024, fp);

	char fname3[17][3];
	for (int i = 0; i < 17; i++)
	{
		fname3[i][0] = 97 + i;
		fname3[i][1] = 97 + i;
		fname3[i][2] = '\0';
		fp = fs_open(fs, fname3[i], G_WRITE);
		fs_write(fs, input + 1024 * i, 1024, fp);
	}

	fp = fs_open(fs, "EA\0", G_WRITE);
	fs_write(fs, input + 1024 * 100, 1024, fp);
	fs_gsys(fs, LS_S);
	
}
