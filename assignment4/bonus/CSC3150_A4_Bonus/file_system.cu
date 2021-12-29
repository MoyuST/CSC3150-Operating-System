#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 global_storage_end = 0;
__device__ __managed__ int level_2  = -1;
__device__ __managed__ int level_3 = -1;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  //init superblock
  for (int i = 0; i < SUPERBLOCK_SIZE; i++) {
	  fs->volume[i] = 0;
  }

  //init FCB
  //FCB: 0-19   name
  //	 20-21  address 
  //	 22-23  size (valid bit at 22)
  //	 24-25  created time
  //	 26-27  modified time
  //	 28		entry type/files num (0xff means file, directory otherwise)
  //	 29-30	identifier
  for (int i = 0; i < FCB_ENTRIES; i++) {
	  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22] = 0xff; //set to invalid
	  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28] = 0xff; //set to file type
	  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 29] = 0xff; //set to identifier
	  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 30] = 0xff; //set to identifier
  }

  //create root directory
  char root[] = "root\0";
  u32 fp = fs_open(fs, root, G_WRITE);
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*fp + 28] = 0;
}

__device__ bool compare_string(char * s1, char * s2) {
	while (*s1 != '\0' && *s2 != '\0' && *s1 == *s2) {
		s1++;
		s2++;
	}

	if (*s1 == '\0' && *s2 == '\0') {
		return true;
	}
	return false;
}

__device__ void prt_string(uchar * s) {
	while (*s != '\0') {
		printf("%c", (char) *s);
		s++;
	}
}

__device__ u32 set_superblock(FileSystem *fs, int block_addr, int type) {
	int row = block_addr / 8;
	int column = block_addr % 8;
	uchar mask;
	mask = (1 << column);

	if (type == 0) {
		mask = ~mask;
		fs->volume[row] &= mask;
	}
	else {
		fs->volume[row] |= mask;
	}
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	u32 FCB_pt = 0x10000000;
	int empty_entry = -1;

	int cur_directory = 0;
	if (level_3 != -1) cur_directory = level_3;
	else if (level_2 != -1) cur_directory = level_2;

	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		//checking valid files
		if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22] != 0xff) {
			if (compare_string((char *) &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i], s)) {
				int id = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 29] * 256;
				id += fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 30];
				if (id == cur_directory) {
					FCB_pt = i;
					break;
				}
			}
		}
		else {
			if (empty_entry == -1) empty_entry = i;
		}
	}

	//find the file
	if (FCB_pt != 0x10000000) {
		return FCB_pt;
	}
	else {
		if (op == G_WRITE) {
			if (empty_entry == -1) {
				printf("files number reach the maximun\n");
				return FCB_pt;
			}
			else {
				uchar temp = *s;
				int FCB_entry_address = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * empty_entry;

				//set new file name
				int name_count = 0;
				char * ss = s;
				while (temp != '\0') {
					fs->volume[FCB_entry_address + name_count] = temp;
					ss++;
					name_count++;
					temp = *ss;
					if (name_count+1 == fs->MAX_FILENAME_SIZE) {
						printf("file name exceed the limit\n");
						return FCB_pt;
					}
				}

				int base_addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*cur_directory;
				if (empty_entry != 0 && fs->volume[base_addr + 28] + 1 > 50) {
					printf("the number of files in the direcoty reach the maximun\n");
					return;
				}

				//increment the directory files count
				fs->volume[base_addr + 28]++;
				
				fs->volume[FCB_entry_address + name_count] = '\0';
				name_count++;

				//set size
				fs->volume[FCB_entry_address + 22] = 0;
				fs->volume[FCB_entry_address + 23] = 0;

				//set create time
				fs->volume[FCB_entry_address + 24] = gtime / 256;
				fs->volume[FCB_entry_address + 25] = gtime % 256;

				//set modified time
				fs->volume[FCB_entry_address + 26] = gtime / 256;
				fs->volume[FCB_entry_address + 27] = gtime % 256;
				gtime++;


				FCB_pt = empty_entry;

				if (empty_entry != 0) {

					int size = fs->volume[base_addr + 22] * 256 + fs->volume[base_addr + 23];

					//set identifier
					fs->volume[FCB_entry_address + 29] = cur_directory / 256;
					fs->volume[FCB_entry_address + 30] = cur_directory % 256;


					//set size for the directory
					int direcort_size = fs->volume[base_addr + 22] * 256;
					direcort_size += fs->volume[base_addr + 23];

					fs->volume[base_addr + 22] = (direcort_size + name_count) / 256;
					fs->volume[base_addr + 23] = (direcort_size + name_count) % 256;
				}
			}
		}
		return FCB_pt;
	}
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	//check whether fp is valid
	if (fp == 0x10000000 || fp >= 1024) {
		printf("invalid file pointer\n");
		return;
	}

	if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22] == 0xff) {
		printf("invalid file\n");
		return;
	}

	//check whether size exceed the file size
	int file_size = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22];
	file_size *= 256;
	file_size += fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 23];
	if (size > file_size) {
		printf("read exceed the file size\n");
		return;
	}

	int storage_addr = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 20];
	storage_addr *= 256;
	storage_addr += fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 21];
	storage_addr *= fs->STORAGE_BLOCK_SIZE;
	storage_addr += fs->FILE_BASE_ADDRESS;

	for (int i = 0; i < size; i++) {
		output[i] = fs->volume[storage_addr + i];
	}
}


__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	if (size >= fs->MAX_FILE_SIZE) {
		printf("new size exceed the limit of the file size\n");
		return 0x10000000;
	}

	//check whether fp is valid
	if (fp == 0x10000000 || fp >= 1024) {
		printf("invalid file pointer\n");
		return 0x10000000;
	}

	if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22] == 0xff) {
		printf("invalid file\n");
		return 0x10000000;
	}

	int new_addr = fs_move(fs, size, fp);

	for (int i = 0; i < size; i++) {
		fs->volume[new_addr + i] = input[i];
	}

	int shift_block = size / fs->STORAGE_BLOCK_SIZE;
	if (size % fs->STORAGE_BLOCK_SIZE != 0) {
		shift_block++;
	}

	int old_size = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22] * 256;
	old_size += fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 23];

	if (size >= old_size) {
		global_storage_end += shift_block * fs->STORAGE_BLOCK_SIZE;
	}

	int start_block = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 20];
	start_block = start_block * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 21];

	//set superblock
	for (int i = 0; i < shift_block; i++) {
		set_superblock(fs, start_block + i, 1);
	}

	start_block = global_storage_end / fs->STORAGE_BLOCK_SIZE;

	//clear superblock
	if (size < old_size) {
		int remain_blocks = old_size - size;
		remain_blocks = remain_blocks / fs->STORAGE_BLOCK_SIZE;
		if (remain_blocks % fs->STORAGE_BLOCK_SIZE != 0) {
			remain_blocks++;
		}

		for (int i = 0; i < remain_blocks; i++) {
			set_superblock(fs, start_block + i, 0);
		}
	}
	
	int fp_modified_time = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 26];
	fp_modified_time = fp_modified_time * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 27];
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		if (i != fp && fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22] != 0xff) {
			int original_modifid_time = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 26];
			original_modifid_time = original_modifid_time * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 27];

			if (original_modifid_time > fp_modified_time) {
				original_modifid_time--;
				fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 26] = original_modifid_time / 256;
				fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 27] = original_modifid_time % 256;
			}
		}
	}

	fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22] = size / 256;
	fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 23] = size % 256;
	

	fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 26] = (gtime - 1) / 256;
	fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 27] = (gtime - 1) % 256;

	if (fp != 0) {
		int cur_directory = 0;
		if (level_3 != -1) cur_directory = level_3;
		else if (level_2 != -1) cur_directory = level_2;

		fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*cur_directory + 28]++;
	}
	
	return 0;
}

__device__ bool less_than(FileSystem *fs, u32 fp1, u32 fp2) {
	int size1 = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp1 + 22];
	size1 = size1 * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp1 + 23];

	int size2 = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp2 + 22];
	size2 = size2 * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp2 + 23];

	if (size1 < size2) {
		return true;
	}
	else if (size1 > size2) {
		return false;
	}
	else {
		int time1 = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp1 + 24];
		time1 = time1 * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp1 + 25];

		int time2 = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp2 + 24];
		time2 = time2 * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp2 + 25];

		if (time1 > time2) {
			return true;
		}
		return false;
	}
}

__device__ void fs_gsys(FileSystem *fs, int op)
{

	int cur_directory = 0;
	if (level_3 != -1) cur_directory = level_3;
	else if (level_2 != -1) cur_directory = level_2;

	if (op == LS_D) {
		printf("===sort by modified time===\n");
		int cur_file_count = gtime - 1;
		for (int i = 0; i < gtime; i++) {
			for (int i = 0; i < fs->FCB_ENTRIES; i++) {
				if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22] != 0xff) {
					int original_modifid_time = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 26];
					original_modifid_time = original_modifid_time * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 27];
					if (original_modifid_time == cur_file_count) {
						uchar* name = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i];
						int id = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 29] * 256;
						id += fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 30];

						if (id == cur_directory) {
							prt_string(name);
							if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28] != 0xff) {
								printf(" d");
							}
							printf("\n");
						}
						cur_file_count--;
						break;
					}
				}
			}
		}	
	}
	else if(op == LS_S){
		printf("===sort by file size===\n");

		int last_max = -1;
		for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22] != 0xff) {
				if (last_max == -1) {
					last_max = i;
				}
				else {
					if (!less_than(fs, i, last_max)) {
						last_max = i;
					}
				}
			}
		}

		uchar* name = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_max];
		int size = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_max + 22];
		size = size * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_max + 23];

		int id = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_max + 29] * 256;
		id += fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_max + 30];



		if (id == cur_directory) {
			prt_string(name);
			printf(" %d", size);
			if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_max + 28] != 0xff) {
				printf(" d");
			}
			printf("\n");
		}

		for (int i = 0; i < gtime - 1; i++) {
			int cur_max = -1;
			for (int j = 0; j < fs->FCB_ENTRIES; j++) {
				if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22] != 0xff) {
					if (cur_max == -1 && less_than(fs, j, last_max)) {
						cur_max = j;
					}
					else {
						if (!less_than(fs, j, cur_max) && less_than(fs, j, last_max)) {
							cur_max = j;
						}
					}
				}
			}

			last_max = cur_max;
			uchar* name = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * cur_max];

			int id = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_max + 29] * 256;
			id += fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_max + 30];

			if (id == cur_directory) {
				int size = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_max + 22];
				size = size * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_max + 23];
				prt_string(name);
				printf(" %d", size);
				if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_max + 28] != 0xff) {
					printf(" d");
				}
				printf("\n", size);
			}
		}
	}
	else if (op == CD_P) {

		if (level_3 != -1) {
			level_3 = -1;
		}
		else if (level_2 != -1) {
			level_2 = -1;
		}

		int directory = 0;
		if (level_3 != -1) directory = level_3;
		else if (level_2 != -1) directory = level_2;
		
	}
	else if (op == PWD) {
		if (level_2 != -1) {
			printf("/");
			uchar * start_addr = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*level_2];
			while (*start_addr != '\0') {
				printf("%c", *start_addr);
				start_addr++;
			}
		}
		if (level_3 != -1) {
			printf("/");
			uchar * start_addr = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*level_3];
			while (*start_addr != '\0') {
				printf("%c", *start_addr);
				start_addr++;
			}
		}
		printf("\n");
	}

}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	if (op == RM) {
		int posi = search_directory(fs, s);

		if (posi != -1) {

			if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * posi + 28] != 0xff) {
				if (op == RM) printf("cannot delete a directory with RM\n");
				return;
			}

			int shift_size = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * posi + 22];
			shift_size = shift_size * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * posi + 23];
			int shift_block_size = shift_size / fs->STORAGE_BLOCK_SIZE;
			if (shift_size % fs->STORAGE_BLOCK_SIZE != 0) {
				shift_block_size++;
			}

			fs_move(fs, 0, posi);
			u32 block_addr = global_storage_end / fs->STORAGE_BLOCK_SIZE;

			for (int i = 0; i < shift_block_size; i++) {
				set_superblock(fs, block_addr + i, 0);
			}

			fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * posi + 22] = 0xff;
			gtime--;

			int rm_time = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * posi + 24];
			rm_time = rm_time * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * posi + 25];

			int rm_time_modified = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * posi + 26];
			rm_time_modified = rm_time_modified * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * posi + 27];

			int file_cnt = 0;

			for (int i = 0; i < fs->FCB_ENTRIES; i++) {
				if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22] != 0xff) {
					int rm_time_i = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 24];
					rm_time_i = rm_time_i * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 25];

					int rm_time_modified_i = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 26];
					rm_time_modified_i = rm_time_modified_i * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 27];

					if (rm_time_i > rm_time) {
						rm_time_i--;
						fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 24] = rm_time_i / 256;
						fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 25] = rm_time_i % 256;
					}
					if (rm_time_modified_i > rm_time_modified) {
						rm_time_modified_i--;
						fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 26] = rm_time_modified_i / 256;
						fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 27] = rm_time_modified_i % 256;
					}

					file_cnt++;
				}
				if (file_cnt == gtime) break;
			}

			int directory = 0;
			if (level_3 != -1) directory = level_3;
			else if (level_2 != -1) directory = level_2;

			int original_size = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * directory + 22] * 256;
			original_size += fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * directory + 23];

			int name_cnt;
			for (name_cnt = 0; name_cnt < 20; name_cnt++) {
				if (s[name_cnt] == '\0') {
					name_cnt++;
					break;
				}
			}

			original_size -= name_cnt;

			fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * directory + 22] = original_size / 256;
			fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * directory + 23] = original_size % 256;

		}
		else {
			if (op == RM) printf("not find such file\n");
		}
	}
	else if (op == MKDIR) {
		int directory = 0;
		if (level_3 != -1) {
			printf("reach the directory depth limit\n");
			return;
		}
		else if (level_2 != -1) directory = level_2;

		int base_addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * directory;
		if (fs->volume[base_addr + 28] == 50) {
			printf("the directory is full\n");
			return;
		}

		//check whether there exist a folder wiith the same name in the current directory
		for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			int entry_i = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*i;
			//check valid entry
			if (fs->volume[entry_i + 22] != 0xff) {
				//check entry with the same name
				if (compare_string(s, (char *) &fs->volume[entry_i])) {
					//check whether the entry with same file is of the same file type
					if (fs->volume[entry_i + 28] != 0xff) {
						int id = fs->volume[entry_i + 29] * 256 + fs->volume[entry_i + 30];
						if (id == directory) {
							printf("the directory with the same name already exist in the folder\n");
							return;
						}
					}
				}
			}
		}

		int cnt = 0;

		for (cnt = 0; cnt < 21; cnt++) {
			if (s[cnt] == '\0') {
				break;
			}
		}

		if (cnt >= 20) {
			printf("directory name exceed the name limit\n");
			return;
		}

		int size = fs->volume[base_addr + 22] * 256;
		size += fs->volume[base_addr + 23];

		if ((size + cnt + 1) >= fs->MAX_FILE_SIZE) {
			printf("directory size reach the maximun\n");
			return;
		}

		if (fs->volume[base_addr + 28] == 0xff) {
			fs->volume[base_addr + 28] = 0;
		}

		//create a directory
		u32 directory_pt = fs_open(fs, s, G_WRITE);
		fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE *  directory_pt + 28] = 0;
	}
	else if (op == CD) {

		int directory = search_directory(fs, s);
		if (directory == -1) {
			printf("the directory does not exist\n");
			return;
		}
		if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*directory + 28] == 0xff) {
			printf("the directory can not be a file\n");
			return;
		}

		if (level_2 == -1) {
			level_2 = directory;
		}
		else level_3 = directory;
	}
	else if (op == RM_RF) {
		int directory_rm = search_directory(fs, s);

		if (directory_rm == -1) {
			printf("the directory does not exist\n");
			return;
		}

		if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*directory_rm + 28] == 0xff) {
			printf("the directory can not be a file\n");
			return;
		}
		
		fs_gsys(fs, CD, s);

		for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			int addr_i = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*i;
			int id = fs->volume[addr_i + 29] * 256;
			id += fs->volume[addr_i + 30];
			if (id == directory_rm) {
				int type = fs->volume[addr_i + 28];
				if (type == 0xff) {
					fs_gsys(fs, RM, (char *) &fs->volume[addr_i]);
				}
				else {
					fs_gsys(fs, RM_RF, (char *) &fs->volume[addr_i]);
				}
			}
		}

		fs_gsys(fs, CD_P);

		if (level_3 != -1) {
			fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * level_3 + 28]--;
		}
		else if (level_2 != -1) {
			fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * level_2 + 28]--;
		}
		else {
			fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * 0 + 28]--;
		}

		//set the type of directory going to removed to be file type
		fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*directory_rm + 28] = 0xff;

		fs_gsys(fs, RM, s);
	}

}

__device__ u32 fs_move(FileSystem *fs, int new_size, u32 fp) {

	int old_size = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22];
	old_size *= 256;
	old_size += fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 23];

	if (old_size == 0) {
		int storage_block_size = global_storage_end / fs->STORAGE_BLOCK_SIZE;

		fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 20] = storage_block_size / 256;
		fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 21] = storage_block_size % 256;
	
		return global_storage_end + fs->FILE_BASE_ADDRESS;
	}

	int storage_addr = 0;
	storage_addr = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 20];
	storage_addr *= 256;
	storage_addr += fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 21];
	storage_addr *= fs->STORAGE_BLOCK_SIZE;
	storage_addr += fs->FILE_BASE_ADDRESS;

	if (old_size == new_size) {
		return storage_addr;
	}

	int shift_size = old_size;
	if (new_size < old_size) {
		shift_size = old_size - new_size;
	}

	shift_size /= fs->STORAGE_BLOCK_SIZE;
	if (old_size % fs->STORAGE_BLOCK_SIZE != 0) {
		shift_size += 1;
	}

	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		if (i != fp && fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22] != 0xff) {
			int tmp = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 20];
			tmp = tmp * 256 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 21];
			int tmp_full = tmp * fs->STORAGE_BLOCK_SIZE + fs->FILE_BASE_ADDRESS;

			//change the addressed of the entries behind the entry
			if (tmp_full > storage_addr) {
				//set superblock bits
				tmp -= shift_size;

				for (int i = 0; i < shift_size; i++) {
					set_superblock(fs, tmp + i, 1);
				}

				fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 20] = tmp / 256;
				fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 21] = tmp % 256;
			}
		}
	}

	int move_start_addr = storage_addr;
	if (new_size < old_size) {
		move_start_addr = storage_addr + new_size;
	}

	//shift storage and superblock after the entry together
	for (int i = 0; i < shift_size; i++) {
		int tmp_addr = move_start_addr + i * fs->STORAGE_BLOCK_SIZE;
		for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++) {
			fs->volume[tmp_addr + j] = fs->volume[tmp_addr + j + shift_size * fs->STORAGE_BLOCK_SIZE];
		}
	}

	global_storage_end -= shift_size * fs->STORAGE_BLOCK_SIZE;

	if (new_size < old_size) {
		return storage_addr;
	}

	int new_block_addr = global_storage_end / fs->STORAGE_BLOCK_SIZE;

	fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 20] = new_block_addr / 256;
	fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 21] = new_block_addr % 256;

	return global_storage_end + fs->FILE_BASE_ADDRESS;
}

__device__ int search_directory(FileSystem *fs, char *s) {

	int directory = 0;
	if (level_3 != -1) directory = level_3;
	else if (level_2 != -1) directory = level_2;

	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		int base_addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i;
		if (fs->volume[base_addr + 22] != 0xff) {
			if (compare_string((char *)&fs->volume[base_addr], s)) {
				int id = fs->volume[base_addr + 29] * 256 + fs->volume[base_addr + 30];
				if (id == directory) {
					return i;
				}
			}
		}
	}
}