#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "device_functions.h"
#include "device_launch_parameters.h"


__device__ __managed__ bool thread1_lock = false;
__device__ __managed__ bool thread2_lock = false;
__device__ __managed__ bool thread3_lock = false;

__device__ void init_invert_page_table(VirtualMemory *vm) {

	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
		vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;

		// initialize the LRU double linked list
		vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] = 0x80000000;
		vm->invert_page_table[i + vm->PAGE_ENTRIES * 3] = 0x80000000;

		//hide write and tail in the first two elements of invert_page_table
		vm->invert_page_table[0] |= 0x00000500;
		vm->invert_page_table[1] |= 0x00000500;
	}
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
	u32 *invert_page_table, int *pagefault_num_ptr,
	int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
	int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
	int PAGE_ENTRIES) {
	// init variables
	vm->buffer = buffer;
	vm->storage = storage;
	vm->invert_page_table = invert_page_table;
	vm->pagefault_num_ptr = pagefault_num_ptr;

	// init constants
	vm->PAGESIZE = PAGESIZE;
	vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
	vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
	vm->STORAGE_SIZE = STORAGE_SIZE;
	vm->PAGE_ENTRIES = PAGE_ENTRIES;

	// before first vm_write or vm_read
	init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
	/* Complate vm_read function to read single element from data buffer */
	  //check whether the address is in the invert page table

	__syncthreads();
	if (addr % 4 != ((int)threadIdx.x)) return;
	printf("[thread %d] reading %d\n", threadIdx.x, addr);

	uchar result;

	int frame_page = -1;
	int empty_page = -1;
	u32 physical_addr;
	int LRU_head = vm->invert_page_table[0] & 0xfff;
	int LRU_tail = vm->invert_page_table[1] & 0xfff;

	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if ((vm->invert_page_table[i] & 0x80000000) != 0x80000000) {
			if (vm->invert_page_table[i + vm->PAGE_ENTRIES] == (addr / vm->PAGESIZE)) {
				frame_page = i;
				break;
			}
		}
		//record the empty page number
		else {
			if (empty_page == -1) {
				empty_page = i;
			}
		}
		if (empty_page != -1 && frame_page != -1) break;
	}

	//if the page is in the table
	if (frame_page != -1) {
		physical_addr = addr % vm->PAGESIZE + frame_page * vm->PAGESIZE;
		result = vm->buffer[physical_addr];
	}

	//if the page not in the table
	else {

		(*vm->pagefault_num_ptr)++;

		//if there exists an empty page
		if (empty_page != -1) {
			vm->invert_page_table[empty_page] &= 0x7fffffff;
		}
		//if there does not exist an empty page
		else {
			empty_page = LRU_tail;
			physical_addr = empty_page * vm->PAGESIZE;
			u32 old_storage_addr = (vm->invert_page_table[empty_page + vm->PAGE_ENTRIES]) * vm->PAGESIZE;
			//swap the page from physical memory to secondary memory
			for (int i = 0; i < vm->PAGESIZE; i++) {
				vm->storage[old_storage_addr + i] = vm->buffer[physical_addr + i];
			}
		}

		u32 new_storage_addr = addr / vm->PAGESIZE * vm->PAGESIZE;
		frame_page = empty_page;
		vm->invert_page_table[empty_page + vm->PAGE_ENTRIES] = addr / vm->PAGESIZE;
		//swap the page from secondary memory to physical memory
		for (int i = 0; i < vm->PAGESIZE; i++) {
			vm->buffer[frame_page*vm->PAGESIZE + i] = vm->storage[new_storage_addr + i];
		}

		physical_addr = addr % vm->PAGESIZE + frame_page * vm->PAGESIZE;
		result = vm->buffer[physical_addr];
	}

	//if current page is not in the LRU list, insert it to the front of the list
	if (vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 2] == 0x80000000) {
		vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 2] = 0xC0000000; //indicate the head

		//if there are no nodes in LRU
		if (LRU_tail == 0x500) {
			vm->invert_page_table[1] &= 0xfffff000;
			vm->invert_page_table[1] += frame_page;
			LRU_tail = frame_page;
			vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 3] = 0xF0000000; //indicate the tail
		}
		//exist some nodes in LRU
		else {
			vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 3] = LRU_head;
			vm->invert_page_table[LRU_head + vm->PAGE_ENTRIES * 2] = frame_page;
		}
		vm->invert_page_table[0] &= 0xfffff000;
		vm->invert_page_table[0] += frame_page;
		LRU_head = frame_page;
	}

	//the node already on the front
	if (frame_page == LRU_head) {
		return result;
	}
	//node on the tail
	else if (frame_page == LRU_tail) {
		int prev_node = vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 2];
		vm->invert_page_table[prev_node + vm->PAGE_ENTRIES * 3] = 0xF0000000; //indicating the tail
		vm->invert_page_table[1] &= 0xfffff000;
		vm->invert_page_table[1] += prev_node;
	}
	//node in the middle
	else {
		int prev_node = vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 2];
		int next_node = vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 3];
		vm->invert_page_table[prev_node + vm->PAGE_ENTRIES * 3] = next_node;
		vm->invert_page_table[next_node + vm->PAGE_ENTRIES * 2] = prev_node;
	}

	vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 3] = LRU_head;
	vm->invert_page_table[LRU_head + vm->PAGE_ENTRIES * 2] = frame_page;

	vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 2] = 0xC0000000; //indicating the head
	
	vm->invert_page_table[0] &= 0xfffff000;
	vm->invert_page_table[0] += frame_page;

	return result;
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
	/* Complete vm_write function to write value into data buffer */

	__syncthreads();
	if (addr % 4 != ((int)threadIdx.x)) return;
	printf("[thread %d] writing %d\n", threadIdx.x, addr);

	  //check whether the address is in the invert page table
	int frame_page = -1;
	int empty_page = -1;
	u32 physical_addr;
	int LRU_head = vm->invert_page_table[0] & 0xfff;
	int LRU_tail = vm->invert_page_table[1] & 0xfff;

	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if ((vm->invert_page_table[i] & 0x80000000) != 0x80000000) {
			if (vm->invert_page_table[i + vm->PAGE_ENTRIES] == (addr / vm->PAGESIZE)) {
				frame_page = i;
				break;
			}
		}
		//record the empty page number
		else {
			if (empty_page == -1) {
				empty_page = i;
			}
		}
		if (empty_page != -1 && frame_page != -1) break;
	}

	//if the page is in the table
	if (frame_page != -1) {
		physical_addr = addr % vm->PAGESIZE + frame_page * vm->PAGESIZE;
		vm->buffer[physical_addr] = value;
	}

	//if the page not in the table
	else {

		(*vm->pagefault_num_ptr)++;

		//if there exists an empty page
		if (empty_page != -1) {
			vm->invert_page_table[empty_page] &= 0x7fffffff;
		}
		//if there does not exist an empty page
		else {
			empty_page = LRU_tail;
			physical_addr = empty_page * vm->PAGESIZE;
			u32 old_storage_addr = (vm->invert_page_table[empty_page + vm->PAGE_ENTRIES]) * vm->PAGESIZE;
			//swap the page from physical memory to secondary memory
			for (int i = 0; i < vm->PAGESIZE; i++) {
				vm->storage[old_storage_addr + i] = vm->buffer[physical_addr + i];
			}
		}

		u32 new_storage_addr = addr / vm->PAGESIZE * vm->PAGESIZE;
		frame_page = empty_page;
		vm->invert_page_table[empty_page + vm->PAGE_ENTRIES] = addr / vm->PAGESIZE;
		//swap the page from secondary memory to physical memory
		for (int i = 0; i < vm->PAGESIZE; i++) {
			vm->buffer[frame_page*vm->PAGESIZE + i] = vm->storage[new_storage_addr + i];
		}

		physical_addr = addr % vm->PAGESIZE + frame_page * vm->PAGESIZE;
		vm->buffer[physical_addr] = value;
	}

	//if current page is not in the LRU list, insert it to the front of the list
	if (vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 2] == 0x80000000) {
		vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 2] = 0xC0000000; //indicate the head

		//if there are no nodes in LRU
		if (LRU_tail == 0x500) {
			vm->invert_page_table[1] &= 0xfffff000;
			vm->invert_page_table[1] += frame_page;
			LRU_tail = frame_page;
			vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 3] = 0xF0000000; //indicate the tail
		}
		//exist some nodes in LRU
		else {
			vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 3] = LRU_head;
			vm->invert_page_table[LRU_head + vm->PAGE_ENTRIES * 2] = frame_page;
		}
		vm->invert_page_table[0] &= 0xfffff000;
		vm->invert_page_table[0] += frame_page;
		LRU_head = frame_page;
	}
	//the node already on the front
	if (frame_page == LRU_head) {
		return;
	}
	//node on the tail
	else if (frame_page == LRU_tail) {
		int prev_node = vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 2];
		vm->invert_page_table[prev_node + vm->PAGE_ENTRIES * 3] = 0xF0000000; //indicating the tail
		vm->invert_page_table[1] &= 0xfffff000;
		vm->invert_page_table[1] += prev_node;
	}
	//node in the middle
	else {
		int prev_node = vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 2];
		int next_node = vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 3];
		vm->invert_page_table[prev_node + vm->PAGE_ENTRIES * 3] = next_node;
		vm->invert_page_table[next_node + vm->PAGE_ENTRIES * 2] = prev_node;
	}

	vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 3] = LRU_head;
	vm->invert_page_table[LRU_head + vm->PAGE_ENTRIES * 2] = frame_page;

	vm->invert_page_table[frame_page + vm->PAGE_ENTRIES * 2] = 0xC0000000; //indicating the head

	vm->invert_page_table[0] &= 0xfffff000;
	vm->invert_page_table[0] += frame_page;
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
	int input_size) {
	/* Complete snapshot function togther with vm_read to load elements from data
	 * to result buffer */

	for (int i = offset; i < input_size / 4; i++) {
		results[i * 4 + (int)threadIdx.x] = vm_read(vm, i * 4 + (int)threadIdx.x);
	}
}

