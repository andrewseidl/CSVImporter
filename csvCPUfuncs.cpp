/******************************************************************************
* Copyright (c) 2016-2018, Brian Kennedy.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
******************************************************************************/

/******************************************************************************
*
* See https://github.com/Simantex/CSVImporter for repository and documentation.
*
******************************************************************************/
#include "csvImporter.h"

// Special memory allocation and free functions to
// Support using HOST PINNED AND ALIGNED MEMORY for the
// fastest possible GPU<->CPU transfers
void AllocateHostMemory(bool bPinGenericMemory, uint32_t **pp_a, uint32_t **ppAligned_a, unsigned int nbytes, bool bUseWriteCombine = false)
{
#if CUDART_VERSION >= 4000
	if (bPinGenericMemory)
	{
		// allocate a generic page-aligned chunk of system memory
#ifdef WIN32
		printf("> VirtualAlloc() allocating %4.2f Mbytes of (generic page-aligned system memory)\n", (float)nbytes / 1048576.0f);
		if(bUseWriteCombine)
			*pp_a = (uint32_t *)VirtualAllocEx(GetCurrentProcess(), NULL, (nbytes + MEMORY_ALIGNMENT), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE | PAGE_WRITECOMBINE);
		else
			*pp_a = (uint32_t *)VirtualAllocEx(GetCurrentProcess(), NULL, (nbytes + MEMORY_ALIGNMENT), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
		printf("> mmap() allocating %4.2f Mbytes (generic page-aligned system memory)\n", (float)nbytes / 1048576.0f);
		*pp_a = (int *)mmap(NULL, (nbytes + MEMORY_ALIGNMENT), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#endif

		*ppAligned_a = (uint32_t *)ALIGN_UP(*pp_a, MEMORY_ALIGNMENT);

		printf("> cudaHostRegister() registering %4.2f Mbytes of generic allocated system memory\n", (float)nbytes / 1048576.0f);

		// pin allocate memory
		checkCudaErrors(cudaHostRegister(*ppAligned_a, nbytes, cudaHostRegisterMapped));
	}
	else
#endif
	{
		printf("> cudaMallocHost() allocating %4.2f Mbytes of system memory\n", (float)nbytes / 1048576.0f);
		// allocate host memory (pinned is required for achieve asynchronicity)
		checkCudaErrors(cudaMallocHost((void **)pp_a, nbytes));
		*ppAligned_a = *pp_a;
	}
}

void FreeHostMemory(bool bPinGenericMemory, uint32_t **pp_a, uint32_t **ppAligned_a, unsigned int nbyte)
{
#if CUDART_VERSION >= 4000

	// CUDA 4.0 support pinning of generic host memory
	if (bPinGenericMemory)
	{
		// unpin and delete host memory
		checkCudaErrors(cudaHostUnregister(*ppAligned_a));
#ifdef WIN32
		VirtualFreeEx(GetCurrentProcess(), *pp_a, 0, MEM_RELEASE);
#else
		munmap(*pp_a, nbytes);
#endif
	}
	else
#endif
	{
		cudaFreeHost(*pp_a);
	}
}

// returns -1 if file open error, returns 1 if file is less than 6 bytes long (assume no records).
// otherwise returns 0 if no problems found.
int CSVfilechunking(char * filepath)
{
	pCsvFileIn = fopen(filepath, "rb");
	if (pCsvFileIn == NULL) return -1;

	_fseeki64(pCsvFileIn, (__int64)0, SEEK_END);
	CsvFileLength = _ftelli64(pCsvFileIn);  // get length of file.
	_fseeki64(pCsvFileIn, (__int64)0, SEEK_SET);  // reset file ptr to beginning for read

	if (CsvFileLength <= MAXCHAR0RECORDS)
	{
		fclose(pCsvFileIn);  // must close the file if the file is empty, so it can be deleted at end.
		return 1;
	}
	// calculate the chunking for the file.

	// default is 1 chunk.
	inumchunks = 1;
	apprx_chunklen = CsvFileLength;

	// if file is longer than target have more than 1 chunk.
	if (CsvFileLength > TARGETBYTES)
	{
		double dnumchunks = (double)CsvFileLength / (double)TARGETBYTES;
		inumchunks = (int)CsvFileLength / TARGETBYTES;
		dnumchunks = dnumchunks - inumchunks;
		// if fraction is > half, add 2 chunks, otherwise 1.
		if (dnumchunks > 0.5) inumchunks += 2;
		else inumchunks += 1;
		apprx_chunklen = (uint32_t)CsvFileLength / inumchunks;
	}

	uint64_t checkbytes = apprx_chunklen + OVERREAD;  // this count should be sufficient for the largest chunk.
	// round up to even 256 for allocations:
	uint64_t lastbyte = checkbytes & 0xff;
	if (lastbyte > 0) checkbytes += (256 - lastbyte);  // this should round up.
	SufficientBytes = checkbytes;

	return 0;
}

bool InitializeCPUElements_REUSABLES(uint64_t totalbytes)
{
	/* Set the useWriteCombine memory flag to true since host is only writing the raw chunks to this buffer and not reading from it.
	   Using writeCombine memory will keep the memory from being stored in the CPU L1/L2 cache (leaving more cache for the rest of the
	   application, and thus access to this memory will not be snooped, resulting in 40% faster transfer speeds to the GPU!
	   NOTE: if the host ever did access this memory for reading it would be VERY SLOW!!  */
	AllocateHostMemory(DEFAULT_PINNED_GENERIC_MEMORY, (uint32_t **)&h_CsvBuffer_NAXX, (uint32_t **)&h_CsvBuffer_a, totalbytes, true);

	return true;
}

void DeinitializeCPUElements_REUSABLES(uint64_t totalbytes)
{
	FreeHostMemory(DEFAULT_PINNED_GENERIC_MEMORY, (uint32_t **)&h_CsvBuffer_NAXX, (uint32_t **)&h_CsvBuffer_a, totalbytes);
	return;
}

