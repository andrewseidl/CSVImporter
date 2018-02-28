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

bool InitializeGPUElements_REUSABLES(uint64_t totalbytes)
{
	// malloc buffer for the test data in the GPU
	checkCudaErrors(cudaMalloc((void **)&(d_CsvBuffer), totalbytes));
	checkCudaErrors(cudaMalloc((void **)&(d_RecordHeaders), totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset((void*)d_RecordHeaders, 0, totalbytes * sizeof(uint32_t)));  // memset this for use in comma processing.

	checkCudaErrors(cudaMalloc((void **)&(d_ColumnHeaders), totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset((void*)d_ColumnHeaders, 0, totalbytes * sizeof(uint32_t)));  // memset this for use in comma processing.

	checkCudaErrors(cudaMalloc((void **)&(d_RecordHeadersSCAN), totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMalloc((void **)&(d_ColumnHeadersSCAN), totalbytes * sizeof(uint32_t)));

	checkCudaErrors(cudaMalloc((void **)&(d_ColumnNumInRecord), totalbytes * sizeof(uint16_t)));
	checkCudaErrors(cudaMalloc((void **)&(d_UTF8Headers), totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMalloc((void **)&(d_UTF8HeadersSCAN), totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMalloc((void **)&(d_CharNumInColumn), totalbytes * sizeof(uint16_t)));

	// memset the new arrays below to ensure default vals will be 0s.
	checkCudaErrors(cudaMalloc((void **)&(d_QuoteBoundaryHeaders), totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMalloc((void **)&(d_QuoteBoundaryHeaders_SCAN), totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMalloc((void **)&(d_printingchars_flags), totalbytes * sizeof(uint32_t)));  // extra 128 just safety cushion.
	checkCudaErrors(cudaMalloc((void **)&(d_CommaHeaders), totalbytes));
	checkCudaErrors(cudaMalloc((void **)&(d_secondquotes), totalbytes));

	// scan printing chars in prep for stream compact.
	checkCudaErrors(cudaMalloc((void **)&(d_printingchars_SCAN), totalbytes * sizeof(uint32_t)));  // extra 128 just safety cushion.
	checkCudaErrors(cudaMalloc((void **)&(d_RecordsToQuoteBoundariesTable), (totalbytes /*recordstablecount_commas + 1*/) * sizeof(uint32_t)));  // 1 element for scans

	checkCudaErrors(cudaMalloc((void **)&(d_CsvBuffer_printing), (totalbytes /*printingcharstemp*/) + 256));

	checkCudaErrors(cudaMalloc((void **)&(d_RecordHeaders_printing), ((totalbytes /*printingcharscount*/)  + 1) * sizeof(uint32_t)));
	checkCudaErrors(cudaMalloc((void **)&(d_ColumnHeaders_printing), ((totalbytes /*printingcharscount*/) + 1) * sizeof(uint32_t)));

	//////######
	// Records table is a table of record header locations.
	// Columns table is a table of column header locations.
	// RecordsToColumns table is a table that maps record headers to locations in the Columns table.
	checkCudaErrors(cudaMalloc((void **)&(d_RecordsTable), (totalbytes/*recordstablecount*/ + 1) * sizeof(uint32_t)));  // 1 element for scans
	checkCudaErrors(cudaMalloc((void **)&(d_RecordsToColumnsTable), (totalbytes/*recordstablecount*/ + 1) * sizeof(uint32_t)));  // 1 element for scans
	checkCudaErrors(cudaMalloc((void **)&(d_ColumnsTable), (totalbytes/*columnstablecount*/ + 1) * sizeof(uint32_t)));  // extra element for scan
	checkCudaErrors(cudaMalloc((void **)&(d_ColumnsToUTF8charsTable), (totalbytes/*columnstablecount*/ + 1) * sizeof(uint32_t)));  // extra element for scan
	checkCudaErrors(cudaMalloc((void **)&(d_RecordLengths), (totalbytes/*recordstablecount*/ * sizeof(uint32_t)) + 128));  // extra 128 just safety cushion
	checkCudaErrors(cudaMalloc((void **)&(d_ColumnCountErrors), (totalbytes/*recordstablecount*/ * sizeof(uint32_t)) + 128));  // extra 128 just safety cushion
	checkCudaErrors(cudaMalloc((void **)&(d_ColumnCharCountErrors), (totalbytes/*recordstablecount*/ * sizeof(uint32_t)) + 128));  // extra 128 just safety cushion
	checkCudaErrors(cudaMalloc((void **)&(d_ColumnCountsPerRecordTable), (totalbytes/*recordstablecount*/ * sizeof(uint32_t)) + 128));  // extra 128 just safety cushion
	checkCudaErrors(cudaMalloc((void **)&(d_ColumnCountErrorsSCAN), (totalbytes/*recordstablecount*/ * sizeof(uint32_t)) + 128));  // extra 128 just safety cushion
	checkCudaErrors(cudaMalloc((void **)&(d_ColumnCountErrorsTable), (totalbytes/*columncounterrorscount*/ * sizeof(uint32_t)) + 128));  // extra 128 just safety cushion
	checkCudaErrors(cudaMalloc((void **)&(d_UTF8CharsTable), (totalbytes/*charstablecount*/ * sizeof(uint32_t)) + 128));  // extra 128 just safety cushion

	return true;
}
bool MemsetGPUElements_REUSABLES(uint64_t totalbytes)
{
	// malloc buffer for the test data in the GPU
	checkCudaErrors(cudaMemset((void *)(d_CsvBuffer), 0, totalbytes));
	checkCudaErrors(cudaMemset((void *)(d_RecordHeaders), 0, totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset((void *)(d_ColumnHeaders), 0, totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset((void *)(d_RecordHeadersSCAN), 0, totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset((void *)(d_ColumnHeadersSCAN), 0, totalbytes * sizeof(uint32_t)));

	checkCudaErrors(cudaMemset((void *)(d_ColumnNumInRecord), 0, totalbytes * sizeof(uint16_t)));
	checkCudaErrors(cudaMemset((void *)(d_UTF8Headers), 0, totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset((void *)(d_UTF8HeadersSCAN), 0, totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset((void *)(d_CharNumInColumn), 0, totalbytes * sizeof(uint16_t)));

	checkCudaErrors(cudaMemset((void*)d_QuoteBoundaryHeaders, 0, totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset((void*)d_QuoteBoundaryHeaders_SCAN, 0, totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset((void*)d_printingchars_flags, 0, totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset((void*)d_printingchars_SCAN, 0, totalbytes * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset((void*)d_CommaHeaders, 0, totalbytes));
	checkCudaErrors(cudaMemset((void*)d_secondquotes, 0, totalbytes));
	checkCudaErrors(cudaMemset((void*)d_RecordsToQuoteBoundariesTable, 0, (totalbytes /*recordstablecount_commas + 1*/) * sizeof(uint32_t)));

	checkCudaErrors(cudaMemset((void*)d_CsvBuffer_printing, 0, (totalbytes /*printingcharstemp*/) + 256));

	checkCudaErrors(cudaMemset((void*)d_RecordHeaders_printing, 0, ((totalbytes /*printingcharscount*/)+1) * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset((void*)d_ColumnHeaders_printing, 0, ((totalbytes /*printingcharscount*/)+1) * sizeof(uint32_t)));

	return true;
}

void DeinitializeGPUElements_REUSABLES()
{
	// Free all the GPU buffers.
	Check_cuda_Free((void **)&d_CsvBuffer);

	Check_cuda_Free((void **)&d_RecordHeaders);
	Check_cuda_Free((void **)&d_ColumnHeaders);
	Check_cuda_Free((void **)&d_RecordHeadersSCAN);
	Check_cuda_Free((void **)&d_ColumnHeadersSCAN);

	Check_cuda_Free((void **)&d_ColumnNumInRecord);
	Check_cuda_Free((void **)&d_UTF8Headers);

	Check_cuda_Free((void **)&d_UTF8HeadersSCAN);
	Check_cuda_Free((void **)&d_CharNumInColumn);

	// first we can clear some memory.
	Check_cuda_Free((void **)&d_QuoteBoundaryHeaders);
	Check_cuda_Free((void **)&d_QuoteBoundaryHeaders_SCAN);
	Check_cuda_Free((void **)&d_printingchars_flags);
	Check_cuda_Free((void **)&d_printingchars_SCAN);
	Check_cuda_Free((void **)&d_CommaHeaders);
	Check_cuda_Free((void **)&d_secondquotes);
	Check_cuda_Free((void **)&d_RecordsToQuoteBoundariesTable);

	Check_cuda_Free((void **)&d_CsvBuffer_printing);
	Check_cuda_Free((void **)&d_RecordHeaders_printing);
	Check_cuda_Free((void **)&d_ColumnHeaders_printing);

	Check_cuda_Free((void **)&d_RecordsTable);
	Check_cuda_Free((void **)&d_RecordsToColumnsTable);
	Check_cuda_Free((void **)&d_ColumnsTable);
	Check_cuda_Free((void **)&d_ColumnsToUTF8charsTable);
	Check_cuda_Free((void **)&d_RecordLengths);
	Check_cuda_Free((void **)&d_ColumnCountErrors);
	Check_cuda_Free((void **)&d_ColumnCharCountErrors);
	Check_cuda_Free((void **)&d_ColumnCountsPerRecordTable);
	Check_cuda_Free((void **)&d_ColumnCountErrorsSCAN);
	Check_cuda_Free((void **)&d_ColumnCountErrorsTable);
	Check_cuda_Free((void **)&d_UTF8CharsTable);

	return;
}

// routine checks for a null pointer and errors out if found.
// also zeros out the pointer to prevent double free.
extern "C" void Check_cuda_Free(void ** memlocaddress)
{
	if (*memlocaddress == (void *)0)
	{
		fprintf(stderr, "cudaFree ATTEMPT TO FREE NULL POINTER.\r\n");
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaFree(*memlocaddress));

	*memlocaddress = (void *)0;  // zero out pointer for next time.

	return;
}

extern "C" void Check_cuda_FreeHost(void ** memlocaddress)
{
	if (*memlocaddress == (void *)0)
	{
		// COMMENTED OUT BELOW BECAUSE OF 0-LENGTH INPUT CSV FILES.
		//fprintf(stderr, "cudaFreeHost ATTEMPT TO FREE NULL POINTER.\r\n");
		//DEVICE_RESET;
		//exit(1);
		return;  // for now simply ignore
	}

	checkCudaErrors(cudaFreeHost(*memlocaddress));

	*memlocaddress = (void *)0;  // zero out pointer for next time.

	return;
}

extern "C" void __Check_cuda_Errors(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString(err));
		DEVICE_RESET
		exit(EXIT_FAILURE);  // redundant.
	}
}