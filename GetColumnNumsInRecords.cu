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
#include "CSV_kernel_declarations.cuh"
#include "csvImporter.h"

extern "C" void
launch_GetColumnNumsInRecords(uint32_t *  d_ScanRecs, uint32_t *  d_ScanCols,
uint32_t *  d_OrdinalsRecsToCols, uint16_t * d_ColNumInRecs, uint32_t ValuesCount)
{

	// Call stream compact kernel.
	int iThreads = 256;
	float fBlocks = (float)ValuesCount / ((float)iThreads);
	int iBlocks = ValuesCount / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	GetColumnNumsInRecords <<< iBlocks, iThreads >>>(d_ScanRecs, d_ScanCols, d_OrdinalsRecsToCols, d_ColNumInRecs, ValuesCount);

	Check_cuda_Errors("GetColumnNumsInRecords");
}

__global__ void GetColumnNumsInRecords(uint32_t *  d_ScanRecs, uint32_t *  d_ScanCols,
	uint32_t *  d_OrdinalsRecsToCols, uint16_t * d_ColNumInRecs, uint32_t ValuesCount )
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= ValuesCount) return;  // ignore anything in last block beyond source arrays length.

	uint32_t firstcol;
	uint32_t rec;
	uint32_t col;
	uint32_t colinrec;
	if (ix == 0)
	{
		rec = 0;
		firstcol = 0;
		col = 0;
	}
	else
	{
		rec = d_ScanRecs[ix];
		col = d_ScanCols[ix];
	}
	if (rec == 0) firstcol = 0;
	else firstcol = d_OrdinalsRecsToCols[rec - 1] + 1;
	colinrec = col - firstcol;
	d_ColNumInRecs[ix] = (uint16_t)colinrec;

	return;
}