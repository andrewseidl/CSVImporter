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

// Goes through the Records Table.  Gets record lengths, Column counts per record, and sets column count errors flag if column nums don't match setting.  
// This version of the launcher and kernel use shared mem to store last lane values in block.  Since need to compare current value to prev, this saves a buffer read.
// However we do not write out last lane of last thread to SMEM (would be throw away), so we don't allocate SMEM for that.
extern "C" void
launch_GetRecLengthsAndColCountErrorsSMEM2(uint32_t *  d_RecsTabl, uint32_t *  d_RecsToColsTabl,
uint32_t *  d_RecLengths, uint32_t *  d_ColCountErrors, uint32_t *  d_ColumnCountsPerRecordTable,
uint32_t ValuesCount, uint16_t numcols)
{

	// Call stream compact kernel.
	int iThreads = 256;
	float fBlocks = (float)ValuesCount / ((float)iThreads);
	int iBlocks = ValuesCount / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	float fwarpsperblock = (float)iThreads / ((float)32);
	int iwarpsperblock = iThreads / 32;
	fwarpsperblock = fwarpsperblock - iwarpsperblock;
	if (fwarpsperblock > 0)
		iwarpsperblock++;
	// need 2 values per warp.
	int smemvalues = (iwarpsperblock - 1) * 2;  // need # warps - 1

	GetRecLengthsAndColCountErrorsSMEM2<<<iBlocks, iThreads, smemvalues * sizeof(uint32_t)>>> (d_RecsTabl, d_RecsToColsTabl, d_RecLengths, d_ColCountErrors, d_ColumnCountsPerRecordTable, ValuesCount, numcols);

	Check_cuda_Errors("GetRecLengthsAndColCountErrorsSMEM2");
}

__global__ void GetRecLengthsAndColCountErrorsSMEM2(uint32_t *  d_RecsTabl, uint32_t *  d_RecsToColsTabl,
	uint32_t *  d_RecLengths, uint32_t *  d_ColCountErrors, uint32_t *  d_ColumnCountsPerRecordTable,
	uint32_t ValuesCount, uint16_t numcols)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= ValuesCount) return;  // ignore anything in last block beyond source arrays length.

	// get current record array idx and recstocols
	uint32_t myrecs = d_RecsTabl[ix];
	uint32_t myrecstocols = d_RecsToColsTabl[ix];

	// look for myrecs (offset 0) and myrecstocols (offset 1).  so 2 uint32_t's per warp.
	extern __shared__ uint32_t finallanevalues[];

	int warpnum = threadIdx.x >> 5;  // int divide by 32, throw away remainder
	int laneid = ix & 0x1f;

	// last lane writes input to shared, unless it is the last thread in the block (value would not be used).
	if ((laneid == 31) && (threadIdx.x != (blockDim.x - 1)))
	{
		finallanevalues[warpnum * 2] = myrecs;
		finallanevalues[(warpnum * 2) + 1] = myrecstocols;
	}

	// ensure all warps in the block have access to these values in shared.
	__syncthreads();

	// try to load prev based on warp shuffles
	uint32_t prevrecs = __shfl_up_sync(0xFFFFFFFF, myrecs, 1);
	uint32_t prevrecstocols = __shfl_up_sync(0xFFFFFFFF, myrecstocols, 1);

	// now mod based on thread and lane
	// set to 0 for very first thread (0) since nothing to look up
	if (ix == 0)
	{
		prevrecs = 0;
		prevrecstocols = 0;
	}

	// for the first thread in the block, must look up from mem.
	else if (threadIdx.x == 0)
	{
		prevrecs = d_RecsTabl[ix - 1];
		prevrecstocols = d_RecsToColsTabl[ix - 1];
	}
	// otherwise if the first thread in the warp (not 1st in block), can get from shared. look back 1 warp position.
	else if ((ix % 32) == 0)
	{
		prevrecs = finallanevalues[(warpnum - 1) * 2];
		prevrecstocols = finallanevalues[((warpnum - 1) * 2) + 1];
	}

	uint32_t reclen = myrecs - prevrecs;
	uint32_t colscount = myrecstocols - prevrecstocols;
	// add 1 if this is the first value to adjust for initial 0-based
	// (rectocols 0 is 1 less than rectocols 1 minus rectocols 0, but consistent from there up)
	if (ix == 0) colscount += 1;

	uint32_t colscounterrorflag = ((uint16_t)colscount == numcols) ? 0 : 1;

	d_RecLengths[ix] = reclen;
	d_ColCountErrors[ix] = colscounterrorflag;
	d_ColumnCountsPerRecordTable[ix] = colscount;

	return;
}

