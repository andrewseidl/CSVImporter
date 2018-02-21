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
launch_GetCharNumsInColumns(uint32_t *  d_ScanCols, uint32_t *  d_ScanChars,
uint32_t *  d_OrdinalsColsToChars, uint16_t * d_CharNumInCols, uint32_t ValuesCount)
{

	// Call stream compact kernel.
	int iThreads = 256;
	float fBlocks = (float)ValuesCount / ((float)iThreads);
	int iBlocks = ValuesCount / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	GetCharNumsInColumns <<<iBlocks, iThreads >>>(d_ScanCols, d_ScanChars, d_OrdinalsColsToChars, d_CharNumInCols, ValuesCount);
}

__global__ void GetCharNumsInColumns(uint32_t *  d_ScanCols, uint32_t *  d_ScanChars,
	uint32_t *  d_OrdinalsColsToChars, uint16_t * d_CharNumInCols, uint32_t ValuesCount)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= ValuesCount) return;  // ignore anything in last block beyond source arrays length.

	uint32_t firstchar;
	uint32_t col;
	uint32_t chr;
	uint32_t charincol;
	if (ix == 0)
	{
		col = 0;
		firstchar = 0;
		chr = 0;
	}
	else
	{
		col = d_ScanCols[ix];
		chr = d_ScanChars[ix];
	}
	if (col == 0) firstchar = 0;
	else firstchar = d_OrdinalsColsToChars[col - 1];
	charincol = chr - firstchar;
	d_CharNumInCols[ix] = (uint16_t)charincol;

	return;
}

