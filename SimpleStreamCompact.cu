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
launch_SimpleStreamCompact(unsigned int *  d_Matches, unsigned int *  d_Scan, unsigned int *  d_ResultsOrdinals, unsigned int ValuesCount)
{

	// Call stream compact kernel.
	int iThreads = 256;
	float fBlocks = (float)ValuesCount / ((float)iThreads);
	int iBlocks = ValuesCount / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	SimpleStreamCompact <<<iBlocks, iThreads >>>(d_Matches, d_Scan, d_ResultsOrdinals, ValuesCount);

	Check_cuda_Errors("SimpleStreamCompact");
}


// simple version of stream compactor simply checks for match headers.
// For each match header the current index of the match header is copied to the ordinals array at the index in the ordinals array which equals the scan value at the same position as the header.

__global__ void SimpleStreamCompact(unsigned int *  d_Matches, unsigned int *  d_Scan, unsigned int *  d_ResultsOrdinals, unsigned int ValuesCount)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= ValuesCount) return;  // ignore anything in last block beyond source arrays length.

	// the index into the result arrays is simply the Exclusive Scan value at the current position.
	if (d_Matches[ix] == 1)
	{
		d_ResultsOrdinals[d_Scan[ix]] = ix;
	}

	return;
}
