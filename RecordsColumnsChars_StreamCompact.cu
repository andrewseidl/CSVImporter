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
launch_RecordsColumnsChars_StreamCompact(uint32_t *  d_MatchesRecs, uint32_t *  d_MatchesCols, uint32_t *  d_MatchesChars,
uint32_t *  d_ScanRecs, uint32_t *  d_ScanCols, uint32_t *  d_ScanChars,
uint32_t *  d_OrdinalsRecs, uint32_t *  d_OrdinalsCols, uint32_t *  d_OrdinalsChars,
uint32_t *  d_OrdinalsRecsToCols, uint32_t *  d_OrdinalsColsToChars, uint32_t ValuesCount)
{

	// Call stream compact kernel.
	int iThreads = 256;
	float fBlocks = (float)ValuesCount / ((float)iThreads);
	int iBlocks = ValuesCount / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	RecordsColumnsChars_StreamCompact <<<iBlocks, iThreads >>> (d_MatchesRecs, d_MatchesCols, d_MatchesChars,
		d_ScanRecs, d_ScanCols, d_ScanChars,
		d_OrdinalsRecs, d_OrdinalsCols, d_OrdinalsChars,
		d_OrdinalsRecsToCols, d_OrdinalsColsToChars, ValuesCount);

	Check_cuda_Errors("RecordsColumnsChars_StreamCompact");
}


// specialized stream compact version that checks for match headers for both records and columns, and builds records, columns, and recordstocolumns tables.
// For each match header the current index of the match header is copied to the ordinals array at the index in the ordinals array which equals the scan value at the same position as the header.
// this assumes the ordinals arrays have been properly sized coming in.

// records to columns would normally be a simple multiple, e.g., if there are 10 columns per record, the array index multiple would be 10.
// as long as that is true, only the columns table would be needed.
// however, we assume there might be column count errors, in which case the different tables provide precise access between records and columns
// and allow us to check for column count errors.

// the new version adds processing utf8 chars.
// it processes char scans and outputs the char table and the columns to chars table.

__global__ void RecordsColumnsChars_StreamCompact(uint32_t *  d_MatchesRecs, uint32_t *  d_MatchesCols, uint32_t *  d_MatchesChars,
	uint32_t *  d_ScanRecs, uint32_t *  d_ScanCols, uint32_t *  d_ScanChars,
	uint32_t *  d_OrdinalsRecs, uint32_t *  d_OrdinalsCols, uint32_t *  d_OrdinalsChars,
	uint32_t *  d_OrdinalsRecsToCols, uint32_t *  d_OrdinalsColsToChars, uint32_t ValuesCount)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= ValuesCount) return;  // ignore anything in last block beyond source arrays length.

	// the index into the result arrays is simply the Exclusive Scan value at the current position.
	if (d_MatchesRecs[ix] == 1)
	{
		d_OrdinalsRecs[d_ScanRecs[ix]] = ix;
		// the recstocols puts in the same relative position as the recs table the SCAN value of the cols, for looking up into the cols table.
		d_OrdinalsRecsToCols[d_ScanRecs[ix]] = d_ScanCols[ix];
	}
	if (d_MatchesCols[ix] == 1)
	{
		d_OrdinalsCols[d_ScanCols[ix]] = ix;

		// the recstocols puts in the same relative position as the recs table the SCAN value of the cols, for looking up into the cols table.
		d_OrdinalsColsToChars[d_ScanCols[ix]] = d_ScanChars[ix];
	}
	if (d_MatchesChars[ix] == 1)
	{
		d_OrdinalsChars[d_ScanChars[ix]] = ix;
	}

	return;
}

