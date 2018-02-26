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
launch_BuildRecsColsCharsHeaders(uint8_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders,
uint32_t *  d_UTF8charHeaders, uint32_t TotalBytes)
{

	// Call build headers kernel.
	int iThreads = 256;
	float fBlocks = (float)TotalBytes / ((float)iThreads);
	int iBlocks = TotalBytes / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	BuildRecsColsCharsHeaders <<<iBlocks, iThreads >>>(d_Buffer, d_RecordHeaders, d_ColumnHeaders, d_UTF8charHeaders, TotalBytes);

	Check_cuda_Errors("BuildRecsColsCharsHeaders");
}

// this kernel builds the header arrays for the record separators and the column separators (which also include the record separators).
// at this point we look for linefeed for record separator (assuming CR LF) and vertical bar column separator.
// note we expect a CR LF after the last record.
// the original input buffer is examined four bytes at a time (using a union to load as).
// the output is written to 2 arrays of uint32_t.
//
// newer version also builds headers for UTF-8 character starts.

__global__ void BuildRecsColsCharsHeaders(uint8_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders,
	uint32_t *  d_UTF8charHeaders, uint32_t TotalBytes)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	// union for loading as...
	union inputbytes_t
	{
		uint32_t inputint;
		unsigned char inputbytes[4];
	};

	// map buffer to 4 bytes.  examine each of those bytes.
	// headers will output to a position re

	int32_t totalints = TotalBytes >> 2;
	int overbytes = TotalBytes & 0x3;
	//int bytestodo = 4;
	if (overbytes > 0)
	{
		totalints++;
	}
	if (ix >= totalints) return;  // ignore anything in last block beyond source arrays length.

	// C++ byte order:  note character order is from least sig to most.
	// 0x61626364  64 63 62 61
	// 0x64636261  61 62 63 64

	inputbytes_t inval;
	inval.inputint = ( (uint32_t *)d_Buffer )[ix];

	unsigned char charval;
	unsigned char colval;
	unsigned char recval;

	bool lastword = false;
	bool lastbyte = false;
	// if at last int32 then only doing overbytes bytes
	if (ix == (totalints - 1))
	{
		//bytestodo = overbytes;
		lastword = true;
	}
	// loop through the 4 (or fewer) bytes and output 
	for (int i = 0; i < 4; i++)
	{
		charval = 0;
		colval = 0;
		recval = 0;

		// if we are on the last word some special handling
		if (lastword)
		{
			// if no overbytes but at last byte, OR in overbyte situation and at last byte (e.g. ob is 1 and i is 0),
			// assuming the file terminates with a CRLF ensure col and rec headers false to mark end.  scan will add 1 in final position for total record/col counts.
			// OTHERWISE force col and rec headers true to mark end.
			if (((overbytes == 0) && (i == 3)) || (i == (overbytes - 1)))
			{
				lastbyte = true;
			}
			else if ((overbytes > 0) && (i == overbytes))  return;  // if finished all overbytes at end, just return here, all done.
		}

		if (!lastbyte)
		{
			unsigned char ival = inval.inputbytes[i];

			// check if have | or LF.  That is column marker.
			if ((ival == (unsigned char)'|') || (ival == (unsigned char)0xa))
			{
				colval = 1;
			}
			// otherwise, rule out CR.  then check for a UTF-8 char start.
			else if (ival != (unsigned char)0xd)
			{
				// check for UTF-8 first byte: either is format 0xxx xxxx  or NOT 10xx xxxx
				if ((ival & 0x80) == 0) charval = 1;
				else if ((ival & 0xc0) != (unsigned char)0x80) charval = 1;
			}

			// check if have LF.  That is row marker.
			if (ival == (unsigned char)0xa)
			{
				recval = 1;
			}
		}
		d_UTF8charHeaders[(ix * 4) + i] = charval;
		d_ColumnHeaders[(ix * 4) + i] = colval;
		d_RecordHeaders[(ix * 4) + i] = recval;
	}

	return;
}


// this routine builds only the record headers.
// this is needed for initial calcs for chunking.
extern "C" void
launch_BuildRecordHeaders(uint8_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t TotalBytes)
{

	// Call build headers kernel.
	int iThreads = 256;
	float fBlocks = (float)TotalBytes / ((float)iThreads);
	int iBlocks = TotalBytes / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	BuildRecordHeaders <<<iBlocks, iThreads >>>(d_Buffer, d_RecordHeaders, TotalBytes);

	Check_cuda_Errors("BuildRecordHeaders");
}

// this kernel builds the header arrays for the record separators
// at this point we look for linefeed for record separator (assuming CR LF) and vertical bar column separator.
// note we expect a CR LF after the last record.
// the original input buffer is examined four bytes at a time (using a union to load as).
// the output is written to array of uint32_t.

__global__ void BuildRecordHeaders(uint8_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t TotalBytes)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	// union for loading as...
	union inputbytes_t
	{
		uint32_t inputint;
		unsigned char inputbytes[4];
	};

	// map buffer to 4 bytes.  examine each of those bytes.
	// headers will output to a position re

	int32_t totalints = TotalBytes >> 2;
	int overbytes = TotalBytes & 0x3;

	if (overbytes > 0)
	{
		totalints++;
	}
	if (ix >= totalints) return;  // ignore anything in last block beyond source arrays length.

	// C++ byte order:  note character order is from least sig to most.
	// 0x61626364  64 63 62 61
	// 0x64636261  61 62 63 64

	inputbytes_t inval;
	inval.inputint = ((uint32_t *)d_Buffer)[ix];

	unsigned char recval;

	bool lastword = false;
	bool lastbyte = false;
	// if at last int32 then only doing overbytes bytes
	if (ix == (totalints - 1))
	{
		lastword = true;
	}
	// loop through the 4 (or fewer) bytes and output 
	for (int i = 0; i < 4; i++)
	{
		recval = 0;

		// if we are on the last word some special handling
		if (lastword)
		{
			// if no overbytes but at last byte, OR in overbyte situation and at last byte (e.g. ob is 1 and i is 0),
			// assuming the file terminates with a CRLF ensure col and rec headers false to mark end.  scan will add 1 in final position for total record/col counts.
			// OTHERWISE force col and rec headers true to mark end.
			if (((overbytes == 0) && (i == 3)) || (i == (overbytes - 1)))
			{
				lastbyte = true;
			}
			else if ((overbytes > 0) && (i == overbytes))  return;  // if finished all overbytes at end, just return here, all done.
		}

		if (!lastbyte)
		{
			unsigned char ival = inval.inputbytes[i];

			// check if have LF.  That is row marker.
			if (ival == (unsigned char)0xa)
			{
				recval = 1;
			}
		}
		d_RecordHeaders[(ix * 4) + i] = recval;
	}

	return;
}
