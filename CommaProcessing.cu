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

// first kernel to run marks commas.  all commas, as well as commas marking a quote comma boundary,
// either single (or odd #) quote going into to comma, or comma going into single (or odd #) quote.
// this kernel also flags odd # quote (before, after, or both) as non-printing

// second kernel picks up quote pairs.  this kernel must run AFTER first one is complete.  
// the purpose is to pick up quote pairs and flag the SECCONDs as non-printing.
// it starts with a first quote.
// this means first in file, first after a non-quote, or first after a non-printing quote.
// this way it picks up quotes in the middle of a long description, a quote set before an ending quote-comma
// boundary, or a quote set starting after an opening quote comma boundary.
// the kernel will handle all consecutive quotes up to non-quote, non-printing, or end of file.
// the idea is to make sure that the threads in this kernel do not step on each other.
// no quote should be handled by more than 1 thread.  critical because no 2 threads should ever be flagging same char as non-printing.

extern "C"
__global__ void MarkCommas(uint8_t *  d_Buffer, uint32_t *  d_QuoteBoundaryHeaders, uint8_t *  d_CommaHeaders, uint32_t * d_LinefeedHeaders,
	uint32_t * d_ColumnHeaders,	uint32_t *  d_printingchars_flags, uint32_t TotalBytes, char delminiter)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= TotalBytes) return;

	int ixlast = (TotalBytes - ix);  // position of last byte in buf

	// max look back and forward from current char pos
	int maxlookback = ix;
	int maxlookforward = ixlast - 1;
	bool havecomma = false;
	bool haveCR = false;
	bool haveLF = false;
	bool atStart = false;
	bool atEnd = false;
	// normal # chars to start looks: if on comma or rec boundary == 1.
	int startlook = 1;

	// now check if this comma is a quoted comma column boundary.
	// if there are an odd number of commas before, after, or both, it is.
	// otherwise it is a comma, but not a quoted boundary.

	if (d_Buffer[ix] == delminiter)
	{
		// flag have a comma or record header
		d_CommaHeaders[ix] = 1;
		d_ColumnHeaders[ix] = 1; // COMMA IS TENTATIVE COLUMN HEADER
		havecomma = true;
	}
	else if (d_Buffer[ix] == 0xa)
	{
		// flag have a column header
		d_ColumnHeaders[ix] = 1;
		// flag have linefeed i.e. record header
		d_LinefeedHeaders[ix] = 1;
		// ALSO set QuoteBoundaryHeaders flag at record header.  This allows a sync up with each record.
		d_QuoteBoundaryHeaders[ix] = 1;
		haveLF = true;
	}
	else
	{
		// no comma or record header
		d_CommaHeaders[ix] = 0;
	}

	// check for the other "edge cases": record boundary, beg or end of file.
	if (d_Buffer[ix] == 0xd)
	{
		haveCR = true;
	}
	if (ix == 0)
	{
		atStart = true;
	}
	if (ix == ixlast)
	{
		atEnd = true;
	}
	if (atStart || atEnd)
	{
		// at file end or beginning, start AT that char looking for odd quotes.
		startlook = 0;
	}

	// unless have 1 of above conditions, nothing to do.
	if (!(havecomma || haveCR || haveLF || atStart || atEnd)) return;
	
	int fwdquotecount = 0;
	int revquotecount = 0;

	// forward look for comma, LF, or start)
	if (havecomma || haveLF || atStart)
	{
		for (int offset = startlook; offset <= maxlookforward; offset++)
		{
			if (d_Buffer[ix + offset] != '"') break;
			fwdquotecount++;
		}
	}

	// reverse look for comma, CR, or end
	if (havecomma || haveCR || atEnd)
	{
	for (int offset = startlook; offset <= maxlookback; offset++)
		{
			if (d_Buffer[ix - offset] != '"') break;
			revquotecount++;
		}
	}

	// odd number of quotes mean have quote boundary.
	bool endboundary = ((revquotecount & 1) == 1);
	bool startboundary = ((fwdquotecount & 1) == 1);

	// check for odd quote counts.  means this is commaquote column boundary.
	// To accomodate possible end and start boundaries on same comma, must mark the Quote not the Comma as a boundary.
	// This way even-odd works.
	// in all cases, the boundary quote is non-printing.
	if (startboundary)
	{
		d_printingchars_flags[ix + startlook] = 1;
		d_QuoteBoundaryHeaders[ix + startlook] = 1;
	}
	if (endboundary)
	{
		d_printingchars_flags[ix - startlook] = 1;
		d_QuoteBoundaryHeaders[ix - startlook] = 1;
	}

}


extern "C" void
launch_MarkCommas(uint8_t *  d_Buffer, uint32_t *  d_QuoteBoundaryHeaders, uint8_t *  d_CommaHeaders, uint32_t * d_LinefeedHeaders,
uint32_t * d_ColumnHeaders, uint32_t *  d_printingchars_flags, uint32_t TotalBytes, char delimiter)
{
	int iThreads = 256;
	float fBlocks = (float)TotalBytes / ((float)iThreads);
	int iBlocks = TotalBytes / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	MarkCommas <<< iBlocks, iThreads >>>
		(d_Buffer, d_QuoteBoundaryHeaders, d_CommaHeaders, d_LinefeedHeaders, d_ColumnHeaders, d_printingchars_flags, TotalBytes, delimiter);

	Check_cuda_Errors("MarkCommas");
}



extern "C"
__global__ void DoubleQuotes(uint8_t *  d_Buffer,
	uint32_t *  d_printingchars_flags, uint8_t *  d_secondquotes, uint32_t TotalBytes)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= TotalBytes) return;

	// max look back and forward from current char pos
	int maxlookforward = (TotalBytes - ix) - 1;

	// look for starting quote.

	if (d_Buffer[ix] != '"')  return;  // exit if not quote
	if (d_printingchars_flags[ix] == 1) return;  // exit if not printing
	
	// exit if this is not a first printing quote (that way threads won't step on each other).
	if (ix != 0)
	{
		if ((d_Buffer[ix - 1] == '"') && (d_printingchars_flags[ix - 1] == 0)) return;
	}

	// so this is a first quote that is currently printing,.

	// loop forward.  every other quote should be a second quote.
	// IMPORTANT NOTE: our intent is to flag even quotes as non-printing.
	// However, can't do that in this kernel as printing status is being used by each
	// thread to make determination.
	// Thus we edit a second array, which a later kernel merges.
	for (int offset = 1; offset <= maxlookforward; offset++)
	{
		if (d_Buffer[ix + offset] != '"') break;
		if ((offset & 1) == 1) d_secondquotes[ix + offset] = 1;
	}

}

extern "C" void
launch_DoubleQuotes(uint8_t *  d_Buffer,
uint32_t *  d_printingchars_flags, uint8_t *  d_secondquotes, uint32_t TotalBytes)
{
	int iThreads = 256;
	float fBlocks = (float)TotalBytes / ((float)iThreads);
	int iBlocks = TotalBytes / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	DoubleQuotes <<< iBlocks, iThreads >>>
		(d_Buffer, d_printingchars_flags, d_secondquotes, TotalBytes);


	Check_cuda_Errors("DoubleQuotes");
}



// simple kernel follows up DoubleQuotes to merge Second Quotes flags with Printing chars.
// this must be done as a second pass to avoid threads stepping on each other.
// here we reverse the meaning of d_printingchars_flags, so that 1 is printing, 0 is non.
// this will allow us to stream compact the buf.
extern "C"
__global__ void Merge2ndQuotesAndNonprinting(
	uint32_t *  d_printingchars_flags, uint8_t *  d_secondquotes, uint32_t TotalBytes)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= TotalBytes) return;

	// read the current printing flag.
	// comes in meaning non-printing.  will turn to printing.
	bool nonprinting = ( d_printingchars_flags[ix] == 1 );

	// if this is a 2nd quote, flag it as non-printing.
	if (d_secondquotes[ix] == 1)
	{
		nonprinting = true;
	}

	uint32_t printingflagval = 1;
	if (nonprinting) printingflagval = 0;
	d_printingchars_flags[ix] = printingflagval;
}

extern "C" void
launch_Merge2ndQuotesAndNonprinting(
uint32_t *  d_printingchars_flags, uint8_t *  d_secondquotes, uint32_t TotalBytes)
{
	int iThreads = 256;
	float fBlocks = (float)TotalBytes / ((float)iThreads);
	int iBlocks = TotalBytes / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	Merge2ndQuotesAndNonprinting <<< iBlocks, iThreads >>>
		(d_printingchars_flags, d_secondquotes, TotalBytes);


	Check_cuda_Errors("Merge2ndQuotesAndNonprinting");
}


// kernel does prelim work toward segmented scan of prospective columns.
// a simplified version of RecordsColumnsChars_StreamCompact() that removes UTF8 char processing.
extern "C"
__global__ void RecordsProspectiveColumns_StreamCompact(uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders,
uint32_t *  d_ScanRecs, uint32_t *  d_ScanCols,
uint32_t *  d_RecordsToColumnsTable_commas, uint32_t TotalBytes)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= TotalBytes) return;  // ignore anything in last block beyond source arrays length.

	// the index into the result arrays is simply the Exclusive Scan value at the current position.
	if (d_RecordHeaders[ix] == 1)
	{
		// the recstocols puts in the same relative position as the recs table the SCAN value of the cols, for looking up into the cols table.
		d_RecordsToColumnsTable_commas[d_ScanRecs[ix]] = d_ScanCols[ix];
	}

	return;
}

extern "C" void
launch_RecordsProspectiveColumns_StreamCompact(uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders,
uint32_t *  d_ScanRecs, uint32_t *  d_ScanCols,
uint32_t *  d_RecordsToColumnsTable_commas, uint32_t TotalBytes)
{
	int iThreads = 256;
	float fBlocks = (float)TotalBytes / ((float)iThreads);
	int iBlocks = TotalBytes / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	RecordsProspectiveColumns_StreamCompact <<< iBlocks, iThreads >>>
		(d_RecordHeaders, d_ColumnHeaders, d_ScanRecs, d_ScanCols, d_RecordsToColumnsTable_commas, TotalBytes);

	Check_cuda_Errors("RecordsProspectiveColumns_StreamCompact");
}


// this kernel uses the segmented scan of the quote boundary scan within records to eliminate
// commas from within quoted columns from counting as column headers.
extern "C"
__global__ void FixColumnHeaderCommas(uint16_t * d_QuoteBoundaryNumInRecord, uint8_t *  d_CommaHeaders, uint32_t * d_ColumnHeaders, uint32_t TotalBytes)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= TotalBytes) return;

	// only process a comma in the buf.
	if (d_CommaHeaders[ix] == 0)  return;

	uint16_t curquoteboundarynuminrec = d_QuoteBoundaryNumInRecord[ix];

	// if this is an odd number, clear the col header.
	if ((curquoteboundarynuminrec & 1) == 1)
	{
		d_ColumnHeaders[ix] = 0;
	}

	return;
}

extern "C" void
launch_FixColumnHeaderCommas(uint16_t * d_QuoteBoundaryNumInRecord, uint8_t *  d_CommaHeaders, uint32_t * d_ColumnHeaders, uint32_t TotalBytes)
{
	int iThreads = 256;
	float fBlocks = (float)TotalBytes / ((float)iThreads);
	int iBlocks = TotalBytes / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	FixColumnHeaderCommas <<< iBlocks, iThreads >>>
		(d_QuoteBoundaryNumInRecord, d_CommaHeaders, d_ColumnHeaders, TotalBytes);

	Check_cuda_Errors("FixColumnHeaderCommas");
}


// kernel stream compacts buffer and records and cols headers eliminating "non-printing" chars.
extern "C"
__global__ void BufferPrinting_StreamCompact(
uint32_t *  d_printingchars_flags, uint32_t *  d_printingchars_SCAN,
uint8_t *  d_CsvBuffer, uint8_t *  d_CsvBuffer_printing,
uint32_t *  d_RecordHeaders, uint32_t *  d_RecordHeaders_printing,
uint32_t *  d_ColumnHeaders, uint32_t *  d_ColumnHeaders_printing,
uint32_t TotalBytes)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= TotalBytes) return;  // ignore anything in last block beyond source arrays length.

	if (d_printingchars_flags[ix] == 0)  return;  // ignore non-printing chars.

	uint32_t compactidx = d_printingchars_SCAN[ix];  // stream compact index

	// stream compacting:
	d_CsvBuffer_printing[compactidx] = d_CsvBuffer[ix];
	d_RecordHeaders_printing[compactidx] = d_RecordHeaders[ix];
	d_ColumnHeaders_printing[compactidx] = d_ColumnHeaders[ix];

	return;
}

extern "C" void
launch_BufferPrinting_StreamCompact(
uint32_t *  d_printingchars_flags, uint32_t *  d_printingchars_SCAN,
uint8_t *  d_CsvBuffer, uint8_t *  d_CsvBuffer_printing,
uint32_t *  d_RecordHeaders, uint32_t *  d_RecordHeaders_printing,
uint32_t *  d_ColumnHeaders, uint32_t *  d_ColumnHeaders_printing,
uint32_t TotalBytes)
{
	int iThreads = 256;
	float fBlocks = (float)TotalBytes / ((float)iThreads);
	int iBlocks = TotalBytes / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	BufferPrinting_StreamCompact <<< iBlocks, iThreads >>>
		(d_printingchars_flags, d_printingchars_SCAN,
		d_CsvBuffer, d_CsvBuffer_printing,
		d_RecordHeaders, d_RecordHeaders_printing,
		d_ColumnHeaders, d_ColumnHeaders_printing, TotalBytes);

	Check_cuda_Errors("launch_BufferPrinting_StreamCompact");
}


// this kernel is a version of BuildRecsColsCharsHeaders that ONLY builds the char headers.
// this is used for true comma-delimited processing, where the record and col headers have already been created.
//
// newer version also builds headers for UTF-8 character starts.

__global__ void BuildCharsHeadersOnly(uint8_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders,
	uint32_t *  d_UTF8charHeaders, uint32_t TotalBytes)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= TotalBytes) return;

	uint32_t charval = 0;  // assume 0.

	// check chars only if not on col or rec header, and not on a CR.
	if ((d_RecordHeaders[ix] == 0) &&
		(d_ColumnHeaders[ix] == 0) &&
		(d_Buffer[ix] != 0xd))
	{
		unsigned char ival = d_Buffer[ix];
		// check for UTF-8 first byte: either is format 0xxx xxxx  or NOT 10xx xxxx
		if ((ival & 0x80) == 0) charval = 1;
		else if ((ival & 0xc0) != (unsigned char)0x80) charval = 1;

		// if not true have a 2nd or 3rd UTF-8 byte, so not a new char.
	}

	// set the header to 0 or 1.
	d_UTF8charHeaders[ix] = charval;

	return;
}

// this kernel builds only the UTF-8 char headers.
// designed to work with true comma-delim processing, where already have record and col headers
// from prior operations.
extern "C" void
launch_BuildCharsHeadersOnly(uint8_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders,
uint32_t *  d_UTF8charHeaders, uint32_t TotalBytes)
{

	// Call build headers kernel.
	int iThreads = 256;
	float fBlocks = (float)TotalBytes / ((float)iThreads);
	int iBlocks = TotalBytes / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	BuildCharsHeadersOnly <<< iBlocks, iThreads >>> (d_Buffer, d_RecordHeaders, d_ColumnHeaders, d_UTF8charHeaders, TotalBytes);

	Check_cuda_Errors("BuildCharsHeadersOnly");
}
