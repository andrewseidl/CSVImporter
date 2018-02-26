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

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <time.h>

using namespace std;

// CUDA runtime
#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

// CUDA atomics
#include "sm_35_atomic_functions.h"

#include "CommonDefinitions.h"

#include "include/util/mgpucontext.h"
#include "include/kernels/scan.cuh"

#include <algorithm>

#include "csvImporter.h"
#include "CSV_kernel_declarations.cuh"

using namespace mgpu;

// function copies only col names on non-omitted columns.
extern "C" bool CopyOnlyUsedColNames(char * columnnames[], bool skucolflagsin[], uint16_t totalcolcount, uint16_t definedcolcount, int16_t * arrayUTF8charwidths, char ** usedcolumnnames[], bool * skucolflagsout[], uint16_t * usedcolcount)
{
	// can't define more cols than you have.
	if (definedcolcount > totalcolcount) return false;

	// count how many pointers we need.
	int cidx = 0;
	for (int idx = 0; idx < definedcolcount; idx++)
	{
		if (arrayUTF8charwidths[idx] > 0)
		{
			cidx++;
		}
	}
	*usedcolcount = cidx;

	// malloc the arrays.
	char ** tnamearray = new char*[cidx];
	bool * skucols = new bool[cidx];

	cidx = 0;
	for (int idx = 0; idx < definedcolcount; idx++)
	{

		if (arrayUTF8charwidths[idx] > 0)
		{
			tnamearray[cidx] = columnnames[idx];  // simply copy over the pointer.  don't reallocate.
			skucols[cidx] = skucolflagsin[idx];
			cidx++;
		}
	}
	*skucolflagsout = skucols;
	*usedcolumnnames = tnamearray;

	return true;
}

// Reads in first record of CSV file to get the column names and total column count.
extern "C" bool PreProcessFileHeaderRecord(char * filepath, char ** columnnames[], uint16_t * colcount, int64_t * seekafterheader, char delim)
{
	pCsvFileIn = fopen(filepath, "rb");
	if (pCsvFileIn == NULL) return false;

	fseek(pCsvFileIn, (int64_t)0, SEEK_END);
	CsvFileLength = ftell(pCsvFileIn);  // get length of file.
	fseek(pCsvFileIn, (int64_t)0, SEEK_SET);  // reset file ptr to beginning for read

	uint64_t readlen = 32768;  // assume this is the biggest record.
	char * tbuf = new char[readlen];
	if (CsvFileLength < readlen) readlen = CsvFileLength;
	fread(tbuf, sizeof(char), readlen, pCsvFileIn);  // read in the data.


	// alloc temp array for up to 1000 cols.
	char ** tnamearray = new char*[1000];


	int colindex = 0;  // count the columns.
	int colstartpos = 0;  // start of current col.
	int pastrecord = 0;
	bool recordfound = false;
	// now parse the data to get first record.
	for (int i = 0; i < (int)readlen; i++)
	{
		// if hit a line feed, this is the end of the record.
		if (tbuf[i] == '\n')
		{
			pastrecord = i + 1;  // position past this first record is next byte after linefeed.
			*colcount = colindex;  // count == index since index should be bumped past last record.
			recordfound = true;
			break;
		}
		// if find a delimiter or a CR, have completed a column.
		if ((tbuf[i] == delim) || (tbuf[i] == '\r'))
		{
			int colnamelen = i - colstartpos;  // get the length of the column name.
			// for a blank name create a name.
			if (colnamelen == 0)
			{
				tnamearray[colindex] = new char[14];
				sprintf(tnamearray[colindex], "##COLUMN %04d", colindex);
			}
			else
			{
				// here we could add a trim string.

				// for now just copy over the characters.
				tnamearray[colindex] = new char[colnamelen + 1];
				strncpy(tnamearray[colindex], tbuf + colstartpos, colnamelen);
				tnamearray[colindex][colnamelen] = 0;
			}
			colindex++;
			colstartpos = i + 1;  // past the delimiter.
		}
	}

	// if read all the way with no record found, return false.
	if (!recordfound) return false;

	// now allocate array to return and copy pointers over from temp array.
	char ** finalcolumnnames = new char*[colindex];  // alloc for number of name pointers.
	for (int i = 0; i < colindex; i++)
	{
		finalcolumnnames[i] = tnamearray[i];
	}
	*columnnames = finalcolumnnames;  // copy new array to pointer.

	*seekafterheader = (int64_t)pastrecord;

	fseek(pCsvFileIn, (int64_t)0, SEEK_SET);  // reset file ptr to beginning for read.

	// FOR NOW CLOSE THE FILE.
	fclose(pCsvFileIn);

	delete tnamearray;
	delete tbuf;

	return true;
}

// Deletes ALL column names (shared by both columnnames and usedcolumnnames) as well as deleting
// and 0-ing the columnnames and usedcolumnnames.
extern "C" bool DeleteFileHeaderNames(char ** columnnames[], uint16_t colcount, char ** usedcolumnnames[], bool * usedskuflags[])
{
	char ** colnames = *columnnames;

	for (int i = 0; i < colcount; i++)
	{
		delete colnames[i];
	}
	delete colnames;
	*columnnames = 0;

	colnames = *usedcolumnnames;
	delete colnames;
	*usedcolumnnames = 0;

	delete *usedskuflags;
	*usedskuflags = 0;

	return true;
}



// This functions is like the prior one, except its purpose is simply to return an initial SEEK point after the first record.
extern "C" bool SkipFileHeaderRecord(char * filepath, int64_t * seekafterheader)
{
	pCsvFileIn = fopen(filepath, "rb");
	if (pCsvFileIn == NULL) return false;

	fseek(pCsvFileIn, (int64_t)0, SEEK_END);
	CsvFileLength = ftell(pCsvFileIn);  // get length of file.
	fseek(pCsvFileIn, (int64_t)0, SEEK_SET);  // reset file ptr to beginning for read

	uint64_t readlen = 32768;  // assume this is the biggest record.
	char * tbuf = new char[readlen];
	if (CsvFileLength < readlen) readlen = CsvFileLength;
	fread(tbuf, sizeof(char), readlen, pCsvFileIn);  // read in the data.

	int pastrecord = 0;
	bool recordfound = false;
	// now parse the data to get first record.
	for (int i = 0; i < (int)readlen; i++)
	{
		// if hit a line feed, this is the end of the record.
		if (tbuf[i] == '\n')
		{
			pastrecord = i + 1;  // position past this first record is next byte after linefeed.
			recordfound = true;
			break;
		}
	}

	// if read all the way with no record found, return false.
	if (!recordfound) return false;

	*seekafterheader = (int64_t)pastrecord;

	fseek(pCsvFileIn, (int64_t)0, SEEK_SET);  // reset file ptr to beginning for read.

	// FOR NOW CLOSE THE FILE.
	fclose(pCsvFileIn);

	delete tbuf;

	return true;
}



// main function
//
// here we read the CSV file in and build the output arrays.
// this is set up for the older vertical bar delimiter or true comma-delimited files.
// pass in:
// the full path of the input CSV file
// an array of character widths per field  (this will determine how many characters to allow)
// the number of columns or fields
// deliminator character to look for.
// GPUResidentFlag = false means copy final data arrays to CPU, otherwise copy final data array to GPU
// Device or Host arrays of final column data,
// a multiplier for how many bytes to allocate per char.  normally 1 for ASCII, 3 for UTF-8.
// a byte alignment for the output arrays.  normally 8 (middleware standard), but 4 works here.
//
// NOTE: This function can be used as a more generic CSV importer into GPU arrays.
// Its use is anticipated for future import jobs.
// As part of Middleware for B2Bx, it will normally be set up for ASCII, 1 byte per char,
// 8-byte alignment, and use only 2 columns (for the Division and the SKU).
// Subsequent functions are specially tailored for Middleware.
//
// The purpose of the domiddleware flag is to limit the final copy to 2 columns, DIV and SKU.
// We also pass in the SKU and DIV col #s (defaults 0 and 1) in the INPUT file.
// The middleware outputs from this function will always put SKU in col 0 and DIV in col 1.
//
uint64_t importer_varcols(CudaContext& context, char * filepath,
	int16_t * arrayUTF8charwidths, uint16_t numdefinedcolumns, uint16_t numtotalcolumns,
	char delimiter, bool GPUResidentFlag, unsigned char ** dataColumnPtrs, unsigned int * dataColumnOffsets, int64_t initialseek = 0, uint8_t charmultiplier = 1, uint8_t bytesalignment = 8)
{
	// BELOW simply opens the file, calculates length and chunk sizes
	// returns 0 if no issues, 1 for no records, -1 for file error.
	int chunkret = CSVfilechunking(filepath);
	if (chunkret == -1)
	{
		printf("Error opening file %s.\r\n", filepath);
		return 0;
	}
	else if (chunkret == 1)
	{
		return 0;  // for no records.
	}
	// now just in case we have a header that is skipped, make sure we are 6 bytes are more.
	if ((CsvFileLength - initialseek) <= MAXCHAR0RECORDS)
	{
		return 0;   // for no records.
	}

	// NOW Initialize the buffers that will be reused in each chunk.
	// CPU versions (some for debugging only).
	InitializeCPUElements_REUSABLES(SufficientBytes);

	h_fieldUTF8charsizes = 0;
	checkCudaErrors(cudaMallocHost((void **)&h_fieldUTF8charsizes, numtotalcolumns * sizeof(uint16_t)));


	// here we simply alloc the array of pointers (not the data).
	checkCudaErrors(cudaMallocHost((void **)&h_fieldptrs, numtotalcolumns * sizeof(unsigned char *)));
	checkCudaErrors(cudaMallocHost((void **)&h_d_fieldptrs, numtotalcolumns * sizeof(unsigned char *)));
	// since these are arrays of pointers, 0 out the pointers.
	for (int xi = 0; xi < numtotalcolumns; xi++)
	{
		h_fieldptrs[xi] = (unsigned char *)0;
		h_d_fieldptrs[xi] = (unsigned char *)0;
	}
	printf("HOST ARRAY CARRIAGES ALLOC'ED AT (host) %llx (device) %llx.\r\n", (int64_t)h_fieldptrs, (int64_t)h_d_fieldptrs);

	int savedfieldcount = 0;  // this tracks the fields that are not discarded.

	// loop through the char sizes array.
	// two objectives:
	// 1. define byte sizes based on char multiplier and alignment.
	// 2. flesh out array for undefined columns at ends of records.
	for (int num = 0; num < numtotalcolumns; num++)
	{
		// if have gone past defined columns we make an ignore column.
		if (num >= numdefinedcolumns)
		{
			h_fieldUTF8charsizes[num] = -1;
		}
		// otherwise copy it over to the new fullsized array.
		else
		{
			// copy over the width as passed in, may not be an aligned value.
			// however, determines the max num of UTF-8 chars.
			h_fieldUTF8charsizes[num] = arrayUTF8charwidths[num];
		}

		// Also -1 is special case, meaning ignore column.  so don't try to align.
		if (h_fieldUTF8charsizes[num] == -1)
		{
			G_h_fieldbytewidths[num] = 0;  // 0 means ignore column.
		}
		else
		{
			// ENFORCE byte alignment for the byte width of the field.
			uint16_t tsiz = arrayUTF8charwidths[num] * charmultiplier;  // multiply by storage multiplier (e.g., 3 for UTF-8, 1 for ASCII)
			uint16_t tlowbits = tsiz & (uint16_t)(bytesalignment - 1);
			if (tlowbits != (uint16_t)0)
			{
				tsiz += ((uint16_t)bytesalignment - tlowbits);
			}
			G_h_fieldbytewidths[num] = tsiz;

			// bump the final field count.
			savedfieldcount++;
		}
	}

	// NOW Initialize the buffers that will be reused in each chunk.
	// Set length to max byte size in chunks plus 1 to pick up final value in exclusive scan.
	// NOTE: no need to do any more since should be ample and also rounded up to 256 byte boundary.
	InitializeGPUElements_REUSABLES(SufficientBytes);

	startseek = initialseek;  // just to make sure starting at the beginning or just after header.
	bool lastchunk = false;
	uint64_t bytestoread;
	uint32_t chunkbytes;
	for (int chunknum = 0; (chunknum < inumchunks) && (!lastchunk); chunknum++)
	{
		// print out 1-based:
		printf("Processing Chunk %d of %d%s.\r\n", chunknum + 1, inumchunks, (lastchunk ? " (last)" : ""));

		// here we clean up the full reusable GPU mem.
		MemsetGPUElements_REUSABLES(SufficientBytes);

		uint64_t testchunksize = SufficientBytes - OVERREAD;  // take off the overread to get the test chunk size.
		if ((CsvFileLength - startseek) <= testchunksize)
		{
			lastchunk = true;
			bytestoread = CsvFileLength - (uint64_t)startseek;
			chunkbytes = (uint32_t)bytestoread;  // this is the exact chunk size.
		}
		else
		{
			bytestoread = testchunksize;
		}

		fseek(pCsvFileIn, startseek, SEEK_SET);  // set file ptr to start for read
		fread(h_CsvBuffer_a, sizeof(char), bytestoread, pCsvFileIn);  // read in the chunk.

		// if not at last chunk, back up to a record terminator (linefeed).
		if (!lastchunk)
		{
			for (int bidx = (int)(bytestoread - 1); bidx >= 0; bidx--)
			{
				if (h_CsvBuffer_a[bidx] == 0x0a)
				{
					chunkbytes = bidx + 1;
					h_CsvBuffer_a[chunkbytes] = 0;  // make sure to 0 out next byte since that will copy.
					break;
				}
			}
		}
		// else if this is lastchunk, must close the file.
		else
		{
			fclose(pCsvFileIn);
		}

		if (chunkbytes == 0)
		{
			printf("Error: record break not found.\r\n");
			exit(0);
		}
		startseek += (uint64_t)chunkbytes;   // for next time.

		/////////////////////
		// Sometimes files are missing the final CR LF we use to identify a record.
		// In this case we patch on a final CR LF.
		// The host buffer has ample memory allocated to do so.
		if (lastchunk && (h_CsvBuffer_a[bytestoread - 1] != 0xa))
		{
			h_CsvBuffer_a[bytestoread] = 0xd;
			h_CsvBuffer_a[bytestoread + 1] = 0xa;
			chunkbytes += 2;
		}
		/////////////////////

		// START PROCESSING OF CHUNK.
		uint32_t chunkbytesplus1 = chunkbytes + 1;
		uint32_t chunkbytesplus1uint32 = (chunkbytes + 1) * sizeof(uint32_t);

		checkCudaErrors(cudaMemcpy((void*)d_CsvBuffer, (void*)h_CsvBuffer_a, (size_t)chunkbytesplus1, cudaMemcpyHostToDevice));

		// handle true comma delim inputs.
		// idea for now is to "preprocess" the file, then clean it up to look more like the
		// older vertical bar delim, except we will substitute (temporarily) a 0 for the vert bar
		// col delim.  the rest of the code will proceed more or less as before.

		launch_MarkCommas(d_CsvBuffer, d_QuoteBoundaryHeaders, d_CommaHeaders, d_RecordHeaders, d_ColumnHeaders, d_printingchars_flags, chunkbytes, delimiter);

		launch_DoubleQuotes(d_CsvBuffer, d_printingchars_flags, d_secondquotes, chunkbytes);

		launch_Merge2ndQuotesAndNonprinting(d_printingchars_flags, d_secondquotes, chunkbytes);

		printf("Starting Scan Record Headers (Commas).\n");
		Scan<MgpuScanTypeExc>(d_RecordHeaders, chunkbytesplus1,
			(uint32_t)0, mgpu::plus<uint32_t>(), (uint32_t *)0, (uint32_t*)0, d_RecordHeadersSCAN, context);

		// get records count.
		uint32_t recordstablecount_commas;
		checkCudaErrors(cudaMemcpy(&recordstablecount_commas, d_RecordHeadersSCAN + chunkbytes, sizeof(uint32_t), cudaMemcpyDeviceToHost));

		printf("Starting Scan Quote Boundaries (Commas).\n");
		Scan<MgpuScanTypeExc>(d_QuoteBoundaryHeaders, chunkbytesplus1,
			(uint32_t)0, mgpu::plus<uint32_t>(), (uint32_t *)0, (uint32_t*)0, d_QuoteBoundaryHeaders_SCAN, context);

		// get quote boundaries count.
		uint32_t quoteboundariescount;
		checkCudaErrors(cudaMemcpy(&quoteboundariescount, d_QuoteBoundaryHeaders_SCAN + chunkbytes, sizeof(uint32_t), cudaMemcpyDeviceToHost));

		// Records table is a table of record header locations.
		// Columns table is a table of PROSPECTIVE column header locations.
		// RecordsToColumns table is a table that maps record headers to locations in the Columns table.

		// do the prelim processing to get segmented scan.
		launch_RecordsProspectiveColumns_StreamCompact(d_RecordHeaders, d_QuoteBoundaryHeaders, d_RecordHeadersSCAN, d_QuoteBoundaryHeaders_SCAN, d_RecordsToQuoteBoundariesTable, chunkbytesplus1);

		// use an existing kernel here.
		launch_GetColumnNumsInRecords(d_RecordHeadersSCAN, d_QuoteBoundaryHeaders_SCAN, d_RecordsToQuoteBoundariesTable, d_ColumnNumInRecord, chunkbytes);
		launch_FixColumnHeaderCommas(d_ColumnNumInRecord, d_CommaHeaders, d_ColumnHeaders, chunkbytes);

		// now we need to stream compact buffer, rec headers, and col headers for use later.
		printf("Starting Scan Printing Chars (Commas).\n");
		Scan<MgpuScanTypeExc>(d_printingchars_flags, chunkbytesplus1,
			(uint32_t)0, mgpu::plus<uint32_t>(), (uint32_t *)0, (uint32_t*)0, d_printingchars_SCAN, context);

		// get the final count of printing chars.
		uint32_t printingcharscount;
		checkCudaErrors(cudaMemcpy(&printingcharscount, d_printingchars_SCAN + chunkbytes, sizeof(uint32_t), cudaMemcpyDeviceToHost));

		// now reallocate buffer, record headers, and col headers.


		////////////***
		// For the replacement buffer must also add some space at the end for the CSV writer to read ahead a full chunk without a memory error.
		// This should only require 128 bytes.  Round up as a sanity check.
		uint32_t printingcharstemp = printingcharscount + 1 + 256;
		uint32_t mod = printingcharstemp % 256;
		if (mod > 0) printingcharstemp += (256 - mod);



		launch_BufferPrinting_StreamCompact(d_printingchars_flags, d_printingchars_SCAN,
			d_CsvBuffer, d_CsvBuffer_printing,
			d_RecordHeaders, d_RecordHeaders_printing,
			d_ColumnHeaders, d_ColumnHeaders_printing, chunkbytes);

		// now reintegrate into prior vertical bar col sep version.
		// essentially shorten up what we normally have.
		// shorten chunkbytes as needed.
		chunkbytes = printingcharscount;
		chunkbytesplus1 = chunkbytes + 1;
		chunkbytesplus1uint32 = chunkbytesplus1 * sizeof(uint32_t);

		launch_BuildCharsHeadersOnly(d_CsvBuffer_printing, d_RecordHeaders_printing, d_ColumnHeaders_printing, d_UTF8Headers, chunkbytesplus1);

		printf("Starting Scan UTF8 Headers.\n");

		// exclusive scan the ends headers so all positions for each zip will have same scan value.
		Scan<MgpuScanTypeExc>(d_UTF8Headers, chunkbytesplus1,
			(uint32_t)0, mgpu::plus<uint32_t>(), (uint32_t *)0, (uint32_t*)0, d_UTF8HeadersSCAN, context);


		// retrieve last value, the one past the end of the actual values.
		uint32_t * h_charstablecount = (uint32_t *)malloc(sizeof(uint32_t));

		// want the record at the end of the scan.  the pointer math adds CsvFileLength as uint32_t so equals a CsvFileLength * 4 bytes adjustment
		cudaStatus = cudaMemcpy((void*)h_charstablecount, (void*)(d_UTF8HeadersSCAN + chunkbytes), 4, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed (utf8 chars scan last)! %s", cudaGetErrorString(cudaStatus));
			return 0;
		}
		uint32_t charstablecount = *h_charstablecount;

		free(h_charstablecount);

		printf("Starting Scan Record Headers.\n");
		Scan<MgpuScanTypeExc>(d_RecordHeaders_printing, chunkbytesplus1,
			(uint32_t)0, mgpu::plus<uint32_t>(), (uint32_t *)0, (uint32_t*)0, d_RecordHeadersSCAN, context);

		// retrieve last value, the one past the end of the actual values.
		uint32_t * h_recordstablecount = (uint32_t *)malloc(sizeof(uint32_t));

		// want the record at the end of the scan.  the pointer math adds CsvFileLength as uint32_t so equals a CsvFileLength * 4 bytes adjustment
		checkCudaErrors(cudaMemcpy((void*)h_recordstablecount, (void*)(d_RecordHeadersSCAN + chunkbytes), 4, cudaMemcpyDeviceToHost));
		uint32_t recordstablecount = *h_recordstablecount;
		free(h_recordstablecount);

		for (int idx = 0; idx < numtotalcolumns; idx++)
		{
			// only malloc when there is a positive width.
			if (G_h_fieldbytewidths[idx] > 0)
			{
				int bytessize = G_h_fieldbytewidths[idx] * recordstablecount;

				checkCudaErrors(cudaMalloc((void **)&h_d_fieldptrs[idx], bytessize));

				checkCudaErrors(cudaMallocHost((void **)&h_fieldptrs[idx], bytessize));
				printf("HOST MALLOC'ED ARR.ELEM %d FOR CHUNK %d: %llx.\r\n", idx, chunknum, (int64_t)h_fieldptrs[idx]);
			}
			else h_d_fieldptrs[idx] = 0;
		}

		bool cpy = FixDestFields((const void*)h_fieldUTF8charsizes, (const void*)G_h_fieldbytewidths, (size_t)(numtotalcolumns * sizeof(uint16_t)), (const void*)h_d_fieldptrs, (size_t)(numtotalcolumns * sizeof(unsigned char *)));

		printf("Starting Scan Column Headers.\n");
		Scan<MgpuScanTypeExc>(d_ColumnHeaders_printing, chunkbytesplus1,
			(uint32_t)0, mgpu::plus<uint32_t>(), (uint32_t *)0, (uint32_t*)0, d_ColumnHeadersSCAN, context);

		// retrieve last value, the one past the end of the actual values.
		// want the column at the end of the scan.  the pointer math adds CsvFileLength as uint32_t so equals a CsvFileLength * 4 bytes adjustment
		uint32_t columnstablecount = 0;
		checkCudaErrors(cudaMemcpy((void*)&columnstablecount, (void*)(d_ColumnHeadersSCAN + chunkbytes), 4, cudaMemcpyDeviceToHost));

		launch_RecordsColumnsChars_StreamCompact(d_RecordHeaders_printing, d_ColumnHeaders_printing, d_UTF8Headers, d_RecordHeadersSCAN, d_ColumnHeadersSCAN, d_UTF8HeadersSCAN,
			d_RecordsTable, d_ColumnsTable, d_UTF8CharsTable, d_RecordsToColumnsTable, d_ColumnsToUTF8charsTable, chunkbytesplus1);

		launch_GetRecLengthsAndColCountErrorsSMEM2(d_RecordsTable, d_RecordsToColumnsTable, d_RecordLengths, d_ColumnCountErrors, d_ColumnCountsPerRecordTable, recordstablecount, numtotalcolumns);

		launch_GetColumnNumsInRecords(d_RecordHeadersSCAN, d_ColumnHeadersSCAN, d_RecordsToColumnsTable, d_ColumnNumInRecord, chunkbytes);

		launch_GetCharNumsInColumns(d_ColumnHeadersSCAN, d_UTF8HeadersSCAN, d_ColumnsToUTF8charsTable, d_CharNumInColumn, chunkbytes);

		// Get the char count in column overflow errors.
		// VERSION THAT MERGES ERRORS WITH COL COUNT ERRORS.
		launch_GetColumnCharCountOverflowErrorsMERGE(d_RecordHeadersSCAN, d_UTF8Headers, d_ColumnNumInRecord, d_CharNumInColumn, d_ColumnCountErrors, chunkbytes, numtotalcolumns);

		printf("Starting Scan Errors Headers.\n");

		Scan<MgpuScanTypeExc>(d_ColumnCountErrors, recordstablecount + 1,
			(uint32_t)0, mgpu::plus<uint32_t>(), (uint32_t *)0, (uint32_t*)0, d_ColumnCountErrorsSCAN, context);

		// retrieve last value, the one past the end of the actual values.
		// want the column at the end of the scan.  the pointer math makes the 4 bytes adjustment by type of pointer.
		uint32_t columncounterrorscount;
		checkCudaErrors(cudaMemcpy((void*)&columncounterrorscount, (void*)(d_ColumnCountErrorsSCAN + recordstablecount), 4, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMallocHost((void**)&h_ColumnCountErrorsTable_a, (columncounterrorscount * sizeof(uint32_t)) + 128));

		launch_SimpleStreamCompact(d_ColumnCountErrors, d_ColumnCountErrorsSCAN, d_ColumnCountErrorsTable, recordstablecount + 1);

		checkCudaErrors(cudaMemcpy((void*)h_ColumnCountErrorsTable_a, (void*)d_ColumnCountErrorsTable, columncounterrorscount * sizeof(uint32_t), cudaMemcpyDeviceToHost));

		printf("Write CSV record.\r\n");

		for (int idx = 0; idx < numtotalcolumns; idx++)
		{
			// only memset when there is a positive width.
			if (G_h_fieldbytewidths[idx] > 0)
			{
				//cuda malloc for char size <<plus 1 (for null term)>> time 3 for utf8 explansion time number of records in CSV file.
				int bytessize = G_h_fieldbytewidths[idx] /* * (int)charmult*/ * recordstablecount;

				checkCudaErrors(cudaMemset((void*)h_d_fieldptrs[idx], 0, bytessize));
			}
		}

		launch_WriteCSVRecord2((uint8_t *)d_CsvBuffer_printing, d_RecordsTable,
			d_ColumnCountErrors, d_ColumnCountErrorsSCAN, d_ColumnsTable, d_RecordsToColumnsTable,
			/* d_fieldcharsizes, d_fieldptrs, */
			numtotalcolumns, recordstablecount/*, charmult*/);

		for (int idx = 0; idx < numtotalcolumns; idx++)
		{
			// only memcpy/free when there is a positive width.
			if (G_h_fieldbytewidths[idx] > 0)
			{
				//cuda malloc for char size <<plus 1 (for null term)>> time 3 for utf8 explansion time number of records in CSV file.
				int bytessize = G_h_fieldbytewidths[idx] /* * (int)charmult*/ * recordstablecount;

				checkCudaErrors(cudaMemcpy((void*)h_fieldptrs[idx], (void*)h_d_fieldptrs[idx], bytessize, cudaMemcpyDeviceToHost));
				Check_cuda_Free((void **)&h_d_fieldptrs[idx]);  // free up the GPU buf now.
			}
		}

		// now manage the link lists of memblocks on host side.
		// for the first chunk, create the link lists.
		if (chunknum == 0)
		{
			// create new bases.

			// allocate for saved field count only, however, reference from original CSV column count.
			h_llbases = 0;
			checkCudaErrors(cudaMallocHost((void **)&h_llbases, savedfieldcount * sizeof(memlinklist_base)));

			int sidx = 0;
			for (int idx = 0; idx < numtotalcolumns; idx++)
			{
				// copy to the link list only saved columns.
				if (G_h_fieldbytewidths[idx] > 0)
				{
					// first create a member record and assign current block to it.
					memlinklist_member * h_ll_memb = 0;
					checkCudaErrors(cudaMallocHost((void **)&h_ll_memb, sizeof(memlinklist_member)));

					(*h_ll_memb).h_cur_block_ptr = (void*)h_fieldptrs[idx];
					(*h_ll_memb).cur_block_validrecordcount = recordstablecount - columncounterrorscount;
					(*h_ll_memb).cur_block_errorcount = columncounterrorscount;
					(*h_ll_memb).next = NULL;

					h_llbases[sidx].recordwidth = G_h_fieldbytewidths[idx];
					h_llbases[sidx].first = h_ll_memb;
					h_llbases[sidx].totalvalidrecords = recordstablecount - columncounterrorscount;
					h_llbases[sidx].totalerrors = columncounterrorscount;

					sidx++;
				}
			}
		}
		// after 1st chunk add to linked lists.
		else
		{
			int sidx = 0;
			for (int idx = 0; idx < numtotalcolumns; idx++)
			{
				// copy to the link list only saved columns.
				if (G_h_fieldbytewidths[idx] > 0)
				{
					// make a new member for the current array block.
					memlinklist_member * h_ll_memb = 0;
					checkCudaErrors(cudaMallocHost((void **)&h_ll_memb, sizeof(memlinklist_member)));

					(*h_ll_memb).h_cur_block_ptr = (void*)h_fieldptrs[idx];
					(*h_ll_memb).cur_block_validrecordcount = recordstablecount - columncounterrorscount;
					(*h_ll_memb).cur_block_errorcount = columncounterrorscount;
					(*h_ll_memb).next = NULL;

					// now find next open slot in link list to place it.
					memlinklist_member * start = h_llbases[sidx].first;
					while ((*start).next != NULL)
					{
						start = (*start).next;
					}

					(*start).next = h_ll_memb;  // the open "next" pointer is set to new member.

					h_llbases[sidx].totalvalidrecords += (recordstablecount - columncounterrorscount);  // bump total records count with this block.
					h_llbases[sidx].totalerrors += columncounterrorscount;  // bump total errors count with this block.

					sidx++;
				}
			}
		}

		// free the memory that is re-used by subsequent chunks.
		validrecordscount = recordstablecount - columncounterrorscount;  // save global for debug purposes below.

		Check_cuda_FreeHost((void **)&h_ColumnCountErrorsTable_a);
	}  // end of chunks loop

	// clean up
	DeinitializeGPUElements_REUSABLES();

	// CPU versions for debugging.
	DeinitializeCPUElements_REUSABLES(SufficientBytes);

	// now build CPU or GPU arrays per the linked list based on GPUResidentFlag.

	// RESET FIELD COUNT NOW TO SAVED FIELD COUNT.  NO MORE NEED TO ACCOUNT FOR UNUSED CSV COLUMNS.
	// First reset the byte widths to their new positions.
	int newfi = 0;
	for (int fi = 0; fi < numtotalcolumns; fi++)
	{
		if (G_h_fieldbytewidths[fi] == 0) continue;  // skip a 0 byte width.
		G_h_fieldbytewidths[newfi] = G_h_fieldbytewidths[fi];  // this should be fine as only copying to same or down.
		// Also set the final offset alignment for the return array
		dataColumnOffsets[newfi] = G_h_fieldbytewidths[fi];
		newfi++;  // bump new index once written.
	}

	// note can use 1 of the bases since all should have the same total count.
	G_totalvalidCSVrecordscount = h_llbases[0].totalvalidrecords;

	for (int idx = 0; idx < savedfieldcount; idx++)
	{
		int bytesperrecord = h_llbases[idx].recordwidth;

		//cuda malloc for the total valid bytes.
		uint64_t totalbytessize = bytesperrecord * G_totalvalidCSVrecordscount;
		if(GPUResidentFlag == true)
			checkCudaErrors(cudaMalloc((void **)&dataColumnPtrs[idx], totalbytessize));
		else
			checkCudaErrors(cudaMallocHost((void **)&dataColumnPtrs[idx], totalbytessize));

		// now copy over each of the chunk arrays from the host.
		int chunkcounter = 0;
		int cumbytescopied = 0;

		// read in the first member.
		memlinklist_member * start = h_llbases[idx].first;
		int curcount = (*start).cur_block_validrecordcount;
		void * h_curblock = (void*)(*start).h_cur_block_ptr;
		if (GPUResidentFlag == true)
			checkCudaErrors(cudaMemcpy((void*)dataColumnPtrs[idx], h_curblock, (size_t)(bytesperrecord*curcount), cudaMemcpyHostToDevice));
		else
			memcpy((void*)dataColumnPtrs[idx], h_curblock, (size_t)(bytesperrecord*curcount));

		// can free host mem block now.
		printf("HOST FREE ARR.ELEM %d COMBINING CHUNKS @ CHUNK %d: %llx.\r\n", idx, chunkcounter, (int64_t)h_curblock);
		Check_cuda_FreeHost((void **)&h_curblock);

		cumbytescopied = (bytesperrecord*curcount);

		while ((*start).next != NULL)
		{
			chunkcounter++;

			start = (*start).next;
			curcount = (*start).cur_block_validrecordcount;
			h_curblock = (void*)(*start).h_cur_block_ptr;
			if (GPUResidentFlag == true)
				checkCudaErrors(cudaMemcpy((void*)(dataColumnPtrs[idx] + cumbytescopied), h_curblock, (size_t)(bytesperrecord*curcount), cudaMemcpyHostToDevice));
			else
				memcpy((void*)(dataColumnPtrs[idx] + cumbytescopied), h_curblock, (size_t)(bytesperrecord*curcount));

			// can free host mem block now.
			printf("HOST FREE ARR.ELEM %d COMBINING CHUNKS @ CHUNK %d: %llx.\r\n", idx, chunkcounter, (int64_t)h_curblock);
			Check_cuda_FreeHost((void **)&h_curblock);

			cumbytescopied += (bytesperrecord*curcount);
		}
	}

	Check_cuda_FreeHost((void **)&h_llbases);  // After freeing all sets of members, free the bases.

	// free up the array of pointers.
	printf("HOST FREE ARRAY CARRIAGE (host): %llx.\r\n", (int64_t)h_fieldptrs);
	Check_cuda_FreeHost((void **)&h_fieldptrs);
	printf("HOST FREE UTF8 CHAR SIZES (host): %llx.\r\n", (int64_t)h_fieldUTF8charsizes);
	Check_cuda_FreeHost((void **)&h_fieldUTF8charsizes);

	return G_totalvalidCSVrecordscount;
}  // end importer_varcols()




extern "C" uint64_t CSVImporterMain(char * filename, char delimiter, uint16_t numTotalColumns, uint16_t numDefinedColumns, int16_t * ColumnCharWidths, unsigned char ** dataColumnPtrs, unsigned int * dataColumnOffsets, int64_t seekafterhdr, uint8_t charmultiplier, bool GPUResidentFlag)
{
	// Get the Cuda device with the most GFLOPS for this operation
	ContextPtr context = CreateCudaDevice(gpuGetMaxGflopsDeviceId());

	checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	uint64_t vcret; // return val for varcols calls.
	vcret = importer_varcols(*context, filename, ColumnCharWidths, numDefinedColumns, numTotalColumns, delimiter, GPUResidentFlag, dataColumnPtrs, dataColumnOffsets, seekafterhdr, charmultiplier);
	
	return vcret;
}


