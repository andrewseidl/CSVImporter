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
using namespace std;
#include <stdint.h>

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

// CUDA runtime
#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

// CUDA atomics
#include "sm_35_atomic_functions.h"

extern "C" __global__ void GetRecLengthsAndColCountErrorsSMEM(uint32_t *  d_RecsTabl, uint32_t *  d_RecsToColsTabl,
	uint32_t *  d_RecLengths, uint32_t *  d_ColCountErrors, uint32_t *  d_ColumnCountsPerRecordTable,
	uint32_t ValuesCount, uint16_t numcols);

extern "C" void launch_GetRecLengthsAndColCountErrorsSMEM(uint32_t *  d_RecsTabl, uint32_t *  d_RecsToColsTabl,
uint32_t *  d_RecLengths, uint32_t *  d_ColCountErrors, uint32_t *  d_ColumnCountsPerRecordTable,
uint32_t ValuesCount, uint16_t numcols);

extern "C" __global__ void GetRecLengthsAndColCountErrorsSMEM2(uint32_t *  d_RecsTabl, uint32_t *  d_RecsToColsTabl,
	uint32_t *  d_RecLengths, uint32_t *  d_ColCountErrors, uint32_t *  d_ColumnCountsPerRecordTable,
	uint32_t ValuesCount, uint16_t numcols);

extern "C" void launch_GetRecLengthsAndColCountErrorsSMEM2(uint32_t *  d_RecsTabl, uint32_t *  d_RecsToColsTabl,
	uint32_t *  d_RecLengths, uint32_t *  d_ColCountErrors, uint32_t *  d_ColumnCountsPerRecordTable,
	uint32_t ValuesCount, uint16_t numcols);


extern "C" __global__ void SimpleStreamCompact(unsigned int *  d_Matches, unsigned int *  d_Scan, unsigned int *  d_ResultsOrdinals, unsigned int ValuesCount);

extern "C" void launch_SimpleStreamCompact(unsigned int *  d_Matches, unsigned int *  d_Scan, unsigned int *  d_ResultsOrdinals, unsigned int ValuesCount);

extern "C" __global__ void BuildMarkerHeaders(uint32_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders, uint32_t TotalBytes);

extern "C" void launch_BuildMarkerHeaders(uint32_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders, uint32_t TotalBytes);

extern "C" void launch_RecordsColumnsChars_StreamCompact(uint32_t *  d_MatchesRecs, uint32_t *  d_MatchesCols, uint32_t *  d_MatchesChars,
uint32_t *  d_ScanRecs, uint32_t *  d_ScanCols, uint32_t *  d_ScanChars,
uint32_t *  d_OrdinalsRecs, uint32_t *  d_OrdinalsCols, uint32_t *  d_OrdinalsChars,
uint32_t *  d_OrdinalsRecsToCols, uint32_t *  d_OrdinalsColsToChars, uint32_t ValuesCount);

extern "C" __global__ void RecordsColumnsChars_StreamCompact(uint32_t *  d_MatchesRecs, uint32_t *  d_MatchesCols, uint32_t *  d_MatchesChars,
	uint32_t *  d_ScanRecs, uint32_t *  d_ScanCols, uint32_t *  d_ScanChars,
	uint32_t *  d_OrdinalsRecs, uint32_t *  d_OrdinalsCols, uint32_t *  d_OrdinalsChars,
	uint32_t *  d_OrdinalsRecsToCols, uint32_t *  d_OrdinalsColsToChars, uint32_t ValuesCount);

extern "C" void launch_GetCharNumsInColumns(uint32_t *  d_ScanCols, uint32_t *  d_ScanChars,
	uint32_t *  d_OrdinalsColsToChars, uint16_t * d_CharNumInCols, uint32_t ValuesCount);
extern "C"  __global__ void GetCharNumsInColumns(uint32_t *  d_ScanCols, uint32_t *  d_ScanChars,
	uint32_t *  d_OrdinalsColsToChars, uint16_t * d_CharNumInCols, uint32_t ValuesCount);

extern "C" void launch_BuildRecsColsCharsHeaders(uint8_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders,
	uint32_t *  d_UTF8charHeaders, uint32_t TotalBytes);

extern "C" __global__ void BuildRecsColsCharsHeaders(uint8_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders,
	uint32_t *  d_UTF8charHeaders, uint32_t TotalBytes);

extern "C" void
launch_BuildRecordHeaders(uint8_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t TotalBytes);
extern "C"
__global__ void BuildRecordHeaders(uint8_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t TotalBytes);




extern "C" void launch_GetColumnNumsInRecords(uint32_t *  d_ScanRecs, uint32_t *  d_ScanCols,
	uint32_t *  d_OrdinalsRecsToCols, uint16_t * d_ColNumInRecs, uint32_t ValuesCount);
extern "C" __global__ void GetColumnNumsInRecords(uint32_t *  d_ScanRecs, uint32_t *  d_ScanCols,
	uint32_t *  d_OrdinalsRecsToCols, uint16_t * d_ColNumInRecs, uint32_t ValuesCount);

extern "C" void launch_UTF8charHeaders(uint32_t *  d_Buffer, uint32_t *  d_UTF8charHeaders, uint32_t TotalBytes);
extern "C" __global__ void UTF8charHeaders(uint32_t *  d_Buffer, uint32_t *  d_UTF8charHeaders, uint32_t TotalBytes);


extern "C" void launch_Tester(int * d_fieldcharsizes, unsigned char ** d_fieldptrs, int fieldcount, uint32_t recordscount);
extern "C" __global__ void Tester(int * d_fieldcharsizes, unsigned char ** d_fieldptrs, int fieldcount, uint32_t recordscount);


extern "C" void launch_WriteCSVRecord(uint8_t *  d_Buffer, uint32_t *  d_RecsTabl, uint32_t *d_ColumnCountErrorsSCAN,
uint32_t *  d_ColsTabl, uint32_t *  d_RecsToColsTabl, uint32_t *  d_RecLengths,
int * d_fieldcharsizes, unsigned char ** d_fieldptrs, int fieldcount, uint32_t recordscount, uint32_t errorscount);

extern "C" __global__ void WriteCSVRecord(uint8_t *  d_Buffer, uint32_t *  d_RecsTabl, uint32_t *d_ColumnCountErrorsSCAN,
uint32_t *  d_ColsTabl, uint32_t *  d_RecsToColsTabl, uint32_t *  d_RecLengths,
int * d_fieldcharsizes, unsigned char ** d_fieldptrs, int fieldcount, uint32_t recordscount, uint32_t errorscount);


extern "C" void
launch_WriteCSVRecord2(uint8_t *  d_Buffer, uint32_t *  d_RecsTabl, uint32_t * d_ColumnCountErrors, uint32_t *d_ColumnCountErrorsSCAN,
uint32_t *  d_ColsTabl, uint32_t *  d_RecsToColsTabl,
/* int * d_fieldcharsizes, unsigned char ** d_fieldptrs, */
int fieldcount, uint32_t recordscount /*, uint8_t charmultiplier = 3 */);

extern "C" __global__ void WriteCSVRecord2(uint8_t *  d_Buffer, uint32_t *  d_RecsTabl, uint32_t * d_ColumnCountErrors, uint32_t *d_ColumnCountErrorsSCAN,
	uint32_t *  d_ColsTabl, uint32_t *  d_RecsToColsTabl,
	/* int * d_fieldcharsizes, unsigned char ** d_fieldptrs, */
	int fieldcount, uint32_t recordscount /*, uint8_t charmultiplier */);


extern "C" bool FixDestFields(const void* fieldUTF8sizesptr, const void* fieldbytesizesptr, size_t fieldsizessize, const void* fieldarrayssptr, size_t fieldarrayssize);



extern "C" void launch_GetColumnCharCountOverflowErrors(uint32_t *d_RecordHeadersSCAN, uint32_t *d_UTF8Headers, uint16_t *d_ColumnNumInRecord,
	uint16_t *d_CharNumInColumn, uint32_t * d_ColumnCountErrors, uint32_t * d_ColumnCharCountErrors, size_t CsvFileLength, int fieldcount);

extern "C" __global__ void GetColumnCharCountOverflowErrors(uint32_t *d_RecordHeadersSCAN, uint32_t *d_UTF8Headers, uint16_t *d_ColumnNumInRecord,
	uint16_t *d_CharNumInColumn, uint32_t * d_ColumnCountErrors, uint32_t * d_ColumnCharCountErrors, size_t CsvFileLength, int fieldcount);

extern "C" void
launch_GetColumnCharCountOverflowErrorsMERGE(uint32_t *d_RecordHeadersSCAN, uint32_t *d_UTF8Headers, uint16_t *d_ColumnNumInRecord,
	uint16_t *d_CharNumInColumn, uint32_t * d_ColumnCountErrors, size_t CsvFileLength, int fieldcount);

extern "C"  __global__ void GetColumnCharCountOverflowErrorsMERGE(uint32_t *d_RecordHeadersSCAN, uint32_t *d_UTF8Headers, uint16_t *d_ColumnNumInRecord,
	uint16_t *d_CharNumInColumn, uint32_t * d_ColumnCountErrors, size_t CsvFileLength, int fieldcount);


extern "C"
__global__ void MarkCommas(uint8_t *  d_Buffer, uint32_t *  d_QuoteBoundaryHeaders, uint8_t *  d_CommaHeaders, uint32_t * d_LinefeedHeaders,
uint32_t * d_ColumnHeaders, uint32_t *  d_printingchars_flags, uint32_t TotalBytes, char delimiter);
extern "C" void
launch_MarkCommas(uint8_t *  d_Buffer, uint32_t *  d_QuoteBoundaryHeaders, uint8_t *  d_CommaHeaders, uint32_t * d_LinefeedHeaders,
uint32_t * d_ColumnHeaders, uint32_t *  d_printingchars_flags, uint32_t TotalBytes, char delimiter);
extern "C"
__global__ void DoubleQuotes(uint8_t *  d_Buffer,
uint32_t *  d_printingchars_flags, uint8_t *  d_secondquotes, uint32_t TotalBytes);
extern "C" void
launch_DoubleQuotes(uint8_t *  d_Buffer,
uint32_t *  d_printingchars_flags, uint8_t *  d_secondquotes, uint32_t TotalBytes);
extern "C"
__global__ void Merge2ndQuotesAndNonprinting(
uint32_t *  d_printingchars_flags, uint8_t *  d_secondquotes, uint32_t TotalBytes);
extern "C" void
launch_Merge2ndQuotesAndNonprinting(
uint32_t *  d_printingchars_flags, uint8_t *  d_secondquotes, uint32_t TotalBytes);
extern "C"
__global__ void RecordsProspectiveColumns_StreamCompact(uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders,
uint32_t *  d_ScanRecs, uint32_t *  d_ScanCols,
uint32_t *  d_RecordsToColumnsTable_commas, uint32_t TotalBytes);
extern "C" void
launch_RecordsProspectiveColumns_StreamCompact(uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders,
uint32_t *  d_ScanRecs, uint32_t *  d_ScanCols,
uint32_t *  d_RecordsToColumnsTable_commas, uint32_t TotalBytes);
extern "C"
__global__ void FixColumnHeaderCommas(uint16_t * d_QuoteBoundaryNumInRecord, uint8_t *  d_CommaHeaders, uint32_t * d_ColumnHeaders, uint32_t TotalBytes);
extern "C" void
launch_FixColumnHeaderCommas(uint16_t * d_QuoteBoundaryNumInRecord, uint8_t *  d_CommaHeaders, uint32_t * d_ColumnHeaders, uint32_t TotalBytes);
extern "C"
__global__ void BufferPrinting_StreamCompact(
uint32_t *  d_printingchars_flags, uint32_t *  d_printingchars_SCAN,
uint8_t *  d_CsvBuffer, uint8_t *  d_CsvBuffer_printing,
uint32_t *  d_RecordHeaders, uint32_t *  d_RecordHeaders_printing,
uint32_t *  d_ColumnHeaders, uint32_t *  d_ColumnHeaders_printing,
uint32_t TotalBytes);
extern "C" void
launch_BufferPrinting_StreamCompact(
uint32_t *  d_printingchars_flags, uint32_t *  d_printingchars_SCAN,
uint8_t *  d_CsvBuffer, uint8_t *  d_CsvBuffer_printing,
uint32_t *  d_RecordHeaders, uint32_t *  d_RecordHeaders_printing,
uint32_t *  d_ColumnHeaders, uint32_t *  d_ColumnHeaders_printing,
uint32_t TotalBytes);
extern "C"
__global__ void BuildCharsHeadersOnly(uint8_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders,
	uint32_t *  d_UTF8charHeaders, uint32_t TotalBytes);
extern "C" void
launch_BuildCharsHeadersOnly(uint8_t *  d_Buffer, uint32_t *  d_RecordHeaders, uint32_t *  d_ColumnHeaders,
uint32_t *  d_UTF8charHeaders, uint32_t TotalBytes);

