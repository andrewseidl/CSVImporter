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
launch_WriteCSVRecord2(uint8_t *  d_Buffer, uint32_t *  d_RecsTabl, uint32_t * d_ColumnCountErrors, uint32_t *d_ColumnCountErrorsSCAN,
	uint32_t *  d_ColsTabl, uint32_t *  d_RecsToColsTabl, int fieldcount, uint32_t recordscount)
{
	// Call kernel.
	uint32_t threadsperwarp = 32;  // one record requires all 32 threads.
	int iThreads = 256;
	float fBlocks = (float)(recordscount * threadsperwarp)/ ((float)iThreads);
	int iBlocks = (recordscount * threadsperwarp) / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	float fwarpsperblock = (float)iThreads / ((float)32);
	int iwarpsperblock = iThreads / 32;
	fwarpsperblock = fwarpsperblock - iwarpsperblock;
	if (fwarpsperblock > 0)
		iwarpsperblock++;

	checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	// shared mem is one uint32 for each lane in each warp in the block
	WriteCSVRecord2 <<<iBlocks, iThreads, iwarpsperblock * 32 * sizeof(uint32_t)>>>(d_Buffer, d_RecsTabl, d_ColumnCountErrors, d_ColumnCountErrorsSCAN, d_ColsTabl, d_RecsToColsTabl,	fieldcount, recordscount);

	Check_cuda_Errors("WriteCSVRecord2");

	return;
}

// declare constant mem accounting for 8k cashe per SM.
__device__ __constant__ int16_t d_fieldUTF8charwidths_C[256];
__device__ __constant__ int16_t d_fieldbytewidths_C[256];
__device__ __constant__ unsigned char * d_fieldptrs_C[256];

// this routine is called from the main program to copy data into constant memory.
// putting it here eliminated declaration issues.
extern "C" bool FixDestFields(const void* fieldUTF8charwidthsptr, const void* fieldbytewidthsptr, size_t fieldsizessize, const void* fieldarrayssptr, size_t fieldarrayssize)
{
	checkCudaErrors(cudaMemcpyToSymbol(d_fieldUTF8charwidths_C, fieldUTF8charwidthsptr, fieldsizessize, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_fieldbytewidths_C, fieldbytewidthsptr, fieldsizessize, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_fieldptrs_C, fieldarrayssptr, fieldarrayssize, 0, cudaMemcpyHostToDevice));
	return true;
}


// pass in:
// the "CSV" buffer, the records table, the col count errors headers, the col count errors scan,
// the columns table, the records to columns table,
// the fields count (# of columns), the records count, and the character multipler (convert char sizes to byte sizes -- normally 3, but can be 1 for ease of debugging).
__global__ void WriteCSVRecord2(uint8_t *  d_Buffer, uint32_t *  d_RecsTabl, uint32_t * d_ColumnCountErrors, uint32_t *d_ColumnCountErrorsSCAN,
	uint32_t *  d_ColsTabl, uint32_t *  d_RecsToColsTabl, int fieldcount, uint32_t recordscount)
{
	// unions for loading as...
	union inputbytes_t
	{
		uint32_t inputuint;
		unsigned char inputbytes[4];
	};
	// similar to above but uses signed chars
	union infou32_i8_t
	{
		uint32_t infouint;
		char infobytes[4];
	};
	union infou32_u8_t
	{
		uint32_t infouint;
		uint8_t infoubytes[4];
	};
	union infou64_u16_t
	{
		uint64_t infouint64;
		uint16_t infouint16s[4];
	};

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned char laneid = (unsigned char)ix & 0x1f;  // lane 0-31 w/in this warp.

	extern __shared__ uint32_t nextchunkvals[];  // available to read 1 chunk ahead.

	// this gets the warp number in the device (not the block).
	uint32_t warpnumindevice = ((blockIdx.x * blockDim.x) + threadIdx.x) / warpSize;
	int warpnuminblock = threadIdx.x >> 5;  // int divide by 32, throw away remainder

	// Since we are making each warp take care of its own record,
	// if we have more warps than records, simply have the whole
	// warp end early - no work to do.
	if (warpnumindevice >= recordscount) return;  // ignore anything in last block beyond source arrays length.

	// however, we only do records without errors, so exit if this record has a col count error.
	if (d_ColumnCountErrors[warpnumindevice] != 0) return;

	// NOW have a non-error record to do.
	// now must adjust destination record index.  subtract off the errors to this point from exclusive scan.
	// DO NOT write out data for a error record.  Skip it.
	uint32_t destination_recindex = warpnumindevice - d_ColumnCountErrorsSCAN[warpnumindevice];  // skip up to adjust for dropped col count error records.

	uint32_t recstartbyte;
	int32_t ColsTableIdxAtRecStart;

	// get rec length.  NOTE: later use records table if have a cap at end.
	uint32_t reclen;

	if (warpnumindevice == 0)
	{
		recstartbyte = 0;
		ColsTableIdxAtRecStart = -1;  // flags on first record.
		reclen = d_RecsTabl[0] - 2;
	}
	else
	{
		recstartbyte = d_RecsTabl[warpnumindevice - 1] + 1;  // get the record for this warp.  add 1 to bump past header (LF)
		ColsTableIdxAtRecStart = d_RecsToColsTabl[warpnumindevice - 1];  // get index in cols table for the rec delim for start of this record.
		reclen = (d_RecsTabl[warpnumindevice] - (recstartbyte - 1)) - 3;
	}

	// this calcs how many chunks reading.
	uint16_t chunk128count = (uint16_t)((reclen + recstartbyte) >> 7) - (uint16_t)(recstartbyte >> 7) + (uint16_t)1;


	// tracking for each lane, by each byte in lane.
	infou32_u8_t RelColNum;  // This is the 1-based relative column num within the warp for the current chunk.
	infou32_i8_t ShufBack;  // This is the number of byte positions to shuffle back for each byte:  0 - 3
	infou32_i8_t Bytes2Write;  // This is the number of bytes to write from the current position when it is shuffled back to byte 0 positon: 0-4
	infou64_u16_t DestByteOffset;  // This tracks the offset in the destination buffer for the current set of bytes, once aligned.

	uint16_t ChunkColumnStart;  // this is the 0-based column # for the start of each chunk.
	uint16_t temp16;  // short term 16 bit num.

	// housekeeping that spans the chunks.
	uint32_t CurColStartingByte, CurColEndingByte;
	uint32_t curcolheader, nextcolheader;
	uint16_t N;  // 1-based col number for entire record.
	char ChunkDone = 0;  // flag used when breaking out of columns loop to go to next chunk.
	char haveColStartEnd = 0;  // flag set between chunks, when already have col start and end in local vars, so don't need to access memory again.

	// big chunk loop.
	nextcolheader = 0;  // set to 0 since for initial chunk cannot reuse the value.
	for (int curchunk = 0; curchunk < chunk128count; curchunk++)
	{
		// let all threads read 4 bytes
		inputbytes_t inval;

		// get the byte position index into the CSV buffer for byte 0 of the current lane.
		uint32_t LaneByte0IndexIntoBuf = (recstartbyte & 0xffffff80) + (curchunk * 128) + (laneid * 4);

		// now if 1st chunk read from buffer.
		if (curchunk == 0)
		{
			// read the buffer 4 bytes at a time as uint32 array.
			inval.inputuint = ((uint32_t *)d_Buffer)[LaneByte0IndexIntoBuf >> 2];
		}
		// otherwise read it from shared mem.
		else
		{
			inval.inputuint = nextchunkvals[(warpnuminblock * warpSize) + laneid];
		}

		// if have at least one more chunk after this one...
		if (curchunk < (chunk128count - 1))
		{
			// read from next chunk into shared.
			nextchunkvals[(warpnuminblock * warpSize) + laneid] = ((uint32_t *)d_Buffer)[(LaneByte0IndexIntoBuf + 128) >> 2];
		}

		// NOW all lanes loaded with raw input.

		// set chunk col start here.  if chunk 0, set to 0.
		if (curchunk == 0) ChunkColumnStart = 0;
		// otherwise build with carryover N.
		else ChunkColumnStart = N - 1;


		RelColNum.infouint = 0;  // 0 means not evaluated yet.
		ShufBack.infouint = 0;  // 0 default no shuffles back.
		Bytes2Write.infouint = 0;  // default no bytes to write.
		DestByteOffset.infouint64 = 0;  // default no offsets.


		// loop through all the columns until all threads have accounted for all bytes.
		// RelCol is column relative to current chunk, will start at 1 for first chunk in a record.
		// loop is to max # rel cols, but normally break earlier.
		// therefore its max is 128 which would be for a 128 byte chunk of only column headers.
		for (uint8_t RelCol = 1; RelCol <= 128; RelCol++)
		{
			// now get N by adding RelCol to ChunkColumnStart
			N = ChunkColumnStart + RelCol;

			// if now have an N > field count (even though relcol is not), done with this chunk and also this record for byte logic.
			if (N > fieldcount)
			{
				ChunkDone = 1;
				break;  // break out of col loop.  below will proceed to next chunk, but this will have been the last one.
			}

			// if coming in on a new chunk will already have calc'ed col params for first col in chunk
			if (haveColStartEnd == 0)
			{
				// at the very beginning, curcol hdr idx = 0, next is first entry.
				if ((ColsTableIdxAtRecStart == -1) && (N == 1))
				{
					nextcolheader = d_ColsTabl[0];
					CurColStartingByte = 0;  // start byte is very first byte
				}
				// not at the very beginning.
				else
				{
					// reuse nextcolheader if have one.
					if (nextcolheader != 0)  curcolheader = nextcolheader;
					else curcolheader = d_ColsTabl[ColsTableIdxAtRecStart + N - 1];

					// read nextcolheader from cols table.
					nextcolheader = d_ColsTabl[ColsTableIdxAtRecStart + N];

					// get start byte index for column
					CurColStartingByte = curcolheader + 1;  // byte after | or LF
				}


				// have specs on Column N.  now get last byte index for col.
				if (N == fieldcount) CurColEndingByte = nextcolheader - 2;  // byte before CR LF
				else CurColEndingByte = nextcolheader - 1;  // byte before |


				// check if at a col that starts after this chunk.
				// NOTE: even though don't need CurColEndingByte (calc'ed above) to do this comparison, set it anyway,
				// so we will be ready for reuse in the next chunk
				if (CurColStartingByte >= ((recstartbyte & 0xffffff80) + ((curchunk + 1) * 128)))
				{
					// ChunkColumnStart = N - 1;
					ChunkDone = 1;
					haveColStartEnd = 1;  // don't need to re-read entering next chunk.
					break;  // out of cols loop.
				}

			}
			else haveColStartEnd = 0;  // ensure this is reset since this is only carried over between chunks.


			// now does this column apply to the bytes in our lane?

			// loop through all four bytes and check individually.
			for (char ByteIdx = 0; ByteIdx < 4; ByteIdx++)
			{
				// if this is the first col, then flag out any bytes that are less.
				if ((N == 1) && (CurColStartingByte >(LaneByte0IndexIntoBuf + (uint32_t)ByteIdx)))
				{
					RelColNum.infoubytes[ByteIdx] = (unsigned char)0xff;  // flag byte prior to start.
				}
				// if this is last col, then flag out any bytes after the content.
				if ((N == fieldcount) && ((LaneByte0IndexIntoBuf + (uint32_t)ByteIdx) > CurColEndingByte))
				{
					RelColNum.infoubytes[ByteIdx] = (unsigned char)0xff;  // flag byte past end.
				}

				// otherwise is the cur col relevant to the current byte?
				// start of col has to be <= curbyte, and end has to be >= curbyte - 1 (the - 1 allows for 0-length cols).
				if ((CurColStartingByte <= (LaneByte0IndexIntoBuf + (uint32_t)ByteIdx)) && (CurColEndingByte >= (LaneByte0IndexIntoBuf + (uint32_t)ByteIdx)))
				{
					RelColNum.infoubytes[ByteIdx] = (char)RelCol;  // flag rel col for cur byte.

					// now how far will cur byte have to be shuffled back for an aligned write.
					// check the position of the start of the column.  we put this on all bytes in the col, since applies to all.
					ShufBack.infobytes[ByteIdx] = (char)(CurColStartingByte & 3);

					// the destination offset for the current byte is its position less the position of the starting byte.
					DestByteOffset.infouint16s[ByteIdx] = (uint16_t)((LaneByte0IndexIntoBuf + (uint32_t)ByteIdx) - CurColStartingByte);

					// finally set bytes to write for cur byte.  this reflects how many bytes to write forward, once this byte is aligned.
					// so should only be set for a byte which will be the first in the aligned shuffle.
					// ex. if the align/shuffle back for this col is 3, the byte in question would be 3.
					if (ShufBack.infobytes[ByteIdx] == ByteIdx)
					{
						temp16 = (uint16_t)((((int32_t)CurColEndingByte + (int32_t)1) - ((int32_t)LaneByte0IndexIntoBuf + (int32_t)ByteIdx)));
						if (temp16 > 4) Bytes2Write.infobytes[ByteIdx] = 4;  // max 4 bytes.
						else Bytes2Write.infobytes[ByteIdx] = (char)temp16;

						// the destination offset for the current byte is its position less the position of the starting byte.
						DestByteOffset.infouint16s[ByteIdx] = (uint16_t)((LaneByte0IndexIntoBuf + (uint32_t)ByteIdx) - CurColStartingByte);
					}
					else Bytes2Write.infobytes[ByteIdx] = 0;  // if 0-length col or not an aligned start, 0 to write.
				}
			}

			// now at end of byte processing, check if our ending byte is in the next chunk.
			if (CurColEndingByte >= ((recstartbyte & 0xffffff80) + ((curchunk + 1) * 128)))
			{
				// ChunkColumnStart = N - 1;
				ChunkDone = 1;
				haveColStartEnd = 1;  // don't need to re-read entering next chunk.
				break;  // out of cols loop.
			}
		}  // end of columns loop.



		// do the shuffles here because shouldn't do inside conditional.
		// will manipulate below using these.
		uint32_t shuffledown = __shfl_down_sync(0xFFFFFFFF, inval.inputuint, 1);
		// can't shuffle into lane 31. so get from shared or set to 0.
		if (laneid == 31)
		{
			if (curchunk < (chunk128count - 1))
			{
				// if not the last chunk read the shuf value from FIRST byte in shared for this warp.
				// ?? Will this cause bank conflict?  If so is it significant?
				shuffledown = nextchunkvals[(warpnuminblock * warpSize)];
			}
			else
			{
				// if there is no next chunk, make the shuf value 0.
				shuffledown = 0;
			}
		}

		// now loop through the "shuffles"
		for (char shufflecount = 0; shufflecount < 4; shufflecount++)
		{
			// NOTE: the vars below may cause undue register pressure.  we can swap out for the values they get set to here in code below.
			// get the bytes to write for the byte in the position.
			char curbytestowrite = Bytes2Write.infobytes[shufflecount];
			uint8_t currelcol = RelColNum.infoubytes[shufflecount];
			uint16_t curdestoffset = DestByteOffset.infouint16s[shufflecount];

			// ALSO: could even get rid of writeout by putting one big expression in the write.
			uint32_t writeout;

			// if (curbytestowrite > 0)
			if (Bytes2Write.infobytes[shufflecount] > 0)
			{
				// First build the raw 32 bits properly shuffled.
				//Shuffle to shift everything "down" 8 bits, trying to achieve alignment.
				if (shufflecount == 0)
				{
					writeout = inval.inputuint;
				}
				else if (shufflecount == 1)
				{
					writeout = inval.inputuint >> 8;  // this will reduce the value by 256
					writeout |= ((shuffledown & 0xff) << 24);  // this puts the low byte of the next into the high byte of the current.
				}
				else if (shufflecount == 2)
				{
					writeout = inval.inputuint >> 16;  // this will reduce the value by 256 * 256
					writeout |= ((shuffledown & 0xffff) << 16);  // this puts the 2 low bytes of the next into the high 2 bytes of the current.
				}
				else  // shufflecount == 3
				{
					writeout = inval.inputuint >> 24;  // this will reduce the value by 256 * 256 * 256
					writeout |= ((shuffledown & 0xffffff) << 8);  // this puts the 3 low bytes of the next into the high 3 bytes of the current.
				}

				// now mask off if there are less than 4 bytes to write.
				if (curbytestowrite == 3) writeout &= 0x00ffffff;
				else if (curbytestowrite == 2) writeout &= 0x0000ffff;
				else if (curbytestowrite == 1) writeout &= 0x000000ff;

				// Correction 2/23/16: removed char multiplier, now using separate char and byte widths
				int16_t curcolbytewidth = d_fieldbytewidths_C[(currelcol + ChunkColumnStart) - 1];
				if ( curcolbytewidth > 0 )
				{
					*(uint32_t*)(d_fieldptrs_C[(currelcol + ChunkColumnStart) - 1] + (destination_recindex * curcolbytewidth) + curdestoffset) = writeout;
				}
			}

		}

		// if chunk done flag set
		if (ChunkDone == 1)
		{
			ChunkDone = 0;
			continue;  // to next chunk.
		}
	}  // end of chunk loop.

	return;
}



// Code to collect Column Char Count Errors moved in here to use constant mem for d_fieldcharsizes_C.

extern "C" void
launch_GetColumnCharCountOverflowErrors(uint32_t *d_RecordHeadersSCAN, uint32_t *d_UTF8Headers, uint16_t *d_ColumnNumInRecord,
uint16_t *d_CharNumInColumn, uint32_t * d_ColumnCountErrors, uint32_t * d_ColumnCharCountErrors, size_t CsvFileLength, int fieldcount )
{

	// Call stream compact kernel.
	int iThreads = 256;
	float fBlocks = (float)CsvFileLength / ((float)iThreads);
	int iBlocks = CsvFileLength / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	GetColumnCharCountOverflowErrors <<<iBlocks, iThreads >>>(d_RecordHeadersSCAN, d_UTF8Headers, d_ColumnNumInRecord, d_CharNumInColumn,
		d_ColumnCountErrors, d_ColumnCharCountErrors, CsvFileLength, fieldcount );

	Check_cuda_Errors("GetColumnCharCountOverflowErrors");

}


__global__ void GetColumnCharCountOverflowErrors(uint32_t *d_RecordHeadersSCAN, uint32_t *d_UTF8Headers, uint16_t *d_ColumnNumInRecord,
	uint16_t *d_CharNumInColumn, uint32_t * d_ColumnCountErrors, uint32_t * d_ColumnCharCountErrors, size_t CsvFileLength, int fieldcount)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix > CsvFileLength) return;  // ignore anything in last block beyond source arrays length.

	// get the 0-based rec num.
	uint32_t recnum = d_RecordHeadersSCAN[ix];

	// if already have a col count error, exit since the alignment of cols to col widths will be off.
	// should not check in that case -- already have an error on the record.
	if (d_ColumnCountErrors[recnum] != 0) return;

	// ignore anything that is not a char start.
	if (d_UTF8Headers[ix] == 0)  return;

	// get the width of the current col in the current rec.
	int16_t mycolwidth = d_fieldUTF8charwidths_C[d_ColumnNumInRecord[ix]];

	// no check where char width is -1.  we ignore.
	if ( mycolwidth != -1)
	{
		// if get an exact match of char num and col width, this is the first char that exceeed the length.
		// this gets flagged as an error.
		if (d_CharNumInColumn[ix] == mycolwidth)
		{
			// flag a col char count error per record
			// NOTE: if there are multiple col char count errors in the record, this will be written by multiple kernels.  hopefully OK.
			d_ColumnCharCountErrors[recnum] = 1;
			// flag a col char count error per record
		}
	}
	return;
}




// Version of the above that merges col char count errors into col count errors.

extern "C" void
launch_GetColumnCharCountOverflowErrorsMERGE(uint32_t *d_RecordHeadersSCAN, uint32_t *d_UTF8Headers, uint16_t *d_ColumnNumInRecord,
uint16_t *d_CharNumInColumn, uint32_t * d_ColumnCountErrors, size_t CsvFileLength, int fieldcount)
{

	// Call stream compact kernel.
	int iThreads = 256;
	float fBlocks = (float)CsvFileLength / ((float)iThreads);
	int iBlocks = CsvFileLength / iThreads;
	fBlocks = fBlocks - iBlocks;
	if (fBlocks > 0)
		iBlocks++;

	GetColumnCharCountOverflowErrorsMERGE <<<iBlocks, iThreads >>>(d_RecordHeadersSCAN, d_UTF8Headers, d_ColumnNumInRecord, d_CharNumInColumn, d_ColumnCountErrors, CsvFileLength, fieldcount);

	Check_cuda_Errors("GetColumnCharCountOverflowErrorsMERGE");

}


__global__ void GetColumnCharCountOverflowErrorsMERGE(uint32_t *d_RecordHeadersSCAN, uint32_t *d_UTF8Headers, uint16_t *d_ColumnNumInRecord,
	uint16_t *d_CharNumInColumn, uint32_t * d_ColumnCountErrors, size_t CsvFileLength, int fieldcount)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix > CsvFileLength) return;  // ignore anything in last block beyond source arrays length.

	// get the 0-based rec num.
	uint32_t recnum = d_RecordHeadersSCAN[ix];

	// since merging errors, if already have a col count error, exit since an error is already flagged.
	if (d_ColumnCountErrors[recnum] != 0) return;

	// ignore anything that is not a char start.
	if (d_UTF8Headers[ix] == 0)  return;

	// get the width of the current col in the current rec.
	int16_t mycolwidth = d_fieldUTF8charwidths_C[d_ColumnNumInRecord[ix]];

	// no check where char width is -1.  we ignore.
	if (mycolwidth != -1)
	{
		// if get an exact match of char num and col width, this is the first char that exceeed the length.
		// this gets flagged as an error.
		if (d_CharNumInColumn[ix] == mycolwidth)
		{
			// flag a col char count error per record
			// NOTE: if there are multiple col char count errors in the record, this will be written by multiple kernels.  hopefully OK.
			d_ColumnCountErrors[recnum] = 1;
			// flag a col char count error per record
		}
	}

	return;
}
