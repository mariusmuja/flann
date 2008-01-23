module util.Allocator;

import tango.stdc.stdlib;

import util.defines;


string allocate_static(string declaration)
{
	int pos;
	for (pos = 0; pos<declaration.length;++pos) {
		if (declaration[pos]=='[') break;
	}
	string type = declaration[0..pos];
	int pos2;	
	pos++;
	for (pos2 = pos; pos2<declaration.length;++pos2) {
		if (declaration[pos2]==']') break;
	}
	string size = declaration[pos..pos2];
	
	for (pos = pos2+1; pos<declaration.length;++pos) {
		if (declaration[pos]==' ') continue;
		break;
	}
	for (pos2 = pos; pos2<declaration.length;++pos2) {
		if (declaration[pos2]==';') break;
	}
	
	string vec = declaration[pos..pos2];

	return 	
	"static "~type~"[] "~vec~";
	if ("~vec~" is null || "~vec~".length!="~size~") {
		"~vec~" = new "~type~"["~size~"];
	}";
}




/*-------------------- Pooled storage allocator ---------------------------*/

/* The following routines allow for the efficient allocation of storage in
     small chunks from a specified pool.  Rather than allowing each structure
     to be freed individually, an entire pool of storage is freed at once.
   This method has two advantages over just using malloc() and free().  First,
     it is far more efficient for allocating small objects, as there is
     no overhead for remembering all the information needed to free each
     object or consolidating fragmented memory.  Second, the decision about 
     how long to keep an object is made at the time of allocation, and there
     is no need to track down all the objects to free them.
*/

/* The memory allocated by this class is not handled by the garbage collector. Be 
carefull not to store in this memory pointers to memory handled by the gc.
*/

class Allocator 
{			
	/* We maintain memory alignment to word boundaries by requiring that all
		allocations be in multiples of the machine wordsize.  */
	private const 	int 	WORDSIZE=16;   /* Size of machine word in bytes.  Must be power of 2. */	
	/* Minimum number of bytes requested at a time from	the system.  Must be multiple of WORDSIZE. */
	private const 	int 	BLOCKSIZE=2048;	
		
	private int 	remaining;  /* Number of bytes left in current block of storage. */
	private void*	base;     /* Pointer to base of current block of storage. */
	private void*	loc;      /* Current location in block to next allocate memory. */
	private int 	blocksize;
	
	alias	allocate opCall;
	
	/* 
		Default constructor. Initializes a new pool.
	*/
	public this(int blocksize = BLOCKSIZE)
	{
		this.blocksize = blocksize;
		remaining = 0;
		base = null;
	}
	
	public ~this()
	{
		free();
	}
	
	/* Returns a pointer to a piece of new memory of the given size in bytes
		allocated from the pool.
	*/
	
	public T* allocate(T)(int count = 1) 
	{
		T* mem = cast(T*) malloc(T.sizeof*count);
		return mem;
	}
	
	
	public T[] allocate(T : T[])(int count) 
	{
		T* mem = cast(T*) malloc(count*T.sizeof);		
		return mem[0..count];
	}
	
	public T[][] allocate(T : T[][])(int rows, int cols = -1) 
	{
		if (cols == -1) {
			T[]* mem = cast(T[]*) malloc(rows*(T[]).sizeof);
			return mem[0..rows];
		} 
		else {
			//if (rows & 1) rows++; // for 16 byte allignment	
			void* mem = malloc(rows*(T[]).sizeof+rows*cols*T.sizeof);
			T[]* index = cast(T[]*) mem;
			T* mat = cast(T*) (mem+rows*(T[]).sizeof);
			
			for (int i=0;i<rows;++i) {
				index[i] = mat[0..cols];
				mat += cols;
			}
			
			return index[0..rows];
		}
	}
	
	
	private void* malloc(int size)
	{
		int blocksize;
		
		/* Round size up to a multiple of wordsize.  The following expression
			only works for WORDSIZE that is a power of 2, by masking last bits of
			incremented size to zero.
		*/
		size = (size + (WORDSIZE - 1)) & ~(WORDSIZE - 1);
	
		/* Check whether a new block must be allocated.  Note that the first word
			of a block is reserved for a pointer to the previous block.
		*/
		if (size > remaining) {
		/* Allocate new storage. */
			blocksize = (size + (void*).sizeof + (WORDSIZE-1) > BLOCKSIZE) ?
						size + (void*).sizeof + (WORDSIZE-1) : BLOCKSIZE;
						
			// use the standard C malloc to allocate memory
			void* m = tango.stdc.stdlib.malloc(blocksize);
			if (! m) {
				throw new Exception("Failed to allocate memory.");
			}
			
			/* Fill first word of new block with pointer to previous block. */
			(cast(void **) m)[0] = base;
			base = m;

			//int shift = 0;
			int shift = (WORDSIZE - ( (cast(int)(m) +(void*).sizeof) & (WORDSIZE-1))) & (WORDSIZE-1);
			
			remaining = blocksize - (void*).sizeof - shift;
			loc = m + (void*).sizeof + shift;
		}
		void* rloc = loc;
		loc += size;
		remaining -= size;
		
		return rloc;
	}
	
	/*
		Free all storage that was previously allocated to this pool.
	*/
	public void free()
	{
		char *prev;
	
		while (base != null) {
			prev = *(cast(char **) base);  /* Get pointer to prev block. */
			tango.stdc.stdlib.free(base);
			base = prev;
		}
	}
	
}