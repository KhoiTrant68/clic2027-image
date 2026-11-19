"""Real innovation envelope: fixed scale is a model asset, all bytes counted."""
import struct
import numpy as np

HEADER=struct.Struct('>4sBBBBIH')
MODES={'conditional':1,'fixed':2,'pixel':3}


def pack(base,mode,coefficients,rank=8,grid=2):
    if mode not in MODES or not base:raise ValueError('Invalid base/mode')
    q=np.asarray(coefficients)
    if q.shape!=(rank,grid,grid) or not np.issubdtype(q.dtype,np.integer):raise ValueError('Invalid coefficients')
    if not (1<=rank<=32 and 1<=grid<=8) or np.any(q < -127) or np.any(q > 127):raise ValueError('Invalid coefficient range')
    payload=q.astype(np.int8).tobytes()
    return HEADER.pack(b'INOV',1,MODES[mode],rank,grid,len(base),len(payload))+base+payload


def unpack(blob):
    if len(blob)<HEADER.size:raise ValueError('Truncated innovation header')
    magic,version,mode,rank,grid,nbase,ncoeff=HEADER.unpack_from(blob)
    if magic!=b'INOV' or version!=1 or mode not in MODES.values():raise ValueError('Unknown innovation format')
    if not (1<=rank<=32 and 1<=grid<=8) or ncoeff!=rank*grid*grid or nbase<1:raise ValueError('Invalid shape')
    if len(blob)!=HEADER.size+nbase+ncoeff:raise ValueError('Bad innovation lengths')
    name=next(k for k,v in MODES.items() if v==mode)
    base=blob[HEADER.size:HEADER.size+nbase]
    q=np.frombuffer(blob[HEADER.size+nbase:],dtype=np.int8).copy().reshape(rank,grid,grid)
    if np.any(q==-128):raise ValueError('Reserved coefficient value')
    return base,name,q


def self_check():
    q=np.arange(32,dtype=np.int8).reshape(8,2,2)-16
    for mode in MODES:
        blob=pack(b'base',mode,q)
        base,name,recovered=unpack(blob)
        assert base==b'base' and name==mode and np.array_equal(q,recovered)
        assert len(blob)==len(b'base')+46
        for invalid in [blob[:-1],blob+b'0',b'']:
            try:unpack(invalid)
            except ValueError:pass
            else:raise AssertionError('Malformed packet accepted')


if __name__=='__main__':self_check();print('Innovation packet checks passed')
