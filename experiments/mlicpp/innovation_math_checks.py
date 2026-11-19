"""Exact finite-model checks and real toy packets; no ML or novelty claims.

All probabilities/arithmetic use Fraction. The decoder reads only bytes and
fixed model constants. This is a mathematical example, never a CLIC codec.
"""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import struct


HEADER=struct.Struct('>4sBBB')
AMPLITUDE=F(2,5)
ETA=F(1,100)
METHODS={0:('conditional_basis',2),1:('fixed_basis',2),2:('prior_variance',1),3:('source_coordinate',1)}


def pack(method,word):
    if method not in METHODS:raise ValueError('Unknown toy model')
    bits=METHODS[method][1]
    if type(word) is not int or not 0<=word<2**bits:raise ValueError('Invalid word')
    return HEADER.pack(b'ICM1',1,method,bits)+bytes([word])


def decode(blob):
    if len(blob)!=HEADER.size+1:raise ValueError('Wrong packet length')
    magic,version,method,bits=HEADER.unpack_from(blob)
    if magic!=b'ICM1' or version!=1 or method not in METHODS or bits!=METHODS[method][1]:
        raise ValueError('Invalid toy header')
    word=blob[-1]
    if word>=2**bits:raise ValueError('Nonzero padding')
    if method in [0,1]:
        b=word>>1;s=1 if word&1 else -1
        if method==0:return (F(1,2)+s*AMPLITUDE*(b==0),F(1,2)+s*AMPLITUDE*(b==1))
        return (F(1,2)+s*AMPLITUDE/2,F(1,2)+s*AMPLITUDE/2)
    if method==2:return (F(1,2),F(1,2)+ETA*(1 if word else -1))
    return (F(word),F(1,2))


def squared(x,y):
    return sum((a-b)**2 for a,b in zip(x,y))


def main(out):
    if out.exists():raise ValueError('Refuse to replace experiment')
    out.mkdir(parents=True)
    records=[]
    for method in [0,1]:
        folder=out/METHODS[method][0];folder.mkdir()
        errors=[]
        for b in [0,1]:
            for s in [-1,1]:
                source=(F(1,2)+s*AMPLITUDE*(b==0),F(1,2)+s*AMPLITUDE*(b==1))
                blob=pack(method,(b<<1)|int(s>0));path=folder/f'b{b}-s{s}.bin';path.write_bytes(blob)
                recon=decode(path.read_bytes());error=squared(source,recon);errors.append(error)
                records.append(dict(family='conditional_vs_fixed',method=METHODS[method][0],base=b,sign=s,
                    bytes=len(blob),information_bits=2,header_bytes=HEADER.size,padding_bits=6,
                    source=[str(x) for x in source],reconstruction=[str(x) for x in recon],squared_error=str(error),hex=blob.hex()))
        expected=F(0) if method==0 else AMPLITUDE**2/2
        assert sum(errors)/len(errors)==expected
    for method in [2,3]:
        folder=out/METHODS[method][0];folder.mkdir();errors=[]
        for x1 in [0,1]:
            for s in [-1,1]:
                source=(F(x1),F(1,2)+s*ETA)
                blob=pack(method,int(s>0) if method==2 else x1)
                path=folder/f'x{x1}-s{s}.bin';path.write_bytes(blob)
                recon=decode(path.read_bytes());error=squared(source,recon);errors.append(error)
                records.append(dict(family='misspecified_variance',method=METHODS[method][0],bytes=len(blob),
                    information_bits=1,header_bytes=HEADER.size,padding_bits=7,squared_error=str(error),hex=blob.hex()))
        assert sum(errors)/len(errors)==(F(1,4) if method==2 else ETA**2)
    # Check malformed packets fail, rather than silently accepting extra bits.
    invalid=[b'',pack(0,0)+b'\0',pack(0,0)[:-1],pack(0,0)[:-1]+bytes([128])]
    for blob in invalid:
        try:decode(blob)
        except ValueError:pass
        else:raise AssertionError('Malformed packet accepted')
    # Non-optimal decoder example verifies the extra approximation-error term.
    pairs=[(0,F(0)),(0,F(2)),(1,F(-1)),(1,F(3))]
    d0=sum((x-b)**2 for b,x in pairs)/4
    bias=sum((F(1)-b)**2 for b,x in pairs)/4
    information=sum((x-F(1))**2 for b,x in pairs)/4
    approximation=F(1,4)**2
    dh=sum((x-(x+F(1,4)))**2 for b,x in pairs)/4
    assert d0-dh==bias+information-approximation
    # The budget condition admits both a win and a loss.
    tax=F(3,100)
    assert tax-F(2,100)>0 and tax-F(4,100)<0
    # Uniform midpoint quantization: certify the error bound, not monotonic error.
    quant=[]
    for r in [1,2,3,4,8]:
        levels=[-F(1)+F(2*j+1,2**r) for j in range(2**r)]
        selected=min(levels,key=lambda c:abs(c-AMPLITUDE))
        error=abs(AMPLITUDE-selected);bound=F(1,2**r)
        assert error<=bound
        quant.append(dict(bits=r,coefficient=str(selected),error=str(error),bound=str(bound)))
    summary=dict(status='complete',arithmetic='exact fractions; exhaustive finite source cases',
        total_packets=len(records),bytes_per_packet=HEADER.size+1,
        conditional_risk='0',fixed_rank1_risk=str(AMPLITUDE**2/2),
        prior_variance_risk='1/4',source_coordinate_risk=str(ETA**2),
        variance_risk_ratio=str(F(1,4)/ETA**2),decomposition_verified=True,
        malformed_packets_rejected=len(invalid),quantization_checks=quant,
        limitations='Toy source and fixed constants; no MLIC++ image evaluation, no human preference, no novelty claim. Context-conditioned residual coding can match the conditional method.')
    (out/'packets.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
