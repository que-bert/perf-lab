import json,sys,urllib.request
port,out=int(sys.argv[1]),sys.argv[2]
data=open('/home/bbuckham/git/perf-lab/harness/corpus/decode_kld.txt','rb').read()
res=[]
for i in range(16):
    off=i*90*1024
    p=data[off:off+24*1024].decode('utf-8','replace')
    b=json.dumps({"prompt":p,"n_predict":256,"temperature":0,"cache_prompt":False}).encode()
    d=json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion",data=b,headers={"Content-Type":"application/json"}),timeout=3600))
    t=d["timings"]; res.append((t.get("draft_n_accepted"),t.get("draft_n"),t.get("predicted_per_second")))
    print(i,res[-1],flush=True)
ok=[r for r in res if r[0] is not None and r[1]]; a=sum(r[0] for r in ok); n=sum(r[1] for r in ok)
print("POOLED",a,n,round(a/n,4))
json.dump(res,open(out,"w"))
