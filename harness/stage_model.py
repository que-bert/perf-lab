#!/usr/bin/env python3
"""Copy a huge file without letting page cache blow up.

Plain `cp` of the 22.9 GB model filled the cache with both read and write
pages and got the job killed for memory pressure at 12.9 GB on a 30 GB box.
This reads in chunks and, after each one, flushes the written range and tells
the kernel it can drop both the source and destination pages it just touched.
Cache stays roughly flat instead of growing to the size of the file.

Resumes: if the destination is a prefix of the source, it continues from there.
"""
import os
import sys
import time

CHUNK = 64 << 20        # 64 MiB
SYNC_EVERY = 512 << 20  # fdatasync + drop cache each 512 MiB


def human(n):
    return f"{n/1e9:.2f} GB"


def main(src, dst):
    total = os.path.getsize(src)
    start = 0
    if os.path.exists(dst):
        have = os.path.getsize(dst)
        if have == total:
            print("destination already complete")
            return
        # resume on a chunk boundary to avoid trusting a partial tail write
        start = max(0, (have // CHUNK) * CHUNK)
        print(f"resuming at {human(start)} of {human(total)}")

    sfd = os.open(src, os.O_RDONLY)
    dfd = os.open(dst, os.O_WRONLY | os.O_CREAT)
    try:
        os.lseek(sfd, start, os.SEEK_SET)
        os.lseek(dfd, start, os.SEEK_SET)
        os.ftruncate(dfd, start)
        pos = start
        last_sync = start
        t0 = time.time()
        while True:
            buf = os.read(sfd, CHUNK)
            if not buf:
                break
            n = 0
            while n < len(buf):
                n += os.write(dfd, buf[n:])
            pos += len(buf)

            if pos - last_sync >= SYNC_EVERY or len(buf) < CHUNK:
                os.fdatasync(dfd)
                # both ranges are done with; let the kernel reclaim them
                os.posix_fadvise(dfd, last_sync, pos - last_sync,
                                 os.POSIX_FADV_DONTNEED)
                os.posix_fadvise(sfd, last_sync, pos - last_sync,
                                 os.POSIX_FADV_DONTNEED)
                last_sync = pos
                el = time.time() - t0
                rate = (pos - start) / el / 1e6 if el else 0
                print(f"  {human(pos)} / {human(total)}  {rate:.0f} MB/s",
                      flush=True)
        os.fdatasync(dfd)
        os.posix_fadvise(dfd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(sfd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(sfd)
        os.close(dfd)

    got = os.path.getsize(dst)
    print(f"done: {human(got)}" + ("" if got == total else "  SIZE MISMATCH"))
    sys.exit(0 if got == total else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
