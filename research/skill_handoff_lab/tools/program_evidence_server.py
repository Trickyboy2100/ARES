"""Loopback-only evidence preview with byte ranges for frame-accurate video seek."""
import argparse,mimetypes,re
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote,urlsplit

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=8795);a=p.parse_args();root=a.root.resolve()
 class Handler(BaseHTTPRequestHandler):
  def do_HEAD(self):self.serve(False)
  def do_GET(self):self.serve(True)
  def serve(self,body):
   path=(root/unquote(urlsplit(self.path).path).lstrip('/')).resolve()
   if not path.is_relative_to(root):self.send_error(403);return
   if path.is_dir():path=path/'REVIEW.html'
   if not path.is_file():self.send_error(404);return
   size=path.stat().st_size;start=0;end=size-1;partial=False;header=self.headers.get('Range')
   if header:
    match=re.fullmatch(r'bytes=(\d*)-(\d*)',header)
    if not match or not any(match.groups()):self.send_error(416);return
    lo,hi=match.groups();start=int(lo) if lo else max(0,size-int(hi));end=min(size-1,int(hi)) if lo and hi else size-1;partial=True
    if start>end or start>=size:self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
   self.send_response(206 if partial else 200);self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream');self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store')
   if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
   self.end_headers()
   if body:
    try:
     with path.open('rb') as f:
      f.seek(start);remaining=end-start+1
      while remaining:
       chunk=f.read(min(1024*1024,remaining))
       if not chunk:break
       self.wfile.write(chunk);remaining-=len(chunk)
    except (BrokenPipeError,ConnectionResetError):pass
 server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);print(f'http://127.0.0.1:{server.server_port}/REVIEW.html',flush=True);server.serve_forever()
if __name__=='__main__':main()
