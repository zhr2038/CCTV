"""Authenticated deployment console; no password/key is written to disk/logs."""
from pathlib import Path
import getpass, json, os, shlex, sys, time
import paramiko

host = os.environ.get('CCTV_NAS_HOST') or input('NAS LAN IP / Host: ').strip()
user = os.environ.get('CCTV_NAS_USER') or input('NAS SSH username: ').strip()
password = getpass.getpass(f'NAS SSH password for {user}@{host} (non-echo): ')
client = paramiko.SSHClient(); client.load_system_host_keys(); client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(host, username=user, password=password, timeout=15, look_for_keys=False, allow_agent=False)
sftp=client.open_sftp()

def stream_put(remote, chunks, mode=384):
 if not remote.startswith('/volume2/cctv-growth-service/'):
  raise ValueError('Transfer target outside the authorized service directory')
 command='umask 077; cat > '+shlex.quote(remote)+' && chmod '+format(mode,'o')+' '+shlex.quote(remote)
 stdin,stdout,stderr=client.exec_command(command,timeout=180)
 for block in chunks:stdin.channel.sendall(block)
 stdin.channel.shutdown_write()
 if stdout.channel.recv_exit_status()!=0:raise RuntimeError('SSH upload failed')

print('READY: JSON operations exec/put/put_secret/get/quit',flush=True)
try:
 while True:
  raw=input('nas> ').strip()
  if not raw:continue
  request=json.loads(raw);op=request['op']
  if op=='quit':break
  try:
   if op=='exec':
    command=request['cmd']
    if request.get('sudo'):command='sudo -S -p \'\' sh -c '+shlex.quote(command)
    stdin,stdout,stderr=client.exec_command(command,timeout=request.get('timeout',60))
    if request.get('sudo'):stdin.write(password+'\n');stdin.flush();stdin.channel.shutdown_write()
    # Long-running build output is returned incrementally, without displaying auth.
    channel=stdout.channel;deadline=time.monotonic()+request.get('timeout',60)
    while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
     if channel.recv_ready():print(channel.recv(16384).decode('utf-8',errors='replace'),end='',flush=True)
     if channel.recv_stderr_ready():print(channel.recv_stderr(8192).decode('utf-8',errors='replace'),end='',flush=True)
     if time.monotonic()>deadline:channel.close();raise TimeoutError('Command wait bound reached; inspect remote process before retry')
     time.sleep(.05)
    print(json.dumps({'exit':channel.recv_exit_status()}),flush=True)
   elif op=='put':
    with open(request['local'],'rb') as source:
     stream_put(request['remote'],iter(lambda:source.read(1024*1024),b''),int(request.get('mode',384)))
    print(json.dumps({'uploaded':request['remote']}),flush=True)
   elif op=='put_secret':
    value=getpass.getpass('Secret value (non-echo): ')
    stream_put(request['remote'],[value.encode()],384);del value
    print(json.dumps({'secret_written':request['remote'],'mode':'0600'}),flush=True)
   elif op=='get':
    if not request['remote'].startswith('/volume2/cctv-growth-service/'):raise ValueError('Transfer source outside service directory')
    stdin,stdout,stderr=client.exec_command('cat '+shlex.quote(request['remote']),timeout=60)
    with open(request['local'],'wb') as target:
     for block in iter(lambda:stdout.read(1024*1024),b''):target.write(block)
    if stdout.channel.recv_exit_status()!=0:raise RuntimeError('SSH download failed')
    print(json.dumps({'downloaded':request['local']}),flush=True)
   else:raise ValueError('Unsupported operation')
  except Exception as exc:
   print(json.dumps({'operation_error':type(exc).__name__}),flush=True)
finally:sftp.close();client.close()
