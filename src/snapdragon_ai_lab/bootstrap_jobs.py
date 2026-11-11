"""Bounded subprocess jobs; used only inside an explicitly verified user unit."""
import os
from pathlib import Path
import selectors
import subprocess
import time
from .bootstrap_guard import Stopped,reap_owned

class Runner:
    def __init__(self,guard,group,directory,log_cap=16*1024**2):
        self.guard=guard;self.group=group;self.directory=Path(directory);self.cap=log_cap;self.logged=0
        self.log=(self.directory/'commands.log').open('xb');self.child=None
        home=self.directory/'isolated-home';home.mkdir(mode=0o700)
        self.env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':str(home),'LANG':'C.UTF-8','LC_ALL':'C.UTF-8',
            'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null','GIT_TERMINAL_PROMPT':'0','GIT_ASKPASS':'/bin/false',
            'GIT_LFS_SKIP_SMUDGE':'1','GIT_NO_LAZY_FETCH':'1','PYTHONDONTWRITEBYTECODE':'1','CMAKE_BUILD_PARALLEL_LEVEL':'1',
            'OMP_NUM_THREADS':'2','OMP_DYNAMIC':'FALSE','OMP_WAIT_POLICY':'PASSIVE'}
    def call(self,command):
        self.guard.require();output=bytearray();selector=selectors.DefaultSelector();eof=False
        self.child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            env=self.env,start_new_session=True,umask=0o077)
        os.set_blocking(self.child.stdout.fileno(),False);selector.register(self.child.stdout,selectors.EVENT_READ)
        try:
            while self.child.poll() is None or not eof:
                self.guard.require()
                for key,_ in selector.select(.1):
                    block=os.read(key.fd,65536)
                    if not block:eof=True;selector.unregister(self.child.stdout);continue
                    if self.logged+len(block)>self.cap:raise Stopped('persistent_log_cap')
                    self.log.write(block);self.log.flush();self.logged+=len(block)
                    output.extend(block[:max(0,65536-len(output))])
            self.child.wait()
            if self.child.returncode:raise Stopped('command_failed_'+Path(command[0]).name+'_'+str(self.child.returncode))
            return output.decode(errors='replace')
        except BaseException:
            reap_owned(self.group);self.child.wait();raise
        finally:
            selector.close();self.child.stdout.close();self.child=None
    def close(self):
        if self.child is not None:
            reap_owned(self.group);self.child.wait()
        self.log.flush();os.fsync(self.log.fileno());self.log.close()
