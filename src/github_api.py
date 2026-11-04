"""GitHub REST helper. Credentials only read from /run/secrets, never logged."""
import json,sys
from urllib.request import Request,urlopen
from urllib.error import HTTPError
BASE='https://api.github.com'
with open('/run/secrets/github-token') as f:TOKEN=f.read().strip()
def api(method,path,body=None):
 d=None if body is None else json.dumps(body).encode()
 req=Request(BASE+path,data=d,headers={'Authorization':'Bearer '+TOKEN,'Accept':'application/vnd.github+json', 'X-GitHub-Api-Version':'2022-11-28','User-Agent':'manual-mnist-exp'},method=method)
 try:
  with urlopen(req,timeout=30) as r:
   return json.load(r)
 except HTTPError as e:
  print('GitHub API HTTP',e.code, e.read().decode()[:700],file=sys.stderr);raise
if __name__=='__main__':
 action=sys.argv[1]
 repo='lovingthenhk-cpu/mnist-direct-weight-edit-experiments'
 if action=='repo':
  print(api('POST','/user/repos',{'name':repo.split('/')[1], 'private':False,'description':'MNIST 10-way: direct LLM edits of explicit weights and biases; each experiment a merged PR', 'auto_init':False})['html_url'])
 elif action=='pr':
  title=sys.argv[2];head=sys.argv[3];body=open(sys.argv[4]).read()
  d=api('POST',f'/repos/{repo}/pulls',{'title':title,'head':head,'base':'main','body':body,'draft':False})
  print(d['number'],d['html_url'])
 elif action=='merge':
  n=sys.argv[2]
  d=api('PUT',f'/repos/{repo}/pulls/{n}/merge',{'merge_method':'merge','commit_title':'Merge experiment PR #'+n})
  print('merged:',d.get('merged'),d.get('sha'))
 elif action=='check':
  n=sys.argv[2];p=api('GET',f'/repos/{repo}/pulls/{n}')
  print('PR',n,'status',p['state'],'mergeable',p.get('mergeable'),'merged',p.get('merged'))
