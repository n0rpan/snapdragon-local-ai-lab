"""Build a prefix-only primer; never include user turns or tool results.

Pure preparation code. Importing this module does not enable caching or write
files. The controller integration remains gated on real hybrid restore tests.
"""
from dataclasses import dataclass, field
import hashlib
import json
import re

@dataclass(frozen=True)
class PrefixPlan:
    key: str
    model_sha256: str
    server_sha256: str
    context_size: int
    primer: dict = field(repr=False)

    def metadata(self):
        return {'schema_version':1,'prefix_sha256':self.key,
                'model_sha256':self.model_sha256,'server_sha256':self.server_sha256,
                'context_size':self.context_size,'scope':'leading system messages and tool definitions only'}

def make_plan(payload,profile,min_characters=4096):
    """Return None when a safe, useful prefix cannot be selected exactly."""
    if not isinstance(payload,dict) or not isinstance(profile,dict):return None
    if payload.get('continue_final_message') or payload.get('add_generation_prompt') is False:return None
    messages=payload.get('messages');tools=payload.get('tools',[])
    if not isinstance(messages,list) or not isinstance(tools,list):return None
    systems=[]
    for message in messages:
        if not isinstance(message,dict):return None
        if message.get('role')!='system':break
        if set(message)-{'role','content','name'} or not isinstance(message.get('content'),str):return None
        if 'name' in message and not isinstance(message['name'],str):return None
        systems.append(message)
    if not systems:return None
    # Later system messages can encode conversation state. Do not silently
    # promote them into a persistent system/tool prefix.
    if any(not isinstance(m,dict) or m.get('role') in ('system','developer') for m in messages[len(systems):]):return None
    model_sha=profile.get('sha256');server_sha=profile.get('server_sha256')
    if any(not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None for value in (model_sha,server_sha)):return None
    context=profile.get('context_size')
    if type(context) is not int or context<1:return None
    template_keys=('chat_template_kwargs','chat_template','reasoning_format','add_generation_prompt','parallel_tool_calls')
    template={key:payload[key] for key in template_keys if key in payload}
    identity={'model_sha256':model_sha,'server_sha256':server_sha,'context_size':context,
              'threads':profile.get('threads'),
              'execution_policy_mode':profile.get('execution_policy_mode','stock'),
              'engine_args':profile.get('extra_args',[]),'environment':profile.get('environment',{}),
              'systems':systems,'tools':tools,'template':template}
    try:
        # Preserve dictionary order because template rendering can depend on
        # tool-schema order. Equivalent but reordered inputs may safely miss.
        encoded=json.dumps(identity,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        prefix_size=len(json.dumps({'systems':systems,'tools':tools},ensure_ascii=False,allow_nan=False))
        if not min_characters<=prefix_size<=512000:return None
        copied=json.loads(encoded)
    except (ValueError,TypeError):return None
    alias=profile.get('model_alias',payload.get('model','qwen-local'))
    if not isinstance(alias,str) or not alias:return None
    primer={'model':alias,
            'messages':copied['systems']+[{'role':'user','content':'Initialize this coding prefix. Reply READY. Do not call tools.'}],
            'tools':copied['tools'],'max_tokens':1,'temperature':0,'seed':42,'stream':False,
            'cache_prompt':True,'tool_choice':'none',**copied['template']}
    return PrefixPlan(hashlib.sha256(encoded.encode()).hexdigest(),model_sha,server_sha,context,primer)
