"""Synthetic smoke/chat/tool inputs reused from the existing Arete comparison."""
import hashlib,json,time
from pathlib import Path

# Format keys/types only. Expected item/value are never supplied to the schema.
FINAL_RESPONSE_FORMAT={'type':'json_schema','json_schema':{'name':'stock_result','strict':True,'schema':{'type':'object','properties':{'item_id':{'type':'string'},'stock':{'type':'integer'}},'required':['item_id','stock'],'additionalProperties':False}}}

def functional(record,request,directory,stage):
 record['functional']=[]
 def call(name,messages,tokens=128,tools=None,expect_tools=False):
  body={'model':'classic-quota-screen','messages':messages,'max_tokens':tokens,'temperature':0,'seed':42,'stream':False,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False}}
  if tools:body.update(tools=tools,tool_choice='auto' if expect_tools else 'none')
  if name=='tool-result-final':body['response_format']=FINAL_RESPONSE_FORMAT
  began=time.monotonic();result=request('/v1/chat/completions',body,timeout=120);data=json.dumps(result,sort_keys=True,ensure_ascii=False).encode()
  if len(data)>2*1024**2:raise ValueError('functional_response_bound')
  path=directory/(name+'-own-synthetic-response.json');path.write_bytes(data);path.chmod(0o600)
  row={'id':name,'passed':False,'request_seconds':time.monotonic()-began,'response_sha256':hashlib.sha256(data).hexdigest(),'timings':result.get('timings'),'usage':result.get('usage'),'response_format':body.get('response_format')};record['functional'].append(row)
  if len(result.get('choices',[]))!=1:raise ValueError('single_choice_required')
  choice=result['choices'][0];message=choice.get('message',{});row['finish_reason']=choice.get('finish_reason')
  if choice.get('finish_reason')!=('tool_calls' if expect_tools else 'stop'):raise ValueError('complete_functional_response_required')
  if not expect_tools and (not isinstance(message.get('content'),str) or not message['content'].strip() or message.get('tool_calls')):raise ValueError('complete_final_text_required')
  return message,row
 msg,row=call('ready',[{'role':'user','content':'Reply with the single word READY, without punctuation, quotation marks, or extra text.'}],32);row['passed']=(msg.get('content') or '').strip()=='READY' and not msg.get('tool_calls')
 if not row['passed']:raise ValueError('ready_answer_required')
 if stage=='smoke':return
 messages=[{'role':'user','content':'A synthetic record says Kestrel has 12 clips and Mora has 7. Output only JSON with keys Kestrel and Mora and their integer counts.'}]
 msg,row=call('chat-record',messages,64);row['passed']=json.loads(msg.get('content') or '')=={'Kestrel':12,'Mora':7}
 messages.extend([msg,{'role':'user','content':'Move three clips from Kestrel to Mora. Output only JSON with keys Kestrel and Mora and their new integer counts.'}])
 msg,row=call('chat-context-json',messages,64);row['passed']=json.loads(msg.get('content') or '')=={'Kestrel':9,'Mora':10}
 tools=[{'type':'function','function':{'name':'lookup_stock','description':'Look up the count for a synthetic item.','parameters':{'type':'object','properties':{'item_id':{'type':'string'}},'required':['item_id'],'additionalProperties':False}}}]
 messages=[{'role':'system','content':'For inventory questions call lookup_stock. Never guess the count. After its result, output only JSON with item_id and stock.'},{'role':'user','content':'How many units of synthetic_adapter are in stock?'}]
 msg,row=call('structured-tool-call',messages,128,tools,True);calls=msg.get('tool_calls',[])
 if len(calls)!=1:raise ValueError('one_tool_call_required')
 item=calls[0];fn=item.get('function',{});row['passed']=fn.get('name')=='lookup_stock' and json.loads(fn.get('arguments','{}'))=={'item_id':'synthetic_adapter'} and isinstance(item.get('id'),str) and bool(item['id'])
 if not row['passed']:raise ValueError('tool_arguments_failed')
 messages.extend([msg,{'role':'tool','tool_call_id':item['id'],'content':json.dumps({'item_id':'synthetic_adapter','stock':17})}])
 msg,row=call('tool-result-final',messages,96,tools,False);row['passed']=json.loads(msg.get('content') or '')=={'item_id':'synthetic_adapter','stock':17}
 if not all(row['passed'] for row in record['functional']):raise ValueError('functional_oracle_failed')
