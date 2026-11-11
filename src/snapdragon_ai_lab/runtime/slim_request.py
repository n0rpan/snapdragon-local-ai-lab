"""Opt-in research request shaping; never changes production configuration."""
import copy

ALLOWED_TOOLS=frozenset(('read','write','edit','shell'))
SYSTEM='You are an offline coding assistant. Read the task and relevant source/tests, implement the requested repair, run its tests, and report the observed result. Preserve public interfaces and protected files. Use available tools; batch independent file reads when practical.'

def client_config(config):
    value=copy.deepcopy(config)
    value.setdefault('agents',{})['build']={'system':SYSTEM}
    value.setdefault('policies',[]).extend([{'action':'tool.use','resource':'*','effect':'deny'}]+[
        {'action':'tool.use','resource':name+':*','effect':'allow'} for name in ('read','edit','shell')])
    return value

def request(body):
    """Keep complete schemas for four local tools; preserve all messages/limits."""
    if not isinstance(body,dict):raise ValueError('request_object_required')
    value=copy.deepcopy(body)
    tools=value.get('tools',[])
    if not isinstance(tools,list):raise ValueError('tool_list_required')
    if tools:
        selected=[tool for tool in tools if isinstance(tool,dict) and tool.get('type')=='function'
                  and isinstance(tool.get('function'),dict) and tool['function'].get('name') in ALLOWED_TOOLS]
        if not selected:raise ValueError('no_local_tools_survived')
        value['tools']=selected
    choice=value.get('tool_choice')
    if isinstance(choice,dict) and choice.get('function',{}).get('name') not in ALLOWED_TOOLS:
        raise ValueError('required_tool_unavailable')
    return value
