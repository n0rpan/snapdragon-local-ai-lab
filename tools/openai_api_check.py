#!/usr/bin/env python3
"""OpenAI-compatibility checks against a local base URL, and the prefix-cache measurement
for an unknown harness. Model-free when pointed at `oxe-server --engine stub` or at the
controller in front of it; it also runs against the live controller (it then loads a model).

    openai_api_check.py audit --base-url http://127.0.0.1:8092 --model local-default [--markdown]
    openai_api_check.py sdk   --base-url http://127.0.0.1:8092 --model local-default   # needs the `openai` package
    openai_api_check.py cache --server <oxe-server> --gguf <model.gguf>                # stub engine, started here

`audit` sends the requests the official `openai` Python client sends (same paths, headers
and bodies) with urllib, so it needs nothing installed. Verdicts: supported, partial
(accepted, but not what the API promises), missing (an error).
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

CLIENT_HEADERS = {'User-Agent': 'OpenAI/Python 1.99.0', 'Accept': 'application/json',
                  'X-Stainless-Lang': 'python', 'X-Stainless-Package-Version': '1.99.0'}
WEATHER = {'type': 'function', 'function': {
    'name': 'get_weather', 'description': 'Weather for a city.',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}


def call(base, method, path, body=None, key='local', headers=None, timeout=120):
    """(status, headers, parsed JSON or raw text). Never raises on HTTP errors."""
    send = dict(CLIENT_HEADERS)
    if key is not None:
        send['Authorization'] = 'Bearer ' + key
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        send['Content-Type'] = 'application/json'
    send.update(headers or {})
    request = urllib.request.Request(base.rstrip('/') + path, data=data, method=method, headers=send)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, reply_headers, raw = response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as error:
        with error:
            status, reply_headers, raw = error.code, dict(error.headers.items()), error.read()
    except (urllib.error.URLError, OSError) as error:
        return 0, {}, str(error)
    text = raw.decode('utf-8', 'replace')
    try:
        return status, reply_headers, json.loads(text)
    except ValueError:
        return status, reply_headers, text


def sse_events(text):
    out = []
    for block in text.split('\n\n') if isinstance(text, str) else []:
        block = block.strip()
        if block.startswith('data:'):
            data = block[5:].strip()
            try:
                out.append(data if data == '[DONE]' else json.loads(data))
            except ValueError:
                out.append({'unparsed': data})
    return out


def openai_error(value):
    error = value.get('error') if isinstance(value, dict) else None
    return isinstance(error, dict) and isinstance(error.get('message'), str) and isinstance(error.get('type'), str)


def audit(base, model, key='local'):
    """[(endpoint, feature, verdict, detail)]."""
    rows = []

    def add(endpoint, feature, verdict, detail=''):
        rows.append((endpoint, feature, verdict, detail))

    def chat(extra=None, messages=None, **kw):
        body = {'model': model, 'messages': messages or [{'role': 'user', 'content': 'Say hello.'}], **(extra or {})}
        return call(base, 'POST', '/v1/chat/completions', body, key, **kw)

    def accepted(endpoint, feature, extra=None, messages=None, note='accepted (effect not checked without a model)'):
        status, _, value = chat(extra, messages)
        ok = status == 200 and isinstance(value, dict) and value.get('choices')
        add(endpoint, feature, 'supported' if ok else 'missing', note if ok else 'HTTP %s' % status)
        return status, value

    # ---- models
    status, _, value = call(base, 'GET', '/v1/models', key=key)
    ids = [row.get('id') for row in value.get('data', [])] if isinstance(value, dict) else []
    add('/v1/models', 'list', 'supported' if status == 200 and value.get('object') == 'list' and ids else 'missing',
        ', '.join(map(str, ids)) or 'HTTP %s' % status)
    status, _, value = call(base, 'GET', '/v1/models/' + model, key=key)
    add('/v1/models/{id}', 'retrieve', 'supported' if status == 200 and isinstance(value, dict) and value.get('id') == model
        else 'missing', 'HTTP %s' % status)

    # ---- chat, not streamed
    C = '/v1/chat/completions'
    status, _, value = chat()
    text = ''
    ok = status == 200 and isinstance(value, dict) and value.get('object') == 'chat.completion' and value.get('choices')
    if ok:
        message = value['choices'][0].get('message', {})
        text = message.get('content') or ''
        ok = message.get('role') == 'assistant' and isinstance(value.get('id'), str) and 'finish_reason' in value['choices'][0]
    add(C, 'non-streaming', 'supported' if ok else 'missing', 'model=%s' % value.get('model') if ok else 'HTTP %s' % status)
    usage = value.get('usage', {}) if isinstance(value, dict) else {}
    add(C, 'usage block', 'supported' if all(type(usage.get(k)) is int for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
        else 'missing', 'cached_tokens reported' if isinstance(usage.get('prompt_tokens_details'), dict) else '')

    # ---- chat, streamed
    status, headers, raw = chat({'stream': True})
    events = sse_events(raw)
    chunks = [e for e in events if isinstance(e, dict)]
    ok = (status == 200 and 'text/event-stream' in headers.get('Content-Type', headers.get('content-type', ''))
          and events and events[-1] == '[DONE]' and chunks and all(c.get('object') == 'chat.completion.chunk' for c in chunks)
          and chunks[0]['choices'][0]['delta'].get('role') == 'assistant'
          and any(c['choices'] and c['choices'][0].get('finish_reason') for c in chunks))
    add(C, 'streaming (SSE, [DONE])', 'supported' if ok else 'missing', '%d chunks' % len(chunks))
    add(C, 'stream without include_usage has no usage chunk', 'supported' if not any('usage' in c for c in chunks) else 'partial')
    status, _, raw = chat({'stream': True, 'stream_options': {'include_usage': True}})
    chunks = [e for e in sse_events(raw) if isinstance(e, dict)]
    last = chunks[-1] if chunks else {}
    add(C, 'stream_options.include_usage', 'supported' if last.get('choices') == [] and isinstance(last.get('usage'), dict)
        else 'missing', 'final chunk: choices [] + usage' if last.get('choices') == [] else '')

    # ---- limits and sampling
    for field in ('max_tokens', 'max_completion_tokens'):
        status, _, value = chat({field: 2})
        ok = status == 200 and value['usage']['completion_tokens'] <= 2 and value['choices'][0]['finish_reason'] == 'length'
        add(C, field, 'supported' if ok else 'missing', 'finish_reason length at 2 tokens' if ok else 'HTTP %s' % status)
    words = text.split()
    if len(words) >= 3:
        status, _, value = chat({'stop': [words[2]]})
        got = value['choices'][0]['message'].get('content') or '' if status == 200 else ''
        ok = status == 200 and words[2] not in got and value['choices'][0]['finish_reason'] == 'stop'
        add(C, 'stop', 'supported' if ok else 'partial', 'cut before %r' % words[2] if ok else 'not applied')
    else:
        accepted(C, 'stop', {'stop': ['###']})
    accepted(C, 'temperature / top_p', {'temperature': 0.7, 'top_p': 0.9})
    accepted(C, 'seed', {'seed': 7})
    accepted(C, 'frequency_penalty / presence_penalty / logit_bias / user',
             {'frequency_penalty': 0.5, 'presence_penalty': 0.5, 'logit_bias': {}, 'user': 'u'},
             note='accepted; penalties and logit_bias are ignored by the engine')
    if rows[-1][2] == 'supported':
        rows[-1] = (*rows[-1][:2], 'partial', rows[-1][3])
    status, _, value = chat({'n': 2})
    count = len(value.get('choices', [])) if status == 200 and isinstance(value, dict) else 0
    add(C, 'n > 1', 'supported' if count == 2 else 'partial' if count == 1 else 'missing',
        'one choice returned' if count == 1 else 'HTTP %s' % status)
    status, _, value = chat({'logprobs': True, 'top_logprobs': 2})
    content = (value['choices'][0].get('logprobs') or {}).get('content') if status == 200 else None
    ok = isinstance(content, list) and content and 'logprob' in content[0] and len(content[0].get('top_logprobs', [])) == 2
    add(C, 'logprobs / top_logprobs', 'supported' if ok else 'missing', '' if ok else 'HTTP %s' % status)

    # ---- tools
    status, value = accepted(C, 'tools + tool_choice auto', {'tools': [WEATHER], 'tool_choice': 'auto'},
                             note='accepted')
    if status == 200:
        calls = value['choices'][0]['message'].get('tool_calls')
        if calls:
            ok = (calls[0].get('type') == 'function' and isinstance(calls[0]['function'].get('arguments'), str)
                  and value['choices'][0]['finish_reason'] == 'tool_calls')
            rows[-1] = (C, 'tools + tool_choice auto', 'supported' if ok else 'partial',
                        'tool call parsed: %s(%s)' % (calls[0]['function'].get('name'), calls[0]['function'].get('arguments')))
            status, _, raw = chat({'tools': [WEATHER], 'stream': True})
            deltas = [c['choices'][0]['delta'] for c in sse_events(raw) if isinstance(c, dict) and c.get('choices')]
            streamed = [d['tool_calls'][0] for d in deltas if d.get('tool_calls')]
            add(C, 'tools, streamed tool_calls deltas', 'supported' if streamed and streamed[0].get('index') == 0 else 'missing',
                '%d deltas' % len(streamed))
    accepted(C, 'tool_choice required / none', {'tools': [WEATHER], 'tool_choice': 'required'}, note='accepted')
    accepted(C, 'tool_choice {function: name}', {'tools': [WEATHER], 'tool_choice': {'type': 'function', 'function': {'name': 'get_weather'}}},
             note='accepted')
    accepted(C, 'parallel_tool_calls', {'tools': [WEATHER], 'parallel_tool_calls': False}, note='accepted')
    accepted(C, 'tool result history (assistant tool_calls + role tool)', {'tools': [WEATHER]}, messages=[
        {'role': 'user', 'content': 'Weather in Oslo?'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {
            'name': 'get_weather', 'arguments': '{"city": "Oslo"}'}}]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': '4 C, rain'}], note='accepted')

    # ---- message shapes
    accepted(C, 'system + developer roles', messages=[{'role': 'developer', 'content': 'Be brief.'},
                                                    {'role': 'user', 'content': 'Hi'}], note='accepted')
    accepted(C, 'content as text parts', messages=[{'role': 'user', 'content': [{'type': 'text', 'text': 'Hi'}]}], note='accepted')
    status, _, value = chat(messages=[{'role': 'user', 'content': [
        {'type': 'text', 'text': 'What is this?'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]}])
    add(C, 'image input', 'partial' if status == 200 else 'missing',
        'accepted; the image is replaced by a text note (text-only models)' if status == 200 else 'HTTP %s' % status)
    accepted(C, 'system message after the first turn', messages=[
        {'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello'},
        {'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Again'}], note='accepted')

    # ---- structured output
    for name, fmt in (('json_object', {'type': 'json_object'}),
                      ('json_schema', {'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': {'type': 'object'}}})):
        status, _, value = chat({'response_format': fmt})
        add(C, 'response_format ' + name, 'supported' if status == 200 else 'missing',
            'HTTP %s%s' % (status, ', OpenAI error shape' if openai_error(value) else ''))
    accepted(C, 'response_format text', {'response_format': {'type': 'text'}}, note='accepted')

    # ---- legacy completions
    P = '/v1/completions'
    status, _, value = call(base, 'POST', P, {'model': model, 'prompt': 'Hello', 'max_tokens': 3}, key)
    ok = (status == 200 and isinstance(value, dict) and value.get('object') == 'text_completion'
          and isinstance(value['choices'][0].get('text'), str) and 'finish_reason' in value['choices'][0]
          and type(value.get('usage', {}).get('total_tokens')) is int)
    add(P, 'non-streaming', 'supported' if ok else 'missing', '' if ok else 'HTTP %s' % status)
    status, _, raw = call(base, 'POST', P, {'model': model, 'prompt': 'Hello', 'max_tokens': 3, 'stream': True}, key)
    events = sse_events(raw)
    chunks = [e for e in events if isinstance(e, dict)]
    ok = (status == 200 and events and events[-1] == '[DONE]' and chunks
          and all(c.get('object') == 'text_completion' and 'text' in c['choices'][0] for c in chunks)
          and chunks[-1]['choices'][0].get('finish_reason'))
    add(P, 'streaming (SSE, [DONE])', 'supported' if ok else 'missing', '%d chunks' % len(chunks) if ok else 'HTTP %s' % status)
    status, _, value = call(base, 'POST', P, {'model': model, 'prompt': 'Hello', 'max_tokens': 3, 'logprobs': 2}, key)
    logprobs = value['choices'][0].get('logprobs') if status == 200 and isinstance(value, dict) and value.get('choices') else None
    add(P, 'logprobs', 'supported' if isinstance(logprobs, dict) and logprobs.get('tokens') else 'missing')
    status, _, value = call(base, 'POST', P, {'model': model, 'prompt': 'a', 'suffix': 'b', 'max_tokens': 3}, key)
    add(P, 'suffix (insert)', 'supported' if status == 200 else 'missing', 'HTTP %s' % status)

    # ---- other endpoints
    for path, body in (('/v1/embeddings', {'model': model, 'input': 'hello'}),
                       ('/v1/responses', {'model': model, 'input': 'hello'})):
        status, _, value = call(base, 'POST', path, body, key)
        add(path, 'any', 'supported' if status == 200 else 'missing',
            'HTTP %s%s' % (status, ', OpenAI error shape' if openai_error(value) else ''))

    # ---- errors, auth, CORS
    status, _, value = chat({'model': 'no-such-model'})
    add('errors', 'unknown model', 'supported' if status == 404 and openai_error(value) and value['error'].get('code') == 'model_not_found'
        else 'partial' if status == 200 else 'missing',
        'HTTP %s%s' % (status, ', served by the loaded model' if status == 200 else ''))
    request = urllib.request.Request(base.rstrip('/') + C, data=b'{not json', method='POST',
                                     headers={**CLIENT_HEADERS, 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + (key or 'x')})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status, value = response.status, {}
    except urllib.error.HTTPError as error:
        with error:
            status = error.code
            try: value = json.loads(error.read())
            except ValueError: value = {}
    add('errors', 'malformed JSON body', 'supported' if status == 400 and openai_error(value) else 'partial', 'HTTP %s' % status)
    status, _, value = call(base, 'GET', '/v1/models', key=None)
    wrong, _, wrong_value = call(base, 'GET', '/v1/models', key='wrong-key')
    if status == 200 and wrong == 200:
        add('auth', 'API key', 'supported', 'no key configured: any or no Bearer token is accepted')
    else:
        add('auth', 'API key', 'supported' if status == 401 and wrong == 401 and openai_error(wrong_value) else 'partial',
            'key required: HTTP %s without, %s with a wrong key' % (status, wrong))
    status, headers, _ = call(base, 'OPTIONS', C, key=None, headers={
        'Origin': 'http://app.example.test', 'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'authorization, content-type'})
    allow = headers.get('Access-Control-Allow-Origin', headers.get('access-control-allow-origin'))
    add('CORS', 'preflight from another origin', 'supported' if status in (200, 204) and allow else 'missing',
        'allowed for %s' % allow if allow else 'HTTP %s (refused unless the origin is configured)' % status)
    return rows


def markdown(rows):
    out = ['| Endpoint | Feature | Status | Detail |', '|---|---|---|---|']
    out += ['| `%s` | %s | %s | %s |' % (e, f, v, str(d).replace('|', '/')) for e, f, v, d in rows]
    return '\n'.join(out)


def cmd_audit(args):
    rows = audit(args.base_url, args.model, args.api_key)
    if args.json:
        print(json.dumps([dict(zip(('endpoint', 'feature', 'status', 'detail'), row)) for row in rows], indent=1))
    else:
        print(markdown(rows) if args.markdown else '\n'.join('%-22s %-58s %-9s %s' % row for row in rows))
    return 0


def cmd_sdk(args):
    """The same surface through the official client (`pip install openai` in a venv)."""
    try:
        import openai
    except ImportError:
        print('the openai package is not installed; `audit` covers the same requests without it')
        return 2
    client = openai.OpenAI(base_url=args.base_url.rstrip('/') + '/v1', api_key=args.api_key or 'local', max_retries=0)
    done = []
    ids = [m.id for m in client.models.list().data]
    done.append(('models.list', ids))
    done.append(('models.retrieve', client.models.retrieve(args.model).id))
    reply = client.chat.completions.create(model=args.model, messages=[{'role': 'user', 'content': 'Say hello.'}], max_tokens=8)
    done.append(('chat.completions.create', (reply.choices[0].message.content, reply.choices[0].finish_reason, reply.usage.total_tokens)))
    text, usage = '', None
    for chunk in client.chat.completions.create(model=args.model, messages=[{'role': 'user', 'content': 'Say hello.'}], max_tokens=8,
                                                stream=True, stream_options={'include_usage': True}):
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
        usage = chunk.usage or usage
    done.append(('chat.completions.create(stream=True)', (text, usage.total_tokens if usage else None)))
    reply = client.chat.completions.create(model=args.model, messages=[{'role': 'user', 'content': 'Weather in Oslo?'}],
                                           tools=[WEATHER], max_tokens=64)
    calls = reply.choices[0].message.tool_calls
    done.append(('chat.completions.create(tools=...)', [(c.function.name, c.function.arguments) for c in calls] if calls
                 else reply.choices[0].message.content))
    reply = client.completions.create(model=args.model, prompt='Hello', max_tokens=4)
    done.append(('completions.create', (reply.choices[0].text, reply.choices[0].finish_reason)))
    text = ''.join(chunk.choices[0].text for chunk in client.completions.create(model=args.model, prompt='Hello', max_tokens=4, stream=True)
                   if chunk.choices)
    done.append(('completions.create(stream=True)', text))
    for label, expected, run in (
            ('unknown model', openai.NotFoundError, lambda: client.chat.completions.create(model='no-such-model', messages=[{'role': 'user', 'content': 'x'}])),
            ('response_format json_object', openai.BadRequestError, lambda: client.chat.completions.create(
                model=args.model, messages=[{'role': 'user', 'content': 'x'}], response_format={'type': 'json_object'})),
            ('embeddings.create', openai.APIStatusError, lambda: client.embeddings.create(model=args.model, input='hello'))):
        try:
            run()
            done.append((label, 'no error'))
        except expected as error:
            done.append((label, '%s %s: %s' % (type(error).__name__, getattr(error, 'status_code', ''), str(error)[:90])))
    print('openai %s' % openai.__version__)
    for label, value in done:
        print('%-40s %s' % (label, value))
    return 0


# ---------------------------------------------------------------- prefix-cache measurement
def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class StubEngine:
    def __init__(self, server, gguf, store, min_tokens=1024):
        self.args = [server, '--engine', 'stub', '--model', gguf, '--alias', 'stub-local', '--ctx-size', '32768',
                     '--reasoning', 'off', '--prefix-cache', store, '--prefix-cache-min-tokens', str(min_tokens)]
        self.process = None
        self.port = free_port()  # kept across restarts
        self.base = 'http://127.0.0.1:%d' % self.port

    def start(self):
        self.process = subprocess.Popen(self.args + ['--port', str(self.port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(200):
            if call(self.base, 'GET', '/health', timeout=2)[0] == 200:
                return self
            time.sleep(0.1)
        raise RuntimeError('stub engine did not start')

    def stop(self):
        if self.process is not None:
            self.process.terminate()  # SIGTERM: the engine writes its stores
            self.process.wait(timeout=30)
            self.process = None

    def chat(self, body):
        status, _, value = call(self.base, 'POST', '/v1/chat/completions',
                                {**body, 'temperature': 0, 'cache_prompt': True, 'max_tokens': 8})
        if status != 200:
            raise RuntimeError('engine answered %s: %s' % (status, value))
        return value['timings']['cache_n'], value['timings']['prompt_n'], value['choices'][0]['message']


def synthetic_harness(name, tools=18, system_words=900):
    """A system prompt and tool list of an agent harness nobody registered (sizes like
    Cline/Continue agent mode: a few thousand tokens)."""
    words = ('workspace file edit search terminal command result diff patch project module function test error '
             'review plan step tool output path line context change apply verify').split()
    system = 'You are %s, a coding agent working in the user\'s repository.\n' % name + ' '.join(
        words[(i * 7 + len(name)) % len(words)] for i in range(system_words))
    tool_list = [{'type': 'function', 'function': {
        'name': '%s_tool_%02d' % (name.lower(), i),
        'description': 'Tool %d of %s: ' % (i, name) + ' '.join(words[(i + j) % len(words)] for j in range(40)),
        'parameters': {'type': 'object', 'properties': {
            'path': {'type': 'string', 'description': 'File path relative to the workspace root.'},
            'content': {'type': 'string', 'description': 'Text to write or search for.'},
            'limit': {'type': 'integer', 'description': 'Maximum number of results.'}}, 'required': ['path']}}}
        for i in range(tools)]
    return system, tool_list


def cmd_cache(args):
    """What the engine's prefix store gives a harness the controller does not know, with no
    learning: token counts (reused / evaluated) per request, through engine restarts."""
    rows = []
    with tempfile.TemporaryDirectory() as store:
        engine = StubEngine(args.server, args.gguf, store, args.min_tokens).start()
        try:
            def session(harness, question, label, turns=3):
                system, tools = harness
                messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': question}]
                for turn in range(turns):
                    cache_n, prompt_n, message = engine.chat({'messages': messages, 'tools': tools})
                    rows.append((label if turn == 0 else '  same session, request %d' % (turn + 1), cache_n, prompt_n))
                    messages = messages + [{'role': 'assistant', 'content': message.get('content') or ''},
                                           {'role': 'user', 'content': 'Continue with step %d of the plan, please.' % (turn + 2)}]

            alpha, beta = synthetic_harness('Alpha'), synthetic_harness('Beta', tools=9, system_words=500)
            session(alpha, 'Fix the failing test in the parser module.', 'A1 first request ever (cold)')
            session(alpha, 'Add a --verbose flag to the CLI.', 'A2 new session, engine still loaded', turns=1)
            session(beta, 'Summarise the README.', 'B1 other unknown harness, first request', turns=1)
            session(alpha, 'Rename the config loader.', 'A3 back to the first harness after the switch', turns=1)
            engine.stop(); engine.start()
            session(alpha, 'Write a changelog entry.', 'A4 new session after engine restart (idle unload)', turns=2)
            session(beta, 'List the public functions.', 'B2 other harness after restart', turns=1)
            system, tools = alpha
            session((system + '\nCurrent time: 21:5%d.' % 1, tools), 'What changed?', 'A5 system prompt with a changing last line', turns=1)
            session((('Session 8841. ' + system), tools), 'What changed?', 'A6 system prompt with a changing FIRST line', turns=1)
        finally:
            engine.stop()
    result = [{'request': label, 'reused_tokens': cache_n, 'evaluated_tokens': prompt_n,
               'reused_share': round(cache_n / (cache_n + prompt_n), 3)} for label, cache_n, prompt_n in rows]
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print('| Request | Reused tokens | Evaluated tokens | Reused |\n|---|---|---|---|')
        for row in result:
            print('| %s | %d | %d | %.0f%% |' % (row['request'], row['reused_tokens'], row['evaluated_tokens'], 100 * row['reused_share']))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, func in (('audit', cmd_audit), ('sdk', cmd_sdk)):
        p = sub.add_parser(name)
        p.add_argument('--base-url', required=True, help='without /v1')
        p.add_argument('--model', default='local-default')
        p.add_argument('--api-key', default=os.environ.get('LOCAL_AI_API_KEY') or 'local')
        p.add_argument('--markdown', action='store_true')
        p.add_argument('--json', action='store_true')
        p.set_defaults(func=func)
    p = sub.add_parser('cache')
    p.add_argument('--server', required=True, help='oxe-server binary')
    p.add_argument('--gguf', required=True, help='model file (only its tokenizer and template are read)')
    p.add_argument('--min-tokens', type=int, default=1024)
    p.add_argument('--json', action='store_true')
    p.set_defaults(func=cmd_cache)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
