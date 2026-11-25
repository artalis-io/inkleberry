#!/usr/bin/env python3
from tokenizer_fixtures import fixture
import json,sys
f=fixture()
f['model']['vocab']['x'*300]=260
for i,name in enumerate(['message_system','message_user','message_model','content_text','content_thinking','end_message','content_model_end_sampling','content_invoke_tool_json','content_image']):
    f['added_tokens'].append(dict(f['added_tokens'][0],id=310+i,content='<|'+name+'|>'))
open(sys.argv[1],'w').write(json.dumps(f,ensure_ascii=False))
