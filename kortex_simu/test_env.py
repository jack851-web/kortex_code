from openai import OpenAI

client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key='ms-0e545d00-2fe3-433b-9a12-6789ad5233e3', # ModelScope Token
)

# set extra_body for thinking control
extra_body = {
    # enable thinking, set to False to disable test
    "enable_thinking": True,
    # use thinking_budget to contorl num of tokens used for thinking
    # "thinking_budget": 4096
}

response = client.chat.completions.create(
    model='Qwen/Qwen3-8B', # ModelScope Model-Id, required
    messages=[
        {
          'role': 'user',
          'content': '9.9和9.11谁大'
        }
    ],
    stream=True,
    extra_body=extra_body
)
done_thinking = False
for chunk in response:
    if chunk.choices:
        thinking_chunk = chunk.choices[0].delta.reasoning_content
        answer_chunk = chunk.choices[0].delta.content
        if thinking_chunk != '':
            print(thinking_chunk, end='', flush=True)
        elif answer_chunk != '':
            if not done_thinking:
                print('\n\n === Final Answer ===\n')
                done_thinking = True
            print(answer_chunk, end='', flush=True)