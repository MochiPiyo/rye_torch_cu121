import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



# rye run python .\src\rye_torch_cuda_test\__init__.py

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"
model_name = "elyza/ELYZA-japanese-Llama-2-13b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    use_cache=True,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()

def generate_response(user_input):
    prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
        bos_token=tokenizer.bos_token,
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        prompt=user_input,
        e_inst=E_INST,
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True)
    return output

# ユーザーとの対話
while True:
    user_input = input("ユーザー: ")
    if user_input.lower() == 'exit':
        print("対話を終了します。")
        break
    
    response = generate_response(user_input)
    print("モデル:", response)




"""

# elyza/ELYZA-japanese-Llama-2-13b-instructのサンプルコード

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"
text = "仕事の熱意を取り戻すためのアイデアを5つ挙げてください。"

model_name = "elyza/ELYZA-japanese-Llama-2-13b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    use_cache=True,
    # 
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()

prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)
token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
print(output)
"""