import ft_config
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

import os
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.0/lib64:/usr/lib/x86_64-linux-gnu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

print("===== ft_config =====")
print(ft_config)
print(ft_config.base_model)
model_id=ft_config.base_model

tokenizer = LlamaTokenizer.from_pretrained(model_id)

# model =LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
model =LlamaForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)


from pathlib import Path
import os
import sys
from utils.dataset_utils import get_preprocessed_dataset
from configs.datasets import samsum_dataset, receipt_dataset

train_dataset = get_preprocessed_dataset(tokenizer, receipt_dataset, 'train')

eval_prompt = """
以下のテキスト一覧は、pdfの請求書ドキュメントからOCRをした結果を左上から順番に並べたものです。テキストから次の項目一覧の値をJson形式で出力してください。存在しない項目に関しては出力しないでください。
### 項目一覧
[請求時分],[消費税額(8%)],[消費税額(10%)],[ページ番号],[支払者名],[請求者FAX],[支払通貨],[請求年月],[合計請求額(税抜)],[口座名義],[口座番号],[口座の種類],[銀行支店名],[銀行名],[支払期日],[消費税額],[合計請求額(税込)],[支払者会社名],[請求者電話番号],[請求者住所],[請求者会社名],[タイトル],[請求番号],[請求日付],[請求額(8%税込)],[請求額(10%税込)],[登録番号]
### テキスト一覧
2023年 6月23日\t令和3年8月分\t前田道路株式会社 御中
下記の通り請求致します
請 求\t書\t(材料その他用)
(業 者 控)
適格請求書株式会社
住\t所
名
発行者登録番号\tT1231231231235
※支払期限\t2022/7/31
みずほ\t銀行 東京\t支店\t普通\t当座\t1234567
⑪
(取引先コード欄)
金額
¥70,200
月日\t品\t名\t納入場所\t工事 №.\t数量\t単位\t単価\t金\t額\t担当
6/23\t*品名1\t三田倉庫\t001\t1.0\t個\t65,000\t65,000
小\t計\t¥65,000
10%
消費税計\t消費税率\t8%\t¥5,200
請求\t金\t額\t¥70,200
(注)1.毎月末日締切で、翌月2日迄に必着するよう提出して下さい。
2.提出用のシートを2枚印刷して、提出してください。
3.取引先コード欄に貴社コードのゴム印を押印または、貴社コードを入力してください。

### Json Output:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")


model.train()

def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"]
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

# create peft config
model, lora_config = create_peft_config(model)

from transformers import TrainerCallback
from contextlib import nullcontext
enable_profiler = False
output_dir = ft_config.output_dir

config = {
    'lora_config': lora_config,
    'learning_rate': 1e-4,
    'num_train_epochs': 40,
    'gradient_accumulation_steps': 2,
    'per_device_train_batch_size': 2,
    'gradient_checkpointing': False,
}

# Set up profiler
if enable_profiler:
    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
    
    class ProfilerCallback(TrainerCallback):
        def __init__(self, profiler):
            self.profiler = profiler
            
        def on_step_end(self, *args, **kwargs):
            self.profiler.step()

    profiler_callback = ProfilerCallback(profiler)
else:
    profiler = nullcontext()



from transformers import default_data_collator, Trainer, TrainingArguments



# Define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    bf16=True,  # Use BF16 if available
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch_fused",
    max_steps=total_steps if enable_profiler else -1,
    **{k:v for k,v in config.items() if k != 'lora_config'}
)

with profiler:
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        callbacks=[profiler_callback] if enable_profiler else [],
    )

    # Star t training
    trainer.train()


model.save_pretrained(output_dir)

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=1000)[0], skip_special_tokens=True))



from evaluate.evaluate_json import evaluate
eval_df = evaluate(model, tokenier, receipt_dataset)
eval_df.to_csv(f"{output_dir}/eval.csv", index=False)
