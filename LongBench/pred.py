import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from transformers import AutoModelForCausalLM
#from transformers.src.transformers.modeling_outputs import BaseModelOutputWithPast
import torch
import tiktoken
import torch.multiprocessing as mp
from multipoleattention.utils.modelutils import shard_model
from transformers import LlamaConfig, AutoConfig
import traceback
from torch import Tensor
from typing import Any, List

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

mp.set_start_method('spawn', force=True)
device = "cuda"

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def generate_pred(
    model_name,
    model,
    tok,
    input_text: str,
    max_tokens: int,
    args,
    barrier,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    if args.inference_tp > 1:
        barrier.wait()

    if input_text == '':
        return

    if 'deepseek' in model_name.lower():
        input_text += "<think>\n"

    max_len = maxlen_map[model_name]
    messages = [{"role": "user", "content": input_text}]

    # check length
    enc = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    enc_ids   = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
    input_ids = enc_ids[0]
    sp_len = input_ids.size(0)

    # truncate
    if sp_len > max_len:
        num_remove = sp_len - max_len
        input_ids_no_chat = tok.encode(input_text, return_tensors="pt")
        half = (len(input_ids_no_chat[0]) - num_remove) // 2
        input_ids_no_chat = torch.cat([input_ids_no_chat[:,:half], input_ids_no_chat[:,-half:]], dim=1)
        input_text = tok.decode(input_ids_no_chat[0], skip_special_tokens=True)

    # Calculate ctx_len before generation.
    messages = [{"role": "user", "content": input_text}]
    chat_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)#, enable_thinking=False)
    input = tok(chat_prompt, return_tensors="pt", truncation=False).to(model.device)
    ctx_len = input.input_ids.shape[-1]

    if  "deepseek" in model_name.lower():
        output = model.generate(
            **input,
            max_new_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            num_beams=1,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )[0]
    else:
        output = model.generate(
            **input,
            max_new_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0,
            do_sample=True,
            num_beams=1,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )[0]
    length = len(output)
    output = tok.decode(output[ctx_len:], skip_special_tokens=True)

    return output, length


def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None


def get_pred(rank, world_size, data, args, out_file, barrier):
    print("Eval worker", rank)
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    model_name = args.model
    if "gpt" in model_name or "o1" in model_name:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_map[model_name], trust_remote_code=True, use_fast=False)

    # Load model
    print("Loading model {}".format(model_name))
    if "chatglm" in model_name or "internlm" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_map[model_name],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device).eval()
    else:
        config = AutoConfig.from_pretrained(model_map[model_name])

        # set attn implementation
        config._flash_attn_2_enabled = True
        config._attn_implementation = "flash_attention_2"
        dtype = torch.bfloat16

        # clustering config parameters
        config.use_centroids = args.use_centroids
        config.use_replacement = args.use_replacement
        config.percentiles_lst = args.percentiles_lst
        config.percent_clusters_lst = args.percent_clusters_lst
        config.inference_tp = args.inference_tp
        config.cluster_interval = args.cluster_interval

        model = AutoModelForCausalLM.from_pretrained(
            model_map[model_name],
            torch_dtype=dtype,
            config=config
        )
        if args.inference_tp > 1:
            print(f"Applying tensor parallelism ({args.inference_tp}) to model on rank {rank}")
            model = shard_model(model, rank, world_size, args.inference_tp, args.port_num)
        model = model.to(device).eval()

    for idx, item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot

        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        max_tokens = 4096

        # Use the generate_pred function from run_eval.py
        output,length = generate_pred(model_name, model, tokenizer, prompt, max_tokens, args, barrier)

        if args.cot: # extract answer
            assert(False) # do not use this w/ reasoning models
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)

            output,length = generate_pred(model_name, model, tokenizer, prompt, 128, args, barrier)

        if output == '':
            continue

        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        item['ctx_length'] = length

        if rank == 0 or args.inference_tp == 1:
            with open(out_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    mp.set_start_method('spawn', force=True)

    os.makedirs(args.save_dir, exist_ok=True)
    print("ARGS: ", args)
    if args.output_file:
        out_file = os.path.join(args.save_dir, args.output_file)
    else:
        model_name_suffix = args.model.split("/")[-1]

        file_parts = [model_name_suffix]

        if args.rag > 0:
            file_parts.append(f"rag_{str(args.rag)}")
        if args.no_context:
            file_parts.append("no_context")
        if args.cot:
            file_parts.append("cot")
        if args.use_centroids:
            file_parts.append("centroids")
        if args.use_replacement:
            file_parts.append("replacement")
        if args.percent_clusters_lst:
            file_parts.append(f"clusters{str(args.percent_clusters_lst)}")
        if args.percentiles_lst:
            file_parts.append(f"lst{str(args.percentiles_lst)}")

        filename = "_".join(file_parts) + ".jsonl"
        out_file = os.path.join(args.save_dir, filename)

        print("Will save to", out_file)

    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    only_short = args.only_short
    if only_short:
        data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset if item["length"] == 'short']
    else:
        data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    world_size = torch.cuda.device_count()
    assert world_size >= args.n_proc, "must have at least enough gpus as processes"

    data_all_with_indices = list(enumerate(data))

    print("inference_tp={}".format(args.inference_tp))
    if args.inference_tp == 1:
        data_subsets = [data_all_with_indices[i::args.n_proc] for i in range(args.n_proc)]
    else:
        data_subsets = [data_all_with_indices for i in range(args.n_proc)]
        assert args.n_proc >= args.inference_tp, "must have at least enough gpus as inference tp"

    barrier = mp.Barrier(args.inference_tp)

    fout = open(out_file, 'a', encoding='utf-8')
    processes = []

    print("Starting ", args.n_proc, " processes")
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(rank, args.n_proc, data_subsets[rank], args, out_file, barrier))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--output_file", "-o", type=str, default=None, help="Specify custom output file name")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--cot", "-cot", action='store_true')
    parser.add_argument("--no_context", "-nc", action='store_true')
    parser.add_argument("--rag", "-rag", type=int, default=0)
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--port_num", "-p", type=int, default=29500)
    parser.add_argument("--inference_tp", "-t", type=int, default=1)

    parser.add_argument("--use_centroids", action="store_true", default=False)
    parser.add_argument("--use_replacement", action="store_true", default=False)
    parser.add_argument("--percent_clusters_lst", type=float, nargs="+", default=[])
    parser.add_argument("--stop_idx", type=int, help="Specify the stop index for debugging on a subset of the full dataset", default=None)
    parser.add_argument("--percentiles_lst", nargs="+", type=float, default=[])
    parser.add_argument("--cluster_interval", type=int, default=1024)

    # only short eval (qwen3)
    parser.add_argument("--only_short", action="store_true", default=False)

    args = parser.parse_args()
    main()
