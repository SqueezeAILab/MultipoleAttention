from model_handler_local import ModelHandler
from no_rag_pipeline import NoRAGPipeline
from transformers import AutoConfig
import os
import json
import argparse
import torch
import torch.multiprocessing as mp
from datasets import Dataset, load_dataset, concatenate_datasets
mp.set_start_method("spawn", force=True)

def dump_dict_to_json(data, filename):
    """Dumps a Python dictionary or list of dicts to a JSON file, creating the directory if needed."""
    try:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            print(f"Successfully dumped results to {filename}")
    except (TypeError, OSError) as e:
        print(f"Error dumping results to JSON: {e}")


def worker(rank, world_size, examples, args):
    # Process examples
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    # Load model config and map
    model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
    model_path = model_map[args.model_name]
    config = AutoConfig.from_pretrained(model_path)
    # enable flash attention
    config._flash_attn_2_enabled = True
    config._attn_implementation = "flash_attention_2"
    # clustering flags
    config.use_centroids = args.use_centroids
    config.use_replacement = args.use_replacement
    config.percentiles_lst = args.percentiles_lst
    config.percent_clusters_lst = args.percent_clusters_lst
    config.inference_tp = args.inference_tp
    config.cluster_interval = args.cluster_interval

    # Initialize model handler and pipeline
    model_handler = ModelHandler(
        model_path=model_path,
        config=config,
        max_tokens=args.max_tokens,
        device=device,
        rank=rank
    )
    pipeline = NoRAGPipeline(
        model_handler=model_handler,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    # collect all queries for this shard
    queries = []
    for example in examples:
        for _ in range(args.num_samples):
            queries.append(example['messages'])

    # run pipeline once (tqdm internal)
    replies = pipeline.process_batch(queries=queries, max_workers=args.batch_size)

    # reconstruct per-example replies
    results = []
    for idx, example in enumerate(examples):
        start = idx * args.num_samples
        end = start + args.num_samples
        out = dict(example)
        out['replies'] = replies[start:end]
        for fld in ('problem', 'question', 'messages'):
            out.pop(fld, None)
        results.append(out)

    # write per-GPU output
    output_dir = f"{args.save_dataset}-{args.save_name}_results"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"results_gpu{rank}.jsonl")
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Parallel eval across GPUs")
    # reuse all original args
    parser.add_argument('--save-name', type=str, default="base")
    parser.add_argument('--save-dataset', type=str, default="base")
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--model-name', type=str, dest='model_name', required=True)
    parser.add_argument('--backend-type', type=str, default="openai")
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--max-tokens', type=int, default=3072)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--length', type=str, default="0")
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--filter-config', type=json.loads, help='Filter config JSON string')
    parser.add_argument('--op-range', type=str, help='Single int or comma-separated ints')
    parser.add_argument("--use_centroids", action="store_true")
    parser.add_argument("--use_replacement", action="store_true")
    parser.add_argument("--percent_clusters_lst", type=float, nargs="+", default=[])
    parser.add_argument("--percentiles_lst", nargs="+", type=float, default=[])
    parser.add_argument("--cluster_interval", type=int, default=1024)
    parser.add_argument("--inference_tp", type=int, default=1)
    args = parser.parse_args()

    # parse op-range into list of ints
    if args.op_range:
        try:
            args.op_range = [int(args.op_range)]
        except ValueError:
            args.op_range = [int(x) for x in args.op_range.split(',')]
    subsets = [f"ops_{x}" for x in args.op_range]

    # load and filter dataset
    full_dataset = load_dataset(f"{args.dataset_name}_{args.length}")
    if args.filter_config:
        filtered = []
        for split in subsets:
            ds_split = full_dataset[split]
            total = min(args.limit, len(ds_split))
            temp = []
            for cfg in args.filter_config:
                count = int(total * cfg["percentage"])
                criteria = {k: v for k, v in cfg.items() if k != 'percentage'}
                subset = ds_split.filter(lambda ex: all(ex[k]==v for k,v in criteria.items()))
                temp.extend(subset.select(range(min(count, len(subset)))))
            filtered.append(Dataset.from_list(temp))
        unprocessed = concatenate_datasets(filtered)
    else:
        splits = []
        for split in subsets:
            ds = full_dataset[split]
            splits.append(ds.select(range(min(args.limit, len(ds)))))
        unprocessed = concatenate_datasets(splits)

    examples = list(unprocessed)
    world_size = torch.cuda.device_count()
    assert world_size > 0, "No GPUs available"

    # split across GPUs
    shards = [[] for _ in range(world_size)]
    for idx, ex in enumerate(examples):
        shards[idx % world_size].append(ex)

    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, shards[rank], args))
        p.start()
        procs.append(p)
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("Received Ctrl+C, terminating worker processes")
        for p in procs:
            p.terminate()
        for p in procs:
            p.join()

    # merge results
    merged = []
    out_dir = f"{args.save_dataset}-{args.save_name}_results"
    for rank in range(world_size):
        path = os.path.join(out_dir, f"results_gpu{rank}.jsonl")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                merged.append(json.loads(line))

    # dump merged
    merged_file = os.path.join(out_dir, f"{args.save_dataset}-{args.save_name}_{args.length}.json")
    dump_dict_to_json(merged, merged_file)


if __name__ == '__main__':
    main()
