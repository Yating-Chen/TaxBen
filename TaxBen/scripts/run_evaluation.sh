TaxBen_path='/root/TaxFBen'
export PYTHONPATH="$TaxBen_path/src:$TaxBen_path/src/tax-evaluation:$TaxBen_path/src/metrics/BARTScore"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"
#cd ..
#python src/eval.py \
#   --model "gpt-3.5-turbo" \
#   --tasks TaxBen_TaxRecite \
#   --no_cache \
#   --batch_size 2 \
#   --write_out \
cd ..
python src/eval.py \
    --model hf-causal-vllm \
    --tasks TaxBen_TaxRecite \
    --model_args use_accelerate=True,pretrained=llama-2-7b-chat-hf,tokenizer=llama-2-7b-chat-hf,use_fast=False,max_gen_toks=1024,dtype=float16 \
    --no_cache \
    --batch_size 2 \
    --write_out 