# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import argparse
import os, re, time
import logging
from transformers import BartTokenizer, PegasusTokenizer, BertTokenizer
from utils import Recorder
from model import RankingLoss, BRIO
from config import cnndm_setting, xsum_setting, cnewsum_setting
from tqdm import tqdm
from utils import cut_sent, split_paragraph
from flask import Flask, jsonify, request, url_for, redirect, render_template

logging.getLogger("transformers.tokenization_utils").setLevel(logging.DEBUG)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.DEBUG)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.DEBUG)



parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument("--cuda", default=False, action="store_true", help="use cuda")
parser.add_argument("--gpuid", nargs='+', default=[1], help="gpu ids 0 1 2 3")
parser.add_argument("--model_pt", default="24-02-20-17/model_generation.bin", type=str, help="model path")
parser.add_argument("--config", default="cnewsum", type=str, help="config path")
parser.add_argument("-l", "--log", action="store_true", default=True, help="logging")
parser.add_argument("-p", "--port", type=int, default=9982, help="port")
parser.add_argument('--address', type=str, default='/summary')
args = parser.parse_args()

id = len(os.listdir("./cache"))
recorder = Recorder(id, args.log)

# load config
if args.config == "cnewsum":
    cnewsum_setting(args)

tokenizer = BertTokenizer.from_pretrained(args.model_type)

# build models
model_path = args.pretrained if args.pretrained is not None else args.model_type
model = BRIO(model_path, tokenizer.pad_token_id, args.is_pegasus)

if args.cuda:
    device = f'cuda:{args.gpuid[0]}'
else:
    device = 'cpu'

model = model.to(device)
model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=device))
model.eval()

# evaluate the model as a generator
model.generation_mode()



def predict(content):
    with torch.no_grad():
        dct = tokenizer.batch_encode_plus([content], max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
        summaries = model.generate(
            input_ids=dct["input_ids"].to(device),
            attention_mask=dct["attention_mask"].to(device),
            max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
            min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            early_stopping=True,
            num_beam_groups=1,
            num_return_sequences=1,
        )
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]

    torch.cuda.empty_cache()
    return dec[0].replace(' ', '')



app = Flask(__name__)

@app.route(args.address, methods=['Post'])
def Summary(text_ori):
    text_ori = re.sub("\s+", " ", text_ori).strip()
    text_ori = re.sub("\n", " ", text_ori)
    text_ori = re.sub("\r", " ", text_ori)
    text_ori = re.sub("\r\n", " ", text_ori)
    zhmodel = re.compile(u'[\u4e00-\u9fa5]')
    match = zhmodel.search(text_ori)

    if not match or len(text_ori) <= 5:
        res = text_ori
    else:
        if len(text_ori) > 510:
            text_ori = split_paragraph(text_ori)

        res = predict(text_ori)

    recorder.print(f"content: {text_ori}\ninference result: {res}")

    # return jsonify({'summary': res})
    return res



@app.route('/', methods=['GET', 'POST'])
def login():
    return redirect(url_for('zhaiyao'))


@app.route('/predict', methods=['GET', 'POST'])
def zhaiyao():
    if request.method == 'POST':
        text = request.form['text']  # 获取输入的文本
        prediction = Summary(text)  # 使用模型进行预测
        return render_template('zhaiyao.html', text=text, prediction=prediction)

    return render_template('zhaiyao.html', text='', prediction='')


def main():
    app.run(debug=False, port=args.port, host='0.0.0.0')
    app.config['JSON_AS_ASCII'] = False


if __name__ == '__main__':
    main()
