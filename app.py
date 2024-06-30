
import os
import sys

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

import streamlit as st
import json
import os.path as osp
from typing import Union

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

print(device)

class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False

class Iteratorize:
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        template_name = "alpaca"
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"
        }
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


base_model      = "D:\project joub\EXPORTMODEL"
lora_weights    = None
load_8bit       = False

prompt_template = ""
server_name     = "0.0.0.0"
share_gradio    = True

prompter  = Prompter(prompt_template)
tokenizer = LlamaTokenizer.from_pretrained(base_model)

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload/"
    )
    if lora_weights:
      model = PeftModel.from_pretrained(
          model,
          lora_weights,
          torch_dtype=torch.float16,
          offload_folder="offload/"
      )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    if lora_weights:
      model = PeftModel.from_pretrained(
          model,
          lora_weights,
          device_map={"": device},
          torch_dtype=torch.float16,
      )
else:
    model = LlamaForCausalLM.from_pretrained(
        base_model, device_map={"": device}, low_cpu_mem_usage=True
    )
    if lora_weights:
      model = PeftModel.from_pretrained(
          model,
          lora_weights,
          device_map={"": device},
      )

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

if not load_8bit:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

def text_evaluate(
      instruction,
      input=None,
      temperature=0.1,
      top_p=0.75,
      top_k=40,
      num_beams=1,
      repetition_penalty=2,
      no_repeat_ngram=5,
      max_new_tokens=500,
      stream_output=False,
      **kwargs,
  ):
      prompt    = prompter.generate_prompt(instruction, input)
      inputs    = tokenizer(prompt, return_tensors="pt")
      input_ids = inputs["input_ids"].to(device)

      generation_config = GenerationConfig(
          temperature=temperature,
          top_p=top_p,
          top_k=top_k,
          num_beams=num_beams,
          **kwargs,
      )

      generate_params = {
          "input_ids": input_ids,
          "generation_config": generation_config,
          "return_dict_in_generate": True,
          "output_scores": True,
          "max_new_tokens": max_new_tokens,
      }

      # Without streaming
      with torch.no_grad():
          generation_output = model.generate(
              input_ids=input_ids,
              generation_config=generation_config,
              return_dict_in_generate=True,
              output_scores=True,
              max_new_tokens=max_new_tokens,
          )
      s = generation_output.sequences[0]
      return tokenizer.decode(s).split("### Response:")[1].strip()
# Define your chatbot logic
def format_chat_prompt(message, chat_history):
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"

    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history):
    formatted_prompt = format_chat_prompt(message, chat_history)

    # Replace this with your actual bot response logic
    bot_message = text_evaluate(message)  # Replace with your text_evaluate logic

    chat_history.append((message, bot_message))
    return bot_message, chat_history

# Streamlit UI code
def main():
    st.title("🇹🇭 chatBot by TANNY")

    chat_history = []
    message = st.text_input("Prompt", key="input_message")

    if st.button("Submit") or message.endswith('\n'):
        if message:
            bot_message, chat_history = respond(message, chat_history)
            st.text_area("Chat History", value=format_chat_prompt(message, chat_history), height=200)
            st.text_area("Assistant", value=f"Assistant: {bot_message}", height=100)
        else:
            st.warning("Please enter a message to submit.")

if __name__ == "__main__":
    main()
