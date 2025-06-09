from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import os




def get_model_processor(model_dir, device='cuda:0'):

    target_class = Qwen2_5_VLForConditionalGeneration


    model = target_class.from_pretrained(
        model_dir, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    # default processer
    processor = AutoProcessor.from_pretrained(
        model_dir, 
        # # if not enough memory:
        # min_pixels=min_pixels, 
        # max_pixels=max_pixels,
    )

    return model, processor 




def get_response(image, question: str, model, processor, device='cuda:0'):
    messages = [
        {
            'role': 'system',
            'content': (
                "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE enclosed within <answer> </answer> tags."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1000, do_sample=False, use_cache=True)
    torch.cuda.empty_cache()
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(output_text)


    return output_text[0]






model_dir = 'yangjie-cv/WeThink-Qwen2.5VL-7B'
device = 'cuda:0'
model, processor = get_model_processor(model_dir, device)



image = os.path.join('./assets/example.jpg')
res = get_response(
    image,
    'What is this building and what is its history?',
    model, 
    processor,
    device=device,
)

print(res)

