import gradio as gr
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
from PIL import Image
import os
from glob import glob


import openai
openai.api_key_path = "openai_key.txt"


class GPTAgent:
    def __init__(self, prompt_type):
        self.prompt = pd.read_csv("gpt_prompts.tsv", sep="\t")
        self.prompt = self.prompt[
            self.prompt["prompt_type"] == prompt_type
            ]["prompt_template"].values[0]

    def __call__(self, query):
        # Get the template prompt
        prompt = self.prompt.replace("<Input_Placeholder>", query)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
            )
        output = response["choices"][0]["message"]["content"]
        return output


def generate_floor_plan_descs(query):
    print("Generating floor plan descriptions...")
    agent = GPTAgent("gen_detailed_descs")
    floor_plan_descs = agent(query)
    floor_plan_descs = json.loads(floor_plan_descs)
    floor_plan_descs = list(floor_plan_descs.values())
    return floor_plan_descs


def generate_z3_code(floor_plan_descs):
    placeholder_output = [
        {'room_type': 'bedroom', 'x': 1.0, 'y': 1.0, 'h': 10.0, 'w': 10.0},
        {'room_type': 'kitchen', 'x': 11.0, 'y': 1.0, 'h': 10.0, 'w': 10.0},
        {'room_type': 'living_room', 'x': 11.0, 'y': 11.0, 'h': 10.0, 'w': 10.0},
        {'room_type': 'bathroom', 'x': 1.0, 'y': 11.0, 'h': 10.0, 'w': 10.0}
        ]
    return [placeholder_output for _ in floor_plan_descs]


def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return np.asarray(grid)


def generate_floor_plan_image_from_spec(floor_plan_spec, save_to):
    fig, ax = plt.subplots()#figsize=(10, 10))

    for room in floor_plan_spec:
        x = room["x"]
        y = room["y"] - room["h"]
        h = room["h"]
        w = room["w"]

        # Generate a random color
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        ax.text(x + w/2, y + h/2, room["room_type"], ha='center', va='center', fontsize=8, color=color)

    # Set the limits of the plot
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Room Layout')
    plt.grid(True)
    plt.savefig(save_to)


def generate_images(floor_plans):
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    for i, floor_plan in enumerate(floor_plans):
        save_to = os.path.join(tmp_dir, f"floor_plan_{i}.png")
        generate_floor_plan_image_from_spec(floor_plan, save_to)
    img_paths = glob("tmp/floor_plan_*.png")
    img = image_grid([Image.open(img_path) for img_path in img_paths], 2, 2)
    return img


def e2e_pipeline(text_input):
    # Step 1: Generate detailed floor plans using GPT
    floor_plan_descs = generate_floor_plan_descs(text_input)

    # Step 2: For each floor plan, generate Z3 code using GPT
    # Step 3: Run each SMTK code to get the floor plan specs
    # Step 4: Tight packing of the floor plan specs
    floor_plans = generate_z3_code(floor_plan_descs)

    # Step 5: Generate the floor plan image using the floor plan specs
    img = generate_images(floor_plans)

    # Return image
    return img


with gr.Blocks() as demo:
    # Define the input component (text prompt)
    text_input = gr.inputs.Textbox(lines=3, label="What type of floor plan would you like to generate?")

    # Define the output components (generated image and score)
    image_output = gr.outputs.Image(label="Generated Image", type="numpy")
    # score_output = gr.outputs.Textbox(label="Generated Text Outputs")

    # Create the Gradio interface
    iface = gr.Interface(
        fn=e2e_pipeline,
        inputs=[text_input],
        outputs=[image_output],
        title="Floor Plan Generation App",
        description="Generate floor plans from a request",
        live=False
    )

if __name__ == "__main__":
    demo.launch()