import gradio as gr
from typing import Dict, List
from gradio.components import Component

from glmtuner.webui.locales import LOCALES


class Manager:

    def __init__(self, elem_list: List[Dict[str, Component]]):
        self.elem_list = elem_list

    def gen_label(self, lang: str) -> Dict[Component, dict]:
        update_dict = {}
        for elems in self.elem_list:
            for name, component in elems.items():
                update_dict[component] = gr.update(**LOCALES[name][lang])
        return update_dict
