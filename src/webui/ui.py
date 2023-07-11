import gradio as gr

class ToolButton(gr.Button, gr.components.IOComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_block_name(self):
        return "button"
    
refresh_symbol = '🔄'
def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_class):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_classes=elem_class)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )

    return refresh_button
