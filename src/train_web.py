from webui import (
    common,
    interface
)

def main():
    demo = interface.create_interface()
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)

if __name__ == "__main__":
    main()
