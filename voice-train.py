import outetts

# Initialize the interface
interface = outetts.Interface(
    config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,
        # For llama.cpp backend
        backend=outetts.Backend.LLAMACPP,
        quantization=outetts.LlamaCppQuantization.FP16
        # For transformers backend
        # backend=outetts.Backend.HF,
    )
)

interface.print_default_speakers()

## Audio must be shorter than 20 seconds, best results if only 15 seconds.
speaker = interface.create_speaker("alexandra-short.wav")

interface.save_speaker(speaker, "alexandra.json")