# Report-format experiment

This optional experiment teaches a consistent Markdown format: **Summary**, **Findings**, and **Next steps**. The app's normal RAG flow already accepts a question that requests these headings; try that before fine-tuning.

The supplied JSONL files contain **12 training examples and 4 held-out evaluation examples** made from fictional operational notes. They match the app's system prompt and document-excerpt layout. Each question explicitly requests the report format. Factual claims include citations, proposed actions are marked as recommendations, and unsupported questions retain the exact missing-information response. These examples demonstrate the pipeline; they are too small to establish a production quality improvement.

No existing documents, conversations, app model settings, or Ollama model weights are changed by these scripts.

## Validate the dataset on this laptop

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe training/train_report.py --validate-only
```

This checks JSON, message roles, report headings, citations, duplicate prompts, and overlap between the training and evaluation splits. It requires only Python and downloads nothing. It does not verify factual accuracy: review examples against their excerpts yourself.

The checked-in dataset is ready to use. To regenerate the fictional examples using the current app prompt:

```powershell
.\.venv\Scripts\python.exe training/prepare_report_data.py
```

Regeneration replaces `training/data/report_train.jsonl` and `report_eval.jsonl`. Store your own examples in other files and pass them using `--train` and `--eval`.

## Train on a compatible GPU

The provided recipe uses **CUDA, NF4 quantization, PEFT LoRA, and TRL's SFTTrainer**. The inspected laptop has Intel UHD graphics and about 15.7 GiB of system RAM, with no detected NVIDIA GPU. This CUDA recipe cannot run here. Other accelerator and CPU approaches require a different recipe; GPU training with this script has not been verified on this machine.

Use a separate environment on a machine with a compatible NVIDIA GPU and enough free VRAM. Install the CUDA-enabled PyTorch build for that machine using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies. Do not replace the app's environment.

```powershell
.\.venv\Scripts\python.exe -m venv .venv-training
$trainPython = '.\.venv-training\Scripts\python.exe'
# Install CUDA PyTorch using the command from the official selector first.
& $trainPython -m pip install -r training/requirements.txt
& $trainPython training/train_report.py --check-env
& $trainPython training/train_report.py --max-steps 20
```

The first training run downloads the original `microsoft/Phi-3-mini-4k-instruct` Hugging Face weights and tokenizer. The quantized Ollama model cache is not a substitute for these training weights. Use `--local-files-only` once the Hugging Face files are cached if you need offline operation. The script does not upload the dataset, adapter, or metrics, and disables training telemetry and external logging integrations.

The script trains only the assistant completion, checks token lengths before loading model weights, and refuses to overwrite a nonempty output directory. It evaluates held-out loss before and after training. Results go into ignored `training/output/report-phi3/`, including the adapter, tokenizer, and `experiment.json` with metrics. Use `--output` to choose a new directory for another run.

## Evaluate and use the result

Compare generated reports before and after training on examples excluded from the training set. Check heading order, factual accuracy, valid citations, and abstention on unsupported questions. A lower loss alone does not prove that report quality improved. The script saves an experimental adapter; it does not automatically deploy it into the app.

To serve the result through Ollama, merge the adapter into the exact matching base model, export a compatible GGUF with the appropriate chat template, and import it following the [Ollama import documentation](https://docs.ollama.com/import). Only after checking its outputs should you set `LLM_MODEL` to the imported local model name and restart the app. This recipe's adapter targets Phi-3; it cannot be applied to Llama or another base model.

Implementation references: [TRL 0.24 SFTTrainer](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer) and [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization).
