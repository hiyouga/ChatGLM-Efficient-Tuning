import json
import datasets
from typing import Any, Dict, List

_DESCRIPTION = "An huatuo of dataset for ChatGLM."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = "FreedomIntelligence/HuatuoGPT-sft-data-v1"


class ExampleDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "instruction": datasets.Value("string"),
            "input": datasets.Value("string"),
            "output": datasets.Value("string")
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_path = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_path
                }
            )
        ]

    def _generate_examples(self, filepath: str) -> Dict[int, Dict[str, Any]]:
        with open(filepath, "r", encoding="utf-8") as f:
            for key, jsonObj in enumerate(f):
                dataset = json.loads(jsonObj)
                data = dataset["data"]
                query = data[0].strip()[2:]
                response = data[1].strip()[2:]
                yield key, {
                    "instruction": query,
                    "input": "",
                    "output": response
                }
