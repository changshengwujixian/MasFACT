from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

Record = Dict[str, Any]
Normalizer = Callable[[Record, str], Record]


@dataclass
class HFSource:
    key: str
    family: str
    stage_name: str
    hf_path: str
    subset: Optional[str] = None
    train_split: str = "train"
    test_split: Optional[str] = None
    label_field: Optional[str] = None
    normalizer: str = "generic"
    extension: str = "jsonl"
    stage: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassProtocol:
    key: str
    family: str
    source_key: str
    target_name: str
    label_field: str
    stages: List[List[str]]
    fine_grained: bool = False
    extension: str = "jsonl"


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _first(record: Record, names: Sequence[str], default: Any = "") -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def normalize_generic(record: Record, source: str) -> Record:
    question = _first(record, ["question", "query", "problem", "prompt", "input", "sentence"], "")
    answer = _first(record, ["answer", "target", "label", "final_answer", "output", "solution"], "")
    normalized = dict(record)
    normalized.setdefault("question", _stringify(question))
    normalized.setdefault("answer", _stringify(answer))
    normalized.setdefault("src", source)
    return normalized


def normalize_qa(record: Record, source: str) -> Record:
    normalized = normalize_generic(record, source)
    choices = _first(record, ["options", "choices", "A", "alternatives"], None)
    if choices is not None:
        normalized["options"] = choices
    category = _first(record, ["category", "subject", "domain", "task", "discipline"], None)
    if category:
        normalized["category"] = str(category).lower()
    return normalized


def normalize_math(record: Record, source: str) -> Record:
    normalized = normalize_generic(record, source)
    normalized["solution"] = _stringify(_first(record, ["solution", "rationale", "explanation"], normalized.get("answer", "")))
    math_type = _first(record, ["type", "subject", "category", "level"], None)
    if math_type:
        normalized["type"] = str(math_type)
    normalized.setdefault("domain", "math")
    return normalized


def normalize_code(record: Record, source: str) -> Record:
    normalized = normalize_generic(record, source)
    normalized["solutions"] = _first(record, ["solutions", "canonical_solution", "solution", "code"], normalized.get("answer", ""))
    normalized["starter_code"] = _first(record, ["starter_code", "starter", "prompt"], "")
    normalized["input_output"] = _first(record, ["input_output", "tests", "public_tests"], "")
    skill = _first(record, ["skill_types", "skills", "tags", "difficulty", "topic"], None)
    if skill is not None:
        normalized["skill_types"] = skill
    return normalized


def normalize_multihop(record: Record, source: str) -> Record:
    normalized = normalize_generic(record, source)
    normalized["context"] = _first(record, ["context", "paragraphs", "evidence", "documents"], [])
    qa_type = _first(record, ["type", "question_type", "category"], None)
    if qa_type:
        normalized["type"] = str(qa_type)
    return normalized


NORMALIZERS: Dict[str, Normalizer] = {
    "generic": normalize_generic,
    "qa": normalize_qa,
    "math": normalize_math,
    "code": normalize_code,
    "multihop": normalize_multihop,
}


SOURCES: Dict[str, HFSource] = {
    "mmlu": HFSource("mmlu", "Knowledge QA", "mmlu", "cais/mmlu", "all", "test", "validation", "subject", "qa", "jsonl", 1),
    "mmlu_pro": HFSource("mmlu_pro", "Knowledge QA", "mmlupro", "TIGER-Lab/MMLU-Pro", None, "test", "validation", "category", "qa", "jsonl", 2),
    "agieval": HFSource("agieval", "Knowledge QA", "agieval", "hails/agieval", None, "train", "test", "task", "qa", "jsonl", 3),
    "gpqa": HFSource("gpqa", "Knowledge QA", "gpqa", "Idavidrein/gpqa", "gpqa_main", "train", None, "high_level_domain", "qa", "jsonl", 4),
    "gsm8k": HFSource("gsm8k", "Math", "gsm8k", "openai/gsm8k", "main", "train", "test", "type", "math", "jsonl", 1, {"difficulty": "basic"}),
    "math": HFSource("math", "Math", "MATH", "hendrycks/competition_math", None, "train", "test", "type", "math", "jsonl", 2, {"difficulty": "advanced"}),
    "olympiad": HFSource("olympiad", "Math", "Olym", "Hothan/OlympiadBench", None, "train", "test", "subject", "math", "jsonl", 3, {"difficulty": "olympiad"}),
    "theoremqa": HFSource("theoremqa", "Math", "TheoremQA", "TIGER-Lab/TheoremQA", None, "train", "test", "type", "math", "jsonl", 4, {"difficulty": "theorem"}),
    "apps": HFSource("apps", "Code", "APPS", "codeparrot/apps", None, "train", "test", "difficulty", "code", "jsonl", 1),
    "humaneval": HFSource("humaneval", "Code", "humaneval", "openai_humaneval", None, "test", None, "task_id", "code", "jsonl", 2),
    "livecodebench": HFSource("livecodebench", "Code", "livecodebench", "livecodebench/code_generation_lite", None, "test", None, "difficulty", "code", "jsonl", 3),
    "mbpp": HFSource("mbpp", "Code", "mbpp", "google-research-datasets/mbpp", "full", "train", "test", "source_file", "code", "jsonl", 4),
    "taco": HFSource("taco", "Code", "TACO", "BAAI/TACO", None, "train", "test", "skill_types", "code", "jsonl", 1),
    "hotpotqa": HFSource("hotpotqa", "RAG-Multi-hop QA", "hotpotqa", "hotpotqa/hotpot_qa", "fullwiki", "train", "validation", "type", "multihop", "json", 1),
    "twowiki": HFSource("twowiki", "RAG-Multi-hop QA", "2Wiki", "voidful/2WikiMultihopQA", None, "train", "validation", "type", "multihop", "json", 2),
    "musique": HFSource("musique", "RAG-Multi-hop QA", "musique", "dgslibisey/MuSiQue", None, "train", "validation", "type", "multihop", "jsonl", 3),
    "strategyqa": HFSource("strategyqa", "RAG-Multi-hop QA", "strategyqa", "ChilleD/StrategyQA", None, "train", "test", "type", "multihop", "json", 4),
}


CLASS_PROTOCOLS: Dict[str, ClassProtocol] = {
    "mmlu_pro_coarse": ClassProtocol("mmlu_pro_coarse", "Knowledge QA", "mmlu_pro", "Class Increment(MMLU-pro)", "category", [["physics", "chemistry", "biology", "math", "computer science"], ["health", "engineering", "business", "law", "economics"], ["psychology", "history", "philosophy"]]),
    "mmlu_pro_fine": ClassProtocol("mmlu_pro_fine", "Knowledge QA", "mmlu_pro", "Class Increment FineGrained(MMLU-pro)", "category", [["physics"], ["chemistry"], ["biology"], ["math"], ["computer science"], ["health"], ["engineering"], ["business"], ["law"], ["economics"], ["psychology"], ["history"], ["philosophy"]], True),
    "math_coarse": ClassProtocol("math_coarse", "Math", "math", "Class Increment(MATH)", "type", [["Algebra", "Prealgebra", "Intermediate Algebra"], ["Geometry", "Precalculus"], ["Counting & Probability", "Number Theory"]]),
    "math_fine": ClassProtocol("math_fine", "Math", "math", "Class Increment FineGrained(MATH)", "type", [["Algebra"], ["Prealgebra"], ["Intermediate Algebra"], ["Geometry"], ["Precalculus"], ["Counting & Probability"], ["Number Theory"]], True),
    "taco_coarse": ClassProtocol("taco_coarse", "Code", "taco", "Class Incremental(TACO)", "skill_types", [["Data structures", "Range queries", "Amortized analysis"], ["Dynamic programming", "Greedy algorithms"], ["Sorting", "Complete search", "Bit manipulation"]]),
    "taco_fine": ClassProtocol("taco_fine", "Code", "taco", "Class Incremental FineGrained(TACO)", "skill_types", [["Data structures"], ["Range queries"], ["Amortized analysis"], ["Dynamic programming"], ["Greedy algorithms"], ["Sorting"], ["Complete search"], ["Bit manipulation"]], True),
    "twowiki_class": ClassProtocol("twowiki_class", "RAG-Multi-hop QA", "twowiki", "Class Increment(2Wiki)", "type", [["comparison"], ["bridge_comparison"], ["inference"], ["compositional"]], False, "json"),
}


DOMAIN_PROTOCOLS: Dict[str, List[str]] = {
    "Knowledge QA": ["mmlu", "mmlu_pro", "agieval", "gpqa"],
    "Math": ["gsm8k", "math", "olympiad", "theoremqa"],
    "Code": ["apps", "humaneval", "livecodebench", "mbpp"],
    "RAG-Multi-hop QA": ["hotpotqa", "twowiki", "musique", "strategyqa"],
}


def load_split(source: HFSource, split: str) -> List[Record]:
    if load_dataset is None:
        raise RuntimeError("The datasets package is required to download HuggingFace datasets.")
    data = load_dataset(source.hf_path, source.subset, split=split) if source.subset else load_dataset(source.hf_path, split=split)
    normalize = NORMALIZERS[source.normalizer]
    output = []
    for record in data:
        normalized = normalize(dict(record), source.stage_name)
        normalized.update(source.metadata)
        normalized["src"] = source.stage_name
        normalized["task_family"] = source.family
        output.append(normalized)
    return output


def deterministic_split(records: List[Record], seed: int, test_ratio: float) -> tuple[List[Record], List[Record]]:
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_ratio))) if len(shuffled) > 1 else 0
    return shuffled[n_test:], shuffled[:n_test]


def load_source_records(source: HFSource, seed: int, test_ratio: float) -> tuple[List[Record], List[Record]]:
    train = load_split(source, source.train_split)
    if source.test_split:
        test = load_split(source, source.test_split)
        return train, test
    return deterministic_split(train, seed + source.stage, test_ratio)


def read_records(path: Path) -> List[Record]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else list(data.values())


def write_records(path: Path, records: Iterable[Record]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if path.suffix == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    else:
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def parse_labels(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except Exception:
                return [text]
        return [text]
    return [str(value)]


def prepare_raw_sources(root: Path, source_keys: Sequence[str], seed: int, test_ratio: float, overwrite: bool) -> Dict[str, Dict[str, Path]]:
    produced = {}
    for key in source_keys:
        source = SOURCES[key]
        family_root = root / source.family
        raw_dir = family_root / "raw_hf" / key
        train_path = raw_dir / f"train.{source.extension}"
        test_path = raw_dir / f"test.{source.extension}"
        if not overwrite and train_path.exists() and test_path.exists():
            produced[key] = {"train": train_path, "test": test_path}
            continue
        train, test = load_source_records(source, seed, test_ratio)
        write_records(train_path, train)
        write_records(test_path, test)
        metadata = {"source": source.hf_path, "subset": source.subset, "train_split": source.train_split, "test_split": source.test_split, "train_count": len(train), "test_count": len(test), "family": source.family, "stage_name": source.stage_name}
        with (raw_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        produced[key] = {"train": train_path, "test": test_path}
    return produced


def build_domain_protocol(root: Path, family: str, source_paths: Dict[str, Dict[str, Path]]) -> Dict[str, Any]:
    target = root / family / ("Domain Incremental" if family in {"Code", "RAG-Multi-hop QA"} else "Domain Increment")
    metadata = {"family": family, "protocol": "domain_incremental", "stages": []}
    for stage, key in enumerate(DOMAIN_PROTOCOLS[family], start=1):
        source = SOURCES[key]
        train = read_records(source_paths[key]["train"])
        test = read_records(source_paths[key]["test"])
        for record in train + test:
            record["stage"] = stage
            record["domain_label"] = f"{family.lower().replace(' ', '_')}_domain_{stage}"
        if family == "RAG-Multi-hop QA":
            stage_dir = target / f"stage_{stage}"
            train_name = stage_dir / "train.json"
            test_name = stage_dir / "test.json"
        else:
            train_name = target / f"domain{stage}_{source.stage_name}_train.{source.extension}"
            test_name = target / f"domain{stage}_{source.stage_name}_test.{source.extension}"
        train_count = write_records(train_name, train)
        test_count = write_records(test_name, test)
        metadata["stages"].append({"stage": stage, "source_key": key, "stage_name": source.stage_name, "train_count": train_count, "test_count": test_count})
    with (target / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def record_matches(record: Record, field: str, labels: Sequence[str]) -> bool:
    values = set(parse_labels(record.get(field)))
    return bool(values.intersection(set(labels)))


def build_class_protocol(root: Path, protocol: ClassProtocol, source_paths: Dict[str, Dict[str, Path]]) -> Dict[str, Any]:
    target = root / protocol.family / protocol.target_name
    source = SOURCES[protocol.source_key]
    train_records = read_records(source_paths[protocol.source_key]["train"])
    test_records = read_records(source_paths[protocol.source_key]["test"])
    metadata = {"family": protocol.family, "protocol": "class_incremental_fine" if protocol.fine_grained else "class_incremental", "source_key": protocol.source_key, "label_field": protocol.label_field, "stages": []}
    ext = protocol.extension
    for stage, labels in enumerate(protocol.stages, start=1):
        train = []
        test = []
        for record in train_records:
            if record_matches(record, protocol.label_field, labels):
                item = dict(record)
                item["stage"] = stage
                item["class_labels"] = labels
                train.append(item)
        for record in test_records:
            if record_matches(record, protocol.label_field, labels):
                item = dict(record)
                item["stage"] = stage
                item["class_labels"] = labels
                test.append(item)
        train_count = write_records(target / f"class{stage}_train.{ext}", train)
        test_count = write_records(target / f"class{stage}_test.{ext}", test)
        metadata["stages"].append({"stage": stage, "labels": labels, "train_count": train_count, "test_count": test_count})
    with (target / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def build_task_protocol(root: Path, source_paths: Dict[str, Dict[str, Path]]) -> Dict[str, Any]:
    target = root / "Task Incremental"
    task_sources = {
        "Knowledge QA": ["mmlu", "mmlu_pro", "agieval", "gpqa"],
        "Math Problem": ["gsm8k", "math", "olympiad", "theoremqa"],
        "Code": ["apps", "humaneval", "livecodebench", "mbpp"],
        "RAG-Multi-hop QA": ["hotpotqa", "twowiki", "musique", "strategyqa"],
    }
    metadata = {"protocol": "task_incremental", "stages": []}
    for stage, (task_name, keys) in enumerate(task_sources.items(), start=1):
        train = []
        test = []
        for key in keys:
            for item in read_records(source_paths[key]["train"]):
                item = dict(item)
                item["stage"] = stage
                item["task_label"] = task_name
                train.append(item)
            for item in read_records(source_paths[key]["test"]):
                item = dict(item)
                item["stage"] = stage
                item["task_label"] = task_name
                test.append(item)
        stage_dir = target / task_name
        train_count = write_records(stage_dir / f"{task_name.lower().replace(' ', '_').replace('-', '_')}_train.jsonl", train)
        test_count = write_records(stage_dir / f"{task_name.lower().replace(' ', '_').replace('-', '_')}_test.jsonl", test)
        metadata["stages"].append({"stage": stage, "task_name": task_name, "train_count": train_count, "test_count": test_count})
    with (target / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def build_protocol(root: Path, seed: int, test_ratio: float, overwrite: bool, include_task: bool) -> Dict[str, Any]:
    source_keys = sorted(SOURCES)
    source_paths = prepare_raw_sources(root, source_keys, seed, test_ratio, overwrite)
    summary = {"root": str(root), "seed": seed, "domain_protocols": {}, "class_protocols": {}, "task_protocol": None}
    for family in DOMAIN_PROTOCOLS:
        summary["domain_protocols"][family] = build_domain_protocol(root, family, source_paths)
    for key, protocol in CLASS_PROTOCOLS.items():
        summary["class_protocols"][key] = build_class_protocol(root, protocol, source_paths)
    if include_task:
        summary["task_protocol"] = build_task_protocol(root, source_paths)
    with (root / "masfact_protocol_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hierarchical continual MAS evaluation protocol files from HuggingFace datasets.")
    parser.add_argument("--output-root", type=Path, default=Path("CL-dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-task-protocol", action="store_true")
    args = parser.parse_args()
    build_protocol(args.output_root, args.seed, args.test_ratio, args.overwrite, not args.no_task_protocol)


if __name__ == "__main__":
    main()
