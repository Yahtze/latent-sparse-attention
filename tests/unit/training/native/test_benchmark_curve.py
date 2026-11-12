import argparse
from pathlib import Path

import pytest

from lsa import io
from lsa.training.native.benchmark import curve, records, tasks
from lsa.training.native.benchmark import run as benchmark

ROOT = Path(__file__).resolve().parents[4]
SWEEP = ROOT / "configs/training/native/qwen35_0_8b_gpt2/sweep_released_lsa_k2048.yaml"


def write_run(root, label, preset, kind, top_k, checkpoint, scores):
    folder = records.run_folder(root, label, preset)
    io.write_json(
        folder / records.CONFIG,
        {"checkpoint": checkpoint, "attention": {"kind": kind, "top_k": top_k}},
    )
    io.write_json(folder / records.RESULTS, {"results": {}})
    for task, score in scores.items():
        io.write_jsonl(
            folder / records.SAMPLES / f"{task}.jsonl",
            [{"doc_id": i, "metrics": {"score": s}} for i, s in enumerate(score)],
        )


def test_a_point_is_the_mean_over_tasks_of_each_tasks_mean(tmp_path):
    preset = "ruler_vt_4096+ruler_cwe_4096"
    scores = {"ruler_vt_4096": [1.0], "ruler_cwe_4096": [0.0, 0.5, 0.5]}
    write_run(tmp_path, "a", preset, "lsa", 256, "/x/r/checkpoint/step-1", scores)

    folder = records.run_folder(tmp_path, "a", preset)

    assert curve.preset_score(folder, ["ruler_vt_4096", "ruler_cwe_4096"]) == pytest.approx((1.0 + 1 / 3) / 2)


def test_the_long_presets_are_the_13_tasks_at_one_length_each():
    for preset, length in [("ruler-64k", 65536), ("ruler-128k", 131072)]:
        names = tasks.expand(preset)
        assert len(names) == 13
        assert {tasks.parse_ruler(name)[1] for name in names} == {length}


def test_a_sweep_is_one_line_per_trained_checkpoint_and_the_dense_run_is_the_baseline(tmp_path):
    task = "ruler_vt_4096"
    for label, k, trained, score in [
        ("lsa-2048_top_k256", 256, "lsa-2048", 0.2),
        ("lsa-2048_top_k4096", 4096, "lsa-2048", 0.8),
        ("lsa-1024_top_k256", 256, "lsa-1024", 0.1),
    ]:
        write_run(tmp_path, label, task, "lsa", k, f"/x/{trained}/checkpoint/step-1", {task: [score]})
    write_run(tmp_path, "dense", task, "dense", 0, "/x/dense/checkpoint/step-1", {task: [0.9]})

    lines, dense = curve.collect(tmp_path, task, "dense")

    assert lines == {"lsa-2048": {256: 0.2, 4096: 0.8}, "lsa-1024": {256: 0.1}}
    assert dense == 0.9


@pytest.mark.parametrize(
    ("label", "expected"),
    [("decay128k-lsa-2048", 2048), ("decay128k-lsa-1024", 1024), ("decay128k-dense", None)],
)
def test_the_trained_k_is_the_number_a_run_name_ends_in(label, expected):
    assert curve.trained_k(label) == expected


def test_the_table_leaves_a_missing_point_blank():
    text = curve.table({"a": {256: 0.5, 512: 0.6}, "b": {256: 0.4}}, 0.9)

    assert text.splitlines()[2].split(" | ") == ["512  ", "0.600", "-    ", "0.900"]


def test_the_plot_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "plots" / "curve.png"

    curve.render({"a": {256: 0.5, 512: 0.6}}, 0.9, "ruler-4k", "dense", out)

    assert out.stat().st_size > 0


def command_args(*arguments):
    parser = argparse.ArgumentParser()
    benchmark.add_arguments(parser)
    return parser.parse_args([str(SWEEP), "--tasks", "ruler-4k", "--dry-run", *arguments])


def test_a_sweep_files_each_point_under_its_run_name(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LSA_ROOT", str(tmp_path))
    monkeypatch.setattr(benchmark.tasks, "check_ruler_tokenizer", lambda names, tokenizer: None)

    benchmark.command(command_args())

    out = capsys.readouterr().out
    for k in (256, 512, 1024, 2048, 4096, 8192):
        assert f"_top_k{k}: step 5866" in out
    assert out.count("released-decay128k-lsa-2048") == 6


def test_a_label_for_a_sweep_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("LSA_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="--label x would file 6 sweep points"):
        benchmark.command(command_args("--label", "x"))
